import time
import json
import re
import os
from typing import Dict, List, Any, Optional, Tuple
from galileo import GalileoLogger
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
import config
from llm_handler import LLMHandler


class AgentSimulation:
    def __init__(
        self,
        agent_model: str,
        domain: str,
        category: str,
        log_to_galileo: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize the agent simulation.

        Args:
            agent_model: The model name for the agent being evaluated
            domain: The domain for the simulation (e.g., 'banking', 'healthcare')
            category: The category of the simulation (e.g., 'tool_coordination')
            log_to_galileo: Whether to log to Galileo
            verbose: Whether to print verbose logs
        """
        self.agent_model = agent_model
        self.domain = domain
        self.category = category
        self.log_to_galileo = log_to_galileo
        self.verbose = verbose

        # Initialize LLM handler
        self.llm_handler = LLMHandler()

        # Initialize LLMs
        self.agent_llm = self.llm_handler.get_llm(
            model_name=agent_model,
            temperature=config.AGENT_TEMPERATURE,
            max_tokens=config.AGENT_MAX_TOKENS,
        )

        self.simulator_llm = self.llm_handler.get_llm(
            model_name=config.SIMULATOR_MODEL,
            temperature=config.SIMULATOR_TEMPERATURE,
            max_tokens=config.SIMULATOR_MAX_TOKENS,
        )

        # Load data
        self.tools = self._load_tools()
        self.personas = self._load_personas()
        self.scenarios = self._load_scenarios()

        # Initialize Galileo logger
        if log_to_galileo:
            log_stream = config.GALILEO_LOG_STREAM.format(
                domain=domain, category=category, model=agent_model.replace("/", "-")
            )

            if self.verbose:
                print(f"Initializing Galileo logger with stream: {log_stream}")
                print(f"Project: {config.GALILEO_PROJECT}")

            self.galileo_logger = GalileoLogger(
                project=config.GALILEO_PROJECT,
                log_stream=log_stream,
            )

            if self.verbose:
                print(f"Galileo logger initialized successfully")
        else:
            self.galileo_logger = None

    def create_tool_selection_prompt(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
    ) -> str:
        """
        Create a prompt that forces the agent to select appropriate tools.

        Args:
            user_message: The user message to respond to
            conversation_history: The conversation history
            tools: Available tools

        Returns:
            A prompt that will force tool selection
        """
        # Format the available tools
        tool_descriptions = []
        for tool in tools:
            description = tool.get("description", "")
            name = tool.get("title", "")
            parameters = tool.get("properties", {})
            required = tool.get("required", [])

            param_descriptions = []
            for param_name, param_info in parameters.items():
                required_str = "(REQUIRED)" if param_name in required else "(OPTIONAL)"
                param_desc = f"- {param_name}: {param_info.get('description', '')} {required_str}"
                param_descriptions.append(param_desc)

            tool_description = f"""
Tool name: {name}
Description: {description}
Parameters:
{chr(10).join(param_descriptions)}
"""
            tool_descriptions.append(tool_description)

        # Format conversation history
        history_text = ""
        for msg in conversation_history:
            role = msg["role"].upper()
            content = msg["content"]
            history_text += f"{role}: {content}\n\n"

        # Create the prompt
        prompt = f"""You are an advanced AI assistant with access to tools that help you fulfill user requests. Based on the user's message and conversation history, you MUST select the most appropriate tool(s) to use.

AVAILABLE TOOLS:
{chr(10).join(tool_descriptions)}

CONVERSATION HISTORY:
{history_text}

CURRENT USER MESSAGE:
{user_message}

INSTRUCTIONS:
1. Analyze the user's message and conversation history carefully.
2. Identify which tool(s) would be most appropriate to fulfill the user's request.
3. Your response MUST be a valid JSON array containing the selected tool(s) and their parameters.
4. You MUST use this exact format:
[
  {{
    "tool": "tool_name",
    "parameters": {{
      "parameter1": "value1",
      "parameter2": "value2"
    }}
  }}
]
5. DO NOT include any explanations, disclaimers, or other text - ONLY the JSON array.
6. If multiple tools are needed, include all of them in the same array.
7. Make sure to provide values for ALL REQUIRED parameters.
8. If the user is asking about account information, you MUST use get_account_details.
9. If the user wants to transfer money, you MUST use transfer_funds.
10. If the user is interested in credit cards, you MUST use apply_for_credit_card.

Now, return a JSON array with the appropriate tool(s) to address the user's request:"""

        return prompt

    def update_agent_prompt(self, tools: List[Dict[str, Any]]) -> str:
        """
        Create a system prompt for the agent that instructs it to use tools.

        Args:
            tools: List of tools available to the agent

        Returns:
            A system prompt instructing the agent to use tools
        """
        tool_descriptions = []
        for tool in tools:
            description = tool.get("description", "")
            name = tool.get("title", "")
            parameters = tool.get("properties", {})
            required = tool.get("required", [])

            param_descriptions = []
            for param_name, param_info in parameters.items():
                param_desc = f"- {param_name}: {param_info.get('description', '')}"
                if param_name in required:
                    param_desc += " (REQUIRED)"
                param_descriptions.append(param_desc)

            tool_description = f"""
Function: {name}
Description: {description}
Parameters:
{chr(10).join(param_descriptions)}
"""
            tool_descriptions.append(tool_description)

        system_prompt = f"""You are a helpful assistant that can use tools to answer user questions.
You have access to the following tools:

{chr(10).join(tool_descriptions)}

IMPORTANT INSTRUCTIONS:
1. ALWAYS use the tools when appropriate rather than making up information.
2. When you need to use a tool, your entire response must start with a JSON array containing tool calls.
3. Use this exact format for tool calls:
[
  {{
    "tool": "tool_name",
    "parameters": {{
      "parameter1": "value1",
      "parameter2": "value2"
    }}
  }}
]
4. After receiving the tool output, you can respond normally to the user with an explanation.
5. When a user asks you to check account information, make a transfer, etc., ALWAYS use the appropriate tool.
6. Do not apologize for using tools - that's what you're supposed to do.

Let's solve the user's request step by step."""

        return system_prompt

    def _load_tools(self) -> List[Dict[str, Any]]:
        """Load tools for the specified domain."""
        path = config.FILE_PATHS["tools"].format(domain=self.domain)
        with open(path, "r") as f:
            return json.load(f)

    def _load_personas(self) -> List[Dict[str, Any]]:
        """Load personas for the specified domain."""
        path = config.FILE_PATHS["personas"].format(domain=self.domain)
        with open(path, "r") as f:
            return json.load(f)

    def _load_scenarios(self) -> List[Dict[str, Any]]:
        """Load scenarios for the specified domain and category."""
        path = config.FILE_PATHS["scenarios"].format(
            domain=self.domain, category=self.category
        )
        with open(path, "r") as f:
            return json.load(f)

    def _get_tool_by_name(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get tool definition by name."""
        for tool in self.tools:
            if tool.get("title") == tool_name:
                return tool
        return None

    def _extract_tool_calls(self, json_str: str) -> List[Dict[str, Any]]:
        """
        Extract tool calls from a JSON string.

        Args:
            json_str: JSON string containing tool calls

        Returns:
            List of extracted tool calls
        """
        tool_calls = []

        try:
            # Parse the JSON array
            data = json.loads(json_str)

            # Process each tool call
            if isinstance(data, list):
                for item in data:
                    if (
                        isinstance(item, dict)
                        and "tool" in item
                        and "parameters" in item
                    ):
                        tool_calls.append(
                            {"name": item["tool"], "parameters": item["parameters"]}
                        )
        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"Failed to parse JSON: {e}")
                print(f"JSON string: {json_str}")

        return tool_calls

    def select_tools(
        self, user_message: str, conversation_history: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """
        Force the agent to select appropriate tools for the given user message.

        Args:
            user_message: The user message to respond to
            conversation_history: The conversation history

        Returns:
            List of selected tools
        """
        # Create tool selection prompt
        prompt = self.create_tool_selection_prompt(
            user_message=user_message,
            conversation_history=conversation_history,
            tools=self.tools,
        )

        if self.verbose:
            print("Creating tool selection prompt...")
            print(f"Prompt length: {len(prompt)}")

        # Call the agent to select tools
        response = self.agent_llm.invoke([HumanMessage(content=prompt)])

        if self.verbose:
            print("Tool selection response:")
            print(response.content)

        # Extract tool calls
        tool_calls = self._extract_tool_calls(response.content)

        if self.verbose:
            print(f"Extracted {len(tool_calls)} tool calls")
            for i, tool_call in enumerate(tool_calls):
                print(
                    f"Tool call {i+1}: {tool_call['name']} - {json.dumps(tool_call['parameters'])}"
                )

        return tool_calls

    def process_tool_calls(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process a list of tool calls.

        Args:
            tool_calls: List of tool calls to process

        Returns:
            List of processed tool call results
        """
        tool_results = []

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_parameters = tool_call["parameters"]

            # Get tool definition
            tool = self._get_tool_by_name(tool_name)
            if not tool:
                if self.verbose:
                    print(f"Tool '{tool_name}' not found")
                continue

            # Simulate tool execution
            tool_result = self.simulate_tool(tool_name, tool_parameters, tool)

            # Add result to the list
            tool_results.append(tool_result)

        return tool_results

    def simulate_tool(
        self,
        tool_name: str,
        tool_parameters: Dict[str, Any],
        tool_definition: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Simulate tool execution using the simulator LLM.

        Args:
            tool_name: The name of the tool to simulate
            tool_parameters: The parameters for the tool call
            tool_definition: The definition of the tool

        Returns:
            Dictionary containing tool execution results and metadata
        """
        # Get response schema
        response_schema = tool_definition.get("response_schema", {})

        # Create prompt for the tool simulator
        prompt = config.TOOL_SIMULATOR_PROMPT.format(
            tool_name=tool_name,
            tool_parameters=json.dumps(tool_parameters, indent=2),
            response_schema=json.dumps(response_schema, indent=2),
        )

        if self.verbose:
            print(f"Simulating tool: {tool_name}")
            print(f"Parameters: {json.dumps(tool_parameters, indent=2)}")

        # Call the simulator LLM
        start_time = time.time()
        response = self.simulator_llm.invoke([HumanMessage(content=prompt)])
        end_time = time.time()
        tool_duration_ns = int((end_time - start_time) * 1_000_000_000)

        # Parse the response as JSON
        try:
            response_content = response.content
            # Extract JSON content if wrapped in markdown code blocks
            if "```json" in response_content:
                json_match = re.search(r"```json\s*([\s\S]+?)```", response_content)
                if json_match:
                    response_content = json_match.group(1)

            tool_response = json.loads(response_content)

            # Log to Galileo
            if self.galileo_logger:
                try:
                    if self.verbose:
                        print(f"Logging tool call to Galileo: {tool_name}")
                        print(f"Tool parameters: {json.dumps(tool_parameters)}")
                        print(f"Tool response: {json.dumps(tool_response)}")

                    self.galileo_logger.add_tool_span(
                        input=tool_parameters,
                        output=tool_response,
                        name=tool_name,
                        duration_ns=tool_duration_ns,
                        metadata={
                            "tool_name": tool_name,
                            "duration_seconds": str(end_time - start_time),
                        },
                    )

                    if self.verbose:
                        print(f"Successfully logged tool call to Galileo")
                except Exception as e:
                    print(f"Error logging tool call to Galileo: {str(e)}")
                    if self.verbose:
                        import traceback

                        traceback.print_exc()

            # Return comprehensive result
            return {
                "tool_name": tool_name,
                "parameters": tool_parameters,
                "response": tool_response,
                "duration_ns": tool_duration_ns,
            }

        except (json.JSONDecodeError, AttributeError) as e:
            error_response = {
                "error": f"Failed to parse tool response: {str(e)}",
                "raw_response": response.content,
            }

            if self.verbose:
                print(f"Error simulating tool: {str(e)}")

            return {
                "tool_name": tool_name,
                "parameters": tool_parameters,
                "response": error_response,
                "duration_ns": tool_duration_ns,
            }

    def generate_final_response(
        self,
        conversation_history: List[Dict[str, str]],
        user_message: str,
        tool_results: List[Dict[str, Any]],
    ) -> str:
        """
        Generate a final response based on tool results.

        Args:
            conversation_history: The conversation history
            user_message: The user's message
            tool_results: Results from tool execution

        Returns:
            The agent's final response
        """
        # Format the tool results for the prompt
        tool_results_text = ""
        for result in tool_results:
            tool_results_text += f"""
Tool: {result['tool_name']}
Parameters: {json.dumps(result['parameters'], indent=2)}
Response: {json.dumps(result['response'], indent=2)}
"""

        # Create a prompt for the agent to generate a final response
        prompt = f"""Based on the conversation history and the results of the tools you used, 
please provide a helpful response to the user's request.

User's message: {user_message}

Tool results:
{tool_results_text}

Your response should:
1. Clearly explain what information you found using the tools
2. Answer the user's question completely based on the tool results
3. Be helpful, clear, and concise

Your response:"""

        # Format messages for the agent
        messages = []

        # Add conversation history (excluding the last assistant message if it exists)
        for msg in (
            conversation_history[:-2]
            if conversation_history[-2]["role"] == "assistant"
            else conversation_history
        ):
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))

        # Add the prompt with tool results
        messages.append(HumanMessage(content=prompt))

        if self.verbose:
            print("Generating final response using tool results")
            print(f"Tool results count: {len(tool_results)}")

        # Call the agent to generate the final response
        response = self.agent_llm.invoke(messages)

        if self.verbose:
            print(f"Final response generated: {response.content[:200]}...")

        return response.content

    def simulate_user(
        self,
        persona: Dict[str, Any],
        scenario: Dict[str, Any],
        conversation_history: List[Dict[str, str]],
        tool_outputs: List[Dict[str, Any]],
    ) -> str:
        """
        Simulate user response using the simulator LLM.

        Args:
            persona: The persona object for the user
            scenario: The scenario object for the simulation
            conversation_history: The conversation history so far
            tool_outputs: List of tool outputs from previous turns

        Returns:
            The simulated user response
        """
        # Format conversation history for the prompt
        formatted_history = ""
        for msg in conversation_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            formatted_history += f"{role.upper()}: {content}\n\n"

        # Create prompt for the user simulator
        prompt = config.USER_SIMULATOR_PROMPT.format(
            persona_json=json.dumps(persona, indent=2),
            scenario_json=json.dumps(scenario, indent=2),
            conversation_history=formatted_history,
            tool_outputs=json.dumps(tool_outputs, indent=2),
        )

        # Call the simulator LLM
        start_time = time.time()
        response = self.simulator_llm.invoke([HumanMessage(content=prompt)])
        end_time = time.time()

        return response.content

    def run_agent(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
    ) -> str:
        """
        Run the agent LLM with the provided user message and tools.

        Args:
            user_message: The user message to respond to
            conversation_history: The conversation history so far
            tools: The tools available to the agent

        Returns:
            The agent's final response
        """
        if self.verbose:
            print(f"Running agent with {len(tools)} tools")

        # Force tool selection first
        tool_calls = self.select_tools(
            user_message=user_message, conversation_history=conversation_history
        )

        start_time = time.time()

        # If no tools were selected, generate a regular response
        if not tool_calls:
            if self.verbose:
                print("No tools selected, generating regular response")

            # Create system message with instructions to use tools
            system_prompt = self.update_agent_prompt(tools)

            # Format conversation history for the agent
            messages = [SystemMessage(content=system_prompt)]

            # Add conversation history
            for msg in conversation_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))

            # Add the current user message
            messages.append(HumanMessage(content=user_message))

            # Call the agent LLM using the LLMHandler
            response = self.agent_llm.invoke(messages)

            # Extract the response content
            agent_response = response.content
        else:
            if self.verbose:
                print(f"Processing {len(tool_calls)} tool calls")

            # Process the tool calls
            tool_results = self.process_tool_calls(tool_calls)

            # Generate final response using tool results
            agent_response = self.generate_final_response(
                conversation_history=conversation_history,
                user_message=user_message,
                tool_results=tool_results,
            )

        end_time = time.time()
        agent_duration_ns = int((end_time - start_time) * 1_000_000_000)

        # Log LLM call to Galileo
        if self.galileo_logger:
            try:
                if self.verbose:
                    print("Logging LLM call to Galileo")

                self.galileo_logger.add_llm_span(
                    input=user_message,
                    output=agent_response,
                    model=self.agent_model,
                    duration_ns=agent_duration_ns,
                    metadata={
                        "tool_calls_count": str(len(tool_calls)),
                        "has_tool_calls": "true" if tool_calls else "false",
                    },
                )

                if self.verbose:
                    print("Successfully logged LLM call to Galileo")
            except Exception as e:
                print(f"Error logging LLM call to Galileo: {str(e)}")
                if self.verbose:
                    import traceback

                    traceback.print_exc()

        return agent_response

    def run_simulation(self, scenario_idx: int = 0) -> Dict[str, Any]:
        """
        Run a full simulation for a single scenario.

        Args:
            scenario_idx: The index of the scenario to run

        Returns:
            A dictionary with the simulation results
        """
        # Get scenario and persona
        scenario = self.scenarios[scenario_idx]
        persona_idx = scenario.get("persona_index", 0)
        persona = self.personas[persona_idx]

        # Prepare initial conversation history
        conversation_history = []

        # Get initial message from scenario
        if isinstance(scenario.get("first_message"), list):
            # If first_message is a list of messages, add them to conversation history
            conversation_history = scenario["first_message"]
            initial_user_message = conversation_history[-1]["content"]
        else:
            # If first_message is a string, use it as the initial message
            initial_user_message = scenario.get("first_message", "Hello")

        # Start trace in Galileo
        if self.galileo_logger:
            if self.verbose:
                print(f"Starting Galileo trace for scenario {scenario_idx}")
            trace = self.galileo_logger.start_trace(
                input=initial_user_message,
                name=f"{self.domain}-{self.category}-scenario-{scenario_idx}",
                tags=[self.domain, self.category],
                metadata={
                    "domain": self.domain,
                    "category": self.category,
                    "scenario_idx": str(scenario_idx),  # Convert to string
                    "agent_model": self.agent_model,
                },
            )

        # If conversation_history is empty, add initial user message
        if not conversation_history:
            conversation_history.append(
                {"role": "user", "content": initial_user_message}
            )

        # Simulation loop
        turn_count = 0
        tool_outputs = []  # Track tool outputs for the user simulator
        all_tool_results = []  # Track all tool results for return
        while turn_count < config.MAX_TURNS:
            turn_count += 1

            # Get the latest user message
            user_message = next(
                (
                    msg["content"]
                    for msg in reversed(conversation_history)
                    if msg["role"] == "user"
                ),
                initial_user_message,
            )

            # Run the agent
            if self.verbose:
                print(f"Turn {turn_count}: Running agent with user message")

            agent_response = self.run_agent(
                user_message=user_message,
                conversation_history=(
                    conversation_history[:-1]
                    if conversation_history[-1]["role"] == "user"
                    else conversation_history
                ),
                tools=self.tools,
            )

            if self.verbose:
                print(f"Turn {turn_count}: Agent responded")
                print(f"Agent response: {agent_response[:200]}...")

            # Add agent response to conversation history
            conversation_history.append(
                {"role": "assistant", "content": agent_response}
            )

            # Simulate user response
            if self.verbose:
                print(f"Turn {turn_count}: Simulating user response...")

            user_response = self.simulate_user(
                persona=persona,
                scenario=scenario,
                conversation_history=conversation_history,
                tool_outputs=tool_outputs,  # Pass all tool outputs to user simulator
            )

            if self.verbose:
                print(f"Turn {turn_count}: User responded")
                print(f"User response: {user_response[:200]}...")

            # Add user response to conversation history
            conversation_history.append({"role": "user", "content": user_response})

            # Check if conversation should end
            if "CONVERSATION_COMPLETE" in user_response:
                print(f"Conversation complete after {turn_count} turns.")
                break

        # Conclude trace in Galileo
        if self.galileo_logger:
            # Calculate duration as an integer in nanoseconds
            # If trace.created_at is a datetime, convert it to nanoseconds since epoch
            start_time_ns = trace.created_at
            if hasattr(start_time_ns, "timestamp"):  # Check if it's a datetime object
                start_time_ns = int(start_time_ns.timestamp() * 1_000_000_000)

            end_time_ns = int(time.time() * 1_000_000_000)
            duration_ns = end_time_ns - start_time_ns

            if self.verbose:
                print(
                    f"Concluding Galileo trace. Duration: {duration_ns/1_000_000_000:.2f}s"
                )

            self.galileo_logger.conclude(
                output=(
                    conversation_history[-1]["content"]
                    if conversation_history[-1]["role"] == "user"
                    else agent_response
                ),
                duration_ns=duration_ns,
            )

            if self.verbose:
                print(f"Flushing Galileo trace...")

            self.galileo_logger.flush()

        # Return results
        return {
            "conversation": conversation_history,
            "turns": turn_count,
            "domain": self.domain,
            "category": self.category,
            "scenario_idx": scenario_idx,
            "agent_model": self.agent_model,
            "tool_results": all_tool_results,
        }

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
from galileo import galileo_context
from galileo.datasets import get_dataset
from galileo.experiments import run_experiment


def ensure_string(value: Any) -> str:
    """
    Ensure that a value is a string suitable for Galileo logging.

    Args:
        value: Any type of value

    Returns:
        A string representation of the value
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        # For dictionaries, lists, etc. use JSON
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value)
        # For other types, convert to string
        return str(value)
    except Exception as e:
        print(f"Error converting to string: {str(e)}")
        return f"[Unconvertible data: {type(value).__name__}]"


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
            self.galileo_logger = galileo_context.get_logger_instance()
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

        # Format conversation history, excluding the current user message
        history_text = ""
        # Use all messages except the last one if it's a user message and matches the current user_message
        messages_to_include = conversation_history
        if (
            conversation_history
            and conversation_history[-1]["role"] == "user"
            and conversation_history[-1]["content"] == user_message
        ):
            messages_to_include = conversation_history[:-1]

        for msg in messages_to_include:
            role = msg["role"].upper()
            content = msg["content"]
            history_text += f"{role}: {content}\n\n"

        # Add a notice about valid JSON formatting to the prompt
        json_formatting_notice = """
IMPORTANT: Your response must be strictly valid JSON without any comments or explanations. 
Do not include '//' comments or any explanatory text in the JSON.
Instead of adding comments within the JSON, use placeholder values and mention requirements in parameter descriptions.
"""

        # Return formatted prompt from config with the JSON formatting notice
        return (
            config.TOOL_SELECTION_PROMPT.format(
                tool_descriptions=chr(10).join(tool_descriptions),
                history_text=history_text,
                user_message=user_message,
            )
            + json_formatting_notice
        )

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

        # Return formatted prompt from config
        return config.AGENT_SYSTEM_PROMPT.format(
            tool_descriptions=chr(10).join(tool_descriptions)
        )

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
            List of extracted tool calls. Returns empty list if clarification needed or request unsupported.
            For unsupported requests, also sets self._last_unsupported_message.
        """
        # Check if response indicates unsupported request
        if json_str.strip().startswith("UNSUPPORTED: "):
            if self.verbose:
                print("Request cannot be handled by available tools")
            self._last_unsupported_message = json_str.strip()[12:]  # Store explanation
            return []

        # Check if response is a clarifying question
        if json_str.strip().startswith("CLARIFY: "):
            if self.verbose:
                print("Agent needs clarification - no tool calls extracted")
            return []

        tool_calls = []

        try:
            # Remove JavaScript-style comments
            json_str_clean = re.sub(r"//.*?(\n|$)", "\n", json_str)
            # Also remove multi-line comments if present
            json_str_clean = re.sub(r"/\*.*?\*/", "", json_str_clean, flags=re.DOTALL)

            # Parse the JSON array
            data = json.loads(json_str_clean)

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
            List of selected tools. Empty list if clarification needed.
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

        # Log the full prompt to Galileo if logger exists
        if self.galileo_logger:
            self.galileo_logger.add_llm_span(
                input=prompt,
                output=response.content,
                model=self.agent_model,
                duration_ns=int(
                    (time.time() - time.time()) * 1_000_000_000
                ),  # Just a placeholder, will be replaced
                name="agent_tool_selection",
                tags=["agent", "tool_selection", self.domain, self.category],
            )

        if self.verbose:
            print("Tool selection response:")
            print(response.content)

        # Extract tool calls
        tool_calls = self._extract_tool_calls(response.content)

        if self.verbose:
            if not tool_calls:
                print("No tool calls extracted - may need clarification")
            else:
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
            tool_start_time = time.time()
            tool_result = self.simulate_tool(tool_name, tool_parameters, tool)
            tool_duration_ns = int((time.time() - tool_start_time) * 1_000_000_000)

            # Add tool span to Galileo if logger exists
            if self.galileo_logger:
                self.galileo_logger.add_tool_span(
                    input=json.dumps(tool_parameters),
                    output=json.dumps(tool_result["response"]),
                    name=tool_name,
                    duration_ns=tool_duration_ns,
                    tags=["tool", tool_name],
                )

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

            # Always print the tool execution to console
            print(f"Tool: {tool_name} - Duration: {end_time - start_time:.4f}s")
            print(
                f"  Parameters: {json.dumps(tool_parameters)[:100]}{'...' if len(json.dumps(tool_parameters)) > 100 else ''}"
            )
            print(
                f"  Response: {json.dumps(tool_response)[:100]}{'...' if len(json.dumps(tool_response)) > 100 else ''}"
            )

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
        # Check if conversation history has at least 2 messages and the second-to-last is from the assistant
        if (
            len(conversation_history) >= 2
            and conversation_history[-2]["role"] == "assistant"
        ):
            messages.extend(
                [
                    (
                        HumanMessage(content=msg["content"])
                        if msg["role"] == "user"
                        else AIMessage(content=msg["content"])
                    )
                    for msg in conversation_history[:-2]
                ]
            )
        else:
            messages.extend(
                [
                    (
                        HumanMessage(content=msg["content"])
                        if msg["role"] == "user"
                        else AIMessage(content=msg["content"])
                    )
                    for msg in conversation_history
                ]
            )

        # Add the prompt with tool results
        messages.append(HumanMessage(content=prompt))

        if self.verbose:
            print("Generating final response using tool results")
            print(f"Tool results count: {len(tool_results)}")

        # Prepare full prompt for logging
        full_prompt = "\n".join([msg.content for msg in messages])

        # Call the agent to generate the final response
        start_time = time.time()
        response = self.agent_llm.invoke(messages)
        end_time = time.time()

        # Log the full prompt to Galileo if logger exists
        if self.galileo_logger:
            self.galileo_logger.add_llm_span(
                input=full_prompt,
                output=response.content,
                model=self.agent_model,
                duration_ns=int((end_time - start_time) * 1_000_000_000),
                name="agent_final_response",
                tags=["agent", "final_response", self.domain, self.category],
            )

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

        # If no tools were selected, check if it was due to unsupported request
        if not tool_calls:
            if hasattr(self, "_last_unsupported_message"):
                return f"I apologize, but {self._last_unsupported_message}"

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

            # Prepare full prompt for logging
            full_prompt = "\n".join([msg.content for msg in messages])

            # Call the agent LLM using the LLMHandler
            response = self.agent_llm.invoke(messages)

            # Extract the response content
            agent_response = response.content

            # Log the full prompt to Galileo
            if self.galileo_logger:
                if self.verbose:
                    print("Logging LLM call to Galileo with full prompt")

                self.galileo_logger.add_llm_span(
                    input=full_prompt,
                    output=agent_response,
                    model=self.agent_model,
                    duration_ns=int((time.time() - start_time) * 1_000_000_000),
                    name="agent_direct_response",
                    tags=["agent", "direct_response", self.domain, self.category],
                )
        else:
            if self.verbose:
                print(f"Processing {len(tool_calls)} tool calls")

            # Process the tool calls
            tool_results = self.process_tool_calls(tool_calls)
            print(f"Processed {len(tool_results)} tool results successfully")

            # Generate final response using tool results
            agent_response = self.generate_final_response(
                conversation_history=conversation_history,
                user_message=user_message,
                tool_results=tool_results,
            )

        end_time = time.time()
        agent_duration_ns = int((end_time - start_time) * 1_000_000_000)

        # No need for additional logging here since we've logged in the specific branches above

        return agent_response

    def run_simulation(self, scenario_idx: int = 0) -> Dict[str, Any]:
        """
        Run a full simulation for a single scenario.
        """
        # Get scenario and persona
        scenario = self.scenarios[scenario_idx]
        persona_idx = scenario.get("persona_index", 0)
        persona = self.personas[persona_idx]

        # Prepare initial conversation history
        conversation_history = []

        # Get initial message from scenario
        if isinstance(scenario.get("first_message"), list):
            conversation_history = scenario["first_message"]
            initial_user_message = conversation_history[-1]["content"]
        else:
            initial_user_message = scenario.get("first_message", "Hello")

        # Get logger from Galileo context
        logger = galileo_context.get_logger_instance()

        # Simulation loop
        turn_count = 0
        tool_outputs = []
        all_tool_results = []
        simulation_start_time = time.time()

        while turn_count < config.MAX_TURNS:
            turn_count += 1
            turn_start_time = time.time()

            # Start a workflow span for this turn
            workflow_name = f"turn_{turn_count}_workflow"
            logger.add_workflow_span(
                input=initial_user_message,
                name=workflow_name,
                tags=["turn", self.domain, self.category],
            )

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

            # Get tool calls
            tool_calls = self.select_tools(
                user_message=user_message, conversation_history=conversation_history
            )

            # Process tools if any were selected
            tool_results = []
            if tool_calls:
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
                    tool_start_time = time.time()
                    tool_result = self.simulate_tool(tool_name, tool_parameters, tool)
                    tool_duration_ns = int(
                        (time.time() - tool_start_time) * 1_000_000_000
                    )

                    # Add tool span to Galileo
                    logger.add_tool_span(
                        input=json.dumps(tool_parameters),
                        output=json.dumps(tool_result["response"]),
                        name=tool_name,
                        duration_ns=tool_duration_ns,
                        tags=["tool", tool_name],
                    )

                    tool_results.append(tool_result)

                all_tool_results.extend(tool_results)

            # Generate agent response
            agent_response = self.generate_final_response(
                conversation_history=conversation_history,
                user_message=user_message,
                tool_results=tool_results if tool_calls else [],
            )

            # Add agent response to conversation history
            conversation_history.append(
                {"role": "assistant", "content": agent_response}
            )

            # Simulate user response
            user_response = self.simulate_user(
                persona=persona,
                scenario=scenario,
                conversation_history=conversation_history,
                tool_outputs=tool_outputs,
            )

            # Add user response to conversation history
            conversation_history.append({"role": "user", "content": user_response})

            # Calculate turn duration
            turn_duration_ns = int((time.time() - turn_start_time) * 1_000_000_000)

            # Conclude the workflow span for this turn
            logger.conclude(
                output=ensure_string(agent_response),
                duration_ns=turn_duration_ns,
            )

            # Check if conversation should end
            if "CONVERSATION_COMPLETE" in user_response:
                print(f"Conversation complete after {turn_count} turns.")
                break

        # After the simulation loop
        # final_result = {
        #     "conversation": conversation_history,
        #     "turns": turn_count,
        #     "domain": self.domain,
        #     "category": self.category,
        #     "scenario_idx": scenario_idx,
        #     "agent_model": self.agent_model,
        #     "tool_results": all_tool_results,
        # }

        # If you need to log the final result
        # if logger:
        #     logger.conclude(
        #         output=ensure_string(final_result),
        #         duration_ns=int((time.time() - simulation_start_time) * 1_000_000_000),
        #     )

        # return json.dumps(final_result)


def create_experiment_runner(
    agent_model: str, domain: str, category: str, verbose: bool = False
):
    """
    Create a runner function for Galileo experiments.

    Args:
        agent_model: The model to use for the agent
        domain: The domain for experiments (e.g., 'banking', 'healthcare')
        category: The category of scenarios to run
        verbose: Whether to print verbose logs

    Returns:
        A function that can be passed to run_experiment
    """

    def runner(input_data):
        """
        Runner function for experiments that processes a single test case.

        Args:
            input_data: A dictionary containing test case data

        Returns:
            Results of the simulation
        """
        # Extract scenario index from input
        scenario_idx = input_data.get("scenario_idx", 0)

        # Initialize simulation
        simulation = AgentSimulation(
            agent_model=agent_model,
            domain=domain,
            category=category,
            log_to_galileo=True,
            verbose=verbose,
        )

        # Run the simulation
        simulation.run_simulation(scenario_idx=scenario_idx)
        print(f"Simulation complete for scenario {scenario_idx}")

    return runner


def run_simulation_experiments(
    models: List[str],
    domains: List[str],
    categories: List[str],
    dataset_name: str = None,
    project: str = "agent-evaluations",
    metrics: List[str] = None,
    verbose: bool = False,
):
    """
    Run experiments for all combinations of models, domains, and categories.

    Args:
        models: List of model names to evaluate
        domains: List of domains to evaluate
        categories: List of categories to evaluate
        dataset_name: Name of the dataset to use
        project: Galileo project name
        metrics: List of metrics to evaluate
        verbose: Whether to print verbose logs

    Returns:
        Dictionary of experiment results
    """
    if metrics is None:
        metrics = ["tool_selection_quality"]

    results = {}

    for model in models:
        for domain in domains:
            for category in categories:
                # Add timestamp to experiment name to ensure uniqueness
                timestamp = int(time.time())
                experiment_name = (
                    f"{model.replace('/', '-')}-{domain}-{category}-{timestamp}"
                )

                if verbose:
                    print(f"Running experiment: {experiment_name}")

                # Create the runner function for this specific combination
                runner = create_experiment_runner(
                    agent_model=model, domain=domain, category=category, verbose=verbose
                )

                # Get or create dataset
                if dataset_name:
                    dataset = get_dataset(name=dataset_name)
                else:
                    # Create custom dataset from scenarios
                    simulation = AgentSimulation(
                        agent_model=model,
                        domain=domain,
                        category=category,
                        log_to_galileo=False,
                        verbose=verbose,
                    )

                    # Convert scenarios to dataset format
                    dataset = [
                        {"scenario_idx": i} for i in range(len(simulation.scenarios))
                    ]

                    if verbose:
                        print(f"Created dataset with {len(dataset)} scenarios")

                # Run the experiment
                result = run_experiment(
                    experiment_name=experiment_name,
                    project=project,
                    dataset=dataset,
                    function=runner,
                    metrics=metrics,
                )

                results[experiment_name] = result

                if verbose:
                    print(f"Completed experiment: {experiment_name}")

    print(f"Completed experiment: {experiment_name}")
    return results

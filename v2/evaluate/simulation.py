import time
import json
import re
import logging
import datetime
import os
from colorama import init, Fore, Style, Back

from typing import Dict, List, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import config
from llm_handler import LLMHandler
from galileo import galileo_context
from galileo.datasets import get_dataset
from galileo.experiments import run_experiment
from dotenv import load_dotenv
from utils import (
    setup_logger,
    log_header,
    log_section,
    format_json_for_display,
    ensure_string,
)

load_dotenv("../.env")


logger = setup_logger()

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
            logger.info(
                log_section(
                    "INITIALIZATION",
                    f"Galileo logger initialized successfully",
                    style=Fore.CYAN,
                )
            )
        else:
            self.galileo_logger = None

        logger.info(
            log_header(
                f"SIMULATION INITIALIZED: {self.agent_model} - {self.domain}/{self.category}",
                style=Fore.CYAN,
            )
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

        # Define a clear schema for tool calls
        tool_call_schema = """
TOOL CALL FORMAT:
When you need to use a tool, use one of these formats:
Format:
[
  {
    "tool": "tool_name",
    "parameters": {
      "param1": "value1",
      "param2": "value2"
    }
  }
]

For multiple tool calls:
[
  {
    "tool": "first_tool_name",
    "parameters": {
      "param1": "value1"
    }
  },
  {
    "tool": "second_tool_name",
    "parameters": {
      "param1": "value1"
    }
  }
]

ALWAYS surround tool calls with square brackets []
ALWAYS use proper JSON format with double quotes
ALWAYS include "tool" or "name" and "parameters" fields
"""

        # Get domain-specific instructions
        domain_instructions = ""
        if self.domain.lower() in config.DOMAIN_SPECIFIC_INSTRUCTIONS:
            domain_instructions = config.DOMAIN_SPECIFIC_INSTRUCTIONS[
                self.domain.lower()
            ]

        # Return formatted prompt from config with schema added
        prompt = config.AGENT_SYSTEM_PROMPT.format(
            tool_descriptions=chr(10).join(tool_descriptions),
            domain_instructions=domain_instructions,
        )

        # Add the tool call schema
        prompt += "\n\n" + tool_call_schema
        # print(f"Agent Prompt: {prompt}")

        return prompt

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
            logger.info(
                log_section(
                    "TOOL SIMULATION",
                    f"Simulating tool: {tool_name}",
                    style=Fore.MAGENTA,
                )
            )
            logger.debug(f"Parameters: {json.dumps(tool_parameters, indent=2)}")

        # Call the simulator LLM
        start_time = time.time()
        response = self.simulator_llm.invoke([HumanMessage(content=prompt)])
        end_time = time.time()
        tool_duration_ns = int((end_time - start_time) * 1_000_000_000)

        # Parse the response as JSON
        response_content = response.content
        # Extract JSON content if wrapped in markdown code blocks
        if "```json" in response_content:
            json_match = re.search(r"```json\s*([\s\S]+?)```", response_content)
            if json_match:
                response_content = json_match.group(1)

        tool_response = json.loads(response_content)

        # Format tool execution info for logging
        tool_info = (
            f"{Fore.MAGENTA}TOOL:{Style.RESET_ALL} {tool_name} | "
            f"{Fore.YELLOW}Duration:{Style.RESET_ALL} {end_time - start_time:.4f}s\n"
            f"{Fore.YELLOW}Parameters:{Style.RESET_ALL} {format_json_for_display(tool_parameters)}\n"
            f"{Fore.YELLOW}Response:{Style.RESET_ALL} {format_json_for_display(tool_response)}"
        )

        # Log the tool execution
        logger.info(log_section("TOOL EXECUTION", tool_info, style=Fore.GREEN))

        # Log tool span to Galileo if logger exists
        if self.galileo_logger:
            tool_span_metadata = {
                "tool_name": tool_name,
                "parameters": json.dumps(tool_parameters),
            }
            self.galileo_logger.add_tool_span(
                input=json.dumps(tool_parameters),
                output=json.dumps(tool_response),
                name=tool_name,
                duration_ns=tool_duration_ns,
                metadata=tool_span_metadata,
                tags=["tool", self.domain, self.category],
                tool_call_id=str(time.time()),
            )

        # Return comprehensive result
        return {
            "tool_name": tool_name,
            "parameters": tool_parameters,
            "response": tool_response,
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
        # If no tool results, we should not generate a new response.
        # This should never actually happen with the current implementation, but keeping as a safeguard.
        if not tool_results:
            return "No tools were used."

        # Format the tool results for the prompt
        tool_results_text = ""
        for result in tool_results:
            tool_results_text += f"""
Tool: {result['tool_name']}
Parameters: {json.dumps(result['parameters'], indent=2)}
Response: {json.dumps(result['response'], indent=2)}
"""

        # Create a prompt for the agent to generate a final response
        prompt = config.FINAL_RESPONSE_PROMPT.format(
            user_message=user_message, tool_results_text=tool_results_text
        )

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
                tools=self.tools,
                duration_ns=int((end_time - start_time) * 1_000_000_000),
                name="agent_final_response",
                tags=["agent", "final_response", self.domain, self.category],
            )

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
        response = self.simulator_llm.invoke([HumanMessage(content=prompt)])

        return response.content

    def detect_and_process_tool_calls(
        self, agent_response: str
    ) -> List[Dict[str, Any]]:
        """
        Detect tool calls in the agent's response by looking for the string "tool":
        and process them using the tool simulator.

        Args:
            agent_response: The response from the agent LLM

        Returns:
            List of tool results
        """
        tool_results = []

        # Look for tool calls in the JSON format: {"tool": "tool_name", "parameters": {...}}
        # or [{"tool": "tool_name", "parameters": {...}}]
        # print(f"Agent response: {agent_response}")
        try:
            # Check if the response contains a tool call
            if '"tool":' in agent_response or '"name":' in agent_response:
                # Try to extract JSON from the response
                json_pattern = r'\[?\s*{.*"(?:tool|name)":.+}\s*\]?'
                json_matches = re.findall(json_pattern, agent_response, re.DOTALL)

                if json_matches:
                    for json_str in json_matches:
                        # Make sure we have valid JSON
                        if not (json_str.startswith("[") and json_str.endswith("]")):
                            if not (
                                json_str.startswith("{") and json_str.endswith("}")
                            ):
                                json_str = f"{json_str}"

                        try:
                            # Parse the JSON
                            tool_calls = json.loads(json_str)

                            # pretty print tool calls
                            if self.verbose:
                                logger.info(
                                    log_section(
                                        "TOOL CALLS",
                                        json.dumps(tool_calls, indent=2),
                                        style=Fore.YELLOW,
                                    )
                                )

                            # Handle both single tool call and array of tool calls
                            if isinstance(tool_calls, dict):
                                tool_calls = [tool_calls]

                            for tool_call in tool_calls:
                                tool_name = tool_call.get("tool") or tool_call.get(
                                    "name"
                                )
                                parameters = tool_call.get("parameters", {})

                                if tool_name:
                                    # Get the tool definition
                                    tool_definition = self._get_tool_by_name(tool_name)

                                    if tool_definition:
                                        # Simulate the tool execution
                                        result = self.simulate_tool(
                                            tool_name=tool_name,
                                            tool_parameters=parameters,
                                            tool_definition=tool_definition,
                                        )
                                        tool_results.append(result)
                                    else:
                                        logger.warning(
                                            f"Tool '{tool_name}' not found in available tools"
                                        )
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse tool call JSON: {e}")
                            logger.debug(f"Problematic JSON string: {json_str}")

            return tool_results

        except Exception as e:
            logger.error(f"Error detecting tool calls: {str(e)}")
            return []

    def run_agent(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
    ) -> tuple:
        """
        Run the agent LLM with the provided user message and tools.

        Args:
            user_message: The user message to respond to
            conversation_history: The conversation history so far
            tools: The tools available to the agent

        Returns:
            A tuple of (final_response, tool_results)
        """
        if self.verbose:
            logger.info(
                log_section(
                    "AGENT", f"Running agent with {len(tools)} tools", style=Fore.GREEN
                )
            )

        start_time = time.time()

        # Create system prompt with tools
        system_prompt = self.update_agent_prompt(tools)

        # Format conversation history
        messages = [SystemMessage(content=system_prompt)]

        # Add conversation history
        for msg in conversation_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        # Add the current user message
        messages.append(HumanMessage(content=user_message))

        # Call the agent
        agent_response = self.agent_llm.invoke(messages).content

        # Log the raw agent response
        # if self.verbose:
        #     logger.info(
        #         log_section("RAW AGENT RESPONSE", agent_response, style=Fore.YELLOW)
        #     )

        # Detect and process any tool calls in the agent's response
        tool_results = self.detect_and_process_tool_calls(agent_response)

        # pretty print tool results
        if self.verbose:
            logger.info(
                log_section(
                    "TOOL RESULTS",
                    json.dumps(tool_results, indent=2),
                    style=Fore.YELLOW,
                )
            )

        # Only generate final response if tools were used, otherwise use raw agent response
        if tool_results:
            final_response = self.generate_final_response(
                conversation_history=conversation_history,
                user_message=user_message,
                tool_results=tool_results,
            )
        else:
            final_response = agent_response

        return final_response, tool_results

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

        # Simulation loop
        turn_count = 0
        tool_outputs = []  # This will be used for simulating user responses
        all_tool_results = []  # This tracks all tool results across turns
        simulation_start_time = time.time()

        # Log simulation start
        sim_start_info = (
            f"{Fore.CYAN}Domain:{Style.RESET_ALL} {self.domain} | "
            f"{Fore.CYAN}Category:{Style.RESET_ALL} {self.category} | "
            f"{Fore.CYAN}Scenario:{Style.RESET_ALL} {scenario_idx}"
        )
        logger.info(log_header("SIMULATION START", style=Fore.BLUE))
        logger.info(log_section("CONFIG", sim_start_info, style=Fore.BLUE))

        while turn_count < config.MAX_TURNS:
            turn_count += 1
            turn_start_time = time.time()

            # Log turn start with clear separator
            logger.info(log_header(f"TURN {turn_count}", style=Fore.GREEN))

            # Start a workflow span for this turn
            workflow_name = f"turn_{turn_count}_workflow"
            if self.galileo_logger:
                self.galileo_logger.add_workflow_span(
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

            # Log user message
            user_msg_display = user_message
            logger.info(log_section("USER MESSAGE", user_msg_display, style=Fore.BLUE))

            # Run the agent and get response
            final_response, tool_results = self.run_agent(
                user_message=user_message,
                conversation_history=conversation_history,
                tools=self.tools,
            )

            # Process tools if any were detected
            if tool_results:
                tools_info = (
                    f"{Fore.YELLOW}Detected {len(tool_results)} tools:{Style.RESET_ALL} "
                    + ", ".join([t["tool_name"] for t in tool_results])
                )
                logger.info(log_section("TOOLS", tools_info, style=Fore.YELLOW))

                # Add to the persistent tool outputs list for user simulation
                tool_outputs.extend(tool_results)

                # Add to overall tool results list
                all_tool_results.extend(tool_results)
            else:
                logger.info(
                    log_section(
                        "TOOLS",
                        f"{Fore.YELLOW}NO TOOLS DETECTED{Style.RESET_ALL}",
                        style=Fore.YELLOW,
                    )
                )

            # Use agent_response as the final response
            final_response = final_response

            # Log agent response
            agent_msg_display = final_response
            logger.info(
                log_section("AGENT RESPONSE", agent_msg_display, style=Fore.GREEN)
            )

            # Add agent response to conversation history
            conversation_history.append(
                {"role": "assistant", "content": final_response}
            )

            # Simulate user response
            user_response = self.simulate_user(
                persona=persona,
                scenario=scenario,
                conversation_history=conversation_history,
                tool_outputs=tool_outputs,  # Pass all accumulated tool outputs
            )

            # Add user response to conversation history
            conversation_history.append({"role": "user", "content": user_response})

            # Calculate turn duration
            turn_duration_ns = int((time.time() - turn_start_time) * 1_000_000_000)
            turn_duration_sec = time.time() - turn_start_time

            # Log turn completion
            logger.info(
                log_section(
                    "TURN END", f"Duration: {turn_duration_sec:.4f}s", style=Fore.GREEN
                )
            )

            # Conclude the workflow span for this turn
            if self.galileo_logger:
                self.galileo_logger.conclude(
                    output=ensure_string(final_response),
                    duration_ns=turn_duration_ns,
                )

            # Check if conversation should end
            if "CONVERSATION_COMPLETE" in user_response:
                logger.info(
                    log_section(
                        "COMPLETION",
                        f"Conversation complete after {turn_count} turns",
                        style=Fore.MAGENTA,
                    )
                )
                break

        # Log simulation end with clear separator
        total_duration = time.time() - simulation_start_time
        sim_end_info = (
            f"{Fore.CYAN}Total turns:{Style.RESET_ALL} {turn_count} | "
            f"{Fore.CYAN}Total time:{Style.RESET_ALL} {total_duration:.2f}s"
        )
        logger.info(log_header("SIMULATION COMPLETE", style=Fore.BLUE))
        logger.info(log_section("SUMMARY", sim_end_info, style=Fore.BLUE))


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
        logger.info(
            log_section(
                "COMPLETE",
                f"Simulation complete for scenario {scenario_idx}",
                style=Fore.GREEN,
            )
        )

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
        metrics = [
            "tool_selection_quality",
            "agentic_workflow_success",
            "agentic_session_success",
        ]

    results = []

    # Format model names for display
    model_names = ", ".join([m.split("/")[-1] if "/" in m else m for m in models])

    start_info = (
        f"{Fore.CYAN}Models:{Style.RESET_ALL} {model_names}\n"
        f"{Fore.CYAN}Domains:{Style.RESET_ALL} {', '.join(domains)}\n"
        f"{Fore.CYAN}Categories:{Style.RESET_ALL} {', '.join(categories)}"
    )

    logger.info(log_header("STARTING EXPERIMENTS", style=Fore.MAGENTA))
    logger.info(log_section("CONFIG", start_info, style=Fore.MAGENTA))

    for model in models:
        for domain in domains:
            for category in categories:
                # Add timestamp to experiment name to ensure uniqueness
                timestamp = int(time.time())
                experiment_name = (
                    f"{model.replace('/', '-')}-{domain}-{category}-{timestamp}"
                )

                exp_info = f"Model: {model} | Domain: {domain} | Category: {category}"
                logger.info(log_section("EXPERIMENT", exp_info, style=Fore.YELLOW))

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

                    logger.info(
                        log_section(
                            "DATASET",
                            f"Created dataset with {len(dataset)} scenarios",
                            style=Fore.CYAN,
                        )
                    )

                # Run the experiment
                result = run_experiment(
                    experiment_name=experiment_name,
                    project=project,
                    dataset=dataset,
                    function=runner,
                    metrics=metrics,
                )
                results.append(result)

                logger.info(
                    log_section(
                        "EXPERIMENT COMPLETE", experiment_name, style=Fore.YELLOW
                    )
                )

    logger.info(log_header("ALL EXPERIMENTS COMPLETED", style=Fore.MAGENTA))
    logger.info(
        log_section("RESULTS", f"Total experiments: {len(results)}", style=Fore.MAGENTA)
    )
    return results

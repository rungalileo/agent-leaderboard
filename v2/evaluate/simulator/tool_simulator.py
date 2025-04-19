import json
import re
import time
import concurrent.futures
from typing import Dict, List, Any, Optional
from langchain_core.messages import HumanMessage
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import logger, log_section, Fore, Style
import config


class ToolSimulator:
    """Simulates tool execution using LLMs."""

    def __init__(
        self,
        domain: str,
        category: str,
        simulator_llm,
        galileo_logger=None,
    ):
        """
        Initialize the tool simulator.

        Args:
            domain: The domain for the tools
            category: The category of tools
            simulator_llm: The LLM instance to use for simulation
            galileo_logger: Optional Galileo logger for telemetry
        """
        self.domain = domain
        self.category = category
        self.simulator_llm = simulator_llm
        self.galileo_logger = galileo_logger

    def simulate_tool(
        self,
        tool_name: str,
        tool_parameters: Dict[str, Any],
        tool_definition: Dict[str, Any],
        conversation_history: List[Dict[str, str]] = None,
        agent_action: str = None,
    ) -> Dict[str, Any]:
        """
        Simulate tool execution using the simulator LLM.

        Args:
            tool_name: The name of the tool to simulate
            tool_parameters: The parameters for the tool call
            tool_definition: The definition of the tool
            conversation_history: The conversation history so far
            agent_action: The agent's action that led to this tool call

        Returns:
            Dictionary containing tool execution results and metadata
        """
        # Get response schema
        response_schema = tool_definition.get("response_schema", {})

        # Format conversation history for the prompt
        formatted_history = ""
        if conversation_history:
            formatted_history = "\n".join(
                [
                    f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
                    for msg in conversation_history[
                        -5:
                    ]  # Only include last 5 messages for context
                ]
            )

        # Format agent action
        agent_action_text = (
            agent_action if agent_action else "No specific action provided"
        )

        # Create prompt for the tool simulator
        prompt = config.TOOL_SIMULATOR_PROMPT.format(
            tool_name=tool_name,
            tool_parameters=json.dumps(tool_parameters, indent=2),
            response_schema=json.dumps(response_schema, indent=2),
            conversation_history=formatted_history,
            agent_action=agent_action_text,
        )

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

        # Format tool execution info for logging with properly formatted JSON
        tool_info = (
            f"{Fore.MAGENTA}TOOL:{Style.RESET_ALL} {tool_name} | "
            f"{Fore.YELLOW}Duration:{Style.RESET_ALL} {end_time - start_time:.4f}s\n"
            f"{Fore.YELLOW}Parameters:{Style.RESET_ALL}\n{json.dumps(tool_parameters, indent=2)}\n"
            f"{Fore.YELLOW}Response:{Style.RESET_ALL}\n{json.dumps(tool_response, indent=2)}"
        )

        # Log the tool execution
        logger.info(log_section("TOOL SIMULATION", tool_info, style=Fore.MAGENTA))

        # Log tool span to Galileo if logger exists
        if self.galileo_logger:
            tool_call_id = (
                f"{self.domain}_{self.category}_{tool_name}_{int(time.time() * 1000)}"
            )

            self.galileo_logger.add_tool_span(
                input=json.dumps(tool_parameters),
                output=json.dumps(tool_response),
                name=tool_name,
                duration_ns=tool_duration_ns,
                tags=["tool_execution", self.domain, self.category],
                tool_call_id=tool_call_id,
            )

        # Return comprehensive result
        return {
            "tool_name": tool_name,
            "parameters": tool_parameters,
            "response": tool_response,
            "duration_ns": tool_duration_ns,
        }


class ToolHandler:
    """Handles tool detection, selection, and execution."""

    def __init__(
        self,
        domain: str,
        category: str,
        tools: List[Dict[str, Any]],
        tool_simulator=None,
        galileo_logger=None,
        use_concurrent_execution: bool = True,
    ):
        """
        Initialize the tool handler.

        Args:
            domain: The domain for the tools
            category: The category of tools
            tools: List of available tools
            tool_simulator: The simulator for tool execution
            galileo_logger: Optional Galileo logger for telemetry
            use_concurrent_execution: Whether to execute multiple tools concurrently
        """
        self.domain = domain
        self.category = category
        self.tools = tools
        self.tool_simulator = tool_simulator
        self.galileo_logger = galileo_logger
        self.use_concurrent_execution = use_concurrent_execution

    def _get_tool_by_name(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get tool definition by name."""
        for tool in self.tools:
            if tool.get("title") == tool_name:
                return tool
        return None

    def detect_and_process_tool_calls(
        self,
        agent_response: str,
        conversation_history: List[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Detect tool calls in the agent's response by looking for the string "tool":
        and process them using the tool simulator. Uses concurrent execution when multiple tools are called
        if use_concurrent_execution is True.

        Args:
            agent_response: The response from the agent LLM
            conversation_history: The conversation history so far

        Returns:
            List of tool results
        """
        tool_calls = []

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
                            parsed_tool_calls = json.loads(json_str)

                            # Handle both single tool call and array of tool calls
                            if isinstance(parsed_tool_calls, dict):
                                parsed_tool_calls = [parsed_tool_calls]

                            for tool_call in parsed_tool_calls:
                                tool_name = tool_call.get("tool") or tool_call.get(
                                    "name"
                                )
                                parameters = tool_call.get("parameters", {})

                                if tool_name:
                                    # Get the tool definition
                                    tool_definition = self._get_tool_by_name(tool_name)

                                    if tool_definition:
                                        # Extract the agent's action from the response
                                        # This is the text before the tool call JSON
                                        agent_action = agent_response.split(json_str)[
                                            0
                                        ].strip()

                                        # Store tool call info for concurrent execution
                                        tool_calls.append(
                                            {
                                                "tool_name": tool_name,
                                                "parameters": parameters,
                                                "tool_definition": tool_definition,
                                                "agent_action": agent_action,
                                            }
                                        )
                                    else:
                                        logger.warning(
                                            f"Tool '{tool_name}' not found in available tools"
                                        )
                        except json.JSONDecodeError as e:
                            logger.warning(
                                log_section(
                                    "ERROR",
                                    f"Failed to parse tool call JSON: {e}\n\nProblematic JSON string: {json_str}",
                                    style=Fore.RED,
                                )
                            )

            # Define a function to execute a single tool
            def execute_tool(tool_info):
                return self.tool_simulator.simulate_tool(
                    tool_name=tool_info["tool_name"],
                    tool_parameters=tool_info["parameters"],
                    tool_definition=tool_info["tool_definition"],
                    conversation_history=conversation_history,
                    agent_action=tool_info["agent_action"],
                )

            # If we have multiple tool calls and concurrent execution is enabled, execute them concurrently
            if len(tool_calls) > 1 and self.use_concurrent_execution:
                logger.info(
                    log_section(
                        "TOOLS",
                        f"Processing {len(tool_calls)} tools concurrently",
                        style=Fore.YELLOW,
                    )
                )

                # Execute tools concurrently
                tool_results = []
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future_to_tool = {
                        executor.submit(execute_tool, tool_info): tool_info
                        for tool_info in tool_calls
                    }

                    for future in concurrent.futures.as_completed(future_to_tool):
                        try:
                            result = future.result()
                            tool_results.append(result)
                        except Exception as exc:
                            tool_info = future_to_tool[future]
                            logger.error(
                                f"Tool execution failed for {tool_info['tool_name']}: {exc}"
                            )

                return tool_results
            elif len(tool_calls) > 0:
                # For sequential execution or a single tool, process them one by one
                tool_results = []

                if len(tool_calls) > 1 and not self.use_concurrent_execution:
                    logger.info(
                        log_section(
                            "TOOLS",
                            f"Processing {len(tool_calls)} tools sequentially",
                            style=Fore.YELLOW,
                        )
                    )

                for tool_info in tool_calls:
                    result = execute_tool(tool_info)
                    tool_results.append(result)

                return tool_results

            return []

        except Exception as e:
            logger.error(f"Error detecting tool calls: {str(e)}")
            return []

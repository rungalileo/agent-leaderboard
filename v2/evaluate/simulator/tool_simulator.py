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
        tools: List[Dict[str, Any]],
        galileo_logger=None,
        verbose: bool = False,
    ):
        """
        Initialize the tool simulator.

        Args:
            domain: The domain for the tools
            category: The category of tools
            simulator_llm: The LLM instance to use for simulation
            galileo_logger: Optional Galileo logger for telemetry
            verbose: Whether to print verbose logs
        """
        self.domain = domain
        self.category = category
        self.simulator_llm = simulator_llm
        self.galileo_logger = galileo_logger
        self.verbose = verbose
        self.tools = tools

    def process_tool_call_response(self, response) -> List[Dict[str, Any]]:
        """
        Process tool call responses when using tool binding.

        Args:
            response: Response from LLM with tool calls

        Returns:
            List of tool call information for simulation
        """
        tool_calls = []

        # Check if response has tool_calls attribute
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tool_call in response.tool_calls:
                # Extract information from the tool call
                tool_name = tool_call.get("name")
                args = tool_call.get("args", {})

                if tool_name:
                    tool_calls.append(
                        {
                            "tool_name": tool_name,
                            "parameters": args,
                            "tool_call_id": tool_call.get("id"),
                        }
                    )

        # If we have verbose mode, print the content and tool calls
        if self.verbose and hasattr(response, "content"):
            content_info = f"{Fore.CYAN}Content:{Style.RESET_ALL} {response.content}"
            tools_info = f"{Fore.CYAN}Tool Calls:{Style.RESET_ALL} " + json.dumps(
                [t["tool_name"] for t in tool_calls], indent=2
            )
            logger.info(log_section("TOOL SELECTION", content_info, style=Fore.CYAN))
            logger.info(log_section("TOOL CALLS", tools_info, style=Fore.CYAN))

        # Handle nested content issue (problem #4)
        if hasattr(response, "content") and isinstance(response.content, str):
            try:
                # Check if content contains nested JSON structure
                parsed_content = json.loads(response.content)
                if isinstance(parsed_content, dict) and "content" in parsed_content:
                    # Extract the actual content from the nested structure
                    if "tool_calls" in parsed_content:
                        # If we also find tool_calls in parsed content, use those
                        for tool_call in parsed_content.get("tool_calls", []):
                            tool_name = tool_call.get("tool_name")
                            parameters = tool_call.get("parameters", {})
                            tool_call_id = tool_call.get("tool_call_id")

                            if tool_name:
                                # Check if this tool_call already exists to avoid duplicates
                                duplicate = False
                                for existing_call in tool_calls:
                                    if (
                                        existing_call["tool_name"] == tool_name
                                        and existing_call["parameters"] == parameters
                                    ):
                                        duplicate = True
                                        break

                                if not duplicate:
                                    tool_calls.append(
                                        {
                                            "tool_name": tool_name,
                                            "parameters": parameters,
                                            "tool_call_id": tool_call_id
                                            or f"call_{int(time.time() * 1000)}",
                                        }
                                    )
            except (json.JSONDecodeError, TypeError):
                # Content is not valid JSON or not a string - ignore and continue
                pass

        return tool_calls

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

        try:
            tool_response = json.loads(response_content)
        except json.JSONDecodeError:
            tool_response = {"Error": "Invalid JSON response from tool"}

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

    def _simulate_tool_task(self, tool_call, tools, conversation_history, agent_action):
        """Helper function for parallel execution of tools"""
        tool_name = tool_call["tool_name"]
        parameters = tool_call["parameters"]

        # Find the tool definition
        tool_definition = None
        for tool in tools:
            if tool.get("title") == tool_name:
                tool_definition = tool
                break

        if tool_definition:
            return self.simulate_tool(
                tool_name=tool_name,
                tool_parameters=parameters,
                tool_definition=tool_definition,
                conversation_history=conversation_history,
                agent_action=agent_action,
            )
        return {"Error": "Tool not found"}

    def simulate_tools(
        self,
        tool_calls: List[Dict[str, Any]],
        conversation_history: List[Dict[str, str]],
        agent_action: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Simulate multiple tool executions in parallel.

        Args:
            tool_calls: List of tool calls to simulate
            conversation_history: The conversation history
            agent_action: The agent action that led to these tool calls

        Returns:
            List of tool execution results
        """
        if not tool_calls or not self.tools:
            return []

        # Initialize results list
        tool_results = []

        # Always use concurrent execution for multiple tool calls
        if len(tool_calls) > 1:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Create tasks for each tool call
                future_to_tool = {
                    executor.submit(
                        self._simulate_tool_task,
                        tool_call,
                        self.tools,
                        conversation_history,
                        agent_action,
                    ): tool_call
                    for tool_call in tool_calls
                }

                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_tool):
                    result = future.result()
                    if result:
                        tool_results.append(result)
        else:
            # For single tool call, use direct execution
            for tool_call in tool_calls:
                tool_name = tool_call["tool_name"]
                parameters = tool_call["parameters"]

                # Find the tool definition
                tool_definition = None
                for tool in self.tools:
                    if tool.get("title") == tool_name:
                        tool_definition = tool
                        break

                if tool_definition:
                    # Simulate the tool
                    tool_result = self.simulate_tool(
                        tool_name=tool_name,
                        tool_parameters=parameters,
                        tool_definition=tool_definition,
                        conversation_history=conversation_history,
                        agent_action=agent_action,
                    )
                    tool_results.append(tool_result)

        return tool_results

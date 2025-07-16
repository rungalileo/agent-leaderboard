import time
import json
from typing import Dict, List, Any, Tuple
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from llm_handler import LLMHandler
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import logger
import config


class LLMAgent:
    """Agent implementation that uses LLMs for decision making and response generation."""

    def __init__(
        self,
        model_name: str,
        agent_llm=None,
        tool_simulator=None,
        domain: str = "",
        category: str = "",
        galileo_logger=None,
        verbose: bool = False,
        history_manager=None,
    ):
        """
        Initialize the LLM agent.

        Args:
            model_name: The model name for the agent
            agent_llm: The LLM instance to use for this agent
            tool_simulator: Tool simulator for executing tool calls
            domain: The domain for the agent
            category: The category of tasks
            galileo_logger: Optional Galileo logger for telemetry
            verbose: Whether to print verbose logs
            history_manager: Conversation history manager
        """
        self.model_name = model_name
        self.domain = domain
        self.category = category
        self.galileo_logger = galileo_logger
        self.verbose = verbose
        self.history_manager = history_manager
        self.agent_llm = agent_llm
        self.tool_simulator = tool_simulator
        self.num_input_tokens = 0
        self.num_output_tokens = 0
        self.total_tokens = 0
        self.total_duration = 0

        # Store workflow context for Galileo logging
        self.workflow_context = {"inputs": [], "outputs": []}

        # Track the current turn number
        self.current_turn = 0

    def update_agent_prompt(self) -> str:
        """
        Create a simplified system prompt for the agent when using tool binding.

        Returns:
            A system prompt for the agent
        """
        # Get domain-specific instructions
        domain_instructions = ""
        if self.domain.lower() in config.DOMAIN_SPECIFIC_INSTRUCTIONS:
            domain_instructions = config.DOMAIN_SPECIFIC_INSTRUCTIONS[
                self.domain.lower()
            ]

        # Return a simplified prompt without tool descriptions since we're using tool binding
        system_prompt = config.AGENT_SYSTEM_PROMPT.format(
            domain_instructions=domain_instructions
        )
            
        return system_prompt

    def generate_final_response(
        self,
        conversation_history: List[Dict[str, str]],
        user_message: str,
        tool_results: List[Dict[str, Any]],
        agent_llm=None,
    ) -> str:
        """
        Generate a final response based on tool results.

        Args:
            conversation_history: The conversation history
            user_message: The user's message
            tool_results: Results from tool execution
            agent_llm: The LLM to use for generation

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
            if "tool_name" in result and "parameters" in result and "response" in result:
                tool_results_text += f"""
    Tool: {result['tool_name']}
    Parameters: {json.dumps(result['parameters'], indent=2)}
    Response: {json.dumps(result['response'], indent=2)}
    """

        # Create a prompt for the agent to generate a final response
        prompt = config.FINAL_RESPONSE_PROMPT.format(
            user_message=user_message, tool_results_text=tool_results_text
        )

        # Get system prompt
        system_prompt = self.update_agent_prompt()  

        # Convert conversation history to LangChain messages, explicitly passing the system prompt
        messages = self.history_manager.to_langchain_messages(
            conversation_history, system_prompt
        )

        # Add the prompt with tool results
        messages.append(HumanMessage(content=prompt))

        if self.verbose:
            logger.info("Generating final response using tool results")
            logger.info(f"Tool results count: {len(tool_results)}")

        # Format for LLM using the history manager
        exact_input = self.history_manager.format_for_llm(messages)

        # Call the agent to generate the final response
        start_time = time.time()
        response = agent_llm.invoke(messages)
        duration = time.time() - start_time   

        input_tokens, output_tokens = LLMHandler.get_token_usage_info(response)
        total_tokens = input_tokens + output_tokens

        self.num_input_tokens += input_tokens
        self.num_output_tokens += output_tokens
        self.total_tokens += total_tokens
        self.total_duration += duration

        # Update workflow context - only add the current input
        self.workflow_context["inputs"].append(exact_input)
        self.workflow_context["outputs"].append(response.content)

        # Log the full prompt to Galileo if logger exists
        if self.galileo_logger:
            self.galileo_logger.add_llm_span(
                input=exact_input,  
                output=response.content,
                model=self.model_name,
                num_input_tokens=input_tokens,
                num_output_tokens=output_tokens,
                total_tokens=total_tokens,
                duration_ns=int(duration*1_000_000_000),
                name="agent_final_response",
            )

        return response.content

    def format_conversation_for_galileo(
        self, conversation_history: List[Dict[str, str]]
    ) -> str:
        """
        Format conversation history in a simple readable format for Galileo logging.

        Args:
            conversation_history: The conversation history

        Returns:
            Formatted conversation history string
        """
        formatted_history = "=== CONVERSATION HISTORY ===\n\n"

        # First message should be system prompt if present
        has_system = False
        for i, message in enumerate(conversation_history):
            role = message.get("role", "unknown")
            content = message.get("content", "")

            # Handle system message at the beginning
            if role == "system" and i == 0:
                has_system = True
                formatted_history += f"SYSTEM: {content}\n\n"
                continue

            # Skip system message if not at beginning
            if role == "system":
                continue

            # For all other messages use standard format
            role_label = "Human" if role == "user" else "Assistant"
            formatted_history += f"{role_label}: {content}\n\n"

        return formatted_history

    def run_agent(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        workflow_input: str = None,
        previous_tool_outputs: List[Dict[str, Any]] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Run the agent LLM with the provided user message and tools.

        Args:
            user_message: The user message to respond to
            conversation_history: The conversation history so far (should already include the user_message)
            tools: The tools available to the agent
            workflow_input: Optional workflow input to use for logging (for consistency)
            previous_tool_outputs: Optional previous tool outputs to include in the prompt for tool selection

        Returns:
            A tuple of (final_response, tool_results)
        """
        start_time = time.time()
        self.current_turn += 1

        # Create system prompt
        system_prompt = self.update_agent_prompt()

        # Initialize history with system prompt if it's not there already
        if not conversation_history or conversation_history[0].get("role") != "system":
            # Set system prompt on history manager
            self.history_manager.system_prompt = system_prompt

            # Also add it to the conversation history
            if not conversation_history:
                conversation_history = []
            conversation_history.insert(0, {"role": "system", "content": system_prompt})

        # Convert conversation history to LangChain messages
        messages = self.history_manager.to_langchain_messages(conversation_history)

        # Create a new system message with updated tool information for each turn
        updated_system_message = SystemMessage(content=system_prompt)

        # Ensure system message is first by checking and replacing or inserting it
        if messages and messages[0].type == "system":
            messages[0] = updated_system_message
        else:
            # Insert at beginning if no system message exists
            messages.insert(0, updated_system_message)

        # Add previous tool outputs if available
        if previous_tool_outputs and len(previous_tool_outputs) > 0:
            # Create a formatted string with previous tool outputs
            tool_outputs_str = "PREVIOUS TOOL OUTPUTS:\n\n"
            for output in previous_tool_outputs:
                tool_outputs_str += f"Tool: {output['tool_name']}\n"
                tool_outputs_str += (
                    f"Parameters: {json.dumps(output['parameters'], indent=2)}\n"
                )
                tool_outputs_str += (
                    f"Response: {json.dumps(output['response'], indent=2)}\n\n"
                )

            # Add tool outputs information just before the last message (which should be the user message)
            if len(messages) > 1:
                messages.insert(-1, tool_outputs_str)
            else:
                # If there's only the system message, add tool outputs after it
                messages.append(tool_outputs_str)

        # Format for LLM using the history manager - now with system message included
        exact_input = self.history_manager.format_for_llm(messages)

        # If we received a workflow_input from the simulation, use that for consistency
        # Otherwise generate our own
        if workflow_input:
            # Use the provided workflow input
            self.workflow_context["inputs"].append(workflow_input)
        else:
            # Generate a simple formatted version for workflow context that doesn't duplicate
            formatted_input = self.format_conversation_for_galileo(conversation_history)

            # For Galileo workflow context
            workflow_input = (
                f"=== TURN {self.current_turn} ===\n\nINPUT: User: {user_message}"
            )
            self.workflow_context["inputs"].append(workflow_input)

        # Call the agent for initial response/tool selection
        start_time = time.time()

        # Tool binding approach - we expect structured tool calls in the response
        agent_response = self.agent_llm.invoke(messages)
        duration = time.time() - start_time

        input_tokens, output_tokens = LLMHandler.get_token_usage_info(agent_response)
        total_tokens = input_tokens + output_tokens

        self.num_input_tokens += input_tokens
        self.num_output_tokens += output_tokens
        self.total_tokens += total_tokens
        self.total_duration += duration
        
        # Update workflow context with the raw response content
        self.workflow_context["outputs"].append(agent_response.content)

        # Process tool calls from the response using the tool simulator's method
        if self.tool_simulator:
            tool_calls = self.tool_simulator.process_tool_call_response(agent_response)
        else:
            # Fallback to processing locally if no tool simulator is provided
            tool_calls = []
            if hasattr(agent_response, "tool_calls") and agent_response.tool_calls:
                for tool_call in agent_response.tool_calls:
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

        # Log the tool selection LLM span if Galileo is enabled
        if self.galileo_logger:
            # Log exactly what was sent to the LLM, and include tool calls in output
            output_with_tool_calls = {
                "content": (
                    agent_response.content
                    if hasattr(agent_response, "content")
                    else str(agent_response)
                ),
                "tool_calls": tool_calls,
            }

            span_name = "tool_selection" if tool_calls else "agent_response"
            self.galileo_logger.add_llm_span(
                input=exact_input,
                output=output_with_tool_calls,
                model=self.model_name,
                num_input_tokens=input_tokens,
                num_output_tokens=output_tokens,
                total_tokens=total_tokens,
                tools=tools,
                duration_ns=int(duration*1_000_000_000),
                name=span_name,
            )

        # Simulate tools using the tool simulator's parallel execution method
        tool_results = []
        if tool_calls and self.tool_simulator:
            # Use the tool simulator to run all tools, potentially in parallel
            tool_results = self.tool_simulator.simulate_tools(
                tool_calls=tool_calls,
                conversation_history=conversation_history,
                agent_action=(
                    agent_response.content
                    if hasattr(agent_response, "content")
                    else str(agent_response)
                ),
            )

        # Only generate final response if tools were used, otherwise use raw agent response
        if tool_results:
            final_response = self.generate_final_response(
                conversation_history=conversation_history,
                user_message=user_message,
                tool_results=tool_results,
                agent_llm=self.agent_llm,
            )

        else:
            if hasattr(agent_response, "content"):
                final_response = agent_response.content
            else:
                final_response = str(agent_response)

        return final_response, tool_results

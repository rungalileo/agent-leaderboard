import time
import json
from typing import Dict, List, Any, Tuple, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import logger, log_section, Fore
import config


class LLMAgent:
    """Agent implementation that uses LLMs for decision making and response generation."""

    def __init__(
        self,
        model_name: str,
        agent_llm=None,
        tool_handler=None,
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
            tool_handler: Tool handler for detecting and processing tool calls
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
        self.tool_handler = tool_handler

        # Store workflow context for Galileo logging
        self.workflow_context = {"inputs": [], "outputs": []}

        # Track the current turn number
        self.current_turn = 0

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

        return prompt

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
            tool_results_text += f"""
Tool: {result['tool_name']}
Parameters: {json.dumps(result['parameters'], indent=2)}
Response: {json.dumps(result['response'], indent=2)}
"""

        # Create a prompt for the agent to generate a final response
        prompt = config.FINAL_RESPONSE_PROMPT.format(
            user_message=user_message, tool_results_text=tool_results_text
        )

        # Use the complete conversation history for response generation
        messages = self.history_manager.to_langchain_messages(conversation_history)

        # Add the prompt with tool results
        messages.append(AIMessage(content=prompt))

        if self.verbose:
            logger.info("Generating final response using tool results")
            logger.info(f"Tool results count: {len(tool_results)}")

        # Format for LLM using the history manager
        exact_input = self.history_manager.format_for_llm(messages)

        # Call the agent to generate the final response
        start_time = time.time()
        response = agent_llm.invoke(messages)
        end_time = time.time()

        # Update workflow context - only add the current input
        self.workflow_context["inputs"].append(exact_input)
        self.workflow_context["outputs"].append(response.content)

        # Log the full prompt to Galileo if logger exists
        if self.galileo_logger:
            self.galileo_logger.add_llm_span(
                input=exact_input,
                output=response.content,
                model=self.model_name,
                duration_ns=int((end_time - start_time) * 1_000_000_000),
                name="agent_final_response",
                tags=["agent", "final_response"],
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
            role_label = "H" if role == "user" else "A"
            formatted_history += f"{role_label}: {content}\n\n"

        return formatted_history

    def run_agent(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        workflow_input: str = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Run the agent LLM with the provided user message and tools.

        Args:
            user_message: The user message to respond to
            conversation_history: The conversation history so far
            tools: The tools available to the agent
            workflow_input: Optional workflow input to use for logging (for consistency)

        Returns:
            A tuple of (final_response, tool_results)
        """
        start_time = time.time()
        self.current_turn += 1

        # Create system prompt with tools
        system_prompt = self.update_agent_prompt(tools)

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

        # Add the current user message
        messages.append(HumanMessage(content=user_message))

        # Format for LLM using the history manager
        exact_input = self.history_manager.format_for_llm(messages)

        # If we received a workflow_input from the simulation, use that for consistency
        # Otherwise generate our own
        if workflow_input:
            # Use the provided workflow input
            self.workflow_context["inputs"].append(workflow_input)
        else:
            # Use a simplified workflow context approach - only add current turn's input
            # Generate a simple formatted version for workflow context that doesn't duplicate
            formatted_input = self.format_conversation_for_galileo(conversation_history)

            # Add the current user message if not already in history
            if (
                conversation_history[-1].get("role") != "user"
                or conversation_history[-1].get("content") != user_message
            ):
                formatted_input += f"Human: {user_message}\n\n"

            # For Galileo workflow context
            workflow_input = (
                f"=== TURN {self.current_turn} ===\n\nINPUT: Human: {user_message}"
            )
            self.workflow_context["inputs"].append(workflow_input)

        # Call the agent for initial response/tool selection
        tool_selection_start_time = time.time()

        # Create a new system message with updated tool information for each turn
        updated_system_message = SystemMessage(content=system_prompt)
        # Replace the first message (system message) with updated version
        if messages and messages[0].type == "system":
            messages[0] = updated_system_message
        else:
            # Insert at beginning if no system message exists
            messages.insert(0, updated_system_message)

        agent_response = self.agent_llm.invoke(messages).content
        tool_selection_end_time = time.time()
        tool_selection_duration_ns = int(
            (tool_selection_end_time - tool_selection_start_time) * 1_000_000_000
        )

        # Update workflow context
        self.workflow_context["outputs"].append(agent_response)

        # Log the tool selection LLM span if Galileo is enabled
        if self.galileo_logger:
            # Log exactly what was sent to the LLM
            self.galileo_logger.add_llm_span(
                input=exact_input,  # Use the exact input sent to the LLM instead of workflow_input
                output=agent_response,
                model=self.model_name,
                tools=tools,
                duration_ns=tool_selection_duration_ns,
                name="tool_selection",
                tags=["tool_selection", self.domain, self.category],
            )

        # Detect and process any tool calls in the agent's response
        tool_results = self.tool_handler.detect_and_process_tool_calls(
            agent_response=agent_response,
            conversation_history=conversation_history,
        )

        # Only generate final response if tools were used, otherwise use raw agent response
        if tool_results:
            response_start_time = time.time()
            final_response = self.generate_final_response(
                conversation_history=conversation_history,
                user_message=user_message,
                tool_results=tool_results,
                agent_llm=self.agent_llm,
            )
            response_end_time = time.time()
            response_duration_ns = int(
                (response_end_time - response_start_time) * 1_000_000_000
            )
        else:
            final_response = agent_response

        return final_response, tool_results

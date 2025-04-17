import json
from typing import Dict, List, Any
from langchain_core.messages import HumanMessage
import config as config


class UserSimulator:
    """Simulates user responses in the conversation."""

    def __init__(
        self,
        simulator_llm,
        history_manager,
    ):
        """
        Initialize the user simulator.

        Args:
            simulator_llm: The LLM instance to use for user simulation
            history_manager: Conversation history manager
        """
        self.simulator_llm = simulator_llm
        self.history_manager = history_manager

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
        formatted_history = self.history_manager.format_for_display(
            conversation_history
        )

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

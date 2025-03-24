"""
Configuration file for Agent Evaluation system
"""

# LLM Configuration
SIMULATOR_MODEL = "claude-3-7-sonnet-20250219"
SIMULATOR_TEMPERATURE = 0.0
SIMULATOR_MAX_TOKENS = 4000

# Agent Configuration (defaults, can be overridden via CLI)
AGENT_TEMPERATURE = 0.0
AGENT_MAX_TOKENS = 4000

# File Paths
FILE_PATHS = {
    "personas": "../data/personas/{domain}.json",
    "scenarios": "../data/scenarios/{domain}/{category}.json",
    "tools": "../data/tools/{domain}.json",
}

# Galileo Configuration
GALILEO_PROJECT = "agent-leaderboard-v2"
GALILEO_LOG_STREAM = "{domain}-{category}-{model}"

# Simulation Configuration
MAX_TURNS = 15  # Maximum number of turns in a conversation
TIMEOUT_SECONDS = 60  # Timeout for each LLM call

# Prompt Templates
TOOL_SIMULATOR_PROMPT = """You are a tool simulator for evaluating AI agents. You will receive a tool name, its parameters, and the tool's response schema. 
Your task is to generate a realistic, valid response that conforms to the given response schema.

TOOL NAME: {tool_name}
TOOL PARAMETERS: {tool_parameters}
RESPONSE SCHEMA: {response_schema}

Please generate a realistic, valid response that would be returned by this tool when called with these parameters.
The response should be a valid JSON object that matches the response schema.
"""

USER_SIMULATOR_PROMPT = """You are simulating a user with the following persona:

{persona_json}

You are participating in a scenario with these details:

{scenario_json}

CONVERSATION HISTORY:
{conversation_history}

TOOL OUTPUTS (these are the results of tools that the assistant has used):
{tool_outputs}

You should respond as this user would, based on their persona and the goals of the scenario. 
If the agent has completed all the tasks or goals successfully, acknowledge that and conclude the conversation by including the exact phrase "CONVERSATION_COMPLETE" somewhere in your response, along with a thank you and goodbye message.
If there are still outstanding goals or tasks, continue the conversation to work towards them.
Your response should be natural and realistic, as if you were the actual user described in the persona.

Your response:"""

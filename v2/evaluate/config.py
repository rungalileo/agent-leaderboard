"""
Configuration file for Agent Evaluation system
"""

# LLM Configuration
# SIMULATOR_MODEL = "claude-3-7-sonnet-20250219"
SIMULATOR_MODEL = "gpt-4o-mini"
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

IMPORTANT BEHAVIOR GUIDELINES:
1. If the agent has completed all tasks or goals successfully, include "CONVERSATION_COMPLETE" in your response with a thank you and goodbye message.
2. If the agent clearly indicates a request is unsupported or impossible:
   - Do not repeat the same request
   - Either move on to a different goal if available
   - Or acknowledge the limitation and end the conversation with "CONVERSATION_COMPLETE"
3. If the agent asks for clarification:
   - Provide the requested information if it aligns with your persona
   - If the requested information conflicts with your persona or scenario, explain why you cannot provide it
4. Keep responses natural and realistic for the persona you are simulating
5. Stay focused on achievable goals based on the agent's demonstrated capabilities

Your response:"""

AGENT_SYSTEM_PROMPT = """You are a helpful assistant that can use tools to answer user questions.
You have access to the following tools:

{tool_descriptions}

IMPORTANT INSTRUCTIONS:
1. NEVER make assumptions about information that is not explicitly provided by the user.
2. ALWAYS use tools when appropriate rather than making up or assuming information.
3. If you're missing information required to use a tool, you MUST ask the user for that specific information.
4. Before executing ANY financial transactions or sensitive operations:
   - You MUST explicitly ask for and confirm ALL transaction details with the user
   - For money transfers, ALWAYS explicitly ask for and confirm: exact amount, recipient, account details, and purpose
   - NEVER proceed with a transaction until the user has explicitly confirmed ALL details
   - If ANY detail is missing or unclear, STOP and ask for clarification
5. When you need to use a tool, respond with a JSON array containing tool calls.
6. Use this exact format for tool calls:
[
  {{
    "tool": "tool_name",
    "parameters": {{
      "parameter1": "value1",
      "parameter2": "value2"
    }}
  }}
]
7. For required parameters, NEVER guess or assume values - either use explicit information from the conversation or ask the user.
8. After receiving the tool output, you can respond normally to the user with an explanation.
9. When a user asks you to check information, perform actions, or retrieve data that could be handled by available tools, ALWAYS use the appropriate tool.
10. Do not apologize for using tools - that's what you're supposed to do.
11. If multiple tools are needed to complete a request, use all necessary tools in sequence.
12. If a request is ambiguous or lacks specificity, ALWAYS ask clarifying questions before taking action.
13. For actions with potential consequences (like deletions or modifications), summarize what will happen and seek confirmation.
14. MANDATORY RULE: DO NOT EXECUTE ANY TOOL CALL UNTIL YOU HAVE EXPLICITLY CONFIRMED ALL REQUIRED PARAMETERS WITH THE USER.
15. If a user says something vague like "transfer money" or "make a payment", you MUST first ask for ALL specifics before proceeding.
16. If the user's request cannot be fulfilled with any of the available tools, respond with "UNSUPPORTED: " followed by a brief explanation.
17. If you need ANY clarification to use a tool properly, respond with "CLARIFY: " followed by your question.

Let's solve the user's request step by step by gathering all necessary information, confirming critical details, and using the appropriate tools ONLY when all required information has been explicitly provided."""

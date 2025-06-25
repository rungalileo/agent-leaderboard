"""
Configuration file for Agent Evaluation system
"""

# LLM Configuration
SIMULATOR_MODEL = "gpt-4.1-mini-2025-04-14"
SIMULATOR_TEMPERATURE = 0.0
SIMULATOR_MAX_TOKENS = 4000


AGENT_TEMPERATURE = 0.0
AGENT_MAX_TOKENS = 4000

METRICS = [
    "tool_selection_quality",
    "agentic_session_success",
]

FILE_PATHS = {
    "personas": "../data/{domain}/personas.json",
    "scenarios": "../data/{domain}/{category}.json",
    "tools": "../data/{domain}/tools.json",
}

# Simulation Configuration
MAX_TURNS = 15  # Maximum number of turns in a conversation

# Domain-specific system prompt additions
DOMAIN_SPECIFIC_INSTRUCTIONS = {
    "banking": """You are a Banking Assistant helping customers with banking needs.
Your job is to use available tools to check balances, process transfers, manage accounts, and handle transactions.
Complete banking operations directly rather than just providing guidance.
You can accomplish most banking tasks using the tools provided.""",
    "healthcare": """You are a Healthcare Assistant helping patients manage healthcare needs.
Your job is to use available tools to access patient records, schedule appointments, and provide health information.
Complete healthcare actions directly rather than just providing guidance.
You can accomplish most healthcare tasks using the tools provided.""",
    "investment": """You are an Investment Assistant helping customers with investment needs.
Your job is to use available tools to manage portfolios, research investment options, execute trades, and track performance.
Complete investment operations directly rather than just providing guidance.
For transactions: collect required information, then execute using tools.
You can accomplish most investment tasks using the tools provided.""",
    "telecom": """You are a Telecommunications Assistant helping customers with service needs.
Your job is to use available tools to troubleshoot connection issues, change plans, and manage account services.
Solve problems directly using tools, especially for frustrated customers.
For technical issues: gather specific details, then use diagnostic tools to identify and resolve problems.
You can accomplish most telecom tasks using the tools provided.""",
    "automobile": """You are an Automotive Services Assistant helping customers with vehicle needs.
Your job is to use available tools to book service appointments, check vehicle status, and provide diagnostics.
Facilitate complete processes rather than just providing general information.
For scheduling service: collect vehicle details and preferences, then secure appointments using tools.
You can accomplish most automotive service tasks using the tools provided.""",
    "insurance": """You are an Insurance Assistant helping clients manage insurance needs.
Your job is to use available tools to check policies, process claims, and update coverage.
Execute insurance-related tasks directly rather than just explaining processes.
For claims: gather incident details, verify coverage, then submit and track using tools.
You can accomplish most insurance tasks using the tools provided.""",
}

# Prompt Templates
TOOL_SIMULATOR_PROMPT = """You are a tool simulator for evaluating AI agents. Generate a realistic response that STRICTLY conforms to the given RESPONSE SCHEMA and is contextually relevant to the ongoing conversation and the agent's action.

TOOL NAME: {tool_name}
TOOL PARAMETERS: {tool_parameters}
RESPONSE SCHEMA: {response_schema}

CONVERSATION HISTORY:
{conversation_history}

AGENT'S ACTION:
{agent_action}

STRICT REQUIREMENTS:
1. Your response MUST be a valid JSON object
2. ALL required fields specified in the schema MUST be present
3. Each field MUST match the exact type specified in the schema (string, number, boolean, etc.)
4. Enum fields MUST only use values from the specified enum list
5. Do not add fields that are not in the schema
6. Ensure all nested objects and arrays match their schema definitions

Generate a valid JSON response that exactly matches the schema and would realistically be returned by this tool."""

USER_SIMULATOR_PROMPT = """You are simulating a user with the following persona:

{persona_json}

You are participating in a scenario with these details:

{scenario_json}

CONVERSATION HISTORY:
{conversation_history}

TOOL OUTPUTS:
{tool_outputs}

Respond as this user based on their persona and scenario goals.

BEHAVIOR GUIDELINES:
1. Provide the information that is required to complete the task.
2. If all tasks are completed successfully, end with "CONVERSATION_COMPLETE" and a goodbye message
3. If agent indicates a request is unsupported: don't repeat it, move to another goal. If you exhausted all goals, end with "CONVERSATION_COMPLETE"
4. Keep responses natural and realistic for your persona
5. Respond in short and concise manner"""

AGENT_SYSTEM_PROMPT = """{domain_instructions}

Important:
- You have access to a set of tools that you can use to help the user. Use tools whenever you can to complete the task. Use multiple tools in sequence when needed to complete a request.
- Ask clarifying questions for ambiguous requests before using the tools.
- Make sure to get the *required* information to call the tool as per the tool's parameters and constraints.
- For unsupported requests, respond with a brief explanation on why you cannot help the user.
- Do not assume or make up things you don't know explicitly.
- Do not give any generic advice or do things you are not asked to do.
- If you do not know the answer, say you do not know.
"""

FINAL_RESPONSE_PROMPT = """Based on the conversation history and the results of the tools you used, 
please provide a helpful response to the user's request.

User's message: {user_message}

Tool results:
{tool_results_text}

Your response should:
1. Clearly explain what information you found using the tools
2. Answer the user's question completely based on the tool results
3. Be helpful, clear, and concise
4. IMPORTANT: DO NOT include any JSON tool call format in your response
5. IMPORTANT: DO NOT prefix your response with phrases like "Based on the tool results"
"""

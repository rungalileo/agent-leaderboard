import json
from dotenv import load_dotenv
from typing import Dict, List, TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from pathlib import Path
import promptquality as pq
from llm_handler import LLMHandler
import asyncio

load_dotenv("../.env")
pq.login("console.demo.rungalileo.io")

# Configuration
DATA_DIR = Path("../data")
DOMAINS = ["banking"]
DEFAULT_MODEL = "claude-3-7-sonnet-20250219"


# State definition
class SimulationState(TypedDict):
    domain: str
    category: str
    persona: Dict
    scenario: Dict
    tools: List[Dict]
    messages: List[Dict[str, str]]
    current_turn: int
    max_turns: int
    user_goals: List[str]
    tool_calls: Optional[List[Dict]]
    tool_results: Optional[List[Dict]]


# Data loading
def load_json(file_path: Path) -> Dict:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    with file_path.open("r") as f:
        return json.load(f)


def load_data(domain: str) -> tuple[List[Dict], List[Dict], List[Dict]]:
    personas = load_json(DATA_DIR / "personas" / f"{domain}.json")
    scenarios = load_json(DATA_DIR / "scenarios" / f"{domain}.json")
    tools = load_json(DATA_DIR / "tools" / f"{domain}.json")
    return personas, scenarios, tools


# LLM setup
llm_handler = LLMHandler()

# Prompts
user_prompt = PromptTemplate(
    input_variables=["persona", "scenario", "history", "turn"],
    template="""
    You are simulating a user based on this persona: {persona}.
    The scenario is: {scenario}.
    Current conversation history: {history}.
    This is turn {turn}. Based on the goals and context, generate the next user message.
    """,
)

agent_prompt = PromptTemplate(
    input_variables=["messages", "tool_results"],
    template="""
    You are an AI agent assisting a user. Respond to the latest user message and, if needed, call appropriate tools.
    
    Previous conversation and tool results: {messages}
    
    If you used tools in your last response, here are the results: {tool_results}
    
    Respond to the user's most recent message. If you need to use a tool, format your response as:
    {{"response": "your explanation to the user", "tool_calls": [{{"name": "tool_name", "args": {{...}}}}]}}
    
    If no tool is needed, simply provide your response as:
    {{"response": "your answer to the user", "tool_calls": []}}
    """,
)

tool_simulator_prompt = PromptTemplate(
    input_variables=["tool_calls", "tools"],
    template="""
    You are a tool simulator. Given the following tool calls and the specification of requested tools, simulate the execution and return realistic responses adhering strictly to each tool's response_schema.
    Tool calls: {tool_calls}
    Requested tool specifications: {tools}
    Output format: [{{"tool": "tool_name", "output": {{...}}}}]
    """,
)


# Helper function
def parse_json(output: str, default: Dict) -> Dict:
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return default


# Simulation nodes
def user(state: SimulationState) -> SimulationState:
    if "messages" not in state:
        state["messages"] = []

    if state["current_turn"] == 0:
        message = state["scenario"]["first_message"]
    else:
        prompt = user_prompt.format(
            persona=json.dumps(state["persona"]),
            scenario=json.dumps(state["scenario"]),
            history=json.dumps(state["messages"]),
            turn=state["current_turn"],
        )
        llm = llm_handler.get_llm(DEFAULT_MODEL, tags=["user"])
        message = llm.invoke(prompt).content

    state["messages"].append({"role": "user", "content": message})
    state["current_turn"] += 1
    return state


def agent(state: SimulationState, model: str) -> SimulationState:
    prompt = agent_prompt.format(
        messages=json.dumps(state["messages"]),
        tool_results=json.dumps(state.get("tool_results", "None")),
    )
    llm = llm_handler.get_llm(model, tags=["agent"])
    output = llm.invoke(prompt).content
    result = parse_json(output, {"response": output, "tool_calls": []})

    state["messages"].append({"role": "assistant", "content": result["response"]})
    state["tool_calls"] = result.get("tool_calls", [])
    state["tool_results"] = None
    return state


def tool_simulator(state: SimulationState) -> SimulationState:
    tool_calls = state.get("tool_calls", [])
    if not tool_calls:
        state["tool_results"] = []
        return state

    relevant_tools = [
        t for t in state["tools"] if t["title"] in [call["name"] for call in tool_calls]
    ]
    prompt = tool_simulator_prompt.format(
        tool_calls=json.dumps(tool_calls), tools=json.dumps(relevant_tools)
    )
    llm = llm_handler.get_llm(DEFAULT_MODEL, tags=["tool"])
    output = llm.invoke(prompt).content
    default_results = [
        {"tool": call["name"], "output": {"error": "Parsing error"}}
        for call in tool_calls
    ]
    state["tool_results"] = parse_json(output, default_results)
    return state


# Graph setup
def build_graph(agent_model: str) -> StateGraph:
    workflow = StateGraph(SimulationState)
    workflow.add_node("user", user)
    workflow.add_node("agent", lambda state: agent(state, agent_model))
    workflow.add_node("tool_simulator", tool_simulator)

    workflow.add_edge("user", "agent")
    workflow.add_edge("agent", "tool_simulator")
    workflow.add_conditional_edges(
        "tool_simulator",
        lambda state: "user" if state["current_turn"] < state["max_turns"] else END,
        {
            "user": "user",
            END: END,
        },
    )
    workflow.set_entry_point("user")
    return workflow.compile()


# Evaluation setup
def get_evaluation_config():
    tool_selection_scorer = pq.CustomizedChainPollScorer(
        scorer_name=pq.CustomizedScorerName.tool_selection_quality,
        model_alias=pq.Models.gpt_4o,
    )
    evaluate_handler = pq.GalileoPromptCallback(
        project_name="agent-leaderboard-v2",
        run_name="test",
        scorers=[pq.Scorers.tool_errors_plus, tool_selection_scorer],
    )
    return {"recursion_limit": 40, "callbacks": [evaluate_handler]}, evaluate_handler

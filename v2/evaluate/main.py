import argparse
import asyncio
from pathlib import Path
from simulation import build_graph, load_data, get_evaluation_config


async def run_simulation(agent_model: str, domain: str):
    graph = build_graph(agent_model)
    personas, scenarios, tools = load_data(domain)

    config, evaluate_handler = get_evaluation_config()

    # Run for first scenario as example (extend to all scenarios as needed)
    initial_state = {
        "domain": domain,
        "category": scenarios[0]["category"],
        "persona": personas[0],
        "scenario": scenarios[0],
        "tools": tools,
        "conversation_history": [],
        "current_turn": 0,
        "max_turns": 10,
        "user_goals": scenarios[0]["goals"],
        "tool_calls": [],
        "responses": [],
    }

    print(f"Running simulation for domain: {domain}, model: {agent_model}")
    print(f"Scenario: {initial_state['scenario']['first_message']}")
    async for event in graph.astream(initial_state, config=config):
        for k, v in event.items():
            if k != "__end__":
                print(f"{k}: {v}")

    evaluate_handler.finish()


def main():
    parser = argparse.ArgumentParser(description="Run Agent Leaderboard v2 simulation")
    parser.add_argument(
        "--model",
        type=str,
        default="claude-3-7-sonnet-20250219",
        help="Model name for the agent",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="banking",
        choices=["banking"],
        help="Domain to simulate",
    )
    args = parser.parse_args()

    asyncio.run(run_simulation(args.model, args.domain))


# python src/main.py --model "gpt-4o-2024-11-20" --domain "banking"
if __name__ == "__main__":
    main()

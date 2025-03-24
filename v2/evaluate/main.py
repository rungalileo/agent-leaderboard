"""
Main entry point for Agent Evaluation system
"""

import os
import sys
import json
import argparse
import time
from typing import List, Dict, Any
from pathlib import Path
from simulation import AgentSimulation


def parse_args():
    parser = argparse.ArgumentParser(description="Agent Leaderboard v2 Evaluation")

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name for the agent being evaluated",
    )

    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        help="Domain for the simulation (e.g., banking, healthcare)",
    )

    parser.add_argument(
        "--category",
        type=str,
        required=True,
        help="Category for the simulation (e.g., tool_coordination, context_retention)",
    )

    parser.add_argument(
        "--scenario-idx",
        type=int,
        default=None,
        help="Index of the specific scenario to run (default: run all)",
    )

    parser.add_argument(
        "--no-galileo", action="store_true", help="Disable logging to Galileo"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Directory to save results (default: ./results)",
    )

    return parser.parse_args()


def run_evaluation(args):
    """Run the evaluation with the provided arguments."""
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Initialize simulation
        simulation = AgentSimulation(
            agent_model=args.model,
            domain=args.domain,
            category=args.category,
            log_to_galileo=not args.no_galileo,
            verbose=args.verbose,
        )

        results = []

        # Run simulation for specific scenario or all scenarios
        if args.scenario_idx is not None:
            # Run single scenario
            print(
                f"Running scenario {args.scenario_idx} for {args.domain}/{args.category} with model {args.model}"
            )
            try:
                result = simulation.run_simulation(scenario_idx=args.scenario_idx)
                results.append(result)
            except Exception as e:
                print(f"Error running scenario {args.scenario_idx}: {str(e)}")
                import traceback

                traceback.print_exc()
        else:
            # Run all scenarios
            print(
                f"Running all scenarios for {args.domain}/{args.category} with model {args.model}"
            )
            total_scenarios = len(simulation.scenarios)
            for i in range(total_scenarios):
                print(f"\n--------------------------------")
                print(f"Running scenario {i+1}/{total_scenarios} (index: {i})")
                print(f"--------------------------------\n")
                try:
                    result = simulation.run_simulation(scenario_idx=i)
                    results.append(result)
                    print(
                        f"\n✅ Scenario {i} completed successfully. Total turns: {result['turns']}"
                    )
                except Exception as e:
                    print(f"\n❌ Error running scenario {i}: {str(e)}")
                    import traceback

                    traceback.print_exc()

        # Save results
        if results:  # Only save if we have results
            timestamp = int(time.time())
            output_file = (
                output_dir
                / f"{args.domain}_{args.category}_{args.model.replace('/', '-')}_{timestamp}.json"
            )

            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)

            print(f"Results saved to {output_file}")
        else:
            print("No results to save - all scenarios failed.")
    except Exception as e:
        print(f"Error occurred during evaluation: {str(e)}")
        import traceback

        traceback.print_exc()


def main():
    """Main entry point."""
    args = parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()

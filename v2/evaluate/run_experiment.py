import argparse
from typing import List
from simulation import run_simulation_experiments
from dotenv import load_dotenv

load_dotenv("../.env")


def parse_list_arg(arg_value: str) -> List[str]:
    """Parse comma-separated list arguments."""
    if not arg_value:
        return []
    return [item.strip() for item in arg_value.split(",")]


def main():
    parser = argparse.ArgumentParser(description="Run agent simulation experiments")

    parser.add_argument(
        "--models",
        type=str,
        default="gpt-4.1-mini-2025-04-14",
        required=True,
        help="Comma-separated list of models to evaluate (e.g., 'gpt-4o,claude-3-opus-20240229')",
    )

    parser.add_argument(
        "--domains",
        type=str,
        default="banking",
        required=True,
        help="Comma-separated list of domains to evaluate (e.g., 'banking,healthcare')",
    )

    parser.add_argument(
        "--categories",
        type=str,
        default="adaptive_tool_use",
        required=True,
        help="Comma-separated list of categories to evaluate (e.g., 'tool_coordination,error_handling')",
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        help="Name of the dataset to use (if not provided, a dataset will be created from scenarios)",
    )

    parser.add_argument(
        "--project_name",
        type=str,
        default=None,
        help="Galileo project name",
    )

    parser.add_argument(
        "--metrics",
        type=str,
        default="tool_selection_quality,agentic_session_success",
        help="Comma-separated list of metrics to evaluate",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    parser.add_argument(
        "--log-to-galileo",
        action="store_true",
        default=False,
        help="Enable logging to Galileo",
    )

    parser.add_argument(
        "--add-timestamp",
        action="store_true",
        default=False,
        help="Add timestamp to experiment name",
    )

    args = parser.parse_args()

    # Parse list arguments
    models = parse_list_arg(args.models)
    domains = parse_list_arg(args.domains)
    categories = parse_list_arg(args.categories)
    metrics = parse_list_arg(args.metrics)

    if args.verbose:
        print(f"Running experiments with:")
        print(f"  Models: {models}")
        print(f"  Domains: {domains}")
        print(f"  Categories: {categories}")
        print(f"  Metrics: {metrics}")
        print(f"  Project: {args.project_name}")
        print(f"  Log to Galileo: {args.log_to_galileo}")
        print(f"  Add timestamp to experiment name: {args.add_timestamp}")
        print(f"  Parallel tool execution: Enabled")
        if args.dataset_name:
            print(f"  Dataset: {args.dataset_name}")

    # Run experiments
    _ = run_simulation_experiments(
        models=models,
        domains=domains,
        categories=categories,
        dataset_name=args.dataset_name,
        project_name=args.project_name,
        metrics=metrics,
        verbose=args.verbose,
        log_to_galileo=args.log_to_galileo,
        add_timestamp=args.add_timestamp,
    )


if __name__ == "__main__":
    main()
#     python run_experiment.py \
#   --models "gpt-4.1-nano-2025-04-14" \
#   --domains "banking" \
#   --categories "tool_coordination" \
#   --project_name "agent-leaderboard-test" \
#   --verbose
#   --log-to-galileo

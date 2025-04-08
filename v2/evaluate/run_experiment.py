import argparse
import json
import os
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
        default="gpt-4o-mini",
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
        default="tool_coordination",
        required=True,
        help="Comma-separated list of categories to evaluate (e.g., 'tool_coordination,error_handling')",
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        help="Name of the dataset to use (if not provided, a dataset will be created from scenarios)",
    )

    parser.add_argument(
        "--project",
        type=str,
        default="agent-leaderboard-v2",
        help="Galileo project name",
    )

    parser.add_argument(
        "--metrics",
        type=str,
        default="tool_selection_quality",
        help="Comma-separated list of metrics to evaluate",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    parser.add_argument(
        "--output-file", type=str, help="File to save experiment results (JSON format)"
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
        print(f"  Project: {args.project}")
        if args.dataset_name:
            print(f"  Dataset: {args.dataset_name}")

    # Run experiments
    results = run_simulation_experiments(
        models=models,
        domains=domains,
        categories=categories,
        dataset_name=args.dataset_name,
        project=args.project,
        metrics=metrics,
        verbose=args.verbose,
    )

    # Format results to include only required fields
    formatted_results = {}

    for result in results:
        exp_data = result["experiment"]
        formatted_results[exp_data.name] = {
            "project_id": exp_data.project_id,
            "id": exp_data.id,
            "name": exp_data.name,
            "created_at": str(exp_data.created_at),
            "link": result["link"],
            "message": result["message"],
        }

    os.makedirs("../data/results", exist_ok=True)
    for exp_name, result in formatted_results.items():
        with open(f"../data/results/{exp_name}.json", "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
#     python run_experiment.py \
#   --models "gpt-4o-mini" \
#   --domains "banking" \
#   --categories "tool_coordination" \
#   --project "test-project" \
#   --verbose

import argparse
import multiprocessing
import os
import random
import itertools
from typing import List, Tuple
from simulation import run_simulation_experiments
from dotenv import load_dotenv
import time

load_dotenv("../.env")


def parse_list_arg(arg_value: str) -> List[str]:
    """Parse comma-separated list arguments."""
    if not arg_value:
        return []
    return [item.strip() for item in arg_value.split(",")]


def run_experiment_worker(args: Tuple):
    """
    Worker function for running a single experiment in a separate process.

    Args:
        args: Tuple containing experiment parameters
            - model: Model to use for the experiment
            - domains: List of domains to evaluate
            - categories: List of categories to evaluate
            - dataset_name: Name of the dataset to use
            - project_name: Galileo project name
            - metrics: List of metrics to evaluate
            - verbose: Whether to print verbose logs
            - log_to_galileo: Whether to log to Galileo
            - add_timestamp: Whether to add timestamp to experiment name
    """
    (
        model,
        domains,
        categories,
        dataset_name,
        project_name,
        metrics,
        verbose,
        log_to_galileo,
        add_timestamp,
    ) = args

    # Run a single experiment with the specified model and one domain/category combination
    run_simulation_experiments(
        models=[model],
        domains=domains,
        categories=categories,
        dataset_name=dataset_name,
        project_name=project_name,
        metrics=metrics,
        verbose=verbose,
        log_to_galileo=log_to_galileo,
        add_timestamp=add_timestamp,
    )





def main():
    parser = argparse.ArgumentParser(
        description="Run parallel agent simulation experiments"
    )

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
        default="banking,telecom,healthcare,insurance,investment",
        required=False,
        help="Comma-separated list of domains to evaluate (e.g., 'banking,healthcare')",
    )

    parser.add_argument(
        "--categories",
        type=str,
        default="adaptive_tool_use",
        required=False,
        help="Comma-separated list of categories to evaluate (e.g., 'tool_coordination,error_handling')",
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        help="Name of the dataset to use (if not provided, a dataset will be created from scenarios)",
    )

    parser.add_argument(
        "--project-name",
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
        default=True,
        help="Add timestamp to experiment name",
    )

    parser.add_argument(
        "--max-processes-per-model",
        type=int,
        default=2,
        help="Maximum number of parallel processes per model",
    )

    parser.add_argument(
        "--parallel-mode",
        type=str,
        choices=["by_model", "by_domain", "by_category", "all"],
        default="all",
        help="How to parallelize experiments: by model, by domain, by category, or all combinations",
    )

    args = parser.parse_args()

    # Parse list arguments
    models = parse_list_arg(args.models)
    domains = parse_list_arg(args.domains)
    categories = parse_list_arg(args.categories)
    metrics = parse_list_arg(args.metrics)

    # Calculate max processes based on number of models and max processes per model
    max_processes = len(models) * args.max_processes_per_model

    if args.verbose:
        print(f"Running parallel experiments with:")
        print(f"  Models: {models}")
        print(f"  Domains: {domains}")
        print(f"  Categories: {categories}")
        print(f"  Metrics: {metrics}")
        print(f"  Project: {args.project_name}")
        print(f"  Log to Galileo: {args.log_to_galileo}")
        print(f"  Add timestamp to experiment name: {args.add_timestamp}")
        print(f"  Parallel tool execution: Enabled")
        print(f"  Max processes per model: {args.max_processes_per_model}")
        print(f"  Total max parallel processes: {max_processes}")
        print(f"  Parallel mode: {args.parallel_mode}")
        if args.dataset_name:
            print(f"  Dataset: {args.dataset_name}")

    start_time = time.time()

    # Organize work based on parallel mode
    if args.parallel_mode == "by_model":
        # Run each model in parallel (with all domain/category combinations)
        with multiprocessing.Pool(
            processes=min(len(models), max_processes)
        ) as pool:
            worker_args = [
                (
                    model,
                    domains,
                    categories,
                    args.dataset_name,
                    args.project_name,
                    metrics,
                    args.verbose,
                    args.log_to_galileo,
                    args.add_timestamp,
                )
                for model in models
            ]

            pool.map(run_experiment_worker, worker_args)

    elif args.parallel_mode == "by_domain":
        # Run each domain in parallel (with all model/category combinations)
        with multiprocessing.Pool(
            processes=min(len(domains), max_processes)
        ) as pool:
            worker_args = [
                (
                    models[0] if len(models) == 1 else models,
                    [domain],
                    categories,
                    args.dataset_name,
                    args.project_name,
                    metrics,
                    args.verbose,
                    args.log_to_galileo,
                    args.add_timestamp,
                )
                for domain in domains
            ]

            pool.map(run_experiment_worker, worker_args)

    elif args.parallel_mode == "by_category":
        # Run each category in parallel (with all model/domain combinations)
        with multiprocessing.Pool(
            processes=min(len(categories), max_processes)
        ) as pool:
            worker_args = [
                (
                    models[0] if len(models) == 1 else models,
                    domains,
                    [category],
                    args.dataset_name,
                    args.project_name,
                    metrics,
                    args.verbose,
                    args.log_to_galileo,
                    args.add_timestamp,
                )
                for category in categories
            ]

            pool.map(run_experiment_worker, worker_args)

    else:  # "all" - run all combinations in parallel
        # Create all experiment combinations
        all_combinations = list(itertools.product(models, domains, categories))
        random.shuffle(all_combinations)
        
        # Prepare worker args for all combinations
        worker_args = [
            (
                model,
                [domain],
                [category],
                args.dataset_name,
                args.project_name,
                metrics,
                args.verbose,
                args.log_to_galileo,
                args.add_timestamp,
            )
            for model, domain, category in all_combinations
        ]

        # Use multiprocessing.Pool to maintain constant number of running processes
        with multiprocessing.Pool(processes=max_processes) as pool:
            pool.map(run_experiment_worker, worker_args)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"All experiments completed in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()

# Example command:
# uv run run_parallel_experiments.py \
#   --models "mistral-small-2506" \
#   --categories adaptive_tool_use \
#   --domains "investment" \
#   --log-to-galileo
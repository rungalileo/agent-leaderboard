"""
Unified results fetching script using cleaner Galileo API.
Replaces both fetch_results.py and update_results_with_parquet_data.py.

Uses:
- get_experiment() + export_records() for detailed span data
- get_experiments() for listing experiments
- Parallel processing with ProcessPoolExecutor (models) and ThreadPoolExecutor (experiments)
- Caching to avoid redundant API calls

Usage:
    python fetch_results_v2.py              # Fetch all models
    python fetch_results_v2.py --version v1 # Fetch only v1 models
    python fetch_results_v2.py --version v2 # Fetch only v2 models
    python fetch_results_v2.py --list-models # List models by version

Output files (always same name, merged incrementally):
    ../data_v2/results.csv      - All experiment data (merged)
    ../data_v2/leaderboard.csv  - Aggregated leaderboard (regenerated from all data)
    ../data_v2/cache/           - Cached experiment data

Workflow:
    1. Run with --version v1 to fetch v1 models → creates results.csv
    2. Run with --version v2 to fetch v2 models → merges into results.csv
    3. Leaderboard always reflects ALL data in results.csv
"""

import argparse
import os
import traceback
import pandas as pd
from galileo.experiments import get_experiment, get_experiments
from galileo.export import export_records
from galileo.projects import get_project
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import time
import json
from pathlib import Path
from datetime import datetime
from functools import wraps

from dotenv import load_dotenv

from model_metadata import model_metadata

load_dotenv("../.env")


# Retry configuration
MAX_RETRIES = 5
RETRY_DELAY = 2  # seconds


def retry_on_failure(func):
    """Decorator to retry a function on failure with exponential backoff."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        last_exception = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < MAX_RETRIES:
                    delay = RETRY_DELAY * attempt  # exponential backoff
                    print(f"    Retry {attempt}/{MAX_RETRIES} after {delay}s: {type(e).__name__}: {str(e)[:100]}")
                    time.sleep(delay)
                else:
                    raise last_exception
    return wrapper


@retry_on_failure
def fetch_project(project_name: str):
    """Fetch project with retry logic."""
    return get_project(name=project_name)


@retry_on_failure
def fetch_experiments(project_id: str):
    """Fetch experiments with retry logic."""
    return get_experiments(project_id=project_id)


@retry_on_failure
def fetch_experiment(project_id: str, experiment_name: str):
    """Fetch single experiment with retry logic."""
    return get_experiment(project_id=project_id, experiment_name=experiment_name)


@retry_on_failure
def fetch_records(project_id: str, experiment_id: str):
    """Fetch experiment records with retry logic. Converts generator to list."""
    return list(export_records(project_id=project_id, experiment_id=experiment_id))

# Configuration
MIN_RESPONSES = 65
DOMAINS = ["banking", "healthcare", "insurance", "investment", "telecom"]
OUTPUT_DIR = Path("../data_v2")
CACHE_DIR = OUTPUT_DIR / "cache"

# Metric name mapping for different API versions
METRIC_MAPPING = {
    "v1": {
        "action_completion": "average_agentic_session_success",
        "tool_selection_quality": "average_tool_selection_quality",
        "total_responses": "total_responses"
    },
    "v2": {
        "action_completion": "average_Action Completion - Agent Leaderboard",
        "tool_selection_quality": "average_Tool Selection Quality - Agent Leaderboard",
        "total_responses": "total_responses"
    }
}


def get_final_model_name(original_model: str) -> str:
    """Derive the final, normalized model name for display and cache directory."""
    # First try to get metadata using the full key
    metadata = model_metadata.get(original_model, {})

    # Extract short name (part after last /)
    short_name = original_model.split('/')[-1] if '/' in original_model else original_model

    # Use model_name from metadata if available, otherwise use short name
    final_model_name = metadata.get('model_name', short_name).lower()
    return final_model_name


def get_metric_value(aggregate_metrics: dict, metric_key: str, api_version: str) -> float | None:
    """Get metric value from aggregate_metrics using the appropriate key for the API version."""
    mapping = METRIC_MAPPING.get(api_version, METRIC_MAPPING["v1"])
    source_key = mapping.get(metric_key)
    if source_key and source_key in aggregate_metrics:
        return aggregate_metrics[source_key]
    return None


def extract_span_metrics(data: list, debug: bool = False) -> dict:
    """Extract average metrics from span data."""
    if not data:
        if debug:
            print(f"    [DEBUG] No data returned from export_records")
        return {}

    try:
        # Debug: show structure of first record
        if debug and len(data) > 0:
            first_record = data[0]
            print(f"    [DEBUG] Data has {len(data)} records")
            print(f"    [DEBUG] First record keys: {list(first_record.keys()) if isinstance(first_record, dict) else type(first_record)}")
            if isinstance(first_record, dict) and "spans" in first_record:
                spans = first_record.get("spans", [])
                print(f"    [DEBUG] First record has {len(spans)} spans")
                if spans:
                    first_span = spans[0]
                    print(f"    [DEBUG] First span keys: {list(first_span.keys()) if isinstance(first_span, dict) else type(first_span)}")
                    output = first_span.get("output")
                    print(f"    [DEBUG] First span output type: {type(output)}")
                    if output:
                        print(f"    [DEBUG] First span output (truncated): {str(output)[:200]}")

        # Parse span outputs - each record has spans array, get first span's output
        span_outputs = []
        for record in data:
            if "spans" in record and record["spans"]:
                output = record["spans"][0].get("output")
                if output:
                    # Output might be JSON string or dict
                    if isinstance(output, str):
                        try:
                            parsed = json.loads(output)
                            span_outputs.append(parsed)
                        except json.JSONDecodeError:
                            continue
                    elif isinstance(output, dict):
                        span_outputs.append(output)

        if not span_outputs:
            if debug:
                print(f"    [DEBUG] No valid span outputs found after parsing")
            return {}

        df = pd.DataFrame(span_outputs)

        if debug:
            print(f"    [DEBUG] DataFrame columns: {list(df.columns)}")

        # Calculate means for available columns
        metrics = {}
        target_columns = [
            'turns_completed',
            'num_input_tokens',
            'num_output_tokens',
            'total_duration_with_tool_calls',
            'total_duration_without_tool_calls'
        ]

        for col in target_columns:
            if col in df.columns:
                metrics[f'avg_{col}'] = round(df[col].mean(), 2)
            else:
                metrics[f'avg_{col}'] = None

        return metrics
    except Exception as e:
        # Only log if there's an actual error message
        if str(e):
            print(f"    Warning: Error extracting span metrics: {type(e).__name__}: {str(e)}")
        return {}


def process_experiment(exp_name: str, project_id: str, project_name: str, original_model: str, api_version: str) -> dict | None:
    """
    Process a single experiment and return combined data.

    Args:
        exp_name: Name of the experiment
        project_id: Project ID from Galileo
        project_name: Name of the project (for logging)
        original_model: Original model key from model_metadata
        api_version: API version for metric name mapping

    Returns:
        dict with all metrics or None if experiment doesn't meet criteria
    """
    try:
        print(f"  Processing experiment: {exp_name}")

        # Get experiment details using project_id (more reliable than project_name)
        experiment = fetch_experiment(project_id=project_id, experiment_name=exp_name)

        if not hasattr(experiment, 'aggregate_metrics') or not experiment.aggregate_metrics:
            print(f"    SKIP {exp_name}: No aggregate_metrics")
            return None

        agg_metrics = experiment.aggregate_metrics.additional_properties

        # Check total responses
        total_responses = get_metric_value(agg_metrics, "total_responses", api_version)
        if total_responses is None:
            print(f"    SKIP {exp_name}: total_responses is None (available keys: {list(agg_metrics.keys())})")
            return None
        if total_responses <= MIN_RESPONSES:
            print(f"    SKIP {exp_name}: only {total_responses} responses (need > {MIN_RESPONSES})")
            return None

        # Get aggregate metrics (AC, TSQ)
        action_completion = get_metric_value(agg_metrics, "action_completion", api_version)
        tool_selection_quality = get_metric_value(agg_metrics, "tool_selection_quality", api_version)

        if action_completion is None:
            print(f"    SKIP {exp_name}: No action_completion metric (available keys: {list(agg_metrics.keys())})")
            return None

        print(f"    OK {exp_name}: {total_responses} responses, AC={action_completion:.2f}")

        # Extract domain and category from experiment name
        parts = exp_name.split("-")
        domain = parts[0] if len(parts) > 0 else ""
        category = parts[1] if len(parts) > 1 else ""

        # Get model metadata (use full key to lookup)
        metadata = model_metadata.get(original_model, {})
        final_model_name = get_final_model_name(original_model)

        # Get span metrics via export_records
        span_metrics = {}
        try:
            data = fetch_records(project_id=experiment.project_id, experiment_id=experiment.id)
            if data:
                # Enable debug for first experiment to see data structure
                debug_mode = os.environ.get("DEBUG_SPANS", "").lower() == "true"
                span_metrics = extract_span_metrics(data, debug=debug_mode)
        except Exception as e:
            if str(e):
                print(f"    Warning: Failed to get span data for {exp_name}: {type(e).__name__}: {str(e)}")

        # Build result
        result = {
            'experiment_name': exp_name,
            'model': final_model_name,
            'category': category,
            'total_responses': total_responses,
            'average_action_completion': round(action_completion, 2) if action_completion else None,
            'average_tool_selection_quality': round(tool_selection_quality, 2) if tool_selection_quality else None,
            # Span-derived metrics
            'avg_turns_completed': span_metrics.get('avg_turns_completed'),
            'avg_num_input_tokens': span_metrics.get('avg_num_input_tokens'),
            'avg_num_output_tokens': span_metrics.get('avg_num_output_tokens'),
            'avg_total_duration_with_tool_calls': span_metrics.get('avg_total_duration_with_tool_calls'),
            'avg_total_duration_without_tool_calls': span_metrics.get('avg_total_duration_without_tool_calls'),
            # Metadata
            'model_type': metadata.get('model_type'),
            'output_type': metadata.get('output_type'),
            'vendor': metadata.get('vendor'),
            'input_cost_per_m_token': metadata.get('input_cost_per_m_token'),
            'output_cost_per_m_token': metadata.get('output_cost_per_m_token'),
            'release_date': metadata.get('release_date'),
            # Cache metadata
            'cached_at': datetime.now().isoformat()
        }

        return result

    except Exception as e:
        print(f"    Error processing experiment {exp_name}: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        return None


def load_from_cache(model_name: str, exp_name: str) -> dict | None:
    """Load cached result for an experiment."""
    cache_file = CACHE_DIR / model_name / f"{exp_name}.json"
    if cache_file.exists():
        try:
            with cache_file.open("r") as f:
                cached = json.load(f)
            if isinstance(cached, dict) and 'experiment_name' in cached:
                return cached
        except Exception as e:
            print(f"    Failed to load cache for {exp_name}: {str(e)}")
    return None


def save_to_cache(model_name: str, result: dict) -> None:
    """Save result to cache."""
    try:
        cache_model_dir = CACHE_DIR / model_name
        cache_model_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_model_dir / f"{result['experiment_name']}.json"
        with cache_file.open("w") as f:
            json.dump(result, f, indent=2)
    except Exception as e:
        print(f"    Failed to save cache for {result.get('experiment_name')}: {str(e)}")


def get_project_name(model_key: str) -> str:
    """
    Get the Galileo project name for a model.
    Replaces '/' with '-' in model_key to match Galileo project naming.

    Example: "moonshotai/Kimi-K2-Instruct" -> "moonshotai-Kimi-K2-Instruct"
    """
    return model_key.replace("/", "-")


def process_model(model_key: str) -> list[dict]:
    """
    Process all experiments for a single model.

    Args:
        model_key: Key from model_metadata dict

    Returns:
        List of experiment result dicts
    """
    # Get model metadata
    metadata = model_metadata.get(model_key, {})
    api_version = metadata.get('api_version', 'v1')
    final_model_name = get_final_model_name(model_key)

    # Get project name (may differ from model_key)
    project_name = get_project_name(model_key)

    print(f"\nProcessing model: {model_key} (project: {project_name}, api_version: {api_version})")

    model_data = []

    # Check if we have a fully cached model (all experiments)
    cache_model_dir = CACHE_DIR / final_model_name
    if cache_model_dir.exists():
        cached_files = list(cache_model_dir.glob("*.json"))
        if cached_files:
            print(f"  Found {len(cached_files)} cached experiments for {final_model_name}")
            # Load all cached data
            for cache_file in cached_files:
                cached = load_from_cache(final_model_name, cache_file.stem)
                if cached:
                    model_data.append(cached)

            if model_data:
                print(f"  Loaded {len(model_data)} experiments from cache")
                return model_data

    # No complete cache - fetch from API
    try:
        project = fetch_project(project_name)
        if not project:
            print(f"  Project not found: {project_name}")
            return []

        project_id = project.id
        experiments = fetch_experiments(project_id=project_id)
        exp_list = list(experiments)
        print(f"  Found {len(exp_list)} experiments")

        # Process experiments with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_exp = {}
            cached_count = 0
            submitted_count = 0

            for exp in exp_list:
                # Check cache first
                cached = load_from_cache(final_model_name, exp.name)
                if cached:
                    model_data.append(cached)
                    cached_count += 1
                    print(f"  [CACHE] {exp.name}")
                else:
                    # Submit for processing - pass project_id for reliable lookup
                    future = executor.submit(
                        process_experiment,
                        exp.name,
                        project_id,
                        project_name,
                        model_key,
                        api_version
                    )
                    future_to_exp[future] = exp.name
                    submitted_count += 1

            print(f"  Status: {cached_count} from cache, {submitted_count} to fetch")

            # Collect results
            success_count = 0
            skip_count = 0
            error_count = 0
            for future in as_completed(future_to_exp):
                exp_name = future_to_exp[future]
                try:
                    result = future.result()
                    if result:
                        model_data.append(result)
                        save_to_cache(final_model_name, result)
                        success_count += 1
                    else:
                        skip_count += 1
                except Exception as e:
                    error_count += 1
                    print(f"    Future error for {exp_name}: {type(e).__name__}: {str(e)}")

            print(f"  Fetch results: {success_count} success, {skip_count} skipped, {error_count} errors")

        print(f"  Completed processing {model_key}: {len(model_data)} valid experiments")

    except Exception as e:
        print(f"  Error processing model {model_key}: {type(e).__name__}: {str(e)}")
        traceback.print_exc()

    return model_data


def get_models_by_version(version_filter: str | None = None, single_model: str | None = None) -> list[str]:
    """
    Get list of models filtered by API version or single model.

    Args:
        version_filter: "v1", "v2", or None for all models
        single_model: If specified, return only this model (for debugging)

    Returns:
        List of model keys
    """
    # Single model overrides version filter
    if single_model:
        if single_model in model_metadata:
            return [single_model]
        else:
            print(f"WARNING: Model '{single_model}' not found in model_metadata!")
            print(f"Available models: {list(model_metadata.keys())}")
            return []

    if version_filter is None:
        return list(model_metadata.keys())

    return [
        model_key for model_key, metadata in model_metadata.items()
        if metadata.get('api_version') == version_filter
    ]


def fetch_all_data(version_filter: str | None = None, single_model: str | None = None) -> list[dict]:
    """
    Main function to fetch all data using parallel processing.

    Args:
        version_filter: "v1", "v2", or None for all models
        single_model: If specified, fetch only this model (for debugging)

    Returns:
        List of all experiment result dicts
    """
    print("Starting data fetching...")
    if single_model:
        print(f"DEBUG MODE: Fetching only model '{single_model}'")
    elif version_filter:
        print(f"Filtering to api_version: {version_filter}")

    start_time = time.time()

    all_data = []
    models = get_models_by_version(version_filter, single_model)
    print(f"Found {len(models)} models to process")

    if not models:
        return []

    # Ensure output directories exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Use ProcessPoolExecutor for models
    with ProcessPoolExecutor(max_workers=min(len(models), 5)) as executor:
        future_to_model = {
            executor.submit(process_model, model): model
            for model in models
        }

        for future in as_completed(future_to_model):
            model = future_to_model[future]
            try:
                model_data = future.result()
                all_data.extend(model_data)
                print(f"Collected {len(model_data)} experiments from model: {model}")
            except Exception as e:
                print(f"Error collecting results for model {model}: {str(e)}")

    end_time = time.time()
    print(f"\nCompleted data fetching in {end_time - start_time:.2f} seconds")
    print(f"Total experiments collected: {len(all_data)}")

    return all_data


def get_release_date_for_model(final_model_name: str) -> str:
    """Look up release date from model_metadata using final model name."""
    # Create a mapping from final model names to release dates
    for model_key, metadata in model_metadata.items():
        if get_final_model_name(model_key) == final_model_name:
            return metadata.get('release_date', '')
    return ''


def create_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create aggregated leaderboard from results DataFrame.

    Args:
        df: Results DataFrame

    Returns:
        Leaderboard DataFrame
    """
    leaderboard_data = []

    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        first_row = model_df.iloc[0]

        # Overall averages
        avg_ac = model_df['average_action_completion'].mean()
        avg_tsq = model_df['average_tool_selection_quality'].mean()
        avg_turns = model_df['avg_turns_completed'].mean()
        avg_duration = model_df['avg_total_duration_without_tool_calls'].mean()
        avg_input_tokens = model_df['avg_num_input_tokens'].mean()
        avg_output_tokens = model_df['avg_num_output_tokens'].mean()

        # Calculate costs
        input_cost_per_m = first_row.get('input_cost_per_m_token')
        output_cost_per_m = first_row.get('output_cost_per_m_token')

        avg_input_cost = None
        avg_output_cost = None
        avg_total_cost = None

        if pd.notna(avg_input_tokens) and pd.notna(input_cost_per_m):
            avg_input_cost = round((avg_input_tokens / 1_000_000) * input_cost_per_m, 4)
        if pd.notna(avg_output_tokens) and pd.notna(output_cost_per_m):
            avg_output_cost = round((avg_output_tokens / 1_000_000) * output_cost_per_m, 4)
        if avg_input_cost is not None and avg_output_cost is not None:
            avg_total_cost = round(avg_input_cost + avg_output_cost, 4)

        # Domain-specific metrics
        domain_metrics = {}
        for domain in DOMAINS:
            domain_df = model_df[model_df['experiment_name'].str.startswith(domain)]
            if len(domain_df) > 0:
                domain_ac = domain_df['average_action_completion'].mean()
                domain_tsq = domain_df['average_tool_selection_quality'].mean()
                domain_turns = domain_df['avg_turns_completed'].mean()
                domain_duration = domain_df['avg_total_duration_without_tool_calls'].mean()
                domain_input = domain_df['avg_num_input_tokens'].mean()
                domain_output = domain_df['avg_num_output_tokens'].mean()

                # Domain cost
                domain_cost = None
                if pd.notna(domain_input) and pd.notna(domain_output) and pd.notna(input_cost_per_m) and pd.notna(output_cost_per_m):
                    domain_input_cost = (domain_input / 1_000_000) * input_cost_per_m
                    domain_output_cost = (domain_output / 1_000_000) * output_cost_per_m
                    domain_cost = round(domain_input_cost + domain_output_cost, 4)

                domain_metrics[domain] = {
                    'ac': round(domain_ac, 2) if pd.notna(domain_ac) else None,
                    'tsq': round(domain_tsq, 2) if pd.notna(domain_tsq) else None,
                    'turns': round(domain_turns, 2) if pd.notna(domain_turns) else None,
                    'duration': round(domain_duration, 2) if pd.notna(domain_duration) else None,
                    'cost': domain_cost,
                    'input_tokens': int(round(domain_input)) if pd.notna(domain_input) else None,
                    'output_tokens': int(round(domain_output)) if pd.notna(domain_output) else None,
                }
            else:
                domain_metrics[domain] = {
                    'ac': None, 'tsq': None, 'turns': None, 'duration': None, 'cost': None,
                    'input_tokens': None, 'output_tokens': None,
                }

        # Build leaderboard row (column order matches reference results_v2.csv)
        row = {
            'Model': model,
            'Vendor': first_row.get('vendor', ''),
            'Avg AC': round(avg_ac, 2) if pd.notna(avg_ac) else None,
            'Avg TSQ': round(avg_tsq, 2) if pd.notna(avg_tsq) else None,
            'Avg Total Cost': avg_total_cost,
            'Avg Session Duration': round(avg_duration, 2) if pd.notna(avg_duration) else None,
            'Avg Turns': round(avg_turns, 2) if pd.notna(avg_turns) else None,
        }

        # Add domain AC columns
        for domain in DOMAINS:
            domain_name = domain.capitalize()
            row[f'{domain_name} AC'] = domain_metrics[domain]['ac']

        # Add domain TSQ columns
        for domain in DOMAINS:
            domain_name = domain.capitalize()
            row[f'{domain_name} TSQ'] = domain_metrics[domain]['tsq']

        # Add cost columns
        row['Avg Input Cost ($)'] = avg_input_cost
        row['Avg Output Cost ($)'] = avg_output_cost

        # Add domain cost, duration, turns
        for domain in DOMAINS:
            domain_name = domain.capitalize()
            row[f'{domain_name} Cost'] = domain_metrics[domain]['cost']

        for domain in DOMAINS:
            domain_name = domain.capitalize()
            row[f'{domain_name} Duration'] = domain_metrics[domain]['duration']

        for domain in DOMAINS:
            domain_name = domain.capitalize()
            row[f'{domain_name} Turns'] = domain_metrics[domain]['turns']

        # Add metadata columns (order matches reference)
        row['Model Type'] = first_row.get('model_type', '')
        row['$/M input token'] = first_row.get('input_cost_per_m_token', '')
        row['$/M output token'] = first_row.get('output_cost_per_m_token', '')
        row['Output Type'] = first_row.get('output_type', '')
        row['Release Date'] = get_release_date_for_model(model)

        # Add token columns (NEW - not in reference)
        row['Avg Input Tokens'] = int(round(avg_input_tokens)) if pd.notna(avg_input_tokens) else None
        row['Avg Output Tokens'] = int(round(avg_output_tokens)) if pd.notna(avg_output_tokens) else None

        for domain in DOMAINS:
            domain_name = domain.capitalize()
            row[f'{domain_name} Input Tokens'] = domain_metrics[domain]['input_tokens']

        for domain in DOMAINS:
            domain_name = domain.capitalize()
            row[f'{domain_name} Output Tokens'] = domain_metrics[domain]['output_tokens']

        leaderboard_data.append(row)

    leaderboard_df = pd.DataFrame(leaderboard_data)

    # Sort by Avg AC descending, then Avg TSQ
    leaderboard_df = leaderboard_df.sort_values(['Avg AC', 'Avg TSQ'], ascending=False)

    return leaderboard_df


def check_domain_coverage(df: pd.DataFrame, models: list[str]) -> None:
    """Check which models don't have runs for all domains."""
    print("\nChecking domain coverage for each model...")

    model_domains = df.groupby('model')['experiment_name'].apply(
        lambda x: set([name.split('-')[0] for name in x])
    ).to_dict()

    all_domains = set(DOMAINS)
    models_missing_domains = []

    for model_key in models:
        final_model_name = get_final_model_name(model_key)

        if final_model_name in model_domains:
            covered = model_domains[final_model_name]
            missing = all_domains - covered

            if missing:
                models_missing_domains.append({
                    'model': model_key,
                    'final_name': final_model_name,
                    'missing_domains': sorted(list(missing)),
                    'covered_domains': sorted(list(covered))
                })
        else:
            models_missing_domains.append({
                'model': model_key,
                'final_name': final_model_name,
                'missing_domains': sorted(list(all_domains)),
                'covered_domains': []
            })

    if models_missing_domains:
        print("\nModels that do NOT have runs for all domains:")
        print("=" * 60)
        for item in models_missing_domains:
            print(f"Model: {item['model']} ({item['final_name']})")
            print(f"  Missing: {item['missing_domains']}")
            print(f"  Covered: {item['covered_domains']}")
            print()
    else:
        print("\nAll models have runs for all domains!")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch experiment results from Galileo API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python fetch_results_v2.py                        # Fetch all models
    python fetch_results_v2.py --version v1           # Fetch only v1 (old metric names)
    python fetch_results_v2.py --version v2           # Fetch only v2 (new metric names)
    python fetch_results_v2.py --model gemini-2.5-pro # Fetch only one model (for debugging)
        """
    )
    parser.add_argument(
        "--version", "-v",
        choices=["v1", "v2"],
        default=None,
        help="Filter models by API version (v1=old metrics, v2=new metrics). Default: all"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Fetch only a specific model (for debugging). Overrides --version filter."
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all models and their API versions, then exit"
    )
    return parser.parse_args()


def list_models():
    """Print all models grouped by API version."""
    v1_models = get_models_by_version("v1")
    v2_models = get_models_by_version("v2")

    print("\n=== Models by API Version ===\n")

    print(f"v1 models ({len(v1_models)}):")
    for model in sorted(v1_models):
        print(f"  - {model}")

    print(f"\nv2 models ({len(v2_models)}):")
    for model in sorted(v2_models):
        print(f"  - {model}")

    print(f"\nTotal: {len(v1_models) + len(v2_models)} models")


def load_existing_results() -> pd.DataFrame | None:
    """Load existing results.csv if it exists."""
    results_path = OUTPUT_DIR / "results.csv"
    if results_path.exists():
        try:
            df = pd.read_csv(results_path)
            print(f"Loaded {len(df)} existing rows from {results_path}")
            return df
        except Exception as e:
            print(f"Failed to load existing results: {e}")
    return None


def merge_results(existing_df: pd.DataFrame | None, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge new results with existing results.
    New data for the same experiment_name will overwrite old data.
    """
    if existing_df is None or existing_df.empty:
        return new_df

    if new_df.empty:
        return existing_df

    # Get experiment names from new data
    new_experiments = set(new_df['experiment_name'].unique())

    # Filter out old data for experiments we're updating
    filtered_existing = existing_df[~existing_df['experiment_name'].isin(new_experiments)]

    # Combine
    merged = pd.concat([filtered_existing, new_df], ignore_index=True)
    print(f"Merged: {len(filtered_existing)} existing + {len(new_df)} new = {len(merged)} total rows")

    return merged


def main():
    """Main entry point."""
    args = parse_args()

    # Handle --list-models flag
    if args.list_models:
        list_models()
        return

    # Fetch data with optional version/model filter
    data = fetch_all_data(version_filter=args.version, single_model=args.model)

    if not data:
        print("No new data collected")
        # Still try to regenerate leaderboard from existing data
        existing_df = load_existing_results()
        if existing_df is not None and not existing_df.empty:
            print("\nRegenerating leaderboard from existing data...")
            leaderboard_df = create_leaderboard(existing_df)
            leaderboard_path = OUTPUT_DIR / "leaderboard.csv"
            leaderboard_df.to_csv(leaderboard_path, index=False)
            print(f"Saved leaderboard with {len(leaderboard_df)} models to {leaderboard_path}")
        return

    # Convert new data to DataFrame
    new_df = pd.DataFrame(data)

    # Load existing results and merge
    existing_df = load_existing_results()
    df = merge_results(existing_df, new_df)

    # Check domain coverage for all models in merged data
    models_to_check = get_models_by_version(args.version, args.model)
    check_domain_coverage(df, models_to_check)

    # Save merged results.csv (no suffix - single combined file)
    results_path = OUTPUT_DIR / "results.csv"
    df.to_csv(results_path, index=False)
    print(f"\nSaved {len(df)} rows to {results_path}")

    # Create and save leaderboard from ALL data
    print("\nCreating leaderboard...")
    leaderboard_df = create_leaderboard(df)
    leaderboard_path = OUTPUT_DIR / "leaderboard.csv"
    leaderboard_df.to_csv(leaderboard_path, index=False)
    print(f"Saved leaderboard with {len(leaderboard_df)} models to {leaderboard_path}")

    # Preview
    print(f"\nResults preview (new data):")
    print(new_df.head(10))

    print(f"\nLeaderboard preview:")
    print(leaderboard_df.head(10))


if __name__ == "__main__":
    main()

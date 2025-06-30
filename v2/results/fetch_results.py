import pandas as pd
from galileo.experiments import get_experiments
from galileo.projects import get_project
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import time

from dotenv import load_dotenv

load_dotenv("../.env")

MIN_RESPONSES = 80

# Model metadata with costs and vendor information
model_metadata = {
    "claude-sonnet-4-20250514": {
        "model_type": "Private",
        "output_type": "Normal", 
        "vendor": "Anthropic",
        "input_cost_per_m_token": 3,
        "output_cost_per_m_token": 15
    },
    "gemini-2.5-pro": {
        "model_type": "Private",
        "output_type": "Reasoning",
        "vendor": "Google", 
        "input_cost_per_m_token": 1.25,
        "output_cost_per_m_token": 10
    },
    "gemini-2.5-flash": {
        "model_type": "Private",
        "output_type": "Reasoning",
        "vendor": "Google",
        "input_cost_per_m_token": 0.3,
        "output_cost_per_m_token": 2.5
    },
    "mistral-small-2506": {
        "model_type": "Open source",
        "output_type": "Normal",
        "vendor": "Mistral",
        "input_cost_per_m_token": 0.1,
        "output_cost_per_m_token": 0.3
    },
    "mistral-medium-2505": {
        "model_type": "Private", 
        "output_type": "Normal",
        "vendor": "Mistral",
        "input_cost_per_m_token": 0.4,
        "output_cost_per_m_token": 2
    },
    "DeepSeek-V3": {  # Maps to deepseek-ai/DeepSeek-V3
        "model_type": "Open source",
        "output_type": "Normal", 
        "vendor": "Deepseek",
        "input_cost_per_m_token": 0.27,
        "output_cost_per_m_token": 1.1
    },
    "Qwen3-235B-A22B-fp8-tput": {
        "model_name": "Qwen3-235B-A22B",
        "model_type": "Open source",
        "output_type": "Reasoning",
        "vendor": "Alibaba",
        "input_cost_per_m_token": 0.2,
        "output_cost_per_m_token": 0.6
    },
    "amazon.nova-pro-v1:0": {  
        "model_name": "nova-pro-v1",
        "model_type": "Private",
        "output_type": "Normal",
        "vendor": "Amazon",
        "input_cost_per_m_token": 0.8,
        "output_cost_per_m_token": 3.2
    },
    "amazon.nova-lite-v1:0": {  
        "model_name": "nova-lite-v1",
        "model_type": "Private", 
        "output_type": "Normal",
        "vendor": "Amazon",
        "input_cost_per_m_token": 0.06,
        "output_cost_per_m_token": 0.24
    },
    "Qwen2.5-72B-Instruct-Turbo": {  # Maps to Qwen/Qwen2.5-72B-Instruct-Turbo
        "model_name": "Qwen2.5-72B-Instruct",
        "model_type": "Open source",
        "output_type": "Normal",
        "vendor": "Alibaba", 
        "input_cost_per_m_token": 0.9,
        "output_cost_per_m_token": 0.9
    },
    "magistral-small-2506": {
        "model_type": "Open source",
        "output_type": "Reasoning",
        "vendor": "Mistral",
        "input_cost_per_m_token": 0.5,
        "output_cost_per_m_token": 1.5
    },
    "magistral-medium-2506": {
        "model_type": "Private",
        "output_type": "Reasoning", 
        "vendor": "Mistral",
        "input_cost_per_m_token": 2,
        "output_cost_per_m_token": 5
    },
    "gpt-4.1-2025-04-14": {
        "model_type": "Private",
        "output_type": "Normal",
        "vendor": "OpenAI",
        "input_cost_per_m_token": 2,
        "output_cost_per_m_token": 8
    },
    "gpt-4.1-mini-2025-04-14": {
        "model_type": "Private",
        "output_type": "Normal", 
        "vendor": "OpenAI",
        "input_cost_per_m_token": 0.4,
        "output_cost_per_m_token": 1.6
    },
    "gpt-4.1-nano-2025-04-14": {
        "model_type": "Private",
        "output_type": "Normal",
        "vendor": "OpenAI", 
        "input_cost_per_m_token": 0.1,
        "output_cost_per_m_token": 0.4
    },
    "caller": {  # Maps to arcee-ai/caller
        "model_type": "Open source",
        "output_type": "Normal",
        "vendor": "Arcee",
        "input_cost_per_m_token": 0.55,
        "output_cost_per_m_token": 0.85
    },
    "granite-3.3-8b-instruct": {  # Maps to ibm-granite/granite-3.3-8b-instruct  
        "model_type": "Open source",
        "output_type": "Normal",
        "vendor": "IBM",
        "input_cost_per_m_token": 0.2,
        "output_cost_per_m_token": 0.2
    }
}

domains = ["banking", "telecom", "healthcare", "insurance", "investment"]
models = [
        "claude-sonnet-4-20250514", "gemini-2.5-pro", 
        "gemini-2.5-flash", "mistral-small-2506", "mistral-medium-2505", "amazon.nova-pro-v1:0", "amazon.nova-lite-v1:0", "deepseek-ai/DeepSeek-V3",
        "Qwen/Qwen2.5-72B-Instruct-Turbo",
        "Qwen/Qwen3-235B-A22B-fp8-tput", "magistral-small-2506", "magistral-medium-2506", 
        "gpt-4.1-2025-04-14", "gpt-4.1-mini-2025-04-14", "gpt-4.1-nano-2025-04-14", "arcee-ai/caller",
        "ibm-granite/granite-3.3-8b-instruct",
        ]

def process_experiment(exp, model):
    """Process a single experiment and return data if it meets criteria"""
    try:
        print(f"Processing experiment: {exp.name} for model: {model}")
        domain, category = exp.name.split("-")[:2]
        
        # Check if aggregate_metrics exists and has the required properties
        if hasattr(exp, 'aggregate_metrics') and exp.aggregate_metrics and hasattr(exp.aggregate_metrics, 'additional_properties'):
            if "total_responses" in exp.aggregate_metrics.additional_properties:
                if exp.aggregate_metrics.additional_properties["total_responses"] > MIN_RESPONSES:
                    # Use the same aggregate_metrics object for getting the metrics
                    if "average_tool_selection_quality" in exp.aggregate_metrics.additional_properties:
                        # Extract model name after first '/' if it exists
                        model_name = model.split('/')[-1] if '/' in model else model
                        
                        # Get metadata for this model
                        metadata = model_metadata.get(model_name, {})
                        
                        # Use model_name from metadata if available, otherwise use processed model_name
                        final_model_name = metadata.get('model_name', model_name).lower()
                        
                        result = {
                            'experiment_name': exp.name, 
                            'total_responses': exp.aggregate_metrics.additional_properties['total_responses'],
                            'average_action_completion': round(exp.aggregate_metrics.additional_properties['average_agentic_session_success'], 2),
                            'average_tool_selection_quality': round(exp.aggregate_metrics.additional_properties['average_tool_selection_quality'], 2), 
                            'model': final_model_name, 
                            'category': category
                        }
                        
                        # Add metadata if available
                        if metadata:
                            result.update({
                                'model_type': metadata.get('model_type'),
                                'output_type': metadata.get('output_type'), 
                                'vendor': metadata.get('vendor'),
                                'input_cost_per_m_token': metadata.get('input_cost_per_m_token'),
                                'output_cost_per_m_token': metadata.get('output_cost_per_m_token')
                            })
                        
                        return result
                    else:
                        print(f"average_tool_selection_quality not found in experiment {exp.name}")
                else:
                    print(f"Experiment {exp.name} has only {exp.aggregate_metrics.additional_properties['total_responses']} responses (< {MIN_RESPONSES})")
            else:
                print(f"total_responses not found in experiment {exp.name}")
        else:
            print(f"aggregate_metrics not properly structured for experiment {exp.name}")
    except Exception as e:
        print(f"Error processing experiment {exp.name}: {str(e)}")
    
    return None

def process_model(model, original_model):
    """Process all experiments for a single model using multithreading"""
    print(f"Starting processing for model: {model}")
    model_data = []
    
    try:
        project_id = get_project(name=model).id
        experiments = get_experiments(project_id=project_id)
        
        # Use ThreadPoolExecutor for experiments within this model
        with ThreadPoolExecutor(max_workers=10) as thread_executor:
            # Submit all experiment processing tasks
            future_to_exp = {
                thread_executor.submit(process_experiment, exp, original_model): exp 
                for exp in experiments
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_exp):
                result = future.result()
                if result is not None:
                    model_data.append(result)
                    
    except Exception as e:
        print(f"Error processing model {model}: {str(e)}")
    
    print(f"Completed processing for model: {model}, found {len(model_data)} valid experiments")
    return model_data

def fetch_all_data():
    """Main function to fetch all data using multiprocessing for models"""
    print("Starting parallel data fetching...")
    start_time = time.time()
    
    all_data = []
    
    # Use ProcessPoolExecutor for models
    with ProcessPoolExecutor(max_workers=min(len(models), 5)) as process_executor:
        # Submit all model processing tasks
        future_to_model = {
            process_executor.submit(process_model, model.replace("/", "-"), model): model 
            for model in models
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_model):
            model = future_to_model[future]
            try:
                model_data = future.result()
                all_data.extend(model_data)
                print(f"Collected {len(model_data)} experiments from model: {model}")
            except Exception as e:
                print(f"Error collecting results for model {model}: {str(e)}")
    
    end_time = time.time()
    print(f"Completed data fetching in {end_time - start_time:.2f} seconds")
    print(f"Total experiments collected: {len(all_data)}")
    
    return all_data

if __name__ == "__main__":
    # Fetch all data using parallel processing
    data = fetch_all_data()
    
    # Convert to DataFrame for further processing if needed
    if data:
        df = pd.DataFrame(data)
        
        # Check which models don't have runs for all domains
        print("\nChecking domain coverage for each model...")
        
        # Get unique combinations of model and domain from the data
        model_domains = df.groupby('model')['experiment_name'].apply(
            lambda x: set([name.split('-')[0] for name in x])
        ).to_dict()
        
        all_domains = set(domains)
        models_missing_domains = []
        
        for model in models:
            # Handle model name formatting (extract name after last "/" to match stored names)
            processed_model = model.split('/')[-1] if '/' in model else model
            
            if processed_model in model_domains:
                model_covered_domains = model_domains[processed_model]
                missing_domains = all_domains - model_covered_domains
                
                if missing_domains:
                    models_missing_domains.append({
                        'model': model,
                        'missing_domains': sorted(list(missing_domains)),
                        'covered_domains': sorted(list(model_covered_domains))
                    })
            else:
                # Model has no experiments at all
                models_missing_domains.append({
                    'model': model,
                    'missing_domains': sorted(list(all_domains)),
                    'covered_domains': []
                })
        
        if models_missing_domains:
            print("\nModels that do NOT have runs for all domains:")
            print("=" * 60)
            for item in models_missing_domains:
                print(f"Model: {item['model']}")
                print(f"  Missing domains: {item['missing_domains']}")
                print(f"  Covered domains: {item['covered_domains']}")
                print()
        else:
            print("\nAll models have runs for all domains!")
        
        # Save DataFrame to CSV
        df.to_csv("../data/results.csv", index=False)
        print(f"\nSaved {len(df)} rows to ../data/results.csv")
        
        # Create leaderboard DataFrame
        print("\nCreating leaderboard...")
        leaderboard_data = []
        
        # Group by model to calculate averages
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            
            # Get metadata for the first row (all rows for same model should have same metadata)
            first_row = model_df.iloc[0]
            
            # Calculate Model Avg (average across all experiments for this model)
            model_avg = model_df['average_action_completion'].mean()
            
            # Calculate domain-specific averages
            domain_scores = {}
            for domain in domains:
                domain_experiments = model_df[model_df['experiment_name'].str.startswith(domain)]
                if len(domain_experiments) > 0:
                    domain_scores[domain] = domain_experiments['average_action_completion'].mean()
                else:
                    domain_scores[domain] = None
            
            # Create leaderboard row
            leaderboard_row = {
                'Model': model.lower(),
                'Model Type': first_row.get('model_type', ''),
                'Model Output Type': first_row.get('output_type', ''),
                'Vendor': first_row.get('vendor', ''),
                'Input cost per M token': first_row.get('input_cost_per_m_token', ''),
                'Output cost per M token': first_row.get('output_cost_per_m_token', ''),
                'Model Avg': round(model_avg, 2) if pd.notna(model_avg) else None,
                'Banking': round(domain_scores['banking'], 2) if domain_scores['banking'] is not None and pd.notna(domain_scores['banking']) else None,
                'Investment': round(domain_scores['investment'], 2) if domain_scores['investment'] is not None and pd.notna(domain_scores['investment']) else None,
                'Telecom': round(domain_scores['telecom'], 2) if domain_scores['telecom'] is not None and pd.notna(domain_scores['telecom']) else None,
                'Healthcare': round(domain_scores['healthcare'], 2) if domain_scores['healthcare'] is not None and pd.notna(domain_scores['healthcare']) else None,
                'Insurance': round(domain_scores['insurance'], 2) if domain_scores['insurance'] is not None and pd.notna(domain_scores['insurance']) else None
            }
            
            leaderboard_data.append(leaderboard_row)
        
        # Create leaderboard DataFrame
        leaderboard_df = pd.DataFrame(leaderboard_data)
        
        # Sort by Model Avg column (descending - highest scores first)
        leaderboard_df = leaderboard_df.sort_values('Model Avg', ascending=False)
        
        # Save leaderboard to CSV
        leaderboard_df.to_csv("../data/leaderboard.csv", index=False)
        print(f"\nSaved leaderboard with {len(leaderboard_df)} models to ../data/leaderboard.csv")
        
        print(f"\nCreated DataFrame with {len(df)} rows")
        print(df.head(200))
        
        print(f"\nLeaderboard preview:")
        print(leaderboard_df.head(10))
    else:
        print("No data collected")
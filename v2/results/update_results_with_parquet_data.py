import pandas as pd
import os
import glob
from pathlib import Path

def process_parquet_files():
    """
    Process all parquet files and extract mean values for the specified columns
    """
    results_dir = "../data/results"
    
    # Target columns to extract means from parquet files
    target_columns = [
        'turns_completed', 
        'num_input_tokens', 
        'num_output_tokens', 
        'total_duration_with_tool_calls', 
        'total_duration_without_tool_calls'
    ]
    
    # Dictionary to store extracted data by experiment name
    parquet_data = {}
    
    # Iterate through all model directories
    for model_dir in os.listdir(results_dir):
        model_path = os.path.join(results_dir, model_dir)
        if os.path.isdir(model_path):
            print(f"Processing model directory: {model_dir}")
            
            # Find all parquet files in this model directory
            parquet_files = glob.glob(os.path.join(model_path, "*.parquet"))
            
            for parquet_file in parquet_files:
                try:
                    # Extract experiment name from filename
                    filename = os.path.basename(parquet_file)
                    experiment_name = filename.replace('.parquet', '')
                    
                    print(f"  Processing file: {filename}")
                    
                    # Read parquet file
                    df = pd.read_parquet(parquet_file)
                    
                    # Calculate means for target columns
                    means = {}
                    for col in target_columns:
                        if col in df.columns:
                            means[f'avg_{col}'] = round(df[col].mean(), 2)
                        else:
                            print(f"    Warning: Column '{col}' not found in {filename}")
                            means[f'avg_{col}'] = None
                    
                    parquet_data[experiment_name] = means
                    
                except Exception as e:
                    print(f"    Error processing {parquet_file}: {str(e)}")
    
    return parquet_data

def update_results_csv(parquet_data):
    """
    Update results.csv with additional columns from parquet data
    """
    # Read current results.csv
    results_df = pd.read_csv("../data/results.csv")
    
    # Add new columns with default values
    new_columns = [
        'avg_turns_completed',
        'avg_num_input_tokens', 
        'avg_num_output_tokens',
        'avg_total_duration_with_tool_calls',
        'avg_total_duration_without_tool_calls'
    ]
    
    for col in new_columns:
        if col not in results_df.columns:
            results_df[col] = None
    
    # Update rows with parquet data
    for idx, row in results_df.iterrows():
        experiment_name = row['experiment_name']
        if experiment_name in parquet_data:
            for col in new_columns:
                results_df.at[idx, col] = parquet_data[experiment_name].get(col)
    
    # Save updated results.csv
    results_df.to_csv("../data/results.csv", index=False)
    print("Updated results.csv with parquet data")
    
    return results_df

def calculate_costs_and_update_leaderboard(results_df):
    """
    Calculate average input cost, output cost, and turns per model and update leaderboard.csv
    """
    # Read current leaderboard.csv
    leaderboard_df = pd.read_csv("../data/leaderboard.csv")
    
    # Convert numeric columns to proper types, handling None/NaN values
    numeric_columns = [
        'avg_turns_completed',
        'avg_num_input_tokens', 
        'avg_num_output_tokens',
        'input_cost_per_m_token',
        'output_cost_per_m_token'
    ]
    
    for col in numeric_columns:
        if col in results_df.columns:
            results_df[col] = pd.to_numeric(results_df[col], errors='coerce')
    
    # Extract domain from experiment_name
    results_df['domain'] = results_df['experiment_name'].str.split('-').str[0]
    
    # Group by model and calculate averages (only for rows with valid data)
    model_stats = results_df.groupby('model').agg({
        'avg_turns_completed': 'mean',
        'avg_num_input_tokens': 'mean',
        'avg_num_output_tokens': 'mean',
        'avg_total_duration_without_tool_calls': 'mean',  # Add session duration average
        'average_tool_selection_quality': 'mean',  # Add TSQ average
        'input_cost_per_m_token': 'first',  # These should be the same for each model
        'output_cost_per_m_token': 'first'
    })
    
    # Calculate average costs per model (only for rows where we have data)
    model_stats['avg_input_cost'] = (
        (model_stats['avg_num_input_tokens'] / 1000000) * 
        model_stats['input_cost_per_m_token']
    ).round(4)
    
    model_stats['avg_output_cost'] = (
        (model_stats['avg_num_output_tokens'] / 1000000) * 
        model_stats['output_cost_per_m_token']
    ).round(4)
    
    model_stats['avg_total_cost'] = (
        model_stats['avg_input_cost'] + model_stats['avg_output_cost']
    ).round(4)
    
    # Round other columns
    model_stats['avg_turns_completed'] = model_stats['avg_turns_completed'].round(2)
    model_stats['avg_total_duration_without_tool_calls'] = model_stats['avg_total_duration_without_tool_calls'].round(2)
    
    # Calculate domain-specific statistics
    domains = ['banking', 'healthcare', 'insurance', 'investment', 'telecom']
    domain_stats = {}
    
    for model in results_df['model'].unique():
        model_df = results_df[results_df['model'] == model]
        domain_stats[model] = {}
        
        for domain in domains:
            domain_df = model_df[model_df['domain'] == domain]
            if len(domain_df) > 0:
                # Calculate domain-specific metrics
                avg_turns = domain_df['avg_turns_completed'].mean()
                avg_duration = domain_df['avg_total_duration_without_tool_calls'].mean()
                avg_input_tokens = domain_df['avg_num_input_tokens'].mean()
                avg_output_tokens = domain_df['avg_num_output_tokens'].mean()
                input_cost_per_m = domain_df['input_cost_per_m_token'].iloc[0]
                output_cost_per_m = domain_df['output_cost_per_m_token'].iloc[0]
                
                # Calculate domain-specific costs
                avg_input_cost = (avg_input_tokens / 1000000) * input_cost_per_m if pd.notna(avg_input_tokens) and pd.notna(input_cost_per_m) else None
                avg_output_cost = (avg_output_tokens / 1000000) * output_cost_per_m if pd.notna(avg_output_tokens) and pd.notna(output_cost_per_m) else None
                avg_total_cost = (avg_input_cost + avg_output_cost) if pd.notna(avg_input_cost) and pd.notna(avg_output_cost) else None
                
                domain_stats[model][domain] = {
                    'turns': round(avg_turns, 2) if pd.notna(avg_turns) else None,
                    'duration': round(avg_duration, 2) if pd.notna(avg_duration) else None,
                    'total_cost': round(avg_total_cost, 4) if pd.notna(avg_total_cost) else None
                }
            else:
                domain_stats[model][domain] = {
                    'turns': None,
                    'duration': None,
                    'total_cost': None
                }
    
    # Add new columns to leaderboard if they don't exist
    new_leaderboard_columns = [
        'Avg Turns',
        'Avg Session Duration',
        'Avg Input Cost ($)',
        'Avg Output Cost ($)',
        'Avg Total Cost ($)'
    ]
    
    # Add domain-specific columns
    for domain in domains:
        domain_name = domain.capitalize()
        new_leaderboard_columns.extend([
            f'{domain_name} Cost',
            f'{domain_name} Duration',
            f'{domain_name} Turns'
        ])
    
    for col in new_leaderboard_columns:
        if col not in leaderboard_df.columns:
            leaderboard_df[col] = None
    
    # Update leaderboard with model statistics
    for idx, row in leaderboard_df.iterrows():
        model_name = row['Model']
        if model_name in model_stats.index:
            leaderboard_df.at[idx, 'Avg Turns'] = model_stats.loc[model_name, 'avg_turns_completed']
            leaderboard_df.at[idx, 'Avg Session Duration'] = round(model_stats.loc[model_name, 'avg_total_duration_without_tool_calls'], 2)
            leaderboard_df.at[idx, 'Avg Input Cost ($)'] = model_stats.loc[model_name, 'avg_input_cost']
            leaderboard_df.at[idx, 'Avg Output Cost ($)'] = model_stats.loc[model_name, 'avg_output_cost']
            leaderboard_df.at[idx, 'Avg Total Cost ($)'] = model_stats.loc[model_name, 'avg_total_cost']
            # Update Avg TSQ if column exists and we have the data
            if 'Avg TSQ' in leaderboard_df.columns:
                tsq_value = model_stats.loc[model_name, 'average_tool_selection_quality']
                leaderboard_df.at[idx, 'Avg TSQ'] = round(tsq_value, 2) if pd.notna(tsq_value) else None
        
        # Update domain-specific columns
        if model_name in domain_stats:
            for domain in domains:
                domain_name = domain.capitalize()
                if domain in domain_stats[model_name]:
                    stats = domain_stats[model_name][domain]
                    leaderboard_df.at[idx, f'{domain_name} Cost'] = stats['total_cost']
                    leaderboard_df.at[idx, f'{domain_name} Duration'] = stats['duration']
                    leaderboard_df.at[idx, f'{domain_name} Turns'] = stats['turns']
                else:
                    leaderboard_df.at[idx, f'{domain_name} Cost'] = None
                    leaderboard_df.at[idx, f'{domain_name} Duration'] = None
                    leaderboard_df.at[idx, f'{domain_name} Turns'] = None
    
    # Clean up any existing duplicate "Avg Total Cost" columns first
    columns_to_drop = [col for col in leaderboard_df.columns if col.startswith('Avg Total Cost') and col != 'Avg Total Cost ($)']
    if columns_to_drop:
        leaderboard_df = leaderboard_df.drop(columns=columns_to_drop)
        print(f"Dropped duplicate columns: {columns_to_drop}")
    
    # Rename columns from old names to new names if they exist (BEFORE organizing columns)
    column_renames = {
        'AC Avg': 'Avg AC',
        'TSQ Avg': 'Avg TSQ',
        'Avg Total Cost ($)': 'Avg Total Cost'
    }
    
    for old_name, new_name in column_renames.items():
        if old_name in leaderboard_df.columns:
            leaderboard_df = leaderboard_df.rename(columns={old_name: new_name})
            print(f"Renamed column '{old_name}' to '{new_name}'")
    
    # Organize columns: group AC columns together, then TSQ columns together
    base_columns = ['Model', 'Vendor']
    
    # AC columns (all accuracy columns together)
    ac_columns = [col for col in leaderboard_df.columns if 'AC' in col and col != 'Model']
    # Get Avg AC and other AC columns separately
    avg_ac = ['Avg AC'] if 'Avg AC' in ac_columns else []
    domain_ac_columns = sorted([col for col in ac_columns if col != 'Avg AC'])
    
    # TSQ columns (all tool selection quality columns together)
    tsq_columns = [col for col in leaderboard_df.columns if 'TSQ' in col]
    # Get Avg TSQ and other TSQ columns separately
    avg_tsq = ['Avg TSQ'] if 'Avg TSQ' in tsq_columns else []
    domain_tsq_columns = sorted([col for col in tsq_columns if col != 'Avg TSQ'])
    
    # Primary performance columns to show right after TSQ
    primary_performance_columns = [
        'Avg Total Cost',
        'Avg Session Duration', 
        'Avg Turns'
    ]
    primary_performance_columns = [col for col in primary_performance_columns if col in leaderboard_df.columns]
    
    # Other performance and cost columns
    other_performance_columns = [
        'Avg Input Cost ($)',
        'Avg Output Cost ($)'
    ]
    other_performance_columns = [col for col in other_performance_columns if col in leaderboard_df.columns]
    
    # Domain-specific performance columns (Cost, Duration, Turns)
    domain_cost_columns = sorted([col for col in leaderboard_df.columns if col.endswith(' Cost') and col != 'Avg Total Cost'])
    domain_duration_columns = sorted([col for col in leaderboard_df.columns if col.endswith(' Duration') and col != 'Avg Session Duration'])
    domain_turns_columns = sorted([col for col in leaderboard_df.columns if col.endswith(' Turns') and col != 'Avg Turns'])
    
    # Columns to keep at the end
    columns_to_keep_at_end = [
        'Model Type',
        'Model Output Type', 
        '$/M input token',
        '$/M output token'
    ]
    columns_to_keep_at_end = [col for col in columns_to_keep_at_end if col in leaderboard_df.columns]
    
    # Final column order: base + Avg AC + Avg TSQ + primary performance + other AC + other TSQ + other performance + domain performance + metadata
    final_column_order = (base_columns + 
                         avg_ac + 
                         avg_tsq + 
                         primary_performance_columns +
                         domain_ac_columns + 
                         domain_tsq_columns + 
                         other_performance_columns +
                         domain_cost_columns +
                         domain_duration_columns +
                         domain_turns_columns +
                         columns_to_keep_at_end)
    
    # Ensure we don't have duplicates and all columns are included
    final_column_order = [col for col in final_column_order if col in leaderboard_df.columns]
    missing_columns = [col for col in leaderboard_df.columns if col not in final_column_order]
    if missing_columns:
        print(f"Warning: Some columns were not included in reordering: {missing_columns}")
        final_column_order.extend(missing_columns)
    
    leaderboard_df = leaderboard_df[final_column_order]
    
    # Save updated results_v2.csv
    leaderboard_df.to_csv("../data/results_v2.csv", index=False)
    print("Updated results_v2.csv with cost and turn statistics")
    
    return leaderboard_df, model_stats

def main():
    """
    Main function to orchestrate the update process
    """
    print("Starting parquet data processing...")
    
    # Step 1: Process all parquet files
    parquet_data = process_parquet_files()
    print(f"Processed {len(parquet_data)} experiment files")
    
    # Step 2: Update results.csv
    updated_results_df = update_results_csv(parquet_data)
    
    # Step 3: Calculate costs and update leaderboard.csv
    updated_leaderboard_df, model_stats = calculate_costs_and_update_leaderboard(updated_results_df)
    
    print("\nModel Statistics Summary:")
    print("=" * 80)
    print(model_stats[['avg_turns_completed', 'avg_total_duration_without_tool_calls', 'avg_input_cost', 'avg_output_cost', 'avg_total_cost']])
    
    print("\nUpdate complete!")
    print("- results.csv updated with parquet data averages")
    print("- results_v2.csv updated with cost and turn statistics")

if __name__ == "__main__":
    main() 
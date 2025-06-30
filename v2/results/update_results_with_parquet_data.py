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
    
    # Group by model and calculate averages (only for rows with valid data)
    model_stats = results_df.groupby('model').agg({
        'avg_turns_completed': 'mean',
        'avg_num_input_tokens': 'mean',
        'avg_num_output_tokens': 'mean',
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
    
    # Add new columns to leaderboard if they don't exist
    new_leaderboard_columns = [
        'Avg Turns',
        'Avg Input Cost ($)',
        'Avg Output Cost ($)',
        'Avg Total Cost ($)'
    ]
    
    for col in new_leaderboard_columns:
        if col not in leaderboard_df.columns:
            leaderboard_df[col] = None
    
    # Update leaderboard with model statistics
    for idx, row in leaderboard_df.iterrows():
        model_name = row['Model']
        if model_name in model_stats.index:
            leaderboard_df.at[idx, 'Avg Turns'] = model_stats.loc[model_name, 'avg_turns_completed']
            leaderboard_df.at[idx, 'Avg Input Cost ($)'] = model_stats.loc[model_name, 'avg_input_cost']
            leaderboard_df.at[idx, 'Avg Output Cost ($)'] = model_stats.loc[model_name, 'avg_output_cost']
            leaderboard_df.at[idx, 'Avg Total Cost ($)'] = model_stats.loc[model_name, 'avg_total_cost']
            # Update TSQ Avg if column exists and we have the data
            if 'TSQ Avg' in leaderboard_df.columns:
                tsq_value = model_stats.loc[model_name, 'average_tool_selection_quality']
                leaderboard_df.at[idx, 'TSQ Avg'] = round(tsq_value, 2) if pd.notna(tsq_value) else None
    
    # Reorder columns to keep specified columns at the end
    columns_to_keep_at_end = [
        'Model Type',
        'Model Output Type', 
        '$/M input token',
        '$/M output token'
    ]
    
    # Get all other columns (excluding the ones to keep at end)
    other_columns = [col for col in leaderboard_df.columns if col not in columns_to_keep_at_end]
    
    # Reorder: other columns first, then the specified columns at the end
    final_column_order = other_columns + [col for col in columns_to_keep_at_end if col in leaderboard_df.columns]
    leaderboard_df = leaderboard_df[final_column_order]
    
    # Save updated leaderboard.csv
    leaderboard_df.to_csv("../data/leaderboard.csv", index=False)
    print("Updated leaderboard.csv with cost and turn statistics")
    
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
    print(model_stats[['avg_turns_completed', 'avg_input_cost', 'avg_output_cost', 'avg_total_cost']])
    
    print("\nUpdate complete!")
    print("- results.csv updated with parquet data averages")
    print("- leaderboard.csv updated with cost and turn statistics")

if __name__ == "__main__":
    main() 
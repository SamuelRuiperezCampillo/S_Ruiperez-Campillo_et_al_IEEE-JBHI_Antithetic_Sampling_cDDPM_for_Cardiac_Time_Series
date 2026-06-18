"""Select patients for the hidden test set (~30% of signals) and plot the distribution.

Faithful migration of
``Diffusion_MAP_fullpipeline_final/select_hidden_test_set.py`` (an argparse
``main()`` entry script). Only mechanical edits were applied: the import was
rewritten to the ``cardiac_map_diffusion`` package layout and this module
docstring was added. All selection logic is otherwise byte-for-byte unchanged.
"""
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# get_MAP_vent_data lives in the diffusion-track loader data_sam (data_diffusion
# only holds the TimeSeriesDataset/IO helpers). Repointed from the mechanical
# data->data_diffusion mapping so this entry script imports cleanly.
from cardiac_map_diffusion.data.data_sam import get_MAP_vent_data

def main():
    parser = argparse.ArgumentParser(description="Select patients for hidden test set (~30% of data).")
    parser.add_argument("--candidate_file", type=str, default=None, help="Optional CSV file with 'pat_ID' of candidate patients to select from.")
    parser.add_argument("--target_percentage", type=float, default=0.30, help="Target percentage of total signals (default: 0.30).")
    parser.add_argument("--output_csv", type=str, default="hidden_test_set_selected.csv", help="Output CSV file name.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Loading dataset...")
    # Handle CWD change in data_sam
    original_cwd = os.getcwd()
    try:
        df_complete = get_MAP_vent_data()
    finally:
        os.chdir(original_cwd)

    total_signals = len(df_complete)
    target_signals = int(total_signals * args.target_percentage)
    
    print(f"Total signals in dataset: {total_signals}")
    print(f"Target signals for hidden set ({args.target_percentage*100}%): {target_signals}")

    # Get counts per patient
    # Ensure pat_ID is string
    df_complete['pat_ID'] = df_complete['pat_ID'].astype(str)
    patient_counts = df_complete['pat_ID'].value_counts()
    
    # Filter candidates if file provided
    if args.candidate_file:
        if os.path.exists(args.candidate_file):
            print(f"Loading candidates from {args.candidate_file}...")
            candidates_df = pd.read_csv(args.candidate_file)
            if 'pat_ID' in candidates_df.columns:
                candidate_ids = candidates_df['pat_ID'].astype(str).unique()
                # Filter patient_counts to only include candidates
                # Note: Some candidates might not be in the dataset (if IDs don't match), check intersection
                valid_candidates = [pid for pid in candidate_ids if pid in patient_counts.index]
                
                if len(valid_candidates) < len(candidate_ids):
                    print(f"Warning: {len(candidate_ids) - len(valid_candidates)} candidates not found in dataset.")
                
                patient_counts = patient_counts[valid_candidates]
                print(f"Candidates available: {len(patient_counts)}")
            else:
                print("Error: Candidate CSV must contain 'pat_ID' column.")
                return
        else:
            print(f"Error: Candidate file {args.candidate_file} not found.")
            return
    else:
        print(f"No candidate file provided. Selecting from all {len(patient_counts)} patients.")

    # Plot Distribution
    plt.figure(figsize=(12, 6))
    plt.hist(patient_counts.values, bins=30, color='skyblue', edgecolor='black')
    plt.title('Distribution of Signals per Patient (Candidates)')
    plt.xlabel('Number of Signals')
    plt.ylabel('Number of Patients')
    plt.grid(axis='y', alpha=0.75)
    
    plot_filename = "patient_signal_distribution.png"
    plt.savefig(plot_filename)
    print(f"Distribution plot saved to {plot_filename}")

    # Selection Algorithm
    # We want to select a subset of 'patient_counts' such that sum(counts) ~ target_signals
    # Since we want a random selection, we can shuffle and accumulate.
    
    patient_ids = list(patient_counts.index)
    random.shuffle(patient_ids)
    
    selected_patients = []
    current_sum = 0
    
    for pid in patient_ids:
        count = patient_counts[pid]
        
        # Check if adding this patient exceeds the target significantly?
        # Or just stop when we cross it?
        # Let's try to get as close as possible.
        
        if current_sum + count <= target_signals:
            selected_patients.append(pid)
            current_sum += count
        else:
            # If we are very close, stop. 
            # If the gap is large, maybe skip this large patient and look for a smaller one?
            # Simple greedy approach: skip if it overshoots, try next.
            if abs((current_sum + count) - target_signals) < abs(current_sum - target_signals):
                 # If adding it makes us closer to target (even if slightly over), take it and stop?
                 # Or just strict "don't exceed too much"?
                 # Let's use a simple logic: if adding it keeps us under target + tolerance, take it.
                 # But "skip if overshoots" is safer to avoid massive overshoot.
                 continue
    
    # Calculate final stats
    final_percentage = (current_sum / total_signals) * 100
    
    print("-" * 30)
    print("Selection Complete")
    print("-" * 30)
    print(f"Selected Patients: {len(selected_patients)}")
    print(f"Selected Signals: {current_sum}")
    print(f"Percentage of Total: {final_percentage:.2f}%")
    print(f"Target was: {args.target_percentage*100}%")
    
    # Save to CSV
    output_df = pd.DataFrame({'pat_ID': selected_patients})
    output_df.to_csv(args.output_csv, index=False)
    print(f"Selected patient list saved to {args.output_csv}")

if __name__ == "__main__":
    main()

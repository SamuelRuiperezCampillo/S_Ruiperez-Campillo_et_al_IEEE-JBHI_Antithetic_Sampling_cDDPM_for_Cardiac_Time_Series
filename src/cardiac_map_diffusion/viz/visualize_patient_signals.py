"""Plot selected MAP signals per patient to a multi-page PDF for visual QC.

Faithful migration of
``Diffusion_MAP_fullpipeline_final/visualize_patient_signals.py``. Only
mechanical edits were applied: the data import was rewritten to the
``cardiac_map_diffusion`` package layout and this module docstring was added.
All plotting logic is otherwise byte-for-byte unchanged.
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import random
import pandas as pd

# Try to import get_MAP_vent_data
try:
    from cardiac_map_diffusion.data.data_sam import get_MAP_vent_data
except ImportError as e:
    import traceback
    traceback.print_exc()
    print(f"Could not import get_MAP_vent_data from data_sam. Error: {e}")
    print("Please ensure the environment is set up correctly.")
    sys.exit(1)

def main():
    # Save the current working directory before data loading changes it
    original_cwd = os.getcwd()
    print(f"Script started in: {original_cwd}")

    # Load data
    print("Loading data...")
    try:
        df_complete = get_MAP_vent_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Get unique patients
    unique_patients = df_complete['pat_ID'].unique()
    print(f"Found {len(unique_patients)} patients.")

    # Select 15 random patients
    # Set seed for reproducibility if needed, or leave random
    # random.seed(42) 
    #if len(unique_patients) < 15:
    #    print(f"Warning: Only {len(unique_patients)} patients found. Selecting all.")
    #    selected_patients = unique_patients
    #else:
        #selected_patients = random.sample(list(unique_patients), 15)
    selected_patients = list(unique_patients)  # Select all patients    
    print(f"Selected patients: {selected_patients}")

    # Create PDF
    pdf_filename = os.path.join(original_cwd, "selected_patient_signals.pdf")
    print(f"Generating PDF: {pdf_filename}")
    
    # Store selected indices for reference
    selection_log = []

    with PdfPages(pdf_filename) as pdf:
        for pat_id in selected_patients:
            # Get signals for this patient
            # We use the original dataframe index to track the global index if needed
            patient_data = df_complete[df_complete['pat_ID'] == pat_id]
            signals = patient_data['MAP_segments'].tolist()
            
            # Get the global indices (from the original dataframe)
            global_indices = patient_data.index.tolist()
            
            N = len(signals)
            if N < 7:
                print(f"Skipping patient {pat_id} (only {N} signals, need at least 7).")
                continue
                
            # Select indices: 3, N/2, N-4 (to match N+3, N/2, N-3 logic roughly)
            # User request: "N+3 N/2 and N-3 where N is the total am ount of indices"
            # Interpreting as: 3rd index, Middle index, 3rd from last index.
            # Using 0-based indexing:
            idx1 = 3
            idx2 = N // 2
            idx3 = N - 4
            
            indices_to_plot = [idx1, idx2, idx3]
            
            fig, axes = plt.subplots(3, 1, figsize=(10, 12))
            fig.suptitle(f"Patient: {pat_id} (Total Signals: {N})", fontsize=16)
            
            for i, idx in enumerate(indices_to_plot):
                signal = signals[idx]
                global_idx = global_indices[idx]
                
                ax = axes[i]
                ax.plot(signal)
                ax.set_title(f"Patient {pat_id} - Local Index: {idx} - Global Index: {global_idx}")
                ax.set_xlabel("Time")
                ax.set_ylabel("Amplitude")
                ax.grid(True)
                
                selection_log.append({
                    'pat_ID': pat_id,
                    'local_index': idx,
                    'global_index': global_idx
                })
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            pdf.savefig(fig)
            plt.close(fig)
            
    print(f"Done. PDF saved to {pdf_filename}")
    
    # Save selection log to CSV
    csv_filename = os.path.join(original_cwd, "selected_indices.csv")
    log_df = pd.DataFrame(selection_log)
    log_df.to_csv(csv_filename, index=False)
    print(f"Selection log saved to {csv_filename}")

if __name__ == "__main__":
    main()

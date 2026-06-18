#!/usr/bin/env python3
"""
Comparison Plotter: DDPM vs VAE vs Wiener vs Butterworth
========================================================
Generates a PDF with 1 page per patient.
- 10 Randomly selected patients from the Test Set (Fold 0).
- 3 Segments per patient (Beginning, Middle, End).
- 4 Columns: DDPM, VAE, Wiener, Butterworth.
- Ensures identical Clean/Noisy signals across all models.

Faithful migration of ``MAP_VAE/test/plot_mechanism_comparison.py``. Only
mechanical edits were applied: the data import was rewritten to the
``cardiac_map_diffusion`` package layout, hardcoded ``/cluster/...`` experiment
roots were routed through ``cardiac_map_diffusion.paths.experiments_root()``, and
this migration note was appended. All plotting logic is otherwise byte-for-byte
unchanged.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import wiener, butter, lfilter, iirnotch

from cardiac_map_diffusion import paths

# Experiment-output root (replaces hardcoded /cluster/.../MAP_VAE/experiments).
_EXP_ROOT = str(paths.experiments_root())

# Import Data Logic
try:
    from cardiac_map_diffusion.data.data_baselines import get_MAP_vent_data
    # We need the exact split logic to map array indices back to patients
    from sklearn.model_selection import KFold
except ImportError:
    print("[ERROR] Could not import data.py from parent directory.")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Filters
# -----------------------------------------------------------------------------
def butterworth_notch(noisy_signal, fs=1000, lowcut=0.01, highcut=400, order=5, f0=60, Q=30.0):
    """Ref: butterworth_notch_filter.py"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # 1. Butterworth
    b_butter, a_butter = butter(order, [low, high], btype='band', analog=False)
    butter_filtered = lfilter(b_butter, a_butter, noisy_signal)

    # 2. Notch
    b_notch, a_notch = iirnotch(f0, Q, fs=fs)
    filtered = lfilter(b_notch, a_notch, butter_filtered)
    return filtered

def apply_wiener(noisy_signal):
    """Apply Wiener filter."""
    return wiener(noisy_signal)

# -----------------------------------------------------------------------------
# Data Mapping
# -----------------------------------------------------------------------------
def get_hidden_patient_mapping():
    """
    Returns the DataFrame for the Hidden Set, matching the order used in generation.
    """
    df = get_MAP_vent_data()
    
    # Try to find the CSV
    csv_name = "hidden_test_set_selected.csv"
    candidates = [
        csv_name
    ]
    
    csv_path = None
    for c in candidates:
        if os.path.exists(c):
            csv_path = c
            break
            
    if csv_path is None:
        print("[ERROR] Could not find hidden_test_set_selected.csv needed for mapping.")
        sys.exit(1)
        
    print(f"[INFO] Using Hidden CSV: {csv_path}")
    exc_df = pd.read_csv(csv_path)
    hidden_ids = exc_df['pat_ID'].astype(str).unique()
    
    df['pat_ID'] = df['pat_ID'].astype(str)
    hidden_df = df[df['pat_ID'].isin(hidden_ids)]
    
    return hidden_df

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_signal(ax, clean, noisy, denoised, title, model_name, color='blue', is_last_row=False):
    x = np.arange(len(clean))
    ax.plot(x, clean, color='black', linewidth=1.5, alpha=0.6, label='Clean')
    ax.plot(x, noisy, color='red', linewidth=0.8, alpha=0.3, label='Noisy')
    ax.plot(x, denoised, color=color, linewidth=1.2, alpha=0.9, label=model_name)
    ax.set_title(title, fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=6)
    if not is_last_row:
        ax.set_xticklabels([])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddpm_npz", default=os.path.join(_EXP_ROOT, "ddpm_with_hidden/denoised_signals/fold0_hidden_test_signals.npz"), help="Path to DDPM Hidden .npz")
    parser.add_argument("--vae_npz", default=os.path.join(_EXP_ROOT, "VAE_with_hidden/denoised_signals/fold0_hidden_test_signals.npz"), help="Path to VAE Hidden .npz")
    parser.add_argument("--wiener_npz", default=os.path.join(_EXP_ROOT, "FILTER_wiener_baseline_nfolds4_noise_allmixed_rs17_rss29_test/denoised_signals/fold0_hidden_test_signals.npz"), help="Path to Wiener Hidden .npz")
    parser.add_argument("--butterworth_npz", default=os.path.join(_EXP_ROOT, "FILTER_butterworth_baseline_nfolds4_noise_allmixed_rs17_rss29_test/denoised_signals/fold0_hidden_test_signals.npz"), help="Path to Butterworth Hidden .npz")
    parser.add_argument("--output", default="figures/hidden_comparison.pdf", help="Output PDF file")
    args = parser.parse_args()
    
    print("Loading Data...")
    try:
        # Load all 4
        ddpm_data = np.load(args.ddpm_npz)
        vae_data = np.load(args.vae_npz)
        wiener_data = np.load(args.wiener_npz)
        bw_data = np.load(args.butterworth_npz)
        
        # Extract Signals
        ddpm_clean = ddpm_data['original_clean']
        ddpm_noisy = ddpm_data['noisy_input']
        ddpm_denoised = ddpm_data['denoised_output']
        
        vae_denoised = vae_data['denoised_output']
        wiener_denoised = wiener_data['denoised_output']
        bw_denoised = bw_data['denoised_output']
        
        # Verify Alignment using Clean Signals
        # Note: Some files might name key differently
        # Assuming all have 'original_clean' or checking alignment against DDPM
        
        # Simple check: Length
        if not (len(ddpm_clean) == len(vae_denoised) == len(wiener_denoised) == len(bw_denoised)):
            print(f"[ERROR] Signal count mismatch: DDPM={len(ddpm_clean)}, VAE={len(vae_denoised)}, Wiener={len(wiener_denoised)}, BW={len(bw_denoised)}")
            sys.exit(1)
            
        print("[INFO] Data loaded and lengths match.")
            
    except Exception as e:
        print(f"[ERROR] Failed to load .npz files: {e}")
        sys.exit(1)
        
    # Get Patient Metadata (For Hidden Set)
    print("Mapping Data to Patients (Hidden Set)...")
    test_df = get_hidden_patient_mapping()
    
    if len(test_df) != len(ddpm_clean):
        print(f"[WARNING] DataFrame length ({len(test_df)}) != Signals length ({len(ddpm_clean)}).")
        print("          This implies the NPZ contains a subset or different split.")
        # Attempt to proceed if indices imply direct mapping, otherwise dangerous.
        # Often with dropped bad signals it might differ?
        # Assuming perfect match for now as per "Test vs Hidden" logic.
    
    # Add index to df to easily grab array rows
    # We assume 'test_df' rows map 1:1 to 'ddpm_clean' rows
    test_df = test_df.reset_index(drop=True)
    test_df['array_idx'] = test_df.index
    
    # 1. Select 10 Patients
    patients = test_df['pat_ID'].unique()
    if len(patients) < 10:
        selected_patients = patients
    else:
        # random.seed(42) # Fixed seed for selection
        selected_patients = random.sample(list(patients), 10)
        
    print(f"Generating PDF for {len(selected_patients)} patients...")
    
    with PdfPages(args.output) as pdf:
        for pat in selected_patients:
            pat_data = test_df[test_df['pat_ID'] == pat]
            
            # Select 3 segments (Beginning, Middle, End)
            if len(pat_data) >= 3:
                indices = [
                    pat_data.iloc[0]['array_idx'],           # Beginning
                    pat_data.iloc[len(pat_data)//2]['array_idx'], # Middle
                    pat_data.iloc[-1]['array_idx']           # End
                ]
                labels = ["Start", "Middle", "End"]
            else:
                # Pad if fewer than 3
                indices = pat_data['array_idx'].tolist()
                while len(indices) < 3:
                    indices.append(indices[-1])
                labels = [f"Seg {i}" for i in range(len(indices))]
            
            # Create Page (3 Rows x 4 Cols)
            fig, axes = plt.subplots(3, 4, figsize=(20, 10))
            fig.suptitle(f"Patient {pat} (Hidden Set)", fontsize=16, fontweight='bold')
            
            # Column headers
            cols = ["DDPM", "VAE", "Wiener", "Butterworth"]
            for ax, col in zip(axes[0], cols):
                ax.set_title(col, fontsize=12, fontweight='bold')

            for row_idx, (idx, label) in enumerate(zip(indices, labels)):
                is_last = (row_idx == len(indices) - 1)
                # Get Signals
                clean = ddpm_clean[idx]
                noisy = ddpm_noisy[idx]
                
                # Models
                sig_ddpm = ddpm_denoised[idx]
                sig_vae = vae_denoised[idx]
                sig_wiener = wiener_denoised[idx]
                sig_butter = bw_denoised[idx]
                
                # Plot
                # Row = Segment, Col = Model
                
                # 1. DDPM
                plot_signal(axes[row_idx, 0], clean, noisy, sig_ddpm, f"{label} - DDPM", "DDPM", color='C0', is_last_row=is_last)
                
                # 2. VAE
                plot_signal(axes[row_idx, 1], clean, noisy, sig_vae, f"{label} - VAE", "VAE", color='C1', is_last_row=is_last)
                
                # 3. Wiener
                # wiener_denoised might be from a baseline run, verify shape
                plot_signal(axes[row_idx, 2], clean, noisy, sig_wiener, f"{label} - Wiener", "Wiener", color='C2', is_last_row=is_last)
                
                # 4. Butterworth
                plot_signal(axes[row_idx, 3], clean, noisy, sig_butter, f"{label} - BW", "BW", color='C4', is_last_row=is_last)
                
                # Add Legend only to first plot of row
                if row_idx == 0:
                     axes[row_idx, 0].legend(loc='upper right', fontsize=6)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            pdf.savefig(fig)
            plt.close()
            
    print(f"Done. Saved to {args.output}")

if __name__ == "__main__":
    main()



#!/usr/bin/env python3
"""
Single Patient Comparison Plotter: DDPM vs VAE vs Wiener vs Butterworth
========================================================================
Generates a PDF with multiple pages for a single patient.
- Focuses on one patient (default: 10044)
- Shows many signal indices from that patient (all available or a configurable max)
- Multiple rows per page, 4 columns: DDPM, VAE, Wiener, Butterworth
- Ensures identical Clean/Noisy signals across all models

Faithful migration of ``MAP_VAE/test/plot_single_patient_comparison.py``. Only
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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import wiener, butter, lfilter, iirnotch

from cardiac_map_diffusion import paths

# Experiment-output root (replaces hardcoded /cluster/.../MAP_VAE/experiments).
_EXP_ROOT = str(paths.experiments_root())

# Import Data Logic
try:
    from cardiac_map_diffusion.data.data_baselines import get_MAP_vent_data
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
def plot_signal(ax, clean, noisy, denoised, title, model_name, color='blue', is_last_row=False, show_legend=False):
    x = np.arange(len(clean))
    ax.plot(x, clean, color='black', linewidth=1.5, alpha=0.6, label='Clean')
    ax.plot(x, noisy, color='red', linewidth=0.8, alpha=0.3, label='Noisy')
    ax.plot(x, denoised, color=color, linewidth=1.2, alpha=0.9, label=model_name)
    ax.set_title(title, fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=6)
    if not is_last_row:
        ax.set_xticklabels([])
    if show_legend:
        ax.legend(loc='upper right', fontsize=6)

def main():
    parser = argparse.ArgumentParser(description="Generate comparison PDF for a single patient with many indices")
    parser.add_argument("--patient_id", default="10044", type=str, help="Patient ID to visualize (default: 10044)")
    parser.add_argument("--max_signals", default=None, type=int, help="Maximum number of signals to plot (default: all available)")
    parser.add_argument("--rows_per_page", default=6, type=int, help="Number of rows (signal indices) per page (default: 6)")
    parser.add_argument("--ddpm_npz", default=os.path.join(_EXP_ROOT, "ddpm_with_hidden/denoised_signals/fold0_hidden_test_signals.npz"), help="Path to DDPM Hidden .npz")
    parser.add_argument("--vae_npz", default=os.path.join(_EXP_ROOT, "VAE_with_hidden/denoised_signals/fold0_hidden_test_signals.npz"), help="Path to VAE Hidden .npz")
    parser.add_argument("--wiener_npz", default=os.path.join(_EXP_ROOT, "FILTER_wiener_baseline_nfolds4_noise_allmixed_rs17_rss29_test/denoised_signals/fold0_hidden_test_signals.npz"), help="Path to Wiener Hidden .npz")
    parser.add_argument("--butterworth_npz", default=os.path.join(_EXP_ROOT, "FILTER_butterworth_baseline_nfolds4_noise_allmixed_rs17_rss29_test/denoised_signals/fold0_hidden_test_signals.npz"), help="Path to Butterworth Hidden .npz")
    parser.add_argument("--output", default=None, help="Output PDF file (default: patient_{ID}_comparison.pdf)")
    args = parser.parse_args()
    
    # Set default output name based on patient ID
    if args.output is None:
        args.output = f"figures/patient_{args.patient_id}_comparison.pdf"
    
    print(f"Single Patient Comparison: Patient {args.patient_id}")
    print("="*60)
    
    # Load Data
    print("Loading Data...")
    try:
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
        
        # Verify Length Match
        if not (len(ddpm_clean) == len(vae_denoised) == len(wiener_denoised) == len(bw_denoised)):
            print(f"[ERROR] Signal count mismatch: DDPM={len(ddpm_clean)}, VAE={len(vae_denoised)}, Wiener={len(wiener_denoised)}, BW={len(bw_denoised)}")
            sys.exit(1)
            
        print(f"[INFO] Data loaded. Total signals: {len(ddpm_clean)}")
            
    except Exception as e:
        print(f"[ERROR] Failed to load .npz files: {e}")
        sys.exit(1)
        
    # Get Patient Metadata (For Hidden Set)
    print("Mapping Data to Patients (Hidden Set)...")
    test_df = get_hidden_patient_mapping()
    
    if len(test_df) != len(ddpm_clean):
        print(f"[WARNING] DataFrame length ({len(test_df)}) != Signals length ({len(ddpm_clean)}).")
        print("          Proceeding assuming direct index mapping.")
    
    # Reset index for mapping
    test_df = test_df.reset_index(drop=True)
    test_df['array_idx'] = test_df.index
    test_df['pat_ID'] = test_df['pat_ID'].astype(str)
    
    # Filter for selected patient
    pat_data = test_df[test_df['pat_ID'] == str(args.patient_id)]
    
    if len(pat_data) == 0:
        print(f"[ERROR] Patient {args.patient_id} not found in the Hidden Set.")
        print(f"        Available patients: {sorted(test_df['pat_ID'].unique().tolist())}")
        sys.exit(1)
    
    print(f"[INFO] Found {len(pat_data)} signals for Patient {args.patient_id}")
    
    # Get all indices for the patient
    all_indices = pat_data['array_idx'].tolist()
    
    # Limit if max_signals is specified
    if args.max_signals is not None and args.max_signals < len(all_indices):
        # Evenly sample across the range
        step = len(all_indices) / args.max_signals
        sampled_positions = [int(i * step) for i in range(args.max_signals)]
        all_indices = [all_indices[pos] for pos in sampled_positions]
        print(f"[INFO] Sampled {len(all_indices)} signals (--max_signals={args.max_signals})")
    
    total_signals = len(all_indices)
    rows_per_page = args.rows_per_page
    num_pages = (total_signals + rows_per_page - 1) // rows_per_page  # Ceiling division
    
    print(f"[INFO] Generating PDF: {num_pages} pages ({rows_per_page} rows per page)")
    
    with PdfPages(args.output) as pdf:
        for page_idx in range(num_pages):
            start_idx = page_idx * rows_per_page
            end_idx = min(start_idx + rows_per_page, total_signals)
            page_indices = all_indices[start_idx:end_idx]
            num_rows = len(page_indices)
            
            # Create Page (num_rows Rows x 4 Cols)
            fig, axes = plt.subplots(num_rows, 4, figsize=(20, 3 * num_rows))
            fig.suptitle(f"Patient {args.patient_id} - Signals {start_idx+1} to {end_idx} of {total_signals}", 
                         fontsize=16, fontweight='bold')
            
            # Ensure axes is 2D even for single row
            if num_rows == 1:
                axes = axes.reshape(1, -1)
            
            # Column headers
            cols = ["DDPM", "VAE", "Wiener", "Butterworth"]
            for ax, col in zip(axes[0], cols):
                ax.annotate(col, xy=(0.5, 1.15), xycoords='axes fraction', 
                           fontsize=12, fontweight='bold', ha='center')

            for row_idx, arr_idx in enumerate(page_indices):
                is_last = (row_idx == num_rows - 1)
                signal_num = start_idx + row_idx + 1
                
                # Get Signals
                clean = ddpm_clean[arr_idx]
                noisy = ddpm_noisy[arr_idx]
                
                # Models
                sig_ddpm = ddpm_denoised[arr_idx]
                sig_vae = vae_denoised[arr_idx]
                sig_wiener = wiener_denoised[arr_idx]
                sig_butter = bw_denoised[arr_idx]
                
                # Plot each column
                # 1. DDPM
                show_legend = (row_idx == 0 and page_idx == 0)
                plot_signal(axes[row_idx, 0], clean, noisy, sig_ddpm, 
                           f"Signal {signal_num} (idx={arr_idx})", "DDPM", 
                           color='C0', is_last_row=is_last, show_legend=show_legend)
                
                # 2. VAE
                plot_signal(axes[row_idx, 1], clean, noisy, sig_vae, 
                           f"Signal {signal_num} (idx={arr_idx})", "VAE", 
                           color='C1', is_last_row=is_last)
                
                # 3. Wiener
                plot_signal(axes[row_idx, 2], clean, noisy, sig_wiener, 
                           f"Signal {signal_num} (idx={arr_idx})", "Wiener", 
                           color='C2', is_last_row=is_last)
                
                # 4. Butterworth
                plot_signal(axes[row_idx, 3], clean, noisy, sig_butter, 
                           f"Signal {signal_num} (idx={arr_idx})", "BW", 
                           color='C4', is_last_row=is_last)

            plt.tight_layout(rect=[0, 0.02, 1, 0.95])
            pdf.savefig(fig)
            plt.close()
            
            print(f"  Page {page_idx+1}/{num_pages} complete")
            
    print("="*60)
    print(f"Done. Saved to {args.output}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Single Signal Comparison: Clean + DDPM + VAE + Wiener + Butterworth + Noisy
===========================================================================
Generates a single-row 6-column figure for one specific signal index.
Also saves each column as a separate SVG.

Faithful migration of ``MAP_VAE/test/plot_single_signal_all_models.py``. Only
mechanical edits were applied: hardcoded ``/cluster/...`` experiment roots were
routed through ``cardiac_map_diffusion.paths.experiments_root()`` and this
migration note was appended. All plotting logic is otherwise byte-for-byte
unchanged.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

from cardiac_map_diffusion import paths

# Experiment-output root (replaces hardcoded /cluster/.../MAP_VAE/experiments).
_EXP_ROOT = str(paths.experiments_root())

def plot_signal_single(ax, signal, title, color='black', linewidth=1.2, noisy_bg=None, alpha_main=1.0):
    """Plot a single signal without legend, axes, or text. Optionally show noisy background."""
    x = np.arange(len(signal))
    if noisy_bg is not None:
        # Noisy background: RGB(255, 170, 0) -> (1.0, 0.667, 0.0), alpha 0.2
        ax.plot(x, noisy_bg, color='black', linewidth=0.8, alpha=0.2)
    ax.plot(x, signal, color=color, linewidth=linewidth, alpha=alpha_main)
    ax.axis('off')

def main():
    parser = argparse.ArgumentParser(description="Generate 6-column SVG for a single signal index")
    parser.add_argument("--idx", default=610, type=int, help="Array index of signal (default: 610)")
    parser.add_argument("--ddpm_npz", default=os.path.join(_EXP_ROOT, "ddpm_with_hidden/denoised_signals/fold0_hidden_test_signals.npz"))
    parser.add_argument("--vae_npz", default=os.path.join(_EXP_ROOT, "VAE_with_hidden/denoised_signals/fold0_hidden_test_signals.npz"))
    parser.add_argument("--wiener_npz", default=os.path.join(_EXP_ROOT, "FILTER_wiener_baseline_nfolds4_noise_allmixed_rs17_rss29_test/denoised_signals/fold0_hidden_test_signals.npz"))
    parser.add_argument("--butterworth_npz", default=os.path.join(_EXP_ROOT, "FILTER_butterworth_baseline_nfolds4_noise_allmixed_rs17_rss29_test/denoised_signals/fold0_hidden_test_signals.npz"))
    parser.add_argument("--output_dir", default="figures", help="Output directory for SVG files")
    parser.add_argument("--output_prefix", default="signal", help="Prefix for output filenames")
    args = parser.parse_args()
    
    idx = args.idx
    print(f"Single Signal Visualization: idx={idx}")
    print("="*60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Data
    print("Loading Data...")
    try:
        ddpm_data = np.load(args.ddpm_npz)
        vae_data = np.load(args.vae_npz)
        wiener_data = np.load(args.wiener_npz)
        bw_data = np.load(args.butterworth_npz)
        
        clean = ddpm_data['original_clean'][idx]
        noisy = ddpm_data['noisy_input'][idx]
        ddpm_denoised = ddpm_data['denoised_output'][idx]
        vae_denoised = vae_data['denoised_output'][idx]
        wiener_denoised = wiener_data['denoised_output'][idx]
        bw_denoised = bw_data['denoised_output'][idx]
        
        print(f"[INFO] Loaded signal idx={idx}, length={len(clean)}")
        
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        sys.exit(1)
    
    # Define colors
    # Noisy: RGB(255, 170, 0) -> (1.0, 0.667, 0.0)
    # Denoised: RGB(140, 180, 225) -> (140/255, 180/255, 225/255) -> (0.549, 0.706, 0.882)
    # Clean: black
    
    color_noisy = 'black'#(1.0, 125/255, 50/255)
    color_denoised = 'black'#(0.549, 0.706, 0.882)
    color_clean = 'black'

    # Columns: name, signal, color, noisy_bg (optional)
    columns = [
        ("Clean", clean, color_clean, None),
        ("DDPM", ddpm_denoised, color_denoised, noisy),
        ("VAE", vae_denoised, color_denoised, noisy),
        ("Wiener", wiener_denoised, color_denoised, noisy),
        ("Butterworth", bw_denoised, color_denoised, noisy),
        ("Noisy", noisy, color_noisy, None), # Noisy column uses noisy color directly
    ]
    
    # ========== Combined 6-column figure ==========
    print("Generating combined 6-column figure...")
    fig, axes = plt.subplots(1, 6, figsize=(16, 2.5), dpi=150)
    # Reduce space between columns (wspace)
    plt.subplots_adjust(wspace=0.001, left=0.01, right=0.99)
    
    for ax, (name, signal, color, noisy_bg) in zip(axes, columns):
        # Special handling for "Noisy" column to ensure alpha=0.5 applies to the main plot line if needed
        # But user requested "noisy signal color ... RGB ... and alpha 0.50"
        # Since plot_signal_single uses alpha=1.0 by default for 'signal', we pass a color tuple with alpha if needed
        # Matplotlib color tuples are (R, G, B, A) or we can just rely on the color being correct.
        # But 'plot' kwarg 'alpha' overrides.
        
        current_alpha = 1.0
        if name == "Noisy":
             current_alpha = 0.7
        
        # 'plot_signal_single' hardcodes plotting the 'signal' with the passed color.
        # So we just pass the color. For alpha, we need to modify plot_signal_single or specific call.
        # Let's modify the call inside the loop to support alpha for the main signal if needed?
        # Actually plot_signal_single takes linewidth but not alpha for the main signal.
        # I will update plot_signal_single to take an optional alpha for the main signal.
        pass

    for ax, (name, signal, color, noisy_bg) in zip(axes, columns):
        # Determine specific alpha for this column's main signal
        alpha_main = 0.5 if name == "Noisy" else 1.0
        plot_signal_single(ax, signal, name, color=color, linewidth=1.0, noisy_bg=noisy_bg, alpha_main=alpha_main)
    
    # plt.tight_layout() # tight_layout might override subplots_adjust, so we might skip it or use rect
    # To be safe with wspace, we skip standard tight_layout or use it with pad argument. 
    # But subplots_adjust is explicit. Let's try removing tight_layout call here to respect subplots_adjust.
    
    combined_path = os.path.join(args.output_dir, f"{args.output_prefix}_idx{idx}_combined.svg")
    plt.savefig(combined_path, format='svg', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {combined_path}")
    
    # ========== Individual SVG for each column ==========
    print("Generating individual SVG files...")
    for name, signal, color, noisy_bg in columns:
        fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.5), dpi=150)
        alpha_main = 0.7 if name == "Noisy" else 1.0
        plot_signal_single(ax, signal, name, color=color, linewidth=1.2, noisy_bg=noisy_bg, alpha_main=alpha_main)
        plt.tight_layout()
        
        # Clean filename (replace spaces, lowercase)
        safe_name = name.lower().replace(" ", "_")
        individual_path = os.path.join(args.output_dir, f"{args.output_prefix}_idx{idx}_{safe_name}.svg")
        plt.savefig(individual_path, format='svg', bbox_inches='tight')
        plt.close()
        print(f"  Saved: {individual_path}")
    
    print("="*60)
    print(f"Done. All files saved to: {args.output_dir}")

if __name__ == "__main__":
    main()

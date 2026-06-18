"""Generate the main MAP-denoising LaTeX results table from per-method summary xlsx.

Faithful migration of ``MAP_VAE/generate_latex_table.py`` (a bare top-level
script run via ``__main__``). Only mechanical edits were applied: the hardcoded
``/cluster/.../MAP_VAE/experiments`` roots in ``ROW_FILES`` were routed through
``cardiac_map_diffusion.paths.experiments_root()`` (joined with the unchanged
relative sub-paths) and this module docstring was added. All table logic is
otherwise byte-for-byte unchanged.
"""
import pandas as pd
import os
import argparse

from cardiac_map_diffusion import paths

# Experiment-output root (replaces hardcoded /cluster/.../MAP_VAE/experiments).
_EXP_ROOT = str(paths.experiments_root())

# ==========================================
# CONFIGURATION
# ==========================================

# Define the rows of the table and the corresponding file paths.
# Set the path to None or "" if the file is not yet available (will show as TBD).
ROW_FILES = {
    "cDDPM (AV)": os.path.join(_EXP_ROOT, "ddpm_with_hidden/summary_values.xlsx"),
    "cDDPM (MC)": r"",
    "$\\beta$-VAE": os.path.join(_EXP_ROOT, "VAE_with_hidden/summary_values.xlsx"), # Example: r"c:\path\to\vae_summary.xlsx"
    "Butterworth": os.path.join(_EXP_ROOT, "FILTER_butterworth_baseline_nfolds4_noise_allmixed_rs17_rss29_test/FILTER_butterworth_butterworth_baseline_summary_simplified.xlsx"),
    "Wiener": os.path.join(_EXP_ROOT, "FILTER_wiener_baseline_nfolds4_noise_allmixed_rs17_rss29_test/FILTER_wiener_wiener_baseline_summary_simplified.xlsx"),
    "TV": os.path.join(_EXP_ROOT, "FILTER_tv_l1_baseline_nfolds4_noise_allmixed_rs17_rss29_test/FILTER_tv_tv_l1_baseline_summary_simplified.xlsx"),
    "Savitzky": os.path.join(_EXP_ROOT, "FILTER_hybrid_savgol_median_baseline_nfolds4_noise_allmixed_rs17_rss29_test/FILTER_hybrid_hybrid_savgol_median_baseline_summary_simplified.xlsx"),
    "Wavelet": os.path.join(_EXP_ROOT, "FILTER_wavelet_baseline_nfolds4_noise_allmixed_rs17_rss29_test/FILTER_wavelet_wavelet_baseline_summary_simplified.xlsx"),
    "Notch": os.path.join(_EXP_ROOT, "FILTER_adaptive_notch_baseline_nfolds4_noise_allmixed_rs17_rss29_test/FILTER_adaptive_adaptive_notch_baseline_summary.xlsx"),
    "U-Net": os.path.join(_EXP_ROOT, "LUNet_baseline_1000max_nfolds4_bs32_lr_5e-5_noise_allmixed_rs17_rss29_es10_test/LUNet_baseline_allmixed_es10_test/LUNet_baseline_summary_values_simplified.xlsx"),
    "AE": os.path.join(_EXP_ROOT, "DAE_baseline_10000max_nfolds4_bs32_lr_5e-5_noise_allmixed_rs17_rss29_ls32_es10_test/DAE_baseline_allmixed_ls32_test/DAE_baseline_summary_values_simplified.xlsx"),
    "D-RNN": os.path.join(_EXP_ROOT, "DRRN_baseline_1000max_nfolds4_bs32_lr_5e-5_noise_allmixed_rs17_rss29_hs64_es10_test/DRRN_baseline_allmixed_hs64_es10/DRRN_baseline_summary_values_simplified.xlsx"),
    "Noisy": os.path.join(_EXP_ROOT, "FILTER_none_baseline_nfolds4_noise_allmixed_rs17_rss29_test/FILTER_none_none_baseline_summary_simplified.xlsx")
}

ROW_FILES = {
    "cDDPM (AV)": os.path.join(_EXP_ROOT, "ddpm_corrected/summary_values.xlsx"),
    "cDDPM (MC)": r"",
    "$\\beta$-VAE": os.path.join(_EXP_ROOT, "2023_0808_allmixed_cyclbeta4_init_1lrelu_50_nfolds4_withepbs_32_lr_5e-3_noise_allmixed_split100_rs17_rss29_test_2/allmixed_beta_0.005/summary_values.xlsx"), # Example: r"c:\path\to\vae_summary.xlsx"
    "Butterworth": os.path.join(_EXP_ROOT, "FILTER_butterworth_baseline_nfolds4_noise_allmixed_rs17_rss29_test_2/butterworth_baseline_summary.xlsx"),
    "Wiener": os.path.join(_EXP_ROOT, "FILTER_wiener_baseline_nfolds4_noise_allmixed_rs17_rss29_test_2/wiener_baseline_summary.xlsx"),
    "TV": os.path.join(_EXP_ROOT, "FILTER_tv_l1_baseline_nfolds4_noise_allmixed_rs17_rss29_test_2/tv_l1_baseline_summary.xlsx"),
    "Savitzky": os.path.join(_EXP_ROOT, "FILTER_hybrid_savgol_median_baseline_nfolds4_noise_allmixed_rs17_rss29_test_2/hybrid_savgol_median_baseline_summary.xlsx"),
    "Wavelet": os.path.join(_EXP_ROOT, "FILTER_wavelet_baseline_nfolds4_noise_allmixed_rs17_rss29_test_2/wavelet_baseline_summary.xlsx"),
    #"Notch": r"/cluster/work/vogtlab/Group/pblasco/MAP_VAE/experiments/FILTER_adaptive_notch_baseline_nfolds4_noise_allmixed_rs17_rss29_test/FILTER_adaptive_adaptive_notch_baseline_summary.xlsx",
    "U-Net": os.path.join(_EXP_ROOT, "LUNet_baseline_1000max_nfolds4_bs32_lr_5e-5_noise_allmixed_rs17_rss29_es10_test_2/LUNet_baseline_allmixed_es10_test/summary_values.xlsx"),
    #"AE": r"/cluster/work/vogtlab/Group/pblasco/MAP_VAE/experiments/DAE_baseline_10000max_nfolds4_bs32_lr_5e-5_noise_allmixed_rs17_rss29_ls32_es10_test/DAE_baseline_allmixed_ls32_test/DAE_baseline_summary_values_simplified.xlsx",
    "D-RNN": os.path.join(_EXP_ROOT, "DRRN_baseline_1000max_nfolds4_bs32_lr_5e-5_noise_allmixed_rs17_rss29_hs64_es10_test_2/DRRN_baseline_allmixed_hs64_es10/summary_values.xlsx"),
    "Noisy": os.path.join(_EXP_ROOT, "FILTER_none_baseline_nfolds4_noise_allmixed_rs17_rss29_test_2/none_baseline_summary.xlsx")
}




# Metric configuration: (Excel Column Name, Scaling Factor)

# ==========================================
# REPORTING MODE
# ==========================================
# Set to True to report Hidden Test metrics
# Set to False to report Validation/Test metrics (e.g. from Cross-Validation)
REPORT_HIDDEN_TEST = False 

if REPORT_HIDDEN_TEST:
    print(">>> MODE: Generating Latex Table for HIDDEN TEST Set")
    METRICS_CONFIG = [
        ("rmse_hidden", 1000),      # RMSE (x10^-3)
        ("pcorr_hidden", 100),      # PCC (x10^2)
        ("psnr_hidden", 1),         # PSNR
        ("spearman_hidden", 100),   # SRC (x10^2)
        ("snr_hidden", 1),          # SNR
        ("nmae_range_hidden", 100), # NMAE (x10^2) - Using Range normalization as default
        ("lsd_hidden", 1),          # LSD
        ("dtw_hidden", 1)           # DTW
    ]
else:
    print(">>> MODE: Generating Latex Table for VALIDATION / STANDARD TEST Set")
    METRICS_CONFIG = [
        ("rmse_test", 1000),      # RMSE (x10^-3)
        ("pcorr_test", 100),      # PCC (x10^2)
        ("psnr_test", 1),         # PSNR
        ("spearman_test", 100),   # SRC (x10^2)
        ("snr_test", 1),          # SNR
        ("nmae_range_test", 100), # NMAE (x10^2) - Using Range normalization as default
        ("lsd_test", 1),          # LSD
        ("dtw_test", 1)           # DTW
    ]

def format_value(avg, std):
    """Formats the average and standard deviation for LaTeX."""
    return f"{avg:.2f} \\scriptsize{{$\\pm$ {std:.2f}}}"

def generate_latex_table(row_files):
    # LaTeX Header
    latex_code = [
        r"\begin{table*}[th]",
        r"\centering",
        r"\caption{MAP Denoising Evaluation on the Test Set}",
        r"\label{tab:map_results_test}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lllllllll}",
        r"\toprule",
        r"Labels & RMSE \scriptsize{(x$10^{-3})$} $\downarrow$ & PCC \scriptsize{(x$10^{2})$} $\uparrow$ & PSNR $\uparrow$ & SRC $\uparrow$ \scriptsize{(x$10^{2})$} & SNR $\uparrow$ & NMAE $\downarrow$ \scriptsize{(x$10^{2})$} & LSD $\downarrow$ & DTW $\downarrow$ \\",
        r"\midrule"
    ]

    for label, file_path in row_files.items():
        row_str = f"{label}"
        
        if not file_path or not os.path.exists(file_path):
            # TBD if file is missing
            tbd_cols = ["TBD"] * len(METRICS_CONFIG)
            row_str += " & " + " & ".join(tbd_cols) + " \\\\"
        else:
            try:
                # Read Excel file
                df = pd.read_excel(file_path)
                
                # Identify column used for 'Split' / 'fold'
                possible_id_cols = ['Split', 'fold', 'condition']
                id_col = next((col for col in possible_id_cols if col in df.columns), None)
                
                if not id_col:
                    print(f"Warning: Could not find identifier column in {file_path}. Filling TBD.")
                    tbd_cols = ["TBD"] * len(METRICS_CONFIG)
                    row_str += " & " + " & ".join(tbd_cols) + " \\\\"
                    latex_code.append(row_str)
                    continue

                # Get Average and Std Dev rows
                # Convert to string to ensure matching "average" or "st. dev."
                df[id_col] = df[id_col].astype(str)
                
                avg_row = df[df[id_col].str.lower().str.contains('average|mean')]
                std_row = df[df[id_col].str.lower().str.contains('st. dev.|std|standard deviation')]
                
                if avg_row.empty or std_row.empty:
                    print(f"Warning: Could not find 'average' or 'st. dev.' rows in {file_path}. Filling TBD.")
                    tbd_cols = ["TBD"] * len(METRICS_CONFIG)
                    row_str += " & " + " & ".join(tbd_cols) + " \\\\"
                    latex_code.append(row_str)
                    continue
                
                # Process each metric
                metric_values = []
                for metric_col, scale in METRICS_CONFIG:
                    col_to_use = None
                    
                    # Robust candidate search to distinguish test/val vs hidden
                    candidates = [metric_col]
                    
                    if "_hidden_test" in metric_col:
                         # Specific hidden test variants
                         candidates.append(metric_col.replace("_hidden_test", "_hidden"))
                    elif "_test" in metric_col and "_hidden" not in metric_col:
                         # Standard test variants (exclude hidden)
                         candidates.append(metric_col.replace("_test", "_val"))
                    elif "_val" in metric_col and "_hidden" not in metric_col:
                         # Standard val variants (exclude hidden)
                         candidates.append(metric_col.replace("_val", "_test"))

                    for cand in candidates:
                        if cand in df.columns:
                            col_to_use = cand
                            break
                                
                    if col_to_use:
                        val_avg = avg_row.iloc[0][col_to_use]
                        val_std = std_row.iloc[0][col_to_use]
                        
                        # Handle comma decimals if read as strings
                        def clean_val(v):
                            if isinstance(v, str):
                                return float(v.replace(',', '.'))
                            return v
                            
                        val_avg = clean_val(val_avg) * scale
                        val_std = clean_val(val_std) * scale
                        
                        metric_values.append(format_value(val_avg, val_std))
                    else:
                        print(f"Warning: Metric '{metric_col}' (candidates: {candidates}) not found in {file_path}.")
                        metric_values.append("N/A")
                
                row_str += " & " + " & ".join(metric_values) + " \\\\"
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                tbd_cols = ["ERROR"] * len(METRICS_CONFIG)
                row_str += " & " + " & ".join(tbd_cols) + " \\\\"

        latex_code.append(row_str)

    # LaTeX Footer
    latex_code.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}"
    ])
    
    # print footer
    # latex_code.append(r"\end{table*}") # User didn't include this in snippet, but valid latex needs it. Keeping snippet format.
    
    return "\n".join(latex_code)

if __name__ == "__main__":
    print("Generating LaTeX table...")
    table_latex = generate_latex_table(ROW_FILES)
    print("\n" + "="*50)
    print("LATEX OUTPUT:")
    print("="*50 + "\n")
    print(table_latex)
    print("\n" + "="*50)
    
    # Optional: Save to file
    with open("results_table.tex", "w") as f:
        f.write(table_latex)
    print(f"Saved to {os.path.abspath('results_table.tex')}")

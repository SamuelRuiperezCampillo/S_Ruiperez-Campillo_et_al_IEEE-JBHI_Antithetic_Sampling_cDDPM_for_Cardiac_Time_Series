"""Generate the AV-vs-Crude-MC sampling LaTeX table (MAP metrics + hardcoded ECG).

Faithful migration of ``MAP_VAE/generate_sampling_table.py`` (a bare top-level
script run via ``__main__``). Only mechanical edits were applied: the hardcoded
``/cluster/.../MAP_VAE/experiments`` ``BASE_PATH`` was routed through
``cardiac_map_diffusion.paths.experiments_root()`` and this module docstring was
added. All table logic is otherwise byte-for-byte unchanged.
"""
import pandas as pd
import os
import argparse

from cardiac_map_diffusion import paths

# ==========================================
# CONFIGURATION
# ==========================================

# Base Directory for Experiments
BASE_PATH = str(paths.experiments_root())
DDPM_DIR = os.path.join(BASE_PATH, "ddpm_with_hidden")

# Experiment Map: (Pairs, Sampling Type) -> Subfolder Name (or full path to summary file)
# Note: "Shots" in the table corresponds to "Pairs" in the filenames.
# "cDDPM (crude)" -> "simple" sampling
# "cDDPM (AV)"    -> "antithetic" sampling

# Example expected paths: 
# .../ddpm_with_hidden_inference_simple_2pairs/summary_values.xlsx
# .../ddpm_with_hidden_inference_antithetic_2pairs/summary_values.xlsx

# MAPPING LOGIC:
# Shots in table = Total Runs
# For "simple" (crude): Runs = Pairs (actually usually 2*Pairs in code but let's map file to table row)
# User instruction: "pairs are shots/2" -> So Shots=4 means look for ..._2pairs...
# BUT wait, the file paths have explicit pair numbers.
# Let's map Table Row (Shots) -> File Path directly based on user provided list.

EXPERIMENTS = {
    # Shots = 2 (Means Pairs=1)
    (2, "cDDPM (crude)"): "ddpm_corrected_inference_simple_1pairs_20260121_215033",
    (2, "cDDPM (AV)"):    "ddpm_corrected_inference_antithetic_1pairs_20260121_215030",
    
    # Shots = 4 (Means Pairs=2)
    (4, "cDDPM (crude)"): "ddpm_corrected_inference_simple_2pairs_20260121_215033",
    (4, "cDDPM (AV)"):    "ddpm_corrected_inference_antithetic_2pairs_20260121_215033",
    
    # Shots = 6 (Means Pairs=3) - Adding this if needed or just sticking to table rows 2, 4, 10, 14
    # User provided 3pairs paths. 
     (6, "cDDPM (crude)"): "ddpm_corrected_inference_simple_3pairs_20260121_215033",
     (6, "cDDPM (AV)"):    "ddpm_corrected_inference_antithetic_3pairs_20260121_215033",

    # Shots = 8 (Means Pairs=4) - Provided but maybe not in table?
     (8, "cDDPM (crude)"): "ddpm_corrected_inference_simple_4pairs_20260121_215031",
     (8, "cDDPM (AV)"):    "ddpm_corrected_inference_antithetic_4pairs_20260121_215033",

    # Shots = 10 (Means Pairs=5)
    (10, "cDDPM (crude)"): "ddpm_corrected_inference_antithetic_5pairs_20260121_220328",
    (10, "cDDPM (AV)"):    "ddpm_corrected_inference_antithetic_5pairs_20260121_220328",
    
    # Shots = 12 (Means Pairs=6)
     (12, "cDDPM (crude)"): "ddpm_corrected_inference_simple_6pairs_20260121_220328",
     (12, "cDDPM (AV)"):    "ddpm_corrected_inference_antithetic_6pairs_20260121_220328",

    # Shots = 14 (Means Pairs=7)
    (14, "cDDPM (crude)"): "ddpm_corrected_inference_simple_7pairs_20260121_220328",
    (14, "cDDPM (AV)"):    "ddpm_corrected_inference_antithetic_7pairs_20260121_220328",
    
    # User requested removing 20 shots
}

# MAP Metrics Configuration
# (Excel Column Name, Scaling Factor)
# "Metric 4" is requested to be DTW
METRICS_MAP = [
    ("mse_test", 1000),      # MSE (x10^-3)
    ("pcorr_test", 1),       # PCC (x1) - Note: check existing table scaling, usually PCC is 0-1, table says x1 in header? No, table says "PCC" no scale? Wait, table body says 0.97. Previous script used 100.
                             # User table shows "0.971", so scale is 1.
    ("psnr_test", 1),        # PSNR
    ("dtw_test", 1)          # Metric 4: DTW
]

# ECG Hardcoded Values (Pairs: (SSD, MAD, PRD, CosSim))
# Copied from user prompt
ECG_VALUES = {
    (2, "cDDPM (crude)"): "4.33$\\pm$6.82 & 0.348$\\pm$0.276 & 42.17$\\pm$26.62 & 0.916$\\pm$0.099",
    (2, "cDDPM (AV)"):    "3.85$\\pm$6.03 & 0.327$\\pm$0.259 & 40.25$\\pm$25.45 & 0.926$\\pm$0.088",
    
    (4, "cDDPM (crude)"): "4.02$\\pm$6.24 & 0.337$\\pm$0.265 & 41.17$\\pm$25.93 & 0.922$\\pm$0.091",
    (4, "cDDPM (AV)"):    "3.79$\\pm$5.96 & 0.326$\\pm$0.258 & 40.14$\\pm$25.42 & 0.927$\\pm$0.086",
    
    (10, "cDDPM (crude)"): "3.85$\\pm$6.11 & 0.330$\\pm$0.260 & 40.43$\\pm$25.49 & 0.926$\\pm$0.087",
    (10, "cDDPM (AV)"):    "3.75$\\pm$5.88 & 0.325$\\pm$0.257 & 40.05$\\pm$25.67 & 0.928$\\pm$0.085",
    
    (14, "cDDPM (crude)"): "3.81$\\pm$5.95 & 0.328$\\pm$0.259 & 40.30$\\pm$25.48 & 0.927$\\pm$0.086",
    (14, "cDDPM (AV)"):    "3.74$\\pm$5.89 & 0.325$\\pm$0.257 & 40.03$\\pm$25.72 & 0.928$\\pm$0.084",
}

# Reporting Mode
REPORT_HIDDEN_TEST = False # Toggle if needed (user code used 'test' metrics usually)

def format_value(avg, std):
    """Formats the average and standard deviation for LaTeX."""
    return f"{avg:.2f}$\\pm${std:.2f}" # Matching user style "3.93\pm0.20" (no scriptsize mentioned in prompt snippet vs previous code)
    # Actually user snippet has: 3.93$\pm$0.20. Let's match that exactly.

def get_map_metrics(experiment_subfolder):
    """
    Reads the summary excel file and retrieves columns defined in METRICS_MAP.
    Returns a string of 4 formatted cells or TBDs.
    """
    if not experiment_subfolder:
        return "TBD & TBD & TBD & TBD"
    
    # Try finding the folder in the base path
    folder_path = os.path.join(BASE_PATH, experiment_subfolder)
    
    # If not absolute, assume base path
    file_path = os.path.join(folder_path, "summary_values.xlsx")
    
    if not os.path.exists(file_path):
        # print(f"[DEBUG] File missing: {file_path}")
        return "TBD & TBD & TBD & TBD"
        
    try:
        df = pd.read_excel(file_path)
        
        # Identify Split/Fold column
        possible_id_cols = ['Split', 'fold', 'condition']
        id_col = next((col for col in possible_id_cols if col in df.columns), None)
        
        if not id_col:
            return "TBD & TBD & TBD & TBD"

        df[id_col] = df[id_col].astype(str)
        
        # Priority: Hidden, then Test, then Val
        # If REPORT_HIDDEN_TEST is True, prioritize hidden
        
        t_suffix = "_hidden" if REPORT_HIDDEN_TEST else "_test"
        
        # Find Average and Std rows
        # We look for "average_test" or "average_hidden" specifically if possible
        avg_row = df[df[id_col].str.lower() == f"average{t_suffix}"]
        std_row = df[df[id_col].str.lower() == f"st_dev{t_suffix}"]
        
        # Fallback to generic "average" if specific split not found
        if avg_row.empty:
            avg_row = df[df[id_col].str.lower().str.contains('average|mean')]
            # Try to filter by split type if column name implies it? No, usually rows are split.
        if std_row.empty:
            std_row = df[df[id_col].str.lower().str.contains('st. dev.|std|standard deviation')]
            
        if avg_row.empty or std_row.empty:
            return "TBD & TBD & TBD & TBD"

        cells = []
        for metric_base, scale in METRICS_MAP:
            # Construct column name candidates
            # e.g. "mse_test", "mse_hidden", "mse"
            # metric_base is like "mse_test"
            
            # If we want hidden, replace _test with _hidden
            if REPORT_HIDDEN_TEST:
                target_metric = metric_base.replace("_test", "_hidden").replace("_val", "_hidden")
            else:
                target_metric = metric_base
                
            candidates = [target_metric]
            
            # Fallbacks
            suffix = "_hidden" if REPORT_HIDDEN_TEST else "_test"
            root = target_metric.replace(suffix, "")
            candidates.append(root + "_test")
            candidates.append(root + "_val") 
            candidates.append(root) # "mse"
            
            col_found = None
            for c in candidates:
                if c in df.columns:
                    col_found = c
                    break
            
            if col_found:
                val = avg_row.iloc[0][col_found]
                std = std_row.iloc[0][col_found]
                
                # Cleanup strings
                if isinstance(val, str): val = float(val.replace(',', '.'))
                if isinstance(std, str): std = float(std.replace(',', '.'))
                
                # Format
                # Special handling for decimal places based on user table
                # MSE: 2 decimals, PCC: 3, PSNR: 2, DTW: maybe 2?
                # User used 2 decimals for MSE/PSNR, 3 for PCC.
                # Let's use format_value which is 2 decimals generally.
                
                # Override format for PCC
                if "pcorr" in col_found:
                     cells.append(f"{val*scale:.3f}$\\pm${std*scale:.3f}")
                else:
                     cells.append(f"{val*scale:.2f}$\\pm${std*scale:.2f}")

            else:
                cells.append("N/A")
                
        return " & ".join(cells)

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return "Err & Err & Err & Err"


def generate_sampling_table():
    header = [
        r"\begin{table*}[htbp]",
        r"  \centering",
        r"  \caption{Denoising Metrics on Test Sets for AV vs Crude MC Sampling in cDDPM for MAPs and ECGs \textcolor{red}{Finalizing}}",
        r"  \label{tab:av_vs_mc_sampling}",
        r"  \begin{tabular}{c l c c c c c c c c}",
        r"    \toprule",
        r"      & & \multicolumn{4}{c}{\textbf{MAP}} & \multicolumn{4}{c}{\textbf{ECG}} \\",
        r"    \cmidrule(lr){3-6}\cmidrule(lr){7-10}",
        r"    Shots & Sampling",
        r"      & MSE ($\times 10^{-3}$)$\downarrow$ & PCC $\uparrow$ & PSNR $\uparrow$ & DTW $\downarrow$",
        r"      & SSD $\downarrow$ & MAD $\downarrow$ & PRD (\%) $\downarrow$ & Cosine Sim $\uparrow$ \\",
        r"    \midrule"
    ]
    
    body = []
    
    pairs_list = [2, 4, 10, 14] # 20 Removed per request
    
    for shots in pairs_list:
        # Tuple keys for experiments
        key_crude = (shots, "cDDPM (crude)")
        key_av    = (shots, "cDDPM (AV)")
        
        # 1. Crude Row
        map_metrics_crude = get_map_metrics(EXPERIMENTS.get(key_crude, ""))
        ecg_metrics_crude = ECG_VALUES.get(key_crude, "TBD & TBD & TBD & TBD")
        
        row_crude = f"    {shots}  & cDDPM (crude) & {map_metrics_crude} & {ecg_metrics_crude} \\\\"
        body.append(row_crude)
        
        # 2. AV Row
        map_metrics_av = get_map_metrics(EXPERIMENTS.get(key_av, ""))
        ecg_metrics_av = ECG_VALUES.get(key_av, "TBD & TBD & TBD & TBD")
        
        row_av = f"       & cDDPM (AV)    & {map_metrics_av} & {ecg_metrics_av} \\\\"
        body.append(row_av)
        
        if shots != pairs_list[-1]:
            body.append(r"    \midrule")
            
    footer = [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table*}"
    ]
    
    return "\n".join(header + body + footer)

if __name__ == "__main__":
    table_latex = generate_sampling_table()
    print("\n" + "="*50)
    print("LATEX OUTPUT:")
    print("="*50 + "\n")
    print(table_latex)
    
    with open("sampling_comparison_table.tex", "w") as f:
        f.write(table_latex)
    print(f"\nSaved to {os.path.abspath('sampling_comparison_table.tex')}")

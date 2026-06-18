"""Generate the APD-prediction LaTeX table from per-method APD summary workbooks.

Faithful migration of ``MAP_VAE/generate_apd_latex_table.py`` (a bare top-level
script run via ``__main__``). Only mechanical edits were applied: the hardcoded
``/cluster/.../MAP_VAE/experiments`` ``BASE_PATH`` was routed through
``cardiac_map_diffusion.paths.experiments_root()`` and the ``ROW_DIRS`` values
were rebuilt by joining that root with the unchanged relative sub-paths, plus
this module docstring was added. All table logic is otherwise byte-for-byte
unchanged.
"""
import pandas as pd
import os
import argparse

from cardiac_map_diffusion import paths

# ==========================================
# CONFIGURATION
# ==========================================

# Map Models to their EXPERIMENT DIRECTORY.
# The script will look for 'summary_values_apd30_gs.xlsx', etc. inside these folders.
# Puts "TBD" if file is missing.

# Helper to extract dir from the previous file paths
def get_dir(filepath):
    if filepath and filepath != "":
        return os.path.dirname(filepath)
    return ""

# Common Base Path (Adjust if needed)
BASE_PATH = str(paths.experiments_root())

ROW_DIRS = {
    # Method Name : Directory Path
    "Clean":        os.path.join(BASE_PATH, "FILTER_wiener_baseline_nfolds4_noise_allmixed_rs17_rss29_test"), # Uses _clean.xlsx from here

    # DL Models
    "cDDPM (AV)":   os.path.join(BASE_PATH, "ddpm_corrected/"),
    "cDDPM (MC)":   r"",
    "$\\beta$-VAE": os.path.join(BASE_PATH, "2023_0808_allmixed_cyclbeta4_init_1lrelu_50_nfolds4_withepbs_32_lr_5e-3_noise_allmixed_split100_rs17_rss29_test_2/allmixed_beta_0.005"),

    # Baselines (Filters) -> Assumes these folders contain the APD excel files now
    "B$_{Butterworth}$": os.path.join(BASE_PATH, "FILTER_butterworth_baseline_nfolds4_noise_allmixed_rs17_rss29_test_2"),
    "Wiener":       os.path.join(BASE_PATH, "FILTER_wiener_baseline_nfolds4_noise_allmixed_rs17_rss29_test_2"),
    "TV":           os.path.join(BASE_PATH, "FILTER_tv_l1_baseline_nfolds4_noise_allmixed_rs17_rss29_test_2"),
    "Savitzky":     os.path.join(BASE_PATH, "FILTER_hybrid_savgol_median_baseline_nfolds4_noise_allmixed_rs17_rss29_test_2"),
    "Wavelet":      os.path.join(BASE_PATH, "FILTER_wavelet_baseline_nfolds4_noise_allmixed_rs17_rss29_test_2"),
    "Notch":        os.path.join(BASE_PATH, "FILTER_adaptive_notch_baseline_nfolds4_noise_allmixed_rs17_rss29_test_2"),

    # Baselines (DL)
    "U-Net":        os.path.join(BASE_PATH, "LUNet_baseline_1000max_nfolds4_bs32_lr_5e-5_noise_allmixed_rs17_rss29_es10_test_2/LUNet_baseline_allmixed_es10_test"),
    "AE":           os.path.join(BASE_PATH, "DAE_baseline_10000max_nfolds4_bs32_lr_5e-5_noise_allmixed_rs17_rss29_ls32_es10_test_2/DAE_baseline_allmixed_ls32_test"),
    "D-RNN":        os.path.join(BASE_PATH, "DRRN_baseline_1000max_nfolds4_bs32_lr_5e-5_noise_allmixed_rs17_rss29_hs64_es10_test_2/DRRN_baseline_allmixed_hs64_es10"),

    "Noisy":        os.path.join(BASE_PATH, "FILTER_none_baseline_nfolds4_noise_allmixed_rs17_rss29_test_2/none_baseline")
}


APD_LABELS = ["APD30", "APD60", "APD90"]
APD_FILE_TEMPLATE = "summary_values_{}_gs.xlsx" # e.g. summary_values_apd30_gs.xlsx (lower case handled in code)

def format_value(avg, std, is_best=False, is_second=False):
    """Formats the average and standard deviation for LaTeX."""
    val_str = f"{avg:.2f} \\scriptsize{{$\\pm$ {std:.2f}}}"
    if is_best:
        return f"\\textbf{{{val_str}}}"
    if is_second:
        return f"\\textit{{{val_str}}}"
    return val_str

def get_apd_metric(folder_path, apd_label, metric_col="root_mean_squared_error", model_name=None):
    """
    Reads the specific APD excel file in the folder and returns (avg, std).
    """
    if not folder_path:
        return None, None
        
    if not os.path.exists(folder_path):
        # Only print debug for non-empty paths to avoid spam for empty placeholders
        if len(folder_path) > 5: 
            print(f"[DEBUG] Folder not found: {folder_path}")
        return None, None
    
    # Select templates based on model type
    if model_name == "Clean":
        templates = ["summary_values_{}_gs_clean.xlsx", "summary_values_{}_clean.xlsx"]
    else:
        # Try multiple filename templates
        templates = [
            "summary_values_{}_gs.xlsx",      # Old/Grid Search
            #"summary_values_{}_seeds.xlsx",   # New/Seeded
            "summary_values_{}.xlsx"          # Generic fallback
        ]
    
    file_path = None
    for tmpl in templates:
        filename = tmpl.format(apd_label.lower())
        
        # 1. Check root folder
        p = os.path.join(folder_path, filename)
        if os.path.exists(p):
            file_path = p
            break
            
        # 2. Check one level of subdirectories
        try:
            subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
            for subdir in subdirs:
                p_sub = os.path.join(folder_path, subdir, filename)
                if os.path.exists(p_sub):
                    file_path = p_sub
                    break
        except OSError:
            pass
            
        if file_path:
            break
            
    if not file_path:
        if model_name != "Clean": # Don't spam if clean hasn't run yet
             print(f"[DEBUG] File not found in {os.path.basename(folder_path)} for {apd_label} (Template: {templates})")
        return None, None
        
    try:
        df = pd.read_excel(file_path)
        
        # Look for Average and Std rows with flexible matching
        # Priority: average_hidden > average_test > generic average
        split_col = df['Split'].astype(str).str.lower().str.strip()
        
        # 1. Try Specific "Hidden" (Gold Standard)
        avg_row = df[split_col == 'average_hidden']
        std_row = df[split_col == 'st_dev_hidden']
        
        # 2. Try Specific "Test"
        if avg_row.empty:
            avg_row = df[split_col == 'average_test']
            std_row = df[split_col == 'st_dev_test']
        
        # 3. Fallback to generic text match
        if avg_row.empty:
            avg_row = df[split_col.str.contains('average') | split_col.str.contains('mean')]
            # std search is tricky, try generic
            std_row = df[split_col.str.contains('st. dev') | split_col.str.contains('st_dev') | split_col.str.contains('std')]
        
        if avg_row.empty:
            print(f"[DEBUG] 'Average' row missing in {os.path.basename(file_path)}")
            return None, None
        if std_row.empty:
            print(f"[DEBUG] 'Std Dev' row missing in {os.path.basename(file_path)}")
            return None, None
            
        # Determine the correct column name to use
        target_col = None
        
        # Priority check for hidden columns if we are in hidden rows
        metric_col_base = metric_col
        if metric_col == "root_mean_squared_error":
             # Standardize metric name search
             metric_search_bases = ["root_mean_squared_error", "rmse"]
        else:
             metric_search_bases = [metric_col]

        # Define candidate columns to searching for
        # If the row is "average_hidden", we MUST look for "rmse_hidden" or similar first if it exists
        is_hidden_row = 'hidden' in avg_row['type'].iloc[0] if 'type' in avg_row.columns else False
        
        candidates = []
        if is_hidden_row:
             for b in metric_search_bases:
                 candidates.append(f"{b}_hidden")
        
        # Fallback candidates (test or generic)
        for b in metric_search_bases:
             candidates.append(f"{b}_test")
             candidates.append(b)

        for cand in candidates:
            if cand in df.columns:
                target_col = cand
                break
        
        if target_col is None:
            # Special fallback for the provided clean format which seems to duplicate columns
            # The user pasted columns like: ... root_mean_squared_error ...
            # Let's just try the base metrics again if nothing matched specific suffixes
            if metric_col in df.columns:
                target_col = metric_col
            elif "root_mean_squared_error" in df.columns:
                 target_col = "root_mean_squared_error"
        
        if target_col is None:
            print(f"[DEBUG] Metric col '{metric_col}' not found in {os.path.basename(file_path)}. Cols: {list(df.columns)}")
            return None, None

        raw_avg = avg_row.iloc[0][target_col]
        raw_std = std_row.iloc[0][target_col]
        
        # Helper to convert comma-strings to float
        def parse_val(v):
            if isinstance(v, str):
                v = v.replace(',', '.')
            try:
                return float(v)
            except:
                return 0.0

        return parse_val(raw_avg), parse_val(raw_std)
        
    except Exception as e:
        print(f"[ERROR] Reading {file_path}: {e}")
        return None, None

def generate_apd_table():
    # LaTeX Header
    latex_code = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{APD Prediction Results from Denoised MAPs}",
        r"%\resizebox{0.5\textwidth}{!}{%",
        r"\begin{tabular}{llll}",
        r"\toprule",
        r"Labels & APD\textsubscript{30} $\downarrow$ & APD\textsubscript{60} $\downarrow$ & APD\textsubscript{90} $\downarrow$ \\",
        r"\midrule"
    ]
    
    # We will collect values to determine best/second best (optional, hard to do row-wise for all)
    # For now, just print values.
    
    for label, folder_path in ROW_DIRS.items():
        row_str = f"{label}"
        
        # Separator lines
        if label == "cDDPM (AV)":
            latex_code.append(r"\cdashline{2-4}")
        if label == "Wiener":
             latex_code.append(r"% --- Baselines ---")
        if label == "Noisy":
            latex_code.append(r"\cdashline{2-4}")

        for apd in APD_LABELS: # APD30, 60, 90
            avg, std = get_apd_metric(folder_path, apd, model_name=label)
            
            if avg is not None:
                cell_content = format_value(avg, std)
            else:
                cell_content = "TBD"
            
            row_str += f" & {cell_content}"
            
        row_str += r" \\"
        latex_code.append(row_str)

    # Footer
    latex_code.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"",
        r"%}"
    ])
    
    print("\n".join(latex_code))
    
    # Also save to file
    with open("apd_table_latex.txt", "w") as f:
        f.write("\n".join(latex_code))
    print("\nSaved LaTeX code to apd_table_latex.txt")

if __name__ == "__main__":
    generate_apd_table()

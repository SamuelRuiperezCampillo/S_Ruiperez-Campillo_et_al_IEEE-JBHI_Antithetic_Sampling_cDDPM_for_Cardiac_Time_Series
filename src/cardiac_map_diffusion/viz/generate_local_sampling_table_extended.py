"""Generate the extended AV-vs-Crude-MC sampling LaTeX table from local xlsx.

Faithful migration of ``MAP_VAE/generate_local_sampling_table_extended.py`` (a
bare top-level script run via ``__main__``). Only a module docstring was added;
no import rewrites were needed. ``BASE_LOCAL_DIR`` is configured via the
``CARDIACDIFF_SAMPLING_DIR`` environment variable (default ``./sample_ddpm``)
and holds the one-off ``summary_values_*.xlsx`` copies.
"""
import pandas as pd
import os

# ==========================================
# CONFIGURATION
# ==========================================

# Base directory holding the one-off summary_values_*.xlsx copies.
# Set via CARDIACDIFF_SAMPLING_DIR (default: ./sample_ddpm).
BASE_LOCAL_DIR = os.environ.get("CARDIACDIFF_SAMPLING_DIR", "sample_ddpm")

# Experiment Map: (Shots, Sampling Type) -> Full Local File Path
# Mapping Logic: File Index X -> 2*X Shots.
# Files without "av" -> Crude. Files with "av" -> AV.


EXPERIMENTS = {
    # File 1 -> Shots 2
    (2, "cDDPM (crude)"): os.path.join(BASE_LOCAL_DIR, "summary_values_1.xlsx"),
    (2, "cDDPM (AV)"):    os.path.join(BASE_LOCAL_DIR, "summary_values_1av.xlsx"),
    
    # File 2 -> Shots 4
    (4, "cDDPM (crude)"): os.path.join(BASE_LOCAL_DIR, "summary_values_2.xlsx"),
    (4, "cDDPM (AV)"):    os.path.join(BASE_LOCAL_DIR, "summary_values_2av.xlsx"),
    
    # File 3 -> Shots 6
    (6, "cDDPM (crude)"): os.path.join(BASE_LOCAL_DIR, "summary_values_3.xlsx"),
    (6, "cDDPM (AV)"):    os.path.join(BASE_LOCAL_DIR, "summary_values_3av.xlsx"),

    # File 4 -> Shots 8
    (8, "cDDPM (crude)"): os.path.join(BASE_LOCAL_DIR, "summary_values_4.xlsx"),
    (8, "cDDPM (AV)"):    os.path.join(BASE_LOCAL_DIR, "summary_values_4av.xlsx"),

    # File 5 -> Shots 10

    # Note: User listed 'summary_values_5_av.xlsx' and 'summary_values_5av.xlsx'.
    # Assuming 'summary_values_5_av.xlsx' is meant to be Crude (as placeholder or renamed) 
    # or one of them is valid. Will use 5_av as Crude based on uniqueness, but check contents if possible.
    (10, "cDDPM (crude)"): os.path.join(BASE_LOCAL_DIR, "summary_values_5_av.xlsx"), 
    (10, "cDDPM (AV)"):    os.path.join(BASE_LOCAL_DIR, "summary_values_5av.xlsx"),
    
    # File 7 -> Shots 14
    (14, "cDDPM (crude)"): os.path.join(BASE_LOCAL_DIR, "summary_values_7.xlsx"),
    (14, "cDDPM (AV)"):    os.path.join(BASE_LOCAL_DIR, "summary_values_7av.xlsx"),
}

# MAP Metrics Configuration
# (Excel Column Name for HIDDEN set, Scaling Factor)
# The Excel files have columns: mse_hidden, pcorr_hidden, psnr_hidden, dtw_hidden
METRICS_MAP = [
    ("mse_hidden", 1000),      # MSE (x10^-3)
    ("pcorr_hidden", 1),       # PCC
    ("psnr_hidden", 1),        # PSNR
    ("dtw_hidden", 1)          # Metric 4: DTW
]

# ECG Hardcoded Values (Pairs: (SSD, MAD, PRD, CosSim))
# Values only available for 2, 4, 10, 14 from previous info. 
# Others set to TBD.
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


def get_map_metrics(file_path):
    """
    Reads the summary excel file and retrieves columns defined in METRICS_MAP.
    Looks for the 'average' row and extracts _hidden columns.
    Returns a string of 4 formatted cells or TBDs.
    """
    if not file_path:
        return "TBD & TBD & TBD & TBD"
        
    if not os.path.exists(file_path):
        print(f"[DEBUG] File missing: {file_path}")
        return "TBD & TBD & TBD & TBD"
        
    try:
        df = pd.read_excel(file_path)
        
        # Identify Split/Fold column (first column)
        possible_id_cols = ['Split', 'fold', 'condition']
        id_col = next((col for col in possible_id_cols if col in df.columns), None)
        
        if not id_col:
            print(f"[DEBUG] No ID column found in {file_path}")
            return "TBD & TBD & TBD & TBD"

        df[id_col] = df[id_col].astype(str).str.strip().str.lower()
        
        # Find Average and St. Dev. rows (there's only ONE average row containing all metrics)
        avg_row = df[df[id_col] == 'average']
        std_row = df[df[id_col] == 'st. dev.']
        
        if avg_row.empty:
            print(f"[DEBUG] No 'average' row found in {file_path}")
            return "TBD & TBD & TBD & TBD"
        if std_row.empty:
            print(f"[DEBUG] No 'st. dev.' row found in {file_path}")
            return "TBD & TBD & TBD & TBD"

        cells = []
        for metric_col, scale in METRICS_MAP:
            # Check if the column exists
            if metric_col in df.columns:
                val = avg_row.iloc[0][metric_col]
                std = std_row.iloc[0][metric_col]
                
                # Cleanup strings (handle comma as decimal separator)
                if isinstance(val, str): val = float(val.replace(',', '.'))
                if isinstance(std, str): std = float(std.replace(',', '.'))
                
                # Format: PCC uses 3 decimals, others use 2 decimals
                if "pcorr" in metric_col:
                    cells.append(f"{val*scale:.3f}$\\pm${std*scale:.3f}")
                else:
                    cells.append(f"{val*scale:.2f}$\\pm${std*scale:.2f}")
            else:
                print(f"[DEBUG] Column '{metric_col}' not found in {file_path}")
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
    
    # Process shots in order
    pairs_list = [2, 4, 6, 8, 10, 14] 
    
    for shots in pairs_list:
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
    
    with open("results_table_local_extended.tex", "w") as f:
        f.write(table_latex)
    print(f"\nSaved to {os.path.abspath('results_table_local_extended.tex')}")

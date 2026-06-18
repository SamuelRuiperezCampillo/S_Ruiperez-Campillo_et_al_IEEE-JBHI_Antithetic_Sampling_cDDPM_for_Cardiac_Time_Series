"""Evaluate cDDPM sampling strategies (Crude MC vs Antithetic Variates) per fold.

Faithful migration of
``Diffusion_MAP_fullpipeline_final/evaluate_sampling_strategies.py`` (an argparse
``main()`` entry script). Only mechanical edits were applied: imports were
rewritten to the ``cardiac_map_diffusion`` package layout and this module
docstring was added. All evaluation logic is otherwise byte-for-byte unchanged.
"""
import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import random
from sklearn.model_selection import KFold

# Import DDPM modules
from cardiac_map_diffusion.diffusion.ddpm_conditional import Diffusion
from cardiac_map_diffusion.diffusion.denoising_net import ConditionalModel
from cardiac_map_diffusion.diffusion.denoising_net_small import ConditionalModelSmall
from cardiac_map_diffusion.metrics.ddpm_metrics import compute_pearson_corr, compute_mse, compute_psnr

# Import VAE modules
import cardiac_map_diffusion.metrics.map_functions as mapf
from cardiac_map_diffusion.data.data_sam import get_MAP_vent_data
from cardiac_map_diffusion.data.ep_noise_sam import get_np_noisearrays
from cardiac_map_diffusion.diffusion.train_ddpm import ensure_tensor_shape, ensure_numpy_shape, retrieveDataSet, compute_comprehensive_metrics

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_fold_data(df, num_folds, split_number, r_seed):
    """
    Reproduce the KFold splitting logic to get the test dataframe for a specific fold.
    """
    random.seed(r_seed)
    unique_patients = df['pat_ID'].unique()
    # Ensure consistent type for shuffling if mixed types exist (though usually they are consistent)
    # unique_patients = sorted(unique_patients) # If we want deterministic order before shuffle? 
    # The original script didn't sort before shuffle, so we shouldn't either.
    # But unique() order depends on data order. Assuming data load is deterministic.
    
    random.shuffle(unique_patients)
    kf = KFold(n_splits=num_folds)

    for i, (train_index, test_index) in enumerate(kf.split(unique_patients)):
        if i == split_number:
            test_patients = unique_patients[test_index]
            test_df = df[df['pat_ID'].isin(test_patients)]
            return test_df
            
    return None

def denoise_batch_crude_mc(model, diffusion, noisy_batch, n_shots, device):
    """
    Denoise a batch using Crude Monte Carlo with n_shots.
    """
    accumulated_signals = 0
    for _ in range(n_shots):
        # inference returns (Batch, 1, Length)
        denoised = diffusion.inference(model, 1, noisy_batch)
        accumulated_signals += denoised
    
    return accumulated_signals / n_shots

def denoise_batch_antithetic(model, diffusion, noisy_batch, n_pairs, device):
    """
    Denoise a batch using Antithetic Variates with n_pairs.
    Total samples = 2 * n_pairs.
    """
    accumulated_signals = 0
    for _ in range(n_pairs):
        # inference_antithetic returns two tensors (Batch, 1, Length)
        denoised, denoised_anti = diffusion.inference_antithetic(model, 1, noisy_batch)
        accumulated_signals += (denoised + denoised_anti)
    
    return accumulated_signals / (2 * n_pairs)

def run_inference_loop(model, diffusion, dataloader, strategy, n_counts, device):
    """
    Run inference on the entire dataloader for a specific strategy and count.
    """
    model.eval()
    denoised_signals_list = []
    clean_signals_list = []
    
    logging.info(f"Running inference: Strategy={strategy}, N={n_counts}")
    
    with torch.no_grad():
        for clean_b, noisy_b in tqdm(dataloader, desc=f"{strategy} N={n_counts}"):
            clean_b = ensure_tensor_shape(clean_b, device)
            noisy_b = ensure_tensor_shape(noisy_b, device)
            
            if strategy == 'crude':
                denoised_batch = denoise_batch_crude_mc(model, diffusion, noisy_b, n_counts, device)
            elif strategy == 'antithetic':
                denoised_batch = denoise_batch_antithetic(model, diffusion, noisy_b, n_counts, device)
            
            # Convert to numpy
            denoised_batch_np = denoised_batch.cpu().numpy()
            clean_batch_np = clean_b.cpu().numpy()
            
            denoised_signals_list.append(ensure_numpy_shape(denoised_batch_np))
            clean_signals_list.append(ensure_numpy_shape(clean_batch_np))
            
    return np.concatenate(clean_signals_list, axis=0), np.concatenate(denoised_signals_list, axis=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", type=str, required=True, help="Path to the experiment directory containing config.json and models/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fold", type=int, default=0, help="Fold number to evaluate (default: 0)")
    parser.add_argument("--eval_patients_file", type=str, default=None, help="Optional CSV file with 'pat_ID' to filter the evaluation set")
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(args.experiment_dir, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Override device if specified
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%d-%b-%y %H:%M:%S'
    )

    # Reconstruct necessary parameters from config
    signal_len = config['signal_length']
    noise_steps = config['noise_steps']
    noise_schedule = config['noise_schedule']
    beta_start = config['beta_start']
    beta_end = config['beta_end']
    feats = config['feats']
    model_small = config['model_small']
    noise_type = config.get('noise_type', 'allmixed') # Default if missing
    seed_split = config.get('seed_split', 29)
    
    # Load Data
    logging.info("Loading data...")
    original_cwd = os.getcwd()
    try:
        df_complete = get_MAP_vent_data()
    finally:
        os.chdir(original_cwd)

    # -------------------------------------------------------------------------
    # REPRODUCE TRAINING EXCLUSION (if any)
    # -------------------------------------------------------------------------
    # Check if the model was trained with an exclusion list
    train_exclude_file = config.get('exclude_patients_file')
    if train_exclude_file:
        # We need to find this file. It might be a relative path from the original run.
        # Or we can check if it exists in the experiment dir (if we copied it there? we didn't).
        # But usually the user provides an absolute path or relative to workspace.
        if os.path.exists(train_exclude_file):
            logging.info(f"Applying training exclusion from: {train_exclude_file}")
            ex_df = pd.read_csv(train_exclude_file)
            if 'pat_ID' in ex_df.columns:
                ex_pats = ex_df['pat_ID'].astype(str).unique()
                df_complete['pat_ID'] = df_complete['pat_ID'].astype(str)
                df_complete = df_complete[~df_complete['pat_ID'].isin(ex_pats)]
                logging.info(f"Data after training exclusion: {len(df_complete)} samples")
            else:
                logging.warning(f"Training exclusion file {train_exclude_file} has no 'pat_ID' column. Ignoring.")
        else:
            logging.warning(f"Training exclusion file {train_exclude_file} not found. Splits might be incorrect!")

    # -------------------------------------------------------------------------
    # GET VALIDATION SPLIT FOR THIS FOLD
    # -------------------------------------------------------------------------
    test_df = get_fold_data(df_complete, num_folds=config['n_splits'], split_number=args.fold, r_seed=seed_split)
    
    if test_df is None:
        logging.error(f"Could not retrieve data for fold {args.fold}")
        return

    logging.info(f"Fold {args.fold} Validation Set: {len(test_df)} samples")

    # -------------------------------------------------------------------------
    # APPLY EVALUATION FILTER (if provided)
    # -------------------------------------------------------------------------
    if args.eval_patients_file:
        if os.path.exists(args.eval_patients_file):
            logging.info(f"Filtering evaluation set using: {args.eval_patients_file}")
            eval_filter_df = pd.read_csv(args.eval_patients_file)
            if 'pat_ID' in eval_filter_df.columns:
                filter_pats = eval_filter_df['pat_ID'].astype(str).unique()
                test_df['pat_ID'] = test_df['pat_ID'].astype(str)
                
                # Intersect
                test_df = test_df[test_df['pat_ID'].isin(filter_pats)]
                logging.info(f"Filtered Evaluation Set: {len(test_df)} samples")
                
                if len(test_df) == 0:
                    logging.warning("No samples remaining after filtering! Check if your patients are in this fold.")
                    return
            else:
                logging.error("Evaluation filter CSV must contain 'pat_ID' column.")
                return
        else:
            logging.error(f"Evaluation filter file {args.eval_patients_file} not found.")
            return

    # Prepare Data for Inference
    X_test = np.array(test_df['MAP_segments'].tolist())
    X_std_test = mapf.normalize_EGM_array(X_test)
    
    # We need dummy train data for retrieveDataSet signature
    # retrieveDataSet(noise_type, noise_params, X_train, X_test, X_std_train, X_std_test, arrays=[])
    # It uses X_train to fit noise params if needed? No, noise_params are passed.
    # It creates train_dataset and test_dataset. We only use test_dataset.
    
    ep_noise_arrays = get_np_noisearrays(df_complete) # Use full (or filtered) df for noise bank
    noise_params = mapf.find_noise_params(noise_type)
    arrays = ep_noise_arrays if noise_type in ['ep', 'allmixed'] else []
    
    _, test_dataset, _ = retrieveDataSet(
        noise_type, noise_params, X_test, X_test, X_std_test, X_std_test, arrays=arrays
    )
    
    dataloader_test = DataLoader(test_dataset, batch_size=config['batch_size_test'], shuffle=False, num_workers=0)

    # Load Model
    logging.info("Loading model...")
    if model_small: 
        model = ConditionalModelSmall(feats).to(device)
    else:
        model = ConditionalModel(feats).to(device)
        
    # Try to load specific fold model first
    model_path = os.path.join(args.experiment_dir, "models", f"model_fold{args.fold}.pth")
    if not os.path.exists(model_path):
        logging.info(f"Model for fold {args.fold} not found, trying model_final.pth")
        model_path = os.path.join(args.experiment_dir, "models", "model_final.pth")
    
    if not os.path.exists(model_path):
        # Fallback for older structure
        model_path = os.path.join(args.experiment_dir, "model_final.pth")

    logging.info(f"Loading weights from {model_path}")
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            model = checkpoint
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return

    diffusion = Diffusion(noise_steps, beta_start, beta_end, signal_len, noise_schedule, device)

    # Define strategies
    crude_shots = [2, 4, 6, 14, 20]
    av_pairs = [2, 4, 6, 14, 20]
    
    results = []

    # 1. Crude Monte Carlo
    for n in crude_shots:
        logging.info(f"Evaluating Crude MC with {n} shots...")
        clean, denoised = run_inference_loop(model, diffusion, dataloader_test, 'crude', n, device)
        metrics = compute_comprehensive_metrics(clean, denoised, args.fold, "Test")
        
        row = {'Strategy': 'Crude MC', 'N_Shots': n, 'N_Pairs': 0, 'Total_Inference_Calls': n}
        row.update(metrics)
        results.append(row)

    # 2. Antithetic Variates
    for n in av_pairs:
        logging.info(f"Evaluating Antithetic Variates with {n} pairs...")
        clean, denoised = run_inference_loop(model, diffusion, dataloader_test, 'antithetic', n, device)
        metrics = compute_comprehensive_metrics(clean, denoised, args.fold, "Test")
        
        row = {'Strategy': 'Antithetic', 'N_Shots': n*2, 'N_Pairs': n, 'Total_Inference_Calls': n} # 1 call to antithetic = 2 shots
        row.update(metrics)
        results.append(row)

    # Save results
    df_results = pd.DataFrame(results)
    output_filename = f"sampling_strategy_comparison_fold{args.fold}"
    if args.eval_patients_file:
        output_filename += "_filtered"
    output_filename += ".xlsx"
    
    output_path = os.path.join(args.experiment_dir, output_filename)
    df_results.to_excel(output_path, index=False)
    logging.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()

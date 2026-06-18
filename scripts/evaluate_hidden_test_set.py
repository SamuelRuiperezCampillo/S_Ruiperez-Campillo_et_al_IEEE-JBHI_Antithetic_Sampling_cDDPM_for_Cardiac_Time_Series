"""Evaluate a trained cDDPM on the held-out hidden test set (Antithetic Variates).

Faithful migration of
``Diffusion_MAP_fullpipeline_final/evaluate_hidden_test_set.py`` (an argparse
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

def run_inference_loop(model, diffusion, dataloader, n_pairs, device):
    """
    Run inference on the entire dataloader using Antithetic Variates.
    """
    model.eval()
    denoised_signals_list = []
    clean_signals_list = []
    noisy_signals_list = []
    
    logging.info(f"Running inference on hidden test set (Antithetic Pairs={n_pairs})...")
    
    with torch.no_grad():
        for clean_b, noisy_b in tqdm(dataloader, desc="Inference"):
            clean_b = ensure_tensor_shape(clean_b, device)
            noisy_b = ensure_tensor_shape(noisy_b, device)
            
            denoised_batch = denoise_batch_antithetic(model, diffusion, noisy_b, n_pairs, device)
            
            # Convert to numpy
            denoised_batch_np = denoised_batch.cpu().numpy()
            clean_batch_np = clean_b.cpu().numpy()
            noisy_batch_np = noisy_b.cpu().numpy()
            
            denoised_signals_list.append(ensure_numpy_shape(denoised_batch_np))
            clean_signals_list.append(ensure_numpy_shape(clean_batch_np))
            noisy_signals_list.append(ensure_numpy_shape(noisy_batch_np))
            
    return (np.concatenate(clean_signals_list, axis=0), 
            np.concatenate(noisy_signals_list, axis=0),
            np.concatenate(denoised_signals_list, axis=0))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", type=str, required=True, help="Path to the experiment directory containing config.json and models/")
    parser.add_argument("--exclude_patients_file", type=str, required=True, help="Path to CSV file containing 'pat_ID' of hidden patients")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_pairs", type=int, default=1, help="Number of antithetic pairs for inference (default: 1)")
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(args.experiment_dir, "config.json")
    if not os.path.exists(config_path):
        logging.error(f"Config file not found at {config_path}")
        return

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
    noise_type = config.get('noise_type', 'allmixed')
    
    # Load Data
    logging.info("Loading data...")
    # Save current CWD because get_MAP_vent_data changes it
    original_cwd = os.getcwd()
    try:
        df_complete = get_MAP_vent_data()
    finally:
        os.chdir(original_cwd)
    
    # Filter to KEEP ONLY excluded patients (Hidden Test Set)
    if os.path.exists(args.exclude_patients_file):
        excluded_df = pd.read_csv(args.exclude_patients_file)
        if 'pat_ID' in excluded_df.columns:
            hidden_pats = excluded_df['pat_ID'].astype(str).unique()
            # Ensure pat_ID in df_complete is string for comparison
            df_complete['pat_ID'] = df_complete['pat_ID'].astype(str)
            
            df_hidden = df_complete[df_complete['pat_ID'].isin(hidden_pats)]
            logging.info(f"Hidden Test Set: {len(hidden_pats)} patients, {len(df_hidden)} samples.")
            
            if len(df_hidden) == 0:
                logging.warning("No samples found for the hidden patients! Check IDs.")
                return
        else:
            logging.error("CSV must contain 'pat_ID' column.")
            return
    else:
        logging.error(f"File {args.exclude_patients_file} not found.")
        return

    # Prepare noise arrays
    ep_noise_arrays = get_np_noisearrays(df_complete) # Use full dataset for noise bank to be safe/consistent
    noise_params = mapf.find_noise_params(noise_type)
    arrays = ep_noise_arrays if noise_type in ['ep', 'allmixed'] else []
    
    # Create Dataset for Hidden Set
    # We treat the hidden set as "Test" data. We don't need training data here.
    # We pass df_hidden data as both train and test arguments just to satisfy the function signature,
    # but we only use the test output.
    X_hidden = np.array(df_hidden['MAP_segments'].tolist())
    X_std_hidden = mapf.normalize_EGM_array(X_hidden) # Normalize
    
    # retrieveDataSet expects train/test splits. We can just pass X_hidden for both.
    # The function returns train_dataset, test_dataset, noisy_test_sample
    _, test_dataset, _ = retrieveDataSet(
        noise_type, noise_params, X_hidden, X_hidden, X_std_hidden, X_std_hidden, arrays=arrays
    )
    
    dataloader_hidden = DataLoader(test_dataset, batch_size=config['batch_size_test'], shuffle=False, num_workers=0)

    # Initialize Diffusion
    diffusion = Diffusion(noise_steps, beta_start, beta_end, signal_len, noise_schedule, device)

    # Iterate over all trained models (folds)
    models_dir = os.path.join(args.experiment_dir, "models")
    if not os.path.exists(models_dir):
         # Maybe models are in the root of experiment dir?
         models_dir = args.experiment_dir
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith(".pth")]
    
    if not model_files:
        logging.error(f"No .pth models found in {models_dir}")
        return

    all_metrics = []
    
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        
        logging.info(f"Evaluating model: {model_file}")
        
        # Load Model
        if model_small: 
            model = ConditionalModelSmall(feats).to(device)
        else:
            model = ConditionalModel(feats).to(device)
            
        try:
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict):
                model.load_state_dict(checkpoint)
            else:
                # Maybe it's the full model
                model = checkpoint
                
        except Exception as e:
            logging.error(f"Failed to load {model_file}: {e}")
            continue

        # Run Inference
        clean, noisy, denoised = run_inference_loop(model, diffusion, dataloader_hidden, args.n_pairs, device)
        
        # Compute Metrics
        metrics = compute_comprehensive_metrics(clean, denoised, 0, "HiddenTest")
        metrics['Model'] = model_file
        all_metrics.append(metrics)
        
        # Save signals for this model
        save_dir = os.path.join(args.experiment_dir, "hidden_test_results")
        os.makedirs(save_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(save_dir, f"signals_{model_file}.npz"),
            clean=clean, noisy=noisy, denoised=denoised
        )

    # Save Summary
    if all_metrics:
        df_metrics = pd.DataFrame(all_metrics)
        output_path = os.path.join(args.experiment_dir, "hidden_test_metrics.xlsx")
        df_metrics.to_excel(output_path, index=False)
        
        # Print average
        logging.info("Average Metrics on Hidden Test Set:")
        logging.info(df_metrics.mean(numeric_only=True))
        logging.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()

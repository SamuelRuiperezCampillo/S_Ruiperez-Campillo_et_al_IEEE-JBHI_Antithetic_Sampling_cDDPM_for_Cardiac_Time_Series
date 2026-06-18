"""Analyse cDDPM averaged sampling (Antithetic vs Simple) across folds and splits.

Faithful migration of ``MAP_VAE/test/ddpm_sampling_analysis.py`` (an argparse
``main()`` entry script). Only mechanical edits were applied: imports were
rewritten to the ``cardiac_map_diffusion`` package layout and this module
docstring was added. All inference/analysis logic is otherwise byte-for-byte
unchanged.
"""
import sys
import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import logging
from datetime import datetime
import json

from cardiac_map_diffusion.diffusion.ddpm_conditional import Diffusion
from cardiac_map_diffusion.diffusion.denoising_net import ConditionalModel
from cardiac_map_diffusion.diffusion.denoising_net_small import ConditionalModelSmall
from cardiac_map_diffusion.metrics.ddpm_metrics import compute_pearson_corr, compute_mse, compute_psnr

# Import VAE modules (from scripts_from_sam)
import cardiac_map_diffusion.metrics.map_functions as mapf
from cardiac_map_diffusion.data.data_sam import get_MAP_vent_data
from cardiac_map_diffusion.data.ep_noise_sam import get_np_noisearrays

# Re-use logic from aligned script
from cardiac_map_diffusion.diffusion.train_ddpm import (
    compute_comprehensive_metrics,
    ensure_numpy_shape,
    get_train_test_kfolds,
    retrieveDataSet,
    str2bool
)

def ddpm_denoise_averaged(model, diffusion, noisy_signals, device, n_pairs, sampling_type="antithetic"):
    """
    Denoise signals by averaging multiple sampling runs.
    """
    model.eval()
    denoised_signals = []
    
    with torch.no_grad():
        # Process in batches
        batch_size = min(32, len(noisy_signals))
        
        for i in range(0, len(noisy_signals), batch_size):
            batch_end = min(i + batch_size, len(noisy_signals))
            noisy_batch = torch.tensor(noisy_signals[i:batch_end]).float().unsqueeze(1).to(device)
            
            # Accumulator for averaging
            batch_sum = torch.zeros_like(noisy_batch)
            total_samples = 0
            
            if sampling_type == "antithetic":
                # Run `n_pairs` times, each produces 2 outputs (crude + anti)
                for _ in range(n_pairs):
                    x_crude, x_anti = diffusion.inference_antithetic(model, 1, noisy_batch)
                    batch_sum += x_crude
                    batch_sum += x_anti
                    total_samples += 2
                    
            elif sampling_type == "simple":
                # Run `2 * n_pairs` times to allow fair comparison of compute/samples vs antithetic
                n_runs = n_pairs * 2
                for _ in range(n_runs):
                    x_simple = diffusion.inference(model, 1, noisy_batch)
                    batch_sum += x_simple
                    total_samples += 1
            
            # Average
            denoised_batch = batch_sum / total_samples
            
            # Remove channel dim (Batch, 1, Length) -> (Batch, Length)
            denoised_batch = denoised_batch.squeeze(1)
            
            # Convert back to numpy
            denoised_batch_np = denoised_batch.cpu().numpy()
            if denoised_batch_np.ndim == 1:
                denoised_batch_np = denoised_batch_np.reshape(1, -1)
                
            denoised_signals.append(denoised_batch_np)
            
    return np.concatenate(denoised_signals, axis=0)

def main():
    parser = argparse.ArgumentParser()
    # Path to the Experiment Folder containing 'models/model_foldX.pth'
    parser.add_argument("--experiment_dir", type=str, required=True, help="Path to the trained experiment folder (input)")
    
    # Inference config
    parser.add_argument("--n_pairs", type=int, default=1, help="Number of pairs for averaging (Antithetic=2*N, Simple=2*N runs)")
    parser.add_argument("--sampling_type", type=str, default="antithetic", choices=["antithetic", "simple"], help="Sampling strategy")
    
    # Environment/Data args (Must match training)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed_split", type=int, default=29) # Default from VAE
    parser.add_argument("--exclude_patients_file", type=str, default=None)
    parser.add_argument("--noise_type", type=str, default="allmixed")
    
    # Model/Diffusion args (Must match training)
    parser.add_argument("--signal_length", type=int, default=370)
    parser.add_argument("--noise_steps", type=int, default=1000)
    parser.add_argument("--noise_schedule", type=str, default="linear")
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--model_small", type=str2bool, default=False)
    parser.add_argument("--feats", type=int, default=32)
    
    # Evaluation Config
    parser.add_argument("--n_splits", type=int, default=4, help="Number of folds to evaluate")
    parser.add_argument("--batch_size_test", type=int, default=500)
    
    args = parser.parse_args()

    # Reproducibility
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)

    # ----------------------------------------------------
    # 0. Setup Output Directory
    # ----------------------------------------------------
    # Name scheme: {OriginalExpName}_inference_{Type}_{Pairs}pairs
    # Removed timestamp to allow resuming (SLURM jobs)
    base_exp_name = os.path.basename(os.path.normpath(args.experiment_dir))
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') 
    inference_folder_name = f"{base_exp_name}_inference_{args.sampling_type}_{args.n_pairs}pairs"
    
    # Create adjacent to original experiment or in a 'results' folder?
    # User said: "create an experiments folder as the ddpm script"
    # Usually ddpm_main creates in ./results/Experiment/Timestamp
    # We will create it inside the SAME parent folder as the original experiment to keep it organized
    parent_dir = os.path.dirname(os.path.normpath(args.experiment_dir))
    output_dir = os.path.join(parent_dir, inference_folder_name)
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "denoised_signals"), exist_ok=True)
    
    print(f"Input Experiment: {args.experiment_dir}")
    print(f"Output Inference Directory: {output_dir}")
    print(f"Resuming capability enabled: Checking for existing files in output dir.")
    
    # Save Inference Config
    with open(os.path.join(output_dir, "inference_config.json"), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # ----------------------------------------------------
    # 1. Load Data
    # ----------------------------------------------------
    print("Loading data...")
    df_complete = get_MAP_vent_data()
    df_hidden = None
    
    # Filter excluded patients
    if args.exclude_patients_file and os.path.exists(args.exclude_patients_file):
        try:
            excluded_df = pd.read_csv(args.exclude_patients_file)
            if 'pat_ID' in excluded_df.columns:
                excluded_pats = excluded_df['pat_ID'].astype(str).unique()
                df_complete['pat_ID'] = df_complete['pat_ID'].astype(str)
                df_hidden = df_complete[df_complete['pat_ID'].isin(excluded_pats)]
                df_complete = df_complete[~df_complete['pat_ID'].isin(excluded_pats)]
                print(f"Excluded {len(excluded_pats)} patients. Hidden set: {len(df_hidden)}")
        except Exception as e:
            print(f"Error loading exclusion file: {e}")
            
    ep_noise_arrays = get_np_noisearrays(df_complete)
    noise_params = mapf.find_noise_params(args.noise_type)
    arrays = ep_noise_arrays if args.noise_type in ['ep', 'allmixed'] else []
    
    # 2. Diffusion Setup
    # ----------------------------------------------------
    diffusion = Diffusion(args.noise_steps, args.beta_start, args.beta_end, args.signal_length, args.noise_schedule, args.device)
    
    summary_values = []
    
    # 3. Loop Folds
    # ----------------------------------------------------
    for fold in range(args.n_splits):
        print(f"\nProcessing Fold {fold}/{args.n_splits}...")
        
        # Load Fold Data
        X_train, X_test, y_train, y_test = get_train_test_kfolds(
            df_complete, num_folds=args.n_splits, split_number=fold, r_seed=args.seed_split
        )
        X_std_train, X_std_test = mapf.normalize_EGM_input(X_train, X_test)
        
        # Create Datasets
        # Note: We need Train AND Test datasets
        # STRATEGY: Try to load exact noisy data from training output to ensure 100% match
        # If not found, fall back to generation (which might differ for Folds > 0 due to RNG state)
        
        train_noisy_path = os.path.join(args.experiment_dir, "denoised_signals", f"fold{fold}_train_signals.npz")
        test_noisy_path = os.path.join(args.experiment_dir, "denoised_signals", f"fold{fold}_test_signals.npz")
        
        train_loaded = False
        test_loaded = False
        
        # Initialize containers
        X_train_clean = None
        X_train_noisy = None
        X_test_clean = None
        X_test_noisy = None
        
        # 1. Try Loading Train
        if os.path.exists(train_noisy_path):
            try:
                print(f"  Loading training data from {train_noisy_path}...")
                data = np.load(train_noisy_path)
                X_train_clean = data['original_clean']
                X_train_noisy = data['noisy_input']
                train_dataset = torch.utils.data.TensorDataset(
                    torch.from_numpy(X_train_clean).float(), 
                    torch.from_numpy(X_train_noisy).float()
                )
                train_loaded = True
            except Exception as e:
                print(f"  Failed to load train npz: {e}")

        # 2. Try Loading Test
        if os.path.exists(test_noisy_path):
            try:
                print(f"  Loading test data from {test_noisy_path}...")
                data = np.load(test_noisy_path)
                X_test_clean = data['original_clean']
                X_test_noisy = data['noisy_input']
                # X_std_test_noisy is needed for retrieveDataSet logic, but here we just use what we loaded
                test_dataset = torch.utils.data.TensorDataset(
                    torch.from_numpy(X_test_clean).float(), 
                    torch.from_numpy(X_test_noisy).float()
                )
                test_loaded = True
            except Exception as e:
                print(f"  Failed to load test npz: {e}")

        # 3. Fallback to Generation if needed
        # We always generate for Hidden (it wasn't in training)
        # We generate for Train/Test only if loading failed
        
        # We always need to call retrieveDataSet to generate noise for Hidden 
        # (and for Train/Test if we didn't load them)
        # Challenge: Calling retrieveDataSet changes RNG state.
        # Ideally, we call it to get "Hidden", and if we needed standard Train/Test we use them.
        
        # To keep RNG consistent for 'Hidden' generation, we should call it exactly as training would...
        # BUT training consumes RNG during training loop, so we can't match Fold > 0 RNG anyway.
        # So we just run it to get generated versions, and overwrite with loaded versions if available.
        
        gen_train_ds, gen_test_ds, gen_X_test_noisy = retrieveDataSet(
            args.noise_type, noise_params, X_train, X_test, X_std_train, X_std_test, arrays=arrays
        )
        
        if not train_loaded:
            print("  Using generated training noise (Warning: might differ from training if RNG state diverged)")
            train_dataset = gen_train_ds
            
        if not test_loaded:
            print("  Using generated test noise (Warning: might differ from training if RNG state diverged)")
            test_dataset = gen_test_ds
            
        # Handle Hidden Data
        hidden_dataset = None
        hidden_noisy_path = os.path.join(args.experiment_dir, "denoised_signals", f"fold{fold}_hidden_test_signals.npz")
        hidden_loaded = False
        
        # 1. Try loading hidden from disk (if training script saved it)
        if os.path.exists(hidden_noisy_path):
            try:
                print(f"  Loading hidden data from {hidden_noisy_path}...")
                data = np.load(hidden_noisy_path)
                X_h_clean = data['original_clean']
                X_h_noisy = data['noisy_input']
                hidden_dataset = torch.utils.data.TensorDataset(
                    torch.from_numpy(X_h_clean).float(), 
                    torch.from_numpy(X_h_noisy).float()
                )
                hidden_loaded = True
            except Exception as e:
                print(f"  Failed to load hidden npz: {e}")
        
        # 2. Fallback to generation
        if not hidden_loaded and df_hidden is not None and not df_hidden.empty:
            print("  Generating hidden data (fallback)...")
            # We need to process hidden data similarly
            # get_train_test_kfolds doesn't return hidden. We take df_hidden full.
            # Normalization? Use stats from this fold's Train set
            X_hidden_raw = np.array(df_hidden['MAP_segments'].tolist())
            X_hidden_std = mapf.normalize_EGM_input_with_stats(
                 X_hidden_raw, 
                 np.mean(X_train), np.std(X_train), 
                 np.min(X_train), np.max(X_train)
            )
            # Or just standard normalization if that's what `normalize_EGM_input` does
            # mapf.normalize_EGM_input calculates min/max/mean/std from the inputs passed.
            # We should probably normalize hidden using its own stats or consistent stats?
            # Standard practice: Normalize independently per signal usually for ECG?
            # Let's see mapf.normalize_EGM_input implementation... 
            # Assuming row-wise normalization (standard scaling per signal)
            X_hidden_std, _ = mapf.normalize_EGM_input(X_hidden_raw, X_hidden_raw) # Hack to normalize
            
            # Generate noise for hidden
            # We need a new Dataset class for Hidden or reuse retrieveDataSet logic?
            # retrieveDataSet handles specific noise types.
            # Ideally we'd have a function `create_noisy_dataset(X_clean, noise_type, params)`
            # We can use the helper from retrieveDataSet logic manually
            
            # For simplicity and robustness, let's reuse retrieveDataSet by passing Hidden as "Test"
            # This is a bit hacky but uses the exact same noise injection code
            _, hidden_dataset_gen, _ = retrieveDataSet(
                args.noise_type, noise_params, X_train, X_hidden_raw, X_std_train, X_hidden_std, arrays=arrays
            )
            hidden_dataset = hidden_dataset_gen

        
        # Load Model
        model_name = f"model_fold{fold}.pth"
        model_path = os.path.join(args.experiment_dir, "models", model_name)
        if not os.path.exists(model_path):
            # Try looking in root (some generic scripts save there)
            model_path_root = os.path.join(args.experiment_dir, model_name)
            if os.path.exists(model_path_root):
                model_path = model_path_root
            else:
                print(f"  Warning: Model not found at {model_path}. Skipping fold.")
                continue
            
        try:
            print(f"  Loading model: {model_path}")
            loaded = torch.load(model_path, map_location=args.device)
            if args.model_small:
                model = ConditionalModelSmall(args.feats).to(args.device)
            else:
                model = ConditionalModel(args.feats).to(args.device)
                
            if isinstance(loaded, (ConditionalModel, ConditionalModelSmall)):
                model = loaded
            else:
                model.load_state_dict(loaded['model_state_dict'] if 'model_state_dict' in loaded else loaded)
        except Exception as e:
            print(f"  Error loading model: {e}")
            continue
            
        # Helper for processing a dataset
        def process_and_save(dataset, split_name):
            # RESUME CHECK
            save_path = os.path.join(output_dir, "denoised_signals", f"fold{fold}_{split_name.lower()}_signals.npz")
            
            if os.path.exists(save_path):
                try:
                    print(f"    [RESUME] Found existing output for {split_name}: {save_path}")
                    print(f"    Loading and computing metrics (Skipping Inference)...")
                    data = np.load(save_path)
                    
                    # Ensure we have the necessary keys
                    if 'original_clean' in data and 'denoised_output' in data:
                        clean_np = data['original_clean']
                        denoised_np = data['denoised_output']
                        # Re-compute metrics (cheap)
                        metrics = compute_comprehensive_metrics(clean_np, denoised_np, fold, split_name)
                        return metrics
                    else:
                        print("    [WARNING] Key missing in .npz, re-running inference...")
                except Exception as e:
                    print(f"    [WARNING] Error loading resume file ({e}), re-running inference...")

            # Run Inference
            loader = DataLoader(dataset, batch_size=args.batch_size_test, shuffle=False)
            clean_all, noisy_all = [], []
            for c, n in loader:
                clean_all.append(c.numpy())
                noisy_all.append(n.numpy())
            
            if not clean_all: return {} # Empty dataset
            
            clean_np = ensure_numpy_shape(np.concatenate(clean_all))
            noisy_np = ensure_numpy_shape(np.concatenate(noisy_all))
            
            print(f"    Denoising {split_name} set ({len(clean_np)} samples)...")
            denoised_np = ddpm_denoise_averaged(model, diffusion, noisy_np, args.device, args.n_pairs, args.sampling_type)
            
            # Compute Metrics
            metrics = compute_comprehensive_metrics(clean_np, denoised_np, fold, split_name)
            
            # Save Signals (Train signals should be saved too)
            np.savez_compressed(
                save_path,
                original_clean=clean_np,
                noisy_input=noisy_np,
                denoised_output=denoised_np,
                fold_number=fold,
                data_type=split_name.lower()
            )
            return metrics
            
        # --- Process TRAIN ---
        # Note: train_dataset from retrieveDataSet returns (clean, noisy)
        train_metrics = process_and_save(train_dataset, "Train")
        
        # --- Process TEST ---
        test_metrics = process_and_save(test_dataset, "Test")
        
        # --- Process HIDDEN ---
        hidden_metrics = {}
        if hidden_dataset is not None:
             hidden_metrics = process_and_save(hidden_dataset, "Hidden")

        # Store Results (Matching DAE/VAE format)
        row = {
            'Split': f"split{fold}",
            
            'pcorr_train': train_metrics.get('pcorr', np.nan),
            'pcorr_test': test_metrics['pcorr'],
            'pcorr_hidden': hidden_metrics.get('pcorr', np.nan),
            
            'rmse_train': train_metrics.get('rmse', np.nan),
            'rmse_test': test_metrics['rmse'],
            'rmse_hidden': hidden_metrics.get('rmse', np.nan),
            
            'psnr_train': train_metrics.get('psnr', np.nan),
            'psnr_test': test_metrics['psnr'],
            'psnr_hidden': hidden_metrics.get('psnr', np.nan),
            
            'mse_train': train_metrics.get('mse', np.nan),
            'mse_test': test_metrics['mse'],
            'mse_hidden': hidden_metrics.get('mse', np.nan),
            
            'spearman_train': train_metrics.get('spearman', np.nan),
            'spearman_test': test_metrics['spearman'],
            'spearman_hidden': hidden_metrics.get('spearman', np.nan),
            
            'snr_train': train_metrics.get('snr', np.nan),
            'snr_test': test_metrics['snr'],
            'snr_hidden': hidden_metrics.get('snr', np.nan),
            
            'dtw_train': train_metrics.get('dtw', np.nan),
            'dtw_test': test_metrics['dtw'],
            'dtw_hidden': hidden_metrics.get('dtw', np.nan),
            
            'lsd_train': train_metrics.get('lsd', np.nan),
            'lsd_test': test_metrics['lsd'],
            'lsd_hidden': hidden_metrics.get('lsd', np.nan),
            
            'nmae_range_train': train_metrics.get('nmae_range', np.nan),
            'nmae_range_test': test_metrics['nmae_range'],
            'nmae_range_hidden': hidden_metrics.get('nmae_range', np.nan),
            
            'nmae_l1_train': train_metrics.get('nmae_l1', np.nan),
            'nmae_l1_test': test_metrics['nmae_l1'],
            'nmae_l1_hidden': hidden_metrics.get('nmae_l1', np.nan),
            
            'nmae_mean_train': train_metrics.get('nmae_mean', np.nan),
            'nmae_mean_test': test_metrics['nmae_mean'],
            'nmae_mean_hidden': hidden_metrics.get('nmae_mean', np.nan),
        }
        summary_values.append(pd.DataFrame([row]))
    
    # Save Summary
    if summary_values:
        df_res = pd.concat(summary_values, ignore_index=True)
        
        # Add Average/Std
        num = df_res.select_dtypes(include=np.number)
        avg = pd.DataFrame(num.mean()).T
        avg['Split'] = 'average'
        std = pd.DataFrame(num.std()).T
        std['Split'] = 'st. dev.'
        
        df_final = pd.concat([df_res, avg, std], ignore_index=True)
        df_final = df_final.round(5)
        
        fname = "summary_values.xlsx"
        out_path = os.path.join(output_dir, fname)
        df_final.to_excel(out_path, index=False)
        print(f"\nSaved summary results to {out_path}")

if __name__ == "__main__":
    main()

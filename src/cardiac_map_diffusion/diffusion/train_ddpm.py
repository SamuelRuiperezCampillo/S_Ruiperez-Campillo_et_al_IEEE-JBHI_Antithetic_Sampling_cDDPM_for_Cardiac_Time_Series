"""Conditional DDPM trainer (VAE-aligned) -- faithful migration of
``Diffusion_MAP_fullpipeline_final/ddpm_main_vae_aligned.py``.

Only the imports and the experiment-output directory were adapted to the package
layout (routed through :mod:`cardiac_map_diffusion.paths`). The model, training
loop, noise schedule, the 2-shot antithetic denoising used for evaluation, the
metrics, and all file I/O are unchanged. Run via ``scripts/train_ddpm.py`` or
``slurm/train_ddpm.sbatch``.

    - This is a modified version of main.py to align EXACTLY with VAE data splits and noise generation.
    - It uses the VAE's data loading and noise generation functions (from scripts_from_sam).
    - It uses the DDPM model and training loop structure.
    - It is designed to be compatible with submit_jobs_cluster.sh arguments.
"""

import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import json

# Resolve experiment-output paths through the package's paths layer.
from cardiac_map_diffusion import paths
current_dir = os.path.dirname(os.path.abspath(__file__))

# Import DDPM modules
from cardiac_map_diffusion.diffusion.ddpm_conditional import Diffusion
from cardiac_map_diffusion.diffusion.denoising_net import ConditionalModel
from cardiac_map_diffusion.diffusion.denoising_net_small import ConditionalModelSmall
from cardiac_map_diffusion.downstream.downstreamAPD import DownstreamAPD
from cardiac_map_diffusion.metrics.ddpm_metrics import compute_pearson_corr, compute_mse, compute_psnr, evaluate_model

# Import data/noise + MAP utilities (diffusion track)
from cardiac_map_diffusion.metrics import map_functions as mapf
from cardiac_map_diffusion.data.data_sam import get_MAP_vent_data
from cardiac_map_diffusion.data.ep_noise_sam import get_np_noisearrays
from cardiac_map_diffusion.data.splits import get_train_test_kfolds

# -----------------------------------------------------------------------------
# Metric Computation Logic (Copied from ddpm_denoising_allfolds.py)
# -----------------------------------------------------------------------------
def compute_comprehensive_metrics(clean_signals, denoised_signals, fold_num, split_type):
    """
    Compute all metrics that are calculated in the VAE script
    """
    logging.info(f'Computing comprehensive metrics for {split_type} set, fold {fold_num}...')
    
    # Compute all metrics using the same functions as VAE
    pcorr = mapf.compute_pearson_corr(clean_signals, denoised_signals, mode='total')
    rmse = mapf.compute_rmse(clean_signals, denoised_signals, mode='total')
    psnr = mapf.compute_psnr(clean_signals, denoised_signals, mode='total')
    mse = mapf.compute_mse(clean_signals, denoised_signals, mode='total')
    spearman = mapf.compute_spearman_corr(clean_signals, denoised_signals, mode='total')
    snr = mapf.compute_snr(clean_signals, denoised_signals, mode='total')
    dtw = mapf.compute_dtw(clean_signals, denoised_signals, mode='total')
    lsd = mapf.compute_lsd(clean_signals, denoised_signals, mode='total')
    nmae_range = mapf.compute_nmae(clean_signals, denoised_signals, norm='range', mode='total')
    nmae_l1 = mapf.compute_nmae(clean_signals, denoised_signals, norm='l1', mode='total')
    nmae_mean = mapf.compute_nmae(clean_signals, denoised_signals, norm='mean', mode='total')
    
    # Log metrics (same format as VAE)
    logging.info(f'{split_type} - PCC: {pcorr:.4f}, RMSE: {rmse:.4f}, PSNR: {psnr:.4f}')
    logging.info(f'{split_type} - MSE: {mse:.4f}, Spearman: {spearman:.4f}, SNR: {snr:.4f}')
    logging.info(f'{split_type} - DTW: {dtw:.4f}, LSD: {lsd:.4f}')
    logging.info(f'{split_type} - NMAE(range): {nmae_range:.4f}, NMAE(l1): {nmae_l1:.4f}, NMAE(mean): {nmae_mean:.4f}')
    
    # Return metrics dictionary (convert to native Python types for JSON serialization)
    return {
        'pcorr': float(pcorr),
        'rmse': float(rmse),
        'psnr': float(psnr),
        'mse': float(mse),
        'spearman': float(spearman),
        'snr': float(snr),
        'dtw': float(dtw),
        'lsd': float(lsd),
        'nmae_range': float(nmae_range),
        'nmae_l1': float(nmae_l1),
        'nmae_mean': float(nmae_mean)
    }

def ddpm_denoise_signals(model, diffusion, noisy_signals, device):
    """
    Use trained DDPM model to denoise signals using the EXACT same method as original main.py
    Uses 2-shot antithetic variates Monte Carlo (inference_antithetic)
    """
    model.eval()
    denoised_signals = []
    
    with torch.no_grad():
        # Process in batches if needed
        batch_size = min(32, len(noisy_signals))  # Limit batch size for memory
        
        for i in range(0, len(noisy_signals), batch_size):
            batch_end = min(i + batch_size, len(noisy_signals))
            noisy_batch = torch.tensor(noisy_signals[i:batch_end]).float().unsqueeze(1).to(device)
            
            # EXACT method from original main.py evaluate_model function:
            # Denoise using 2-shot Antithetic Variates Monte Carlo
            signals_denoised_crude, signals_denoised_anti = diffusion.inference_antithetic(model, 1, noisy_batch)
            denoised_batch = 0.5 * (signals_denoised_crude.squeeze(1) + signals_denoised_anti.squeeze(1))
            
            # Convert back to numpy
            denoised_batch_np = denoised_batch.cpu().numpy()
            if denoised_batch_np.ndim == 1:
                denoised_batch_np = denoised_batch_np.reshape(1, -1)
            
            denoised_signals.append(denoised_batch_np)
    
    return np.vstack(denoised_signals) if len(denoised_signals) > 1 else denoised_signals[0]

# -----------------------------------------------------------------------------
# VAE Data Splitting Logic (Copied from MAP_VAE/data.py)
# -----------------------------------------------------------------------------
# get_train_test_kfolds is centralised in cardiac_map_diffusion.data.splits
# (it was byte-identical across the diffusion loader, the baseline loader, and the
# trainer); imported at the top of this module.

# -----------------------------------------------------------------------------
# VAE Dataset Classes (Copied from MAP_VAE/utils.py)
# -----------------------------------------------------------------------------
class NumpyDataSet(torch.utils.data.Dataset):
    def __init__(self, array):
        self.array = array
    def __len__(self):
        return len(self.array)

    def __getitem__(self, i):
        clean = self.array[i]
        return clean

class NumpyDataSet_gaussian(torch.utils.data.Dataset):
    def __init__(self, array, noise=0.2):
        self.array = array
        self.noise = noise
    def __len__(self):
        return len(self.array)

    def __getitem__(self, i):
        clean = self.array[i]
        noisy = mapf.introduce_gaussian_noise(0, self.noise, self.array[i])
        return clean, noisy

class NumpyDataSet_spike(torch.utils.data.Dataset):
    def __init__(self, array):
        self.array = array

    def __len__(self):
        return len(self.array)

    def __getitem__(self, i):
        clean = self.array[i]
        noisy = mapf.introduce_spike_noise(self.array[i])
        return clean, noisy

class NumpyDataSet_bwander(torch.utils.data.Dataset):
    def __init__(self, array, min_freq=0.01, max_freq=0.3, min_sins=1, max_sins=4, max_amplitude=1):
        self.array = array
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.min_sins = min_sins
        self.max_sins = max_sins
        self.max_amplitude = max_amplitude
    def __len__(self):
        return len(self.array)

    def __getitem__(self, i):
        clean = self.array[i]
        noisy = mapf.introduce_baseline_wander(self.array[i], min_freq=self.min_freq, max_freq=self.max_freq,
                                               min_sins=self.min_sins, max_sins=self.max_sins,
                                               max_amplitude=self.max_amplitude)
        return clean, noisy

class NumpyDataSet_truncation(torch.utils.data.Dataset):
    def __init__(self, array, option, percent, var):
        self.array = array
        self.option = option
        self.percent = percent
        self.var = var
        self.trunc_func = mapf.introduce_truncation_noise
        self.normalize_func = mapf.normalize_EGM_array
    def __len__(self):
        return len(self.array)

    def __getitem__(self, i):
        clean = self.array[i]
        clean_std = self.normalize_func(clean)
        truncated = self.trunc_func(self.array[i], self.option, self.percent, self.var)
        truncated_std = self.normalize_func(truncated)
        return clean_std, truncated_std

class NumpyDataSet_allmixed(torch.utils.data.Dataset):
    def __init__(self, array, noise_ids, min_number_noises, max_number_noises, arrays):
        self.array = array
        self.all_noises_func = mapf.introduce_several_noises
        self.noise_ids = noise_ids
        self.min_number_noises = min_number_noises
        self.max_number_noises = max_number_noises
        self.normalize_func = mapf.normalize_EGM_array
        self.arrays = arrays
    
    def __len__(self):
        return len(self.array)

    def __getitem__(self, i):
        clean = self.array[i]
        clean_std = self.normalize_func(clean)
        noisy = self.all_noises_func(self.array[i], noise_ids=self.noise_ids,
                                     min_number_noises=self.min_number_noises,
                                     max_number_noises=self.max_number_noises,
                                     arrays=self.arrays)

        # Clip the input between 0 and 1 using numpy.clip
        clean_std = np.clip(clean_std, 0, 1)
        noisy = np.clip(noisy, 0, 1)
        return clean_std, noisy

class NumpyDataSet_epnoise(torch.utils.data.Dataset):
    def __init__(self, array, arrays, noise=0.25):
        self.array = array
        self.noise = noise
        self.arrays = arrays
    def __len__(self):
        return len(self.array)

    def __getitem__(self, i):
        clean = self.array[i]
        noisy = mapf.introduce_epnoise(self.arrays, clean, noise_boundary=self.noise)
        return clean, noisy

class NumpyDataSet_test_noisy(torch.utils.data.Dataset):
    def __init__(self, array, array_noise):
        self.array = array
        self.array_noise = array_noise

    def __len__(self):
        return len(self.array)

    def __getitem__(self, i):
        clean = self.array[i]
        noisy = self.array_noise[i]

        # Clip the input between 0 and 1 using numpy.clip
        clean = np.clip(clean, 0, 1)
        noisy = np.clip(noisy, 0, 1)
        return clean, noisy

# -----------------------------------------------------------------------------
# VAE Data Retrieval Logic (Copied from MAP_VAE/utils.py)
# -----------------------------------------------------------------------------
def retrieveDataSet(noise_type, noise_params, X_train, X_test, X_std_train, X_std_test, arrays=[]):
    if noise_type == 'none':
        train = NumpyDataSet(X_std_train)  # Returns original and noisy set of signals
        test = NumpyDataSet(X_std_test)  # Returns original and noisy set of signals
        X_std_test_noisy = X_std_test # No noise
    elif noise_type == 'gaussian':
        train = NumpyDataSet_gaussian(X_std_train,
                                      noise=noise_params[0])  # Returns original and noisy set of signals
        X_std_test_noisy = mapf.introduce_gaussian_noise(0, noise_params[0], X_std_test)
        test = NumpyDataSet_test_noisy(X_std_test, X_std_test_noisy)  # Returns original and noisy set of signals
        # Does not change noise every batch
    elif noise_type == 'spike':
        train = NumpyDataSet_spike(X_std_train)  # Returns original and noisy set of signals
        X_std_test_noisy = mapf.introduce_spike_noise(X_std_test)
        test = NumpyDataSet_test_noisy(X_std_test, X_std_test_noisy)  # Returns original and noisy set of signals
        # Does not change noise every batch
    elif noise_type == 'bwander' or noise_type == 'powerline':
        train = NumpyDataSet_bwander(X_std_train, min_freq=noise_params[0], max_freq=noise_params[1],
                                     min_sins=noise_params[2], max_sins=noise_params[3],
                                     max_amplitude=noise_params[4])
        # Returns original and noisy set of signals
        X_std_test_noisy = mapf.introduce_baseline_wander(X_std_test, min_freq=noise_params[0],
                                                          max_freq=noise_params[1],
                                                          min_sins=noise_params[2],
                                                          max_sins=noise_params[3],
                                                          max_amplitude=noise_params[4])
        test = NumpyDataSet_test_noisy(X_std_test, X_std_test_noisy)  # Returns original and noisy set of signals
        # Does not change noise every batch
    elif noise_type == 'truncation':  # Introduce the truncated function in the NumpyDataSet function
        train = NumpyDataSet_truncation(X_train, option=noise_params[2], percent=noise_params[0],
                                        var=noise_params[1])  # Returns original and noisy set of signals
        # Changes noise every batch
        X_std_test_noisy = mapf.introduce_truncation_noise(np.array(X_test), option=noise_params[2], percent=noise_params[0],
                                                           var=noise_params[1])
        test = NumpyDataSet_test_noisy(X_std_test, X_std_test_noisy)  # Returns original and noisy set of signals
    elif noise_type == 'allmixed':
        train = NumpyDataSet_allmixed(X_train, noise_ids=[1, 2, 3, 4, 5, 6],
                                      min_number_noises=1, max_number_noises=6,
                                      arrays=arrays)
        X_std_test_noisy = mapf.introduce_several_noises(X_test, noise_ids=[1, 2, 3, 4, 5, 6],
                                                         min_number_noises=1, max_number_noises=6,
                                                         arrays=arrays)
        test = NumpyDataSet_test_noisy(X_std_test, X_std_test_noisy)
    elif noise_type == 'ep':
        train = NumpyDataSet_epnoise(X_std_train, arrays=arrays, noise=noise_params[0])
        X_std_test_noisy = mapf.introduce_epnoise(arrays, X_std_test, noise_boundary=noise_params[0])
        test = NumpyDataSet_test_noisy(X_std_test, X_std_test_noisy)
    
    return train, test, X_std_test_noisy

# Function to correctly parse boolean values
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def ensure_tensor_shape(x, device):
    """
    Ensure input tensor has shape (Batch, 1, Length)
    Handles (Batch, Length), (Batch, 1, Length), (Batch, Length, 1)
    """
    x = x.to(device).float()
    if x.ndim == 2:
        x = x.unsqueeze(1)
    elif x.ndim == 3:
        if x.shape[2] == 1: # (Batch, Length, 1) -> (Batch, 1, Length)
            x = x.permute(0, 2, 1)
    return x

def ensure_numpy_shape(x):
    """
    Ensure input numpy array has shape (N, Length)
    """
    if x.ndim == 3:
        if x.shape[1] == 1:
            return x.squeeze(1)
        elif x.shape[2] == 1:
            return x.squeeze(2)
    return x

def train(args, train_dataset, test_dataset, fold_num, experiment_dir):
    ##############
    # Preparation for train 
    ##############

    # Unpack args
    device = args.device
    signal_len = args.signal_length
    lr = args.lr
    noise_steps = args.noise_steps
    noise_schedule = args.noise_schedule
    beta_start = args.beta_start
    beta_end = args.beta_end
    model_small = args.model_small
    feats = args.feats 
    use_pretrained = args.use_pretrained
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    batch_size_test = args.batch_size_test
    step_size = args.step_size
    gamma = args.gamma
    seed = args.seed

    # Prepare data loaders
    generator = torch.Generator(device=device) 
    
    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0) # num_workers=0 for safety on windows/cluster
    dataloader_test = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False, num_workers=0)

    # Load the model
    if model_small: 
        model = ConditionalModelSmall(feats).to(device)
    else:
        model = ConditionalModel(feats).to(device)

    if use_pretrained:
        checkpoint_path = os.path.join(current_dir, "model_pretrained.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            adjusted_checkpoint = {key.replace('model.', ''): value for key, value in checkpoint.items()}
            missing_keys, unexpected_keys = model.load_state_dict(adjusted_checkpoint, strict=False)
            print("Missing keys:", missing_keys)

    # Initialisation of optimisation
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    mse = nn.MSELoss()
    diffusion = Diffusion(noise_steps, beta_start, beta_end, signal_len, noise_schedule, device)

    # Setup logging 
    # Use shared experiment directory
    log_file_path = experiment_dir
    
    # Configure Python logging
    logging.basicConfig(
        filename=os.path.join(log_file_path, f"fold_{fold_num}.log"),
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%d-%b-%y %H:%M:%S',
        force=True
    )
    
    # Create a unique logger for this fold to avoid conflicts if running in parallel (though here it's sequential)
    # But we want to write to the same log file as VAE does? VAE appends.
    # Tensorboard writer needs a unique directory usually, or it mixes plots.
    # Let's give Tensorboard a subfolder per fold to be safe/clean
    writer = SummaryWriter(os.path.join(log_file_path, f"fold_{fold_num}"))
    l = len(dataloader_train)
    
    ##############
    # Train loop
    ##############

    for epoch in range(num_epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader_train)
        for i, (clean_batch, noisy_batch) in enumerate(pbar):
            # Data from VAE dataset is already (clean, noisy)
            # Ensure float and correct shape (Batch, 1, Length)
            signals = ensure_tensor_shape(clean_batch, device)
            x_noisy = ensure_tensor_shape(noisy_batch, device)
            
            t = diffusion.sample_timesteps(signals.shape[0]).to(device) # dim: [batch_size]
            sqrt_alpha_hat = diffusion.alpha_hat[t].sqrt()              # dim: [batch_size]
            x_t, noise = diffusion.noise_signal(signals, t)             # dim: [batch_size, channels, signal_len]
            
            predicted_noise = model(x_t, x_noisy, sqrt_alpha_hat)       # dim: [batch_size, channels, signal_len]
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            writer.add_scalar("MSE Train", loss.item(), global_step=epoch * l + i)
        
        scheduler.step()
        
        # Evaluate model on the test set
        if (epoch+1) % 40 == 0 and (epoch+1) < num_epochs: 
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for clean_b, noisy_b in dataloader_test:
                    clean_b = ensure_tensor_shape(clean_b, device)
                    noisy_b = ensure_tensor_shape(noisy_b, device)
                    t = diffusion.sample_timesteps(clean_b.shape[0]).to(device)
                    x_t, noise = diffusion.noise_signal(clean_b, t)
                    sqrt_alpha_hat = diffusion.alpha_hat[t].sqrt()
                    predicted_noise = model(x_t, noisy_b, sqrt_alpha_hat)
                    val_loss += mse(noise, predicted_noise).item()
                writer.add_scalar("MSE Test", val_loss / len(dataloader_test), global_step=epoch)
            model.train()

    # Save final model and close logger
    model_path = os.path.join(log_file_path, f"models", f"model_fold{fold_num}.pth")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model, model_path)
    logging.info(f"Model saved to {model_path}")
    
    # -------------------------------------------------------------------------
    # Final Comprehensive Evaluation (Same as VAE)
    # -------------------------------------------------------------------------
    logging.info("Starting final comprehensive evaluation...")
    
    # Collect all test data
    clean_test_all = []
    noisy_test_all = []
    for clean_b, noisy_b in dataloader_test:
        clean_test_all.append(clean_b.numpy())
        noisy_test_all.append(noisy_b.numpy())
    
    clean_test_np = np.concatenate(clean_test_all, axis=0)
    noisy_test_np = np.concatenate(noisy_test_all, axis=0)
    
    # Ensure correct shape for metric computation and denoising
    clean_test_np = ensure_numpy_shape(clean_test_np)
    noisy_test_np = ensure_numpy_shape(noisy_test_np)
    
    # Denoise using trained DDPM (EXACT method: 2-shot antithetic variates)
    denoised_test_np = ddpm_denoise_signals(model, diffusion, noisy_test_np, device)
    
    # Compute metrics for Test set
    test_metrics = compute_comprehensive_metrics(clean_test_np, denoised_test_np, fold_num, "Test")
    
    # Also compute for Train set (subset to save time if needed, but VAE does full)
    # For full comparison, we should do full train set, but it might be large.
    # Let's do full train set as requested "output the same as for the vae".
    clean_train_all = []
    noisy_train_all = []
    # Use a non-shuffled loader for evaluation
    train_eval_loader = DataLoader(train_dataset, batch_size=batch_size_test, shuffle=False, num_workers=0)
    for clean_b, noisy_b in train_eval_loader:
        clean_train_all.append(clean_b.numpy())
        noisy_train_all.append(noisy_b.numpy())
        
    clean_train_np = np.concatenate(clean_train_all, axis=0)
    noisy_train_np = np.concatenate(noisy_train_all, axis=0)
    
    # Ensure correct shape for metric computation and denoising
    clean_train_np = ensure_numpy_shape(clean_train_np)
    noisy_train_np = ensure_numpy_shape(noisy_train_np)
    
    denoised_train_np = ddpm_denoise_signals(model, diffusion, noisy_train_np, device)
    train_metrics = compute_comprehensive_metrics(clean_train_np, denoised_train_np, fold_num, "Train")
    
    # Save results to disk (JSON and NPZ) matching VAE structure
    denoised_dir = os.path.join(log_file_path, "denoised_signals")
    os.makedirs(denoised_dir, exist_ok=True)
    
    # Save training data and denoised outputs
    train_signals_dict = {
        'original_clean': clean_train_np,          
        'noisy_input': noisy_train_np,
        'denoised_output': denoised_train_np,   
        'fold_number': fold_num,
        'data_type': 'train'
    }
    
    train_file_path = os.path.join(denoised_dir, f"fold{fold_num}_train_signals.npz")
    np.savez_compressed(train_file_path, **train_signals_dict)
    logging.info(f'Training signals saved: {train_file_path}')
    
    # Save test data and denoised outputs  
    test_signals_dict = {
        'original_clean': clean_test_np,           
        'noisy_input': noisy_test_np,
        'denoised_output': denoised_test_np,    
        'fold_number': fold_num,
        'data_type': 'test'
    }
    
    test_file_path = os.path.join(denoised_dir, f"fold{fold_num}_test_signals.npz")
    np.savez_compressed(test_file_path, **test_signals_dict)
    logging.info(f'Test signals saved: {test_file_path}')
    
    # Save metadata for easy reference (matching VAE structure)
    metadata = {
        'fold_number': fold_num,
        'model_type': 'DDPM',
        'architecture': 'ConditionalModel', # or Small
        'noise_steps': noise_steps,
        'noise_type': args.noise_type,
        'num_epochs': num_epochs,
        'train_samples': int(clean_train_np.shape[0]),
        'test_samples': int(clean_test_np.shape[0]),
        'signal_length': int(clean_train_np.shape[1]),
        'final_metrics': {
            # Core metrics
            'train_pcc': float(train_metrics['pcorr']),
            'train_rmse': float(train_metrics['rmse']),
            'train_psnr': float(train_metrics['psnr']),
            'test_pcc': float(test_metrics['pcorr']),
            'test_rmse': float(test_metrics['rmse']),
            'test_psnr': float(test_metrics['psnr']),
            # Additional comprehensive metrics
            'train_mse': float(train_metrics['mse']),
            'train_spearman': float(train_metrics['spearman']),
            'train_snr': float(train_metrics['snr']),
            'train_nmae_range': float(train_metrics['nmae_range']),
            'train_nmae_l1': float(train_metrics['nmae_l1']),
            'train_nmae_mean': float(train_metrics['nmae_mean']),
            'test_mse': float(test_metrics['mse']),
            'test_spearman': float(test_metrics['spearman']),
            'test_snr': float(test_metrics['snr']),
            'test_nmae_range': float(test_metrics['nmae_range']),
            'test_nmae_l1': float(test_metrics['nmae_l1']),
            'test_nmae_mean': float(test_metrics['nmae_mean']),
            # DTW and LSD metrics
            'train_dtw': float(train_metrics['dtw']),
            'train_lsd': float(train_metrics['lsd']),
            'test_dtw': float(test_metrics['dtw']),
            'test_lsd': float(test_metrics['lsd'])
        }
    }
    
    metadata_file_path = os.path.join(denoised_dir, f"fold{fold_num}_metadata.json")
    with open(metadata_file_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    logging.info(f'Metadata saved: {metadata_file_path}')
    
    writer.close()

    # Return metrics for summary table
    return {
        'fold': fold_num,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'log_file_path': log_file_path
    }

def main():
    parser = argparse.ArgumentParser()
    # Arguments from main.py
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--signal_length", type=int, required=True)
    parser.add_argument("--n_prints", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--noise_batchwise", type=str2bool, required=True)
    parser.add_argument("--noise_steps", type=int, required=True)
    parser.add_argument("--noise_schedule", type=str, required=True)
    parser.add_argument("--beta_start", type=float, required=True)
    parser.add_argument("--beta_end", type=float, required=True)
    parser.add_argument("--use_pretrained", type=str2bool, required=True)
    parser.add_argument("--model_small", type=str2bool, required=True)
    parser.add_argument("--feats", type=int, required=True)  
    parser.add_argument("--num_epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--batch_size_test", type=int, required=True)
    parser.add_argument("--normalise", type=str2bool, required=True)
    parser.add_argument("--test_size", type=float, required=True)
    parser.add_argument("--n_splits", type=int, required=True)
    parser.add_argument("--step_size", type=int, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    
    # Additional arguments for VAE alignment (with defaults)
    parser.add_argument("--noise_type", type=str, default="allmixed")
    parser.add_argument("--seed_split", type=int, default=29)
    parser.add_argument("--exclude_patients_file", type=str, default=None, help="Path to CSV file containing 'pat_ID' column of patients to exclude from training")
    
    args = parser.parse_args()

    # Ensure reproducibility
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
        
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Get data using the function from Sam (VAE data loading)
    df_complete = get_MAP_vent_data()
    
    # Hidden test set logic
    df_hidden = None
    
    # Filter out excluded patients if provided
    if args.exclude_patients_file:
        if os.path.exists(args.exclude_patients_file):
            print(f"Loading excluded patients from {args.exclude_patients_file}...")
            try:
                excluded_df = pd.read_csv(args.exclude_patients_file)
                # Ensure pat_ID is string
                if 'pat_ID' in excluded_df.columns:
                    excluded_pats = excluded_df['pat_ID'].astype(str).unique()
                    
                    # Store hidden test set before filtering
                    df_complete['pat_ID'] = df_complete['pat_ID'].astype(str)
                    df_hidden = df_complete[df_complete['pat_ID'].isin(excluded_pats)]
                    
                    # Filter df_complete
                    initial_len = len(df_complete)
                    df_complete = df_complete[~df_complete['pat_ID'].isin(excluded_pats)]
                    final_len = len(df_complete)
                    
                    print(f"Excluded {len(excluded_pats)} patients.")
                    print(f"Dataframe reduced from {initial_len} to {final_len} samples.")
                    print(f"Hidden test set has {len(df_hidden)} samples.")
                else:
                    print(f"Warning: 'pat_ID' column not found in {args.exclude_patients_file}. No patients excluded.")
            except Exception as e:
                print(f"Error reading excluded patients file: {e}")
        else:
            print(f"Warning: Excluded patients file {args.exclude_patients_file} not found.")
    
    # Get electrophysiological noise using the function from Sam
    ep_noise_arrays = get_np_noisearrays(df_complete)
    
    # Prepare noise params (similar to VAE utils.py)
    noise_params = mapf.find_noise_params(args.noise_type)
    if args.noise_type == 'ep' or args.noise_type == 'allmixed':
        arrays = ep_noise_arrays
    else:
        arrays = []

    # Setup experiment directory (similar to VAE)
    # Use experiment name from args
    experiment_name = args.experiment
    
    # Log to a 'results' folder under the experiments root (see cardiac_map_diffusion.paths)
    log_dir = os.path.join(str(paths.experiments_root()), "ddpm", "results")
    
    # Create a unique experiment folder with timestamp to avoid overwriting
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = os.path.join(log_dir, experiment_name, f"{experiment_name}_{run_timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    print(f"Experiment directory: {experiment_dir}")
    
    # Save config
    with open(os.path.join(experiment_dir, "config.json"), 'w') as f:
        json.dump(vars(args), f, indent=4)

    summary_values = []

    # Train model for every fold
    # Note: main.py iterates over splits. We iterate over folds using VAE logic.
    # args.n_splits in main.py corresponds to num_folds here.
    for fold in range(args.n_splits):
        print(f"Processing Fold {fold}/{args.n_splits}")
        
        # Get splits using VAE logic
        X_train, X_test, y_train, y_test = get_train_test_kfolds(
            df_complete, num_folds=args.n_splits, split_number=fold, r_seed=args.seed_split
        )
        
        # Normalize (VAE logic)
        X_std_train, X_std_test = mapf.normalize_EGM_input(X_train, X_test)
        
        # Create Datasets (VAE logic)
        train_dataset, test_dataset, _ = retrieveDataSet(
            args.noise_type, noise_params, X_train, X_test, X_std_train, X_std_test, arrays=arrays
        )
        
        results = train(args, train_dataset, test_dataset, fold, experiment_dir)
        
        # Create summary row matching VAE structure
        train_m = results['train_metrics']
        test_m = results['test_metrics'] # This corresponds to the K-Fold validation set (referred to as "test" in VAE script)

        # ---------------------------------------------------------------------
        # Start of Hidden Test Set Evaluation
        # ---------------------------------------------------------------------
        hidden_m = {} # Empty dict if no hidden set
        if df_hidden is not None and len(df_hidden) > 0:
            print(f"Evaluating Fold {fold} on Hidden Test Set...")
            
            # Prepare hidden test set (similar to how we prepare other datasets)
            X_hidden = np.array(df_hidden['MAP_segments'].tolist())
            X_std_hidden = mapf.normalize_EGM_array(X_hidden)
            
            # We use retrieveDataSet to get consistency in noise generation
            # We treat X_hidden as "test" in retrieveDataSet
            # We don't care about "train" return here
            _, hidden_dataset, _ = retrieveDataSet(
                args.noise_type, noise_params, X_hidden, X_hidden, X_std_hidden, X_std_hidden, arrays=arrays
            )
            
            dataloader_hidden = DataLoader(hidden_dataset, batch_size=args.batch_size_test, shuffle=False, num_workers=0)
            
            # Inference Loop
            clean_hidden_all = []
            noisy_hidden_all = []
            
            # Use model (it is already loaded in memory after training? No, train() returns results dict only)
            # Need to reload model or pass it modify train() to return model?
            # Or load model from disk since we just saved it.
            model_path = os.path.join(experiment_dir, "models", f"model_fold{fold}.pth")
            
            # Reload model to be safe and ensure consistent state
            if args.model_small: 
                model = ConditionalModelSmall(args.feats).to(args.device)
            else:
                model = ConditionalModel(args.feats).to(args.device)
            
            # Load weights
            # Note: We saved the full model with torch.save(model, ...) inside train(), 
            # but usually it's better to save/load state_dict. The script uses torch.save(model).
            try:
                # We need to make sure we load what we saved.
                # In train(): torch.save(model, model_path)
                loaded_model = torch.load(model_path, map_location=args.device)
                if isinstance(loaded_model, (ConditionalModelSmall, ConditionalModel)):
                    model = loaded_model
                elif isinstance(loaded_model, dict): # state_dict
                     model.load_state_dict(loaded_model['model_state_dict'] if 'model_state_dict' in loaded_model else loaded_model)
            except Exception as e:
                print(f"Error loading model for hidden test evaluation: {e}")
            
            model.eval()
            diffusion_eval = Diffusion(args.noise_steps, args.beta_start, args.beta_end, args.signal_length, args.noise_schedule, args.device)
            
            for clean_b, noisy_b in dataloader_hidden:
                clean_hidden_all.append(clean_b.numpy())
                noisy_hidden_all.append(noisy_b.numpy())
            
            if clean_hidden_all:
                clean_hidden_np = np.concatenate(clean_hidden_all, axis=0)
                noisy_hidden_np = np.concatenate(noisy_hidden_all, axis=0)
                
                clean_hidden_np = ensure_numpy_shape(clean_hidden_np)
                noisy_hidden_np = ensure_numpy_shape(noisy_hidden_np)
                
                denoised_hidden_np = ddpm_denoise_signals(model, diffusion_eval, noisy_hidden_np, args.device)
                
                hidden_m = compute_comprehensive_metrics(clean_hidden_np, denoised_hidden_np, fold, "HiddenTest")
                
                # Save Hidden Signals
                denoised_dir = os.path.join(experiment_dir, "denoised_signals")
                hidden_signals_dict = {
                    'original_clean': clean_hidden_np,           
                    'noisy_input': noisy_hidden_np,
                    'denoised_output': denoised_hidden_np,    
                    'fold_number': fold,
                    'data_type': 'hidden_test'
                }
                np.savez_compressed(os.path.join(denoised_dir, f"fold{fold}_hidden_test_signals.npz"), **hidden_signals_dict)
        # ---------------------------------------------------------------------

        last_values = {
            'Split': f"split{fold}",
            'pcorr_train': train_m['pcorr'],
            'pcorr_val': test_m['pcorr'], # Renamed to 'val' to distinguish from hidden test
            'pcorr_hidden_test': hidden_m.get('pcorr', np.nan),
            
            'rmse_train': train_m['rmse'],
            'rmse_val': test_m['rmse'],
            'rmse_hidden_test': hidden_m.get('rmse', np.nan),
            
            'psnr_train': train_m['psnr'],
            'psnr_val': test_m['psnr'],
            'psnr_hidden_test': hidden_m.get('psnr', np.nan),
            
            'mse_train': train_m['mse'],
            'mse_val': test_m['mse'],
            'mse_hidden_test': hidden_m.get('mse', np.nan),
            
            'spearman_train': train_m['spearman'],
            'spearman_val': test_m['spearman'],
            'spearman_hidden_test': hidden_m.get('spearman', np.nan),
            
            'snr_train': train_m['snr'],
            'snr_val': test_m['snr'],
            'snr_hidden_test': hidden_m.get('snr', np.nan),
            
            'dtw_train': train_m['dtw'],
            'dtw_val': test_m['dtw'],
            'dtw_hidden_test': hidden_m.get('dtw', np.nan),
            
            'lsd_train': train_m['lsd'],
            'lsd_val': test_m['lsd'],
            'lsd_hidden_test': hidden_m.get('lsd', np.nan),
            
            'nmae_range_train': train_m['nmae_range'],
            'nmae_range_val': test_m['nmae_range'],
            'nmae_range_hidden_test': hidden_m.get('nmae_range', np.nan),
            
            'nmae_l1_train': train_m['nmae_l1'],
            'nmae_l1_val': test_m['nmae_l1'],
            'nmae_l1_hidden_test': hidden_m.get('nmae_l1', np.nan),
            
            'nmae_mean_train': train_m['nmae_mean'],
            'nmae_mean_val': test_m['nmae_mean'],
            'nmae_mean_hidden_test': hidden_m.get('nmae_mean', np.nan)
        }
        
        summary_values.append(pd.DataFrame([last_values]))

    # Save summary Excel file (EXACTLY as VAE)
    if summary_values:
        combined_df = pd.concat(summary_values, ignore_index=True)
        
        # Calculate average and std
        numeric_cols = combined_df.select_dtypes(include=np.number)
        average_row = numeric_cols.mean()
        std_dev_row = numeric_cols.std()
        
        average_df = pd.DataFrame(average_row).transpose()
        average_df['Split'] = 'average'
        std_dev_df = pd.DataFrame(std_dev_row).transpose()
        std_dev_df['Split'] = 'st. dev.'
        
        combined_df = pd.concat([combined_df, average_df, std_dev_df], ignore_index=True)
        combined_df = combined_df.round(5)
        
        output_path = os.path.join(experiment_dir, "summary_values.xlsx")
        combined_df.to_excel(output_path, index=False)
        print(f"Excel file 'summary_values.xlsx' saved with data to {output_path}")

if __name__ == "__main__":
    main()

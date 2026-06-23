#!/usr/bin/env python3
"""
Comprehensive Metrics Calculator for All Baseline Models

This script processes experiment folders with the standard structure and calculates
all metrics (psnr, snr, rmse, mse, pcorr, spearman, lsd, dwt) for fair comparison
across traditional filters, deep learning baselines, and VAE models.

Faithful migration of ``MAP_VAE/compute_comprehensive_metrics.py`` (an argparse
``main()`` entry script). Only mechanical edits were applied: the hardcoded
``--base_path`` default of ``/cluster/.../MAP_VAE/experiments/`` was routed
through ``cardiac_map_diffusion.paths.experiments_root()`` and this migration
note was added. The ``from metrics import compute_lsd, compute_dwt_distance``
line could NOT be mapped (no ``metrics`` module is in the import map for
``MAP_VAE`` files and ``compute_dwt_distance`` is absent from the package); it
is flagged below and left unchanged -- the surrounding ``try/except`` already
falls back to in-file implementations. All metric logic is otherwise
byte-for-byte unchanged.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from scipy import stats
import pywt
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

from cardiac_map_diffusion import paths

# NOTE: there is no top-level `metrics` module in this package, so the optional
# import below is expected to fail and the try/except installs self-contained
# fallback implementations of compute_lsd / compute_dwt_distance. No sys.path
# manipulation is needed.
try:
    from metrics import compute_lsd, compute_dwt_distance  # noqa: F401 (intentional optional import)
except ImportError:
    print("Warning: Could not import LSD/DWT functions from metrics.py")
    print("Implementing fallback versions...")
    
    def compute_lsd(clean: np.ndarray, denoised: np.ndarray) -> float:
        """
        Log Spectral Distance - measures spectral similarity
        """
        try:
            # Compute power spectral densities
            clean_fft = np.fft.fft(clean, axis=-1)
            denoised_fft = np.fft.fft(denoised, axis=-1)
            
            clean_psd = np.abs(clean_fft) ** 2
            denoised_psd = np.abs(denoised_fft) ** 2
            
            # Add small epsilon to avoid log(0)
            eps = 1e-10
            clean_psd = np.maximum(clean_psd, eps)
            denoised_psd = np.maximum(denoised_psd, eps)
            
            # Compute log spectral distance
            lsd = np.mean((np.log(clean_psd) - np.log(denoised_psd)) ** 2)
            return float(np.sqrt(lsd))
        except Exception as e:
            print(f"Warning: LSD computation failed: {e}")
            return np.nan
    
    def compute_dwt_distance(clean: np.ndarray, denoised: np.ndarray, wavelet: str = 'db4', levels: int = 3) -> float:
        """
        Discrete Wavelet Transform distance - measures time-frequency similarity
        """
        try:
            # Ensure signals have the same length
            min_len = min(len(clean), len(denoised))
            clean_trim = clean[:min_len]
            denoised_trim = denoised[:min_len]
            
            # Compute DWT coefficients
            clean_coeffs = pywt.wavedec(clean_trim, wavelet, level=levels)
            denoised_coeffs = pywt.wavedec(denoised_trim, wavelet, level=levels)
            
            # Compute distance between coefficient sets
            total_distance = 0.0
            for clean_c, denoised_c in zip(clean_coeffs, denoised_coeffs):
                total_distance += np.mean((clean_c - denoised_c) ** 2)
            
            return float(total_distance)
        except Exception as e:
            print(f"Warning: DWT computation failed: {e}")
            return np.nan


def compute_all_metrics_comprehensive(clean: np.ndarray, denoised: np.ndarray) -> Dict[str, float]:
    """
    Compute all 8 comprehensive metrics for fair comparison
    
    Args:
        clean: Clean reference signals [N, signal_length] 
        denoised: Denoised signals [N, signal_length]
        
    Returns:
        Dictionary with all metric values
    """
    metrics = {}
    
    try:
        # Ensure proper shapes
        if clean.ndim == 1:
            clean = clean.reshape(1, -1)
        if denoised.ndim == 1:
            denoised = denoised.reshape(1, -1)
            
        # Flatten for some metrics that work on 1D
        clean_flat = clean.flatten()
        denoised_flat = denoised.flatten()
        
        # 1. Pearson Correlation Coefficient (higher is better)
        try:
            pcorr, _ = stats.pearsonr(clean_flat, denoised_flat)
            metrics['pcorr'] = float(pcorr) if not np.isnan(pcorr) else 0.0
        except:
            metrics['pcorr'] = 0.0
            
        # 2. Spearman Correlation (higher is better) 
        try:
            spearman, _ = stats.spearmanr(clean_flat, denoised_flat)
            metrics['spearman'] = float(spearman) if not np.isnan(spearman) else 0.0
        except:
            metrics['spearman'] = 0.0
            
        # 3. RMSE (lower is better)
        try:
            rmse = np.sqrt(mean_squared_error(clean_flat, denoised_flat))
            metrics['rmse'] = float(rmse)
        except:
            metrics['rmse'] = np.inf
            
        # 4. MSE (lower is better)
        try:
            mse = mean_squared_error(clean_flat, denoised_flat)
            metrics['mse'] = float(mse)
        except:
            metrics['mse'] = np.inf
            
        # 5. PSNR (higher is better)
        try:
            mse_val = metrics['mse']
            if mse_val > 0:
                max_val = np.max(clean_flat) if len(clean_flat) > 0 else 1.0
                psnr = 20 * np.log10(max_val / np.sqrt(mse_val))
                metrics['psnr'] = float(psnr)
            else:
                metrics['psnr'] = 100.0  # Perfect reconstruction
        except:
            metrics['psnr'] = 0.0
            
        # 6. SNR (higher is better)
        try:
            signal_power = np.mean(clean_flat ** 2)
            noise_power = np.mean((clean_flat - denoised_flat) ** 2)
            if noise_power > 0 and signal_power > 0:
                snr = 10 * np.log10(signal_power / noise_power)
                metrics['snr'] = float(snr)
            else:
                metrics['snr'] = 100.0  # Perfect reconstruction
        except:
            metrics['snr'] = 0.0
            
        # 7. Log Spectral Distance (lower is better)
        try:
            # Compute LSD per sample and average
            lsd_values = []
            for i in range(clean.shape[0]):
                lsd_val = compute_lsd(clean[i], denoised[i])
                if not np.isnan(lsd_val):
                    lsd_values.append(lsd_val)
            
            if lsd_values:
                metrics['lsd'] = float(np.mean(lsd_values))
            else:
                metrics['lsd'] = np.inf
        except Exception as e:
            print(f"LSD computation failed: {e}")
            metrics['lsd'] = np.inf
            
        # 8. DWT Distance (lower is better)
        try:
            # Compute DWT distance per sample and average
            dwt_values = []
            for i in range(clean.shape[0]):
                dwt_val = compute_dwt_distance(clean[i], denoised[i])
                if not np.isnan(dwt_val):
                    dwt_values.append(dwt_val)
                    
            if dwt_values:
                metrics['dwt'] = float(np.mean(dwt_values))
            else:
                metrics['dwt'] = np.inf
        except Exception as e:
            print(f"DWT computation failed: {e}")
            metrics['dwt'] = np.inf
            
    except Exception as e:
        print(f"Error in comprehensive metrics computation: {e}")
        # Return safe defaults
        metrics = {
            'pcorr': 0.0, 'spearman': 0.0, 'rmse': np.inf, 'mse': np.inf,
            'psnr': 0.0, 'snr': 0.0, 'lsd': np.inf, 'dwt': np.inf
        }
    
    return metrics


def load_signals_from_folder(experiment_folder: str) -> Tuple[Dict, bool]:
    """
    Load denoised signals from experiment folder
    
    Args:
        experiment_folder: Path to experiment directory
        
    Returns:
        Tuple of (fold_data_dict, success_flag)
    """
    signals_folder = os.path.join(experiment_folder, 'denoised_signals')
    
    if not os.path.exists(signals_folder):
        print(f"❌ Signals folder not found: {signals_folder}")
        return {}, False
        
    fold_data = {}
    
    # Load all fold signal files
    for fold_idx in range(4):  # Assuming 4-fold CV
        for split in ['train', 'test']:
            signal_file = os.path.join(signals_folder, f'fold{fold_idx}_{split}_signals.npz')
            
            if not os.path.exists(signal_file):
                print(f"❌ Missing signal file: {signal_file}")
                return {}, False
                
            try:
                data = np.load(signal_file)
                
                # Expected keys: original_clean, denoised_output, noisy_input
                if 'original_clean' in data and 'denoised_output' in data:
                    fold_data[f'fold{fold_idx}_{split}'] = {
                        'clean': data['original_clean'],
                        'denoised': data['denoised_output'],
                        'noisy': data['noisy_input'] if 'noisy_input' in data else None
                    }
                else:
                    print(f"❌ Invalid signal file format: {signal_file}")
                    print(f"   Available keys: {list(data.keys())}")
                    return {}, False
                    
            except Exception as e:
                print(f"❌ Error loading {signal_file}: {e}")
                return {}, False
                
    print(f"✅ Successfully loaded signals from {len(fold_data)} files")
    return fold_data, True


def process_experiment_folder(experiment_folder: str) -> Optional[Dict]:
    """
    Process single experiment folder and compute comprehensive metrics
    
    Args:
        experiment_folder: Path to experiment directory
        
    Returns:
        Dictionary with computed metrics or None if failed
    """
    print(f"\n🔍 Processing: {os.path.basename(experiment_folder)}")
    
    # Load signals
    fold_data, success = load_signals_from_folder(experiment_folder)
    if not success:
        return None
        
    # Compute metrics for each fold and split
    all_metrics = {}
    
    for fold_split, signals in fold_data.items():
        print(f"  📊 Computing metrics for {fold_split}...")
        
        clean = signals['clean']
        denoised = signals['denoised']
        
        # Compute comprehensive metrics
        metrics = compute_all_metrics_comprehensive(clean, denoised)
        all_metrics[fold_split] = metrics
        
        # Print key metrics for verification
        print(f"    PCC: {metrics['pcorr']:.4f}, RMSE: {metrics['rmse']:.4f}, LSD: {metrics['lsd']:.4f}")
    
    # Calculate cross-fold statistics
    metric_names = ['pcorr', 'spearman', 'rmse', 'mse', 'psnr', 'snr', 'lsd', 'dwt']
    summary_stats = {}
    
    for metric in metric_names:
        for split in ['train', 'test']:
            values = [all_metrics[f'fold{i}_{split}'][metric] for i in range(4)]
            
            # Handle infinite values
            finite_values = [v for v in values if np.isfinite(v)]
            
            if finite_values:
                summary_stats[f'{metric}_{split}_mean'] = np.mean(finite_values)
                summary_stats[f'{metric}_{split}_std'] = np.std(finite_values)
            else:
                summary_stats[f'{metric}_{split}_mean'] = np.nan
                summary_stats[f'{metric}_{split}_std'] = np.nan
    
    return {
        'fold_metrics': all_metrics,
        'summary_stats': summary_stats,
        'experiment_folder': experiment_folder
    }


def save_comprehensive_results(results: Dict, experiment_folder: str):
    """
    Save comprehensive metrics to Excel file in DAE-compatible format
    
    Args:
        results: Dictionary with metrics results
        experiment_folder: Path to experiment directory
    """
    model_name = os.path.basename(experiment_folder)
    output_file = os.path.join(experiment_folder, f'summary_values_{model_name}.xlsx')
    
    # Create DAE-format results 
    dae_format_rows = []
    
    # Metric order from comprehensive metrics (add loss=mse, nmae placeholders for compatibility)
    base_metrics = ['pcorr', 'spearman', 'rmse', 'mse', 'psnr', 'snr', 'lsd', 'dwt']
    
    # Add fold-by-fold results in DAE format (split0, split1, etc.)
    for fold_idx in range(4):
        row = {'Split': f'split{fold_idx}'}
        
        # Add metrics for train and test
        for metric in base_metrics:
            train_key = f'fold{fold_idx}_train'
            test_key = f'fold{fold_idx}_test'
            
            if train_key in results['fold_metrics'] and metric in results['fold_metrics'][train_key]:
                row[f'{metric}_train'] = results['fold_metrics'][train_key][metric]
            else:
                row[f'{metric}_train'] = 0.0
                
            if test_key in results['fold_metrics'] and metric in results['fold_metrics'][test_key]:
                row[f'{metric}_test'] = results['fold_metrics'][test_key][metric]
            else:
                row[f'{metric}_test'] = 0.0
        
        # Add loss as mse for compatibility with DAE format
        row['loss_train'] = row.get('mse_train', 0.0)
        row['loss_test'] = row.get('mse_test', 0.0)
        
        # Add nmae placeholders (not computed in comprehensive metrics)
        for nmae_type in ['nmae_range', 'nmae_l1', 'nmae_mean']:
            row[f'{nmae_type}_train'] = 0.0
            row[f'{nmae_type}_test'] = 0.0
        
        dae_format_rows.append(row)
    
    # Add average row
    avg_row = {'Split': 'average'}
    for metric in base_metrics:
        avg_row[f'{metric}_train'] = results['summary_stats'].get(f'{metric}_train_mean', 0.0)
        avg_row[f'{metric}_test'] = results['summary_stats'].get(f'{metric}_test_mean', 0.0)
    
    # Add loss as mse
    avg_row['loss_train'] = avg_row.get('mse_train', 0.0)
    avg_row['loss_test'] = avg_row.get('mse_test', 0.0)
    
    # Add nmae placeholders
    for nmae_type in ['nmae_range', 'nmae_l1', 'nmae_mean']:
        avg_row[f'{nmae_type}_train'] = 0.0
        avg_row[f'{nmae_type}_test'] = 0.0
    
    dae_format_rows.append(avg_row)
    
    # Add std dev row
    std_row = {'Split': 'st. dev.'}
    for metric in base_metrics:
        std_row[f'{metric}_train'] = results['summary_stats'].get(f'{metric}_train_std', 0.0)
        std_row[f'{metric}_test'] = results['summary_stats'].get(f'{metric}_test_std', 0.0)
    
    # Add loss as mse std
    std_row['loss_train'] = std_row.get('mse_train', 0.0)
    std_row['loss_test'] = std_row.get('mse_test', 0.0)
    
    # Add nmae std placeholders
    for nmae_type in ['nmae_range', 'nmae_l1', 'nmae_mean']:
        std_row[f'{nmae_type}_train'] = 0.0
        std_row[f'{nmae_type}_test'] = 0.0
    
    dae_format_rows.append(std_row)
    
    # Create DataFrame with DAE-compatible column ordering
    df = pd.DataFrame(dae_format_rows)
    
    # Reorder columns to match DAE format exactly
    dae_metric_order = ['loss', 'pcorr', 'psnr', 'rmse', 'mse', 'spearman', 'snr', 
                        'nmae_range', 'nmae_l1', 'nmae_mean', 'lsd', 'dwt']
    
    ordered_columns = ['Split']
    for metric in dae_metric_order:
        if f'{metric}_train' in df.columns:
            ordered_columns.extend([f'{metric}_train', f'{metric}_test'])
    
    # Only include columns that exist
    final_columns = [col for col in ordered_columns if col in df.columns]
    df = df[final_columns]
    
    df.to_excel(output_file, index=False)
    print(f"💾 DAE-format comprehensive metrics saved to: {output_file}")
    
    # Also create legacy detailed format for backward compatibility
    legacy_file = os.path.join(experiment_folder, f'detailed_results_{model_name}.xlsx')
    
    legacy_rows = []
    # Add fold-by-fold results in legacy format
    for fold_idx in range(4):
        for split in ['train', 'test']:
            fold_key = f'fold{fold_idx}_{split}'
            if fold_key in results['fold_metrics']:
                row = {
                    'fold': fold_idx,
                    'split': split,
                    'condition': 'denoised'
                }
                row.update(results['fold_metrics'][fold_key])
                legacy_rows.append(row)
    
    # Add summary statistics in legacy format
    metric_names = ['pcorr', 'spearman', 'rmse', 'mse', 'psnr', 'snr', 'lsd', 'dwt']
    for split in ['train', 'test']:
        # Mean row
        mean_row = {'fold': 'mean', 'split': split, 'condition': 'denoised'}
        for metric in metric_names:
            mean_row[metric] = results['summary_stats'][f'{metric}_{split}_mean']
        legacy_rows.append(mean_row)
        
        # Std row
        std_row = {'fold': 'std', 'split': split, 'condition': 'denoised'}
        for metric in metric_names:
            std_row[metric] = results['summary_stats'][f'{metric}_{split}_std']
        legacy_rows.append(std_row)
    
    legacy_df = pd.DataFrame(legacy_rows)
    ordered_cols = ['fold', 'split', 'condition'] + metric_names
    legacy_df = legacy_df[ordered_cols]
    legacy_df.to_excel(legacy_file, index=False)
    print(f"💾 Legacy format saved to: {legacy_file}")


def main():
    parser = argparse.ArgumentParser(description='Compute comprehensive metrics for all baseline models')
    parser.add_argument('--experiment_folders', nargs='+', required=True,
                       help='List of experiment folder paths')
    parser.add_argument('--base_path', type=str,
                       default=str(paths.experiments_root()),
                       help='Base path for experiments')
    
    args = parser.parse_args()
    
    print("🚀 Comprehensive Metrics Calculator")
    print("=" * 60)
    print(f"Processing {len(args.experiment_folders)} experiment folders...")
    
    successful_folders = []
    failed_folders = []
    
    for folder_name in args.experiment_folders:
        # Handle both full paths and relative names
        if folder_name.startswith('/'):
            experiment_folder = folder_name
        else:
            experiment_folder = os.path.join(args.base_path, folder_name)
            
        if not os.path.exists(experiment_folder):
            print(f"❌ Folder not found: {experiment_folder}")
            failed_folders.append(folder_name)
            continue
            
        try:
            results = process_experiment_folder(experiment_folder)
            
            if results is not None:
                save_comprehensive_results(results, experiment_folder)
                successful_folders.append(folder_name)
                print(f"✅ Successfully processed: {os.path.basename(experiment_folder)}")
            else:
                failed_folders.append(folder_name)
                print(f"❌ Failed to process: {os.path.basename(experiment_folder)}")
                
        except Exception as e:
            print(f"❌ Error processing {folder_name}: {e}")
            failed_folders.append(folder_name)
    
    # Final summary
    print(f"\n📋 PROCESSING SUMMARY")
    print("=" * 40)
    print(f"✅ Successful: {len(successful_folders)}")
    print(f"❌ Failed: {len(failed_folders)}")
    
    if successful_folders:
        print(f"\n🎉 Successfully processed:")
        for folder in successful_folders:
            print(f"   - {folder}")
            
    if failed_folders:
        print(f"\n⚠️ Failed folders:")
        for folder in failed_folders:
            print(f"   - {folder}")
    
    print(f"\n📊 Each successful folder now contains:")
    print(f"   - summary_values_<model>.xlsx (detailed metrics)")
    print(f"   - summary_comparison_<model>.xlsx (quick comparison)")


if __name__ == "__main__":
    # Default folder list for your experiments
    default_folders = [
        "TVL1_baseline_nfolds4_noise_allmixed_rs17_rss29_weight_0_02",
        "TVL1_baseline_nfolds4_noise_allmixed_rs17_rss29_weight_0_01", 
        "TVL1_baseline_nfolds4_noise_allmixed_rs17_rss29_weight_0_05",
        "TVL1_baseline_nfolds4_noise_allmixed_rs17_rss29_weight_0_1",
        "WAVELET_baseline_nfolds4_noise_allmixed_rs17_rss29_db4_L5_soft_sure",
        "WAVELET_baseline_nfolds4_noise_allmixed_rs17_rss29_sym4_L5_soft_sure",
        "WAVELET_baseline_nfolds4_noise_allmixed_rs17_rss29_sym4_L5_soft_visushrink",
        "WAVELET_baseline_nfolds4_noise_allmixed_rs17_rss29_db4_L5_soft_visushrink",
        "FILTER_tv_l1_baseline_nfolds4_noise_allmixed_rs17_rss29",
        "FILTER_hybrid_savgol_median_baseline_nfolds4_noise_allmixed_rs17_rss29",
        "FILTER_butterworth_baseline_nfolds4_noise_allmixed_rs17_rss29",
        "FILTER_wiener_baseline_nfolds4_noise_allmixed_rs17_rss29",
        "FILTER_wavelet_baseline_nfolds4_noise_allmixed_rs17_rss29",
        "FILTER_adaptive_notch_baseline_nfolds4_noise_allmixed_rs17_rss29",
        "reg_DRRN_baseline_1000max_nfolds4_bs32_lr_5e-5_noise_allmixed_rs17_rss29_hs64_es10",
        "DRRN_baseline_1000max_nfolds4_bs32_lr_5e-5_noise_allmixed_rs17_rss29_hs64_es10",
        "LUNet_baseline_1000max_nfolds4_bs32_lr_5e-5_noise_allmixed_rs17_rss29_es10",
        "DAE_baseline_10000max_nfolds4_bs32_lr_5e-5_noise_allmixed_rs17_rss29_ls32_es10",
        "2023_0808_allmixed_cyclbeta4_init_1lrelu_50_nfolds4_withepbs_32_lr_5e-3_noise_allmixed_split100_rs17_rss29_final"
    ]
    
    if len(sys.argv) == 1:
        print("🎯 No arguments provided, using default experiment folders...")
        import sys
        sys.argv.extend(['--experiment_folders'] + default_folders)
        
    main()
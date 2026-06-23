#!/usr/bin/env python3
"""
Results Aggregator for Comprehensive Model Comparison

This script collects all the individual model results and creates a unified
comparison table across all baseline methods (filters, deep learning, VAE).

Faithful migration of ``MAP_VAE/aggregate_model_comparison.py`` (an argparse
``main()`` entry script). Only mechanical edits were applied: the hardcoded
``--base_path`` default of ``/cluster/.../MAP_VAE/experiments/`` was routed
through ``cardiac_map_diffusion.paths.experiments_root()`` and this migration
note was appended. All aggregation logic is otherwise byte-for-byte unchanged.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

from cardiac_map_diffusion import paths


def extract_model_info(folder_name: str) -> Tuple[str, str, Dict]:
    """
    Extract model type, variant, and parameters from folder name
    
    Args:
        folder_name: Name of experiment folder
        
    Returns:
        Tuple of (model_type, model_variant, parameters)
    """
    folder_name = folder_name.strip('/')
    
    # VAE model
    if 'allmixed_cyclbeta4' in folder_name:
        return 'VAE', 'CyclicalBeta', {'beta': '4', 'lr': '5e-3'}
    
    # Deep learning baselines
    elif 'DAE_baseline' in folder_name:
        return 'DAE', 'Standard', {'epochs': '10000', 'lr': '5e-5', 'latent': '32'}
    elif 'LUNet_baseline' in folder_name:
        return 'LUNet', 'Standard', {'epochs': '1000', 'lr': '5e-5'}
    elif 'DRRN_baseline' in folder_name and 'reg_' in folder_name:
        return 'DRRN', 'Regularized', {'epochs': '1000', 'lr': '5e-5', 'hidden': '64'}
    elif 'DRRN_baseline' in folder_name:
        return 'DRRN', 'Standard', {'epochs': '1000', 'lr': '5e-5', 'hidden': '64'}
    
    # Traditional filters with parameter sweeps
    elif 'TVL1_baseline' in folder_name:
        if 'weight_0_01' in folder_name:
            return 'TV-L1', 'λ=0.01', {'weight': '0.01'}
        elif 'weight_0_02' in folder_name:
            return 'TV-L1', 'λ=0.02', {'weight': '0.02'}
        elif 'weight_0_05' in folder_name:
            return 'TV-L1', 'λ=0.05', {'weight': '0.05'}
        elif 'weight_0_1' in folder_name:
            return 'TV-L1', 'λ=0.1', {'weight': '0.1'}
    
    elif 'WAVELET_baseline' in folder_name:
        if 'db4' in folder_name and 'sure' in folder_name:
            return 'Wavelet', 'DB4-SURE', {'wavelet': 'db4', 'method': 'sure'}
        elif 'db4' in folder_name and 'visushrink' in folder_name:
            return 'Wavelet', 'DB4-VisuShrink', {'wavelet': 'db4', 'method': 'visushrink'}
        elif 'sym4' in folder_name and 'sure' in folder_name:
            return 'Wavelet', 'Sym4-SURE', {'wavelet': 'sym4', 'method': 'sure'}
        elif 'sym4' in folder_name and 'visushrink' in folder_name:
            return 'Wavelet', 'Sym4-VisuShrink', {'wavelet': 'sym4', 'method': 'visushrink'}
    
    # Standard filter baselines
    elif 'FILTER_' in folder_name:
        filter_type = folder_name.split('FILTER_')[1].split('_baseline')[0]
        
        filter_map = {
            'butterworth': ('Butterworth', 'Notch+Bandpass'),
            'wiener': ('Wiener', 'Adaptive'),
            'tv_l1': ('TV-L1', 'Standard'),
            'hybrid_savgol_median': ('Hybrid', 'SavGol+Median'),
            'wavelet': ('Wavelet', 'Standard'),
            'adaptive_notch': ('Adaptive', 'Notch+LMS')
        }
        
        if filter_type in filter_map:
            return filter_map[filter_type][0], filter_map[filter_type][1], {}
    
    # Fallback
    return 'Unknown', folder_name, {}


def load_model_results(experiment_folder: str) -> Dict:
    """
    Load results from a single model's experiment folder (supports both legacy and DAE formats)
    
    Args:
        experiment_folder: Path to experiment folder
        
    Returns:
        Dictionary with model results or None if failed
    """
    folder_name = os.path.basename(experiment_folder)
    
    # Look for the comprehensive results file
    results_file = os.path.join(experiment_folder, f'summary_values_{folder_name}.xlsx')
    
    if not os.path.exists(results_file):
        print(f"❌ Results file not found: {results_file}")
        return None
    
    try:
        # Load the Excel file
        df = pd.read_excel(results_file)
        
        # Define all possible metric names (comprehensive set)
        all_metrics = ['loss', 'pcorr', 'psnr', 'rmse', 'mse', 'spearman', 'snr', 
                      'nmae_range', 'nmae_l1', 'nmae_mean', 'lsd', 'dwt']
        
        results = {}
        
        # Check if this is DAE format (has 'Split' column) or legacy format (has 'fold' and 'split' columns)
        if 'Split' in df.columns:
            # DAE format: Split column with 'average' and 'st. dev.' rows
            print(f"  📊 Loading DAE format from {folder_name}")
            
            avg_row = df[df['Split'] == 'average']
            std_row = df[df['Split'] == 'st. dev.']
            
            if len(avg_row) > 0 and len(std_row) > 0:
                for metric in all_metrics:
                    train_col = f'{metric}_train'
                    test_col = f'{metric}_test'
                    
                    if train_col in df.columns and test_col in df.columns:
                        # Extract train results
                        train_mean = avg_row[train_col].iloc[0] if len(avg_row) > 0 else 0.0
                        train_std = std_row[train_col].iloc[0] if len(std_row) > 0 else 0.0
                        results[f'{metric}_train_mean'] = train_mean
                        results[f'{metric}_train_std'] = train_std
                        
                        # Extract test results
                        test_mean = avg_row[test_col].iloc[0] if len(avg_row) > 0 else 0.0
                        test_std = std_row[test_col].iloc[0] if len(std_row) > 0 else 0.0
                        results[f'{metric}_test_mean'] = test_mean
                        results[f'{metric}_test_std'] = test_std
                        
        elif 'fold' in df.columns and 'split' in df.columns:
            # Legacy format: fold and split columns with 'mean' and 'std' fold values
            print(f"  📊 Loading legacy format from {folder_name}")
            
            legacy_metrics = ['pcorr', 'spearman', 'rmse', 'mse', 'psnr', 'snr', 'lsd', 'dwt']
            
            for split in ['train', 'test']:
                mean_row = df[(df['fold'] == 'mean') & (df['split'] == split)]
                std_row = df[(df['fold'] == 'std') & (df['split'] == split)]
                
                if len(mean_row) > 0 and len(std_row) > 0:
                    for metric in legacy_metrics:
                        if metric in mean_row.columns and metric in std_row.columns:
                            mean_val = mean_row[metric].iloc[0]
                            std_val = std_row[metric].iloc[0]
                            
                            results[f'{metric}_{split}_mean'] = mean_val
                            results[f'{metric}_{split}_std'] = std_val
        else:
            print(f"⚠️ Unrecognized format in {results_file}")
            return None
        
        return results
        
    except Exception as e:
        print(f"❌ Error loading results from {results_file}: {e}")
        return None


def create_comparison_table(all_results: List[Dict]) -> pd.DataFrame:
    """
    Create comprehensive comparison table across all models
    
    Args:
        all_results: List of result dictionaries
        
    Returns:
        DataFrame with comparison results
    """
    comparison_rows = []
    
    metric_names = ['pcorr', 'spearman', 'rmse', 'mse', 'psnr', 'snr', 'lsd', 'dwt']
    
    for result_dict in all_results:
        model_type = result_dict['model_type']
        model_variant = result_dict['model_variant']
        results = result_dict['results']
        
        if results is None:
            continue
            
        # Create rows for train and test
        for split in ['train', 'test']:
            row = {
                'Model_Type': model_type,
                'Model_Variant': model_variant,
                'Split': split.title()
            }
            
            # Add all metrics with mean±std format
            for metric in metric_names:
                mean_key = f'{metric}_{split}_mean'
                std_key = f'{metric}_{split}_std'
                
                if mean_key in results and std_key in results:
                    mean_val = results[mean_key]
                    std_val = results[std_key]
                    
                    if pd.notna(mean_val) and pd.notna(std_val):
                        row[metric.upper()] = f"{mean_val:.4f}±{std_val:.4f}"
                    else:
                        row[metric.upper()] = "N/A"
                else:
                    row[metric.upper()] = "N/A"
            
            comparison_rows.append(row)
    
    return pd.DataFrame(comparison_rows)


def create_ranking_table(all_results: List[Dict], split: str = 'test') -> pd.DataFrame:
    """
    Create ranking table for specified split (focusing on test performance)
    
    Args:
        all_results: List of result dictionaries
        split: 'train' or 'test'
        
    Returns:
        DataFrame with model rankings
    """
    ranking_data = []
    
    for result_dict in all_results:
        model_type = result_dict['model_type']
        model_variant = result_dict['model_variant']
        results = result_dict['results']
        
        if results is None:
            continue
            
        row = {
            'Model_Type': model_type,
            'Model_Variant': model_variant,
            'Model_Full': f"{model_type}-{model_variant}"
        }
        
        # Extract key metrics for ranking
        key_metrics = {
            'pcorr': 'higher',      # Higher is better
            'rmse': 'lower',        # Lower is better  
            'psnr': 'higher',       # Higher is better
            'lsd': 'lower',         # Lower is better
            'dwt': 'lower'          # Lower is better
        }
        
        for metric, direction in key_metrics.items():
            mean_key = f'{metric}_{split}_mean'
            if mean_key in results:
                val = results[mean_key]
                if pd.notna(val) and np.isfinite(val):
                    row[metric.upper()] = val
                else:
                    # Use worst possible value for missing/invalid data
                    row[metric.upper()] = -999 if direction == 'higher' else 999
            else:
                row[metric.upper()] = -999 if direction == 'higher' else 999
        
        ranking_data.append(row)
    
    ranking_df = pd.DataFrame(ranking_data)
    
    # Add rankings for each metric
    for metric in ['PCORR', 'PSNR']:  # Higher is better
        ranking_df[f'{metric}_Rank'] = ranking_df[metric].rank(ascending=False, method='min')
    
    for metric in ['RMSE', 'LSD', 'DWT']:  # Lower is better
        ranking_df[f'{metric}_Rank'] = ranking_df[metric].rank(ascending=True, method='min')
    
    # Calculate average rank
    rank_cols = [col for col in ranking_df.columns if col.endswith('_Rank')]
    ranking_df['Average_Rank'] = ranking_df[rank_cols].mean(axis=1)
    ranking_df['Overall_Rank'] = ranking_df['Average_Rank'].rank(method='min')
    
    # Sort by overall rank
    ranking_df = ranking_df.sort_values('Overall_Rank')
    
    return ranking_df


def main():
    parser = argparse.ArgumentParser(description='Aggregate comprehensive metrics across all models')
    parser.add_argument('--base_path', type=str,
                       default=str(paths.experiments_root()),
                       help='Base path for experiments')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory for comparison tables')
    
    args = parser.parse_args()
    
    # Default experiment folders
    experiment_folders = [
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
    
    print("🚀 Results Aggregator for Comprehensive Model Comparison")
    print("=" * 70)
    print(f"Processing {len(experiment_folders)} experiment folders...")
    
    all_results = []
    
    for folder_name in experiment_folders:
        experiment_folder = os.path.join(args.base_path, folder_name)
        
        if not os.path.exists(experiment_folder):
            print(f"❌ Folder not found: {experiment_folder}")
            continue
            
        print(f"📊 Loading results from: {os.path.basename(experiment_folder)}")
        
        # Extract model information
        model_type, model_variant, parameters = extract_model_info(folder_name)
        
        # Load results
        results = load_model_results(experiment_folder)
        
        all_results.append({
            'folder_name': folder_name,
            'model_type': model_type,
            'model_variant': model_variant,
            'parameters': parameters,
            'results': results
        })
        
        if results is not None:
            print(f"   ✅ {model_type}-{model_variant}")
        else:
            print(f"   ❌ Failed to load results")
    
    if not all_results:
        print("❌ No results loaded successfully!")
        return
        
    # Create comprehensive comparison table
    print(f"\n📋 Creating comprehensive comparison table...")
    comparison_df = create_comparison_table(all_results)
    
    # Save comprehensive comparison
    comparison_file = os.path.join(args.output_dir, 'comprehensive_model_comparison.xlsx')
    comparison_df.to_excel(comparison_file, index=False)
    print(f"💾 Comprehensive comparison saved to: {comparison_file}")
    
    # Create ranking tables for train and test
    print(f"\n🏆 Creating performance rankings...")
    
    for split in ['train', 'test']:
        ranking_df = create_ranking_table(all_results, split=split)
        ranking_file = os.path.join(args.output_dir, f'model_rankings_{split}.xlsx')
        ranking_df.to_excel(ranking_file, index=False)
        print(f"💾 {split.title()} rankings saved to: {ranking_file}")
        
        # Print top 5 models for this split
        print(f"\n🥇 Top 5 Models ({split.title()} Performance):")
        top_5 = ranking_df.head(5)
        for idx, row in top_5.iterrows():
            print(f"   {int(row['Overall_Rank'])}. {row['Model_Full']} (Avg Rank: {row['Average_Rank']:.1f})")
    
    # Create summary statistics table
    print(f"\n📈 Creating summary statistics...")
    summary_data = []
    
    model_types = comparison_df['Model_Type'].unique()
    
    for model_type in model_types:
        subset = comparison_df[comparison_df['Model_Type'] == model_type]
        test_subset = subset[subset['Split'] == 'Test']
        
        if len(test_subset) > 0:
            summary_data.append({
                'Model_Type': model_type,
                'Variants': len(test_subset),
                'Best_PCORR': test_subset['PCORR'].str.extract(r'(\d+\.\d+)')[0].astype(float).max(),
                'Best_RMSE': test_subset['RMSE'].str.extract(r'(\d+\.\d+)')[0].astype(float).min(),
                'Best_PSNR': test_subset['PSNR'].str.extract(r'(\d+\.\d+)')[0].astype(float).max(),
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(args.output_dir, 'model_type_summary.xlsx')
    summary_df.to_excel(summary_file, index=False)
    print(f"💾 Model type summary saved to: {summary_file}")
    
    print(f"\n✅ All comparison tables generated successfully!")
    print(f"📂 Output files in: {args.output_dir}")
    print(f"   - comprehensive_model_comparison.xlsx")
    print(f"   - model_rankings_train.xlsx")  
    print(f"   - model_rankings_test.xlsx")
    print(f"   - model_type_summary.xlsx")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
APD Results Aggregator for Denoised Signal Analysis

This script collects APD prediction results from all baseline models and creates
a comprehensive comparison table for downstream task performance evaluation.

Faithful migration of ``MAP_VAE/aggregate_apd_results.py`` (an argparse
``main()`` entry script). The hardcoded ``--apd_base_path`` default
(``/cluster/.../MAP_VAE/apd_experiments``) was routed through
``paths.experiments_root()``. All aggregation logic is otherwise byte-for-byte
unchanged.
"""

import os
import json
import pandas as pd
import numpy as np
import argparse
from typing import Dict

from cardiac_map_diffusion import paths


def extract_model_info_apd(experiment_name: str) -> tuple:
    """
    Extract model information from APD experiment name
    
    Args:
        experiment_name: APD experiment directory name
        
    Returns:
        Tuple of (model_type, model_variant, apd_label)
    """
    # Extract APD label
    if 'APD30_gs' in experiment_name:
        apd_label = 'APD30'
    elif 'APD60_gs' in experiment_name:
        apd_label = 'APD60'
    elif 'APD90_gs' in experiment_name:
        apd_label = 'APD90'
    else:
        apd_label = 'Unknown'
    
    # Extract model info
    experiment_clean = experiment_name.replace(f'_APD_{apd_label}_gs', '').replace(f'_APD_{apd_label}', '')
    
    if experiment_clean.startswith('TVL1_'):
        if 'lambda001' in experiment_clean:
            return 'TV-L1', 'λ=0.01', apd_label
        elif 'lambda002' in experiment_clean:
            return 'TV-L1', 'λ=0.02', apd_label
        elif 'lambda005' in experiment_clean:
            return 'TV-L1', 'λ=0.05', apd_label
        elif 'lambda01' in experiment_clean:
            return 'TV-L1', 'λ=0.1', apd_label
        else:
            return 'TV-L1', 'Standard', apd_label
    
    elif experiment_clean.startswith('Wavelet_'):
        if 'DB4_SURE' in experiment_clean:
            return 'Wavelet', 'DB4-SURE', apd_label
        elif 'DB4_VisuShrink' in experiment_clean:
            return 'Wavelet', 'DB4-VisuShrink', apd_label
        elif 'Sym4_SURE' in experiment_clean:
            return 'Wavelet', 'Sym4-SURE', apd_label
        elif 'Sym4_VisuShrink' in experiment_clean:
            return 'Wavelet', 'Sym4-VisuShrink', apd_label
        else:
            return 'Wavelet', 'Standard', apd_label
    
    elif experiment_clean.startswith('Filter_'):
        filter_type = experiment_clean.replace('Filter_', '')
        return 'Filter', filter_type, apd_label
    
    elif experiment_clean.startswith('DRRN_'):
        if 'Regularized' in experiment_clean:
            return 'DRRN', 'Regularized', apd_label
        else:
            return 'DRRN', 'Standard', apd_label
    
    elif experiment_clean.startswith('LUNet'):
        return 'LUNet', 'Standard', apd_label
    
    elif experiment_clean.startswith('DAE'):
        return 'DAE', 'Standard', apd_label
    
    elif experiment_clean.startswith('VAE_'):
        return 'VAE', 'CyclicalBeta', apd_label
    
    else:
        return 'Unknown', experiment_clean, apd_label


def load_apd_results(apd_experiment_dir: str) -> Dict:
    """
    Load APD analysis results from experiment directory
    
    Args:
        apd_experiment_dir: Path to APD experiment directory
        
    Returns:
        Dictionary with APD results or None if failed
    """
    results_file = os.path.join(apd_experiment_dir, 'apd_results_denoised.json')
    
    if not os.path.exists(results_file):
        return None
    
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        return results
    except Exception as e:
        print(f"Error loading {results_file}: {e}")
        return None


def create_apd_comparison_table(apd_base_path: str) -> pd.DataFrame:
    """
    Create comprehensive APD comparison table
    
    Args:
        apd_base_path: Base path containing APD experiment directories
        
    Returns:
        DataFrame with APD comparison results
    """
    if not os.path.exists(apd_base_path):
        print(f"APD base path not found: {apd_base_path}")
        return pd.DataFrame()
    
    # Find all APD experiment directories
    apd_experiments = [d for d in os.listdir(apd_base_path) 
                      if os.path.isdir(os.path.join(apd_base_path, d)) and 'APD' in d]
    
    print(f"Found {len(apd_experiments)} APD experiments")
    
    comparison_rows = []
    
    for exp_name in apd_experiments:
        exp_path = os.path.join(apd_base_path, exp_name)
        
        # Extract model information
        model_type, model_variant, apd_label = extract_model_info_apd(exp_name)
        
        # Load results
        results = load_apd_results(exp_path)
        
        if results is None:
            print(f"❌ No results for {exp_name}")
            continue
            
        # Extract best performance
        best_params = results.get('best_params', {})
        
        if not best_params:
            print(f"❌ No best params for {exp_name}")
            continue
            
        row = {
            'Model_Type': model_type,
            'Model_Variant': model_variant,
            'APD_Label': apd_label,
            'Experiment_Name': exp_name,
            # Primary metrics (original scale)
            'Test_RMSE_Mean': best_params.get('mean_test_rmse', np.nan),
            'Test_RMSE_Std': best_params.get('std_test_rmse', np.nan),
            'Test_MAE_Mean': best_params.get('test_mae_original_mean', np.nan),
            'Test_MAE_Std': best_params.get('test_mae_original_std', np.nan),
            'Test_R2_Mean': best_params.get('test_r2_score_original_mean', np.nan),
            'Test_R2_Std': best_params.get('test_r2_score_original_std', np.nan),
            'Test_MAPE_Mean': best_params.get('test_mape_original_mean', np.nan),
            'Test_MAPE_Std': best_params.get('test_mape_original_std', np.nan),
            'Test_MedianAE_Mean': best_params.get('test_median_ae_original_mean', np.nan),
            'Test_MedianAE_Std': best_params.get('test_median_ae_original_std', np.nan),
            'Test_MaxError_Mean': best_params.get('test_max_error_original_mean', np.nan),
            'Test_ExplainedVar_Mean': best_params.get('test_explained_variance_original_mean', np.nan),
            # Training metrics
            'Train_RMSE_Mean': best_params.get('mean_train_rmse', np.nan),
            'Train_RMSE_Std': best_params.get('std_train_rmse', np.nan),
            'Train_R2_Mean': best_params.get('train_r2_score_original_mean', np.nan),
            'Train_R2_Std': best_params.get('train_r2_score_original_std', np.nan),
            # Hyperparameters
            'Best_C': best_params.get('c', 'N/A'),
            'Best_Kernel': best_params.get('k', 'N/A'),
            'Best_Gamma': best_params.get('g', 'N/A'),
            'Best_Tolerance': best_params.get('t', 'N/A'),
            'Best_Degree': best_params.get('d', 'N/A')
        }
        
        # Format metrics with error bars
        metrics_to_format = [
            ('RMSE', 'Test_RMSE_Mean', 'Test_RMSE_Std'),
            ('MAE', 'Test_MAE_Mean', 'Test_MAE_Std'),
            ('R2', 'Test_R2_Mean', 'Test_R2_Std'),
            ('MAPE', 'Test_MAPE_Mean', 'Test_MAPE_Std')
        ]
        
        for metric_name, mean_key, std_key in metrics_to_format:
            mean_val = row[mean_key]
            std_val = row[std_key]
            
            if not np.isnan(mean_val) and not np.isnan(std_val):
                row[f'Test_{metric_name}_Formatted'] = f"{mean_val:.4f}±{std_val:.4f}"
            else:
                row[f'Test_{metric_name}_Formatted'] = "N/A"
        
        # Format training metrics
        if not np.isnan(row['Train_RMSE_Mean']) and not np.isnan(row['Train_RMSE_Std']):
            row['Train_RMSE_Formatted'] = f"{row['Train_RMSE_Mean']:.4f}±{row['Train_RMSE_Std']:.4f}"
        else:
            row['Train_RMSE_Formatted'] = "N/A"
            
        if not np.isnan(row['Train_R2_Mean']) and not np.isnan(row['Train_R2_Std']):
            row['Train_R2_Formatted'] = f"{row['Train_R2_Mean']:.4f}±{row['Train_R2_Std']:.4f}"
        else:
            row['Train_R2_Formatted'] = "N/A"
        
        comparison_rows.append(row)
        print(f"✅ {model_type}-{model_variant} ({apd_label}): RMSE={row['Test_RMSE_Formatted']}, R²={row['Test_R2_Formatted']}")
    
    if not comparison_rows:
        print("No valid APD results found!")
        return pd.DataFrame()
    
    df = pd.DataFrame(comparison_rows)
    
    # Sort by model type, variant, and APD label for better organization
    df = df.sort_values(['Model_Type', 'Model_Variant', 'APD_Label'])
    
    return df


def create_apd_summary_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary table showing best APD performance by model across all APD labels
    
    Args:
        df: APD comparison dataframe
        
    Returns:
        Summary dataframe by model
    """
    if df.empty:
        return pd.DataFrame()
    
    summary_rows = []
    
    # Group by model type and variant
    for (model_type, model_variant), group in df.groupby(['Model_Type', 'Model_Variant']):
        
        # Get performance for each APD label
        apd_performance = {}
        for apd_label in ['APD30', 'APD60', 'APD90']:
            apd_data = group[group['APD_Label'] == apd_label]
            if not apd_data.empty:
                best_row = apd_data.loc[apd_data['Test_RMSE_Mean'].idxmin()]
                apd_performance[f'{apd_label}_RMSE'] = best_row['Test_RMSE_Formatted']
            else:
                apd_performance[f'{apd_label}_RMSE'] = "N/A"
        
        # Calculate average RMSE across APD labels (for ranking)
        valid_rmse = []
        for apd_label in ['APD30', 'APD60', 'APD90']:
            apd_data = group[group['APD_Label'] == apd_label]
            if not apd_data.empty and not np.isnan(apd_data['Test_RMSE_Mean'].min()):
                valid_rmse.append(apd_data['Test_RMSE_Mean'].min())
        
        avg_rmse = np.mean(valid_rmse) if valid_rmse else np.nan
        
        row = {
            'Model_Type': model_type,
            'Model_Variant': model_variant,
            'Model_Full': f"{model_type}-{model_variant}",
            'Average_Test_RMSE': avg_rmse,
            **apd_performance
        }
        
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Add ranking
    if not summary_df.empty:
        summary_df['Rank'] = summary_df['Average_Test_RMSE'].rank(method='min', na_option='bottom')
        summary_df = summary_df.sort_values('Rank')
    
    return summary_df


def main():
    parser = argparse.ArgumentParser(description='Aggregate APD analysis results across all models')
    parser.add_argument('--apd_base_path', type=str,
                       default=str(paths.experiments_root()),
                       help='Base path containing APD experiment directories')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory for APD comparison tables')
    
    args = parser.parse_args()
    
    print("🚀 APD Results Aggregator")
    print("=" * 50)
    print(f"APD experiments path: {args.apd_base_path}")
    print(f"Output directory: {args.output_dir}")
    
    # Create APD comparison table
    print(f"\n📊 Loading APD results...")
    comparison_df = create_apd_comparison_table(args.apd_base_path)
    
    if comparison_df.empty:
        print("❌ No APD results found!")
        return
    
    # Save detailed comparison
    os.makedirs(args.output_dir, exist_ok=True)
    comparison_file = os.path.join(args.output_dir, 'apd_detailed_comparison.xlsx')
    comparison_df.to_excel(comparison_file, index=False)
    print(f"💾 Detailed APD comparison saved to: {comparison_file}")
    
    # Create and save summary by model
    print(f"\n🏆 Creating model summary...")
    summary_df = create_apd_summary_by_model(comparison_df)
    
    if not summary_df.empty:
        summary_file = os.path.join(args.output_dir, 'apd_model_summary.xlsx')
        summary_df.to_excel(summary_file, index=False)
        print(f"💾 APD model summary saved to: {summary_file}")
        
        # Print top 10 models
        print(f"\n🥇 Top 10 Models (Average APD Performance):")
        top_10 = summary_df.head(10)
        for idx, row in top_10.iterrows():
            rank = int(row['Rank'])
            model = row['Model_Full']
            avg_rmse = row['Average_Test_RMSE']
            if not np.isnan(avg_rmse):
                print(f"   {rank:2d}. {model:<25} (Avg RMSE: {avg_rmse:.4f})")
            else:
                print(f"   {rank:2d}. {model:<25} (Avg RMSE: N/A)")
    
    # Create APD label specific rankings
    print(f"\n📈 Creating APD-specific rankings...")
    
    for apd_label in ['APD30', 'APD60', 'APD90']:
        apd_specific = comparison_df[comparison_df['APD_Label'] == apd_label].copy()
        
        if not apd_specific.empty:
            apd_specific['Rank'] = apd_specific['Test_RMSE_Mean'].rank(method='min', na_option='bottom')
            apd_specific = apd_specific.sort_values('Rank')
            
            apd_file = os.path.join(args.output_dir, f'apd_{apd_label.lower()}_ranking.xlsx')
            apd_specific.to_excel(apd_file, index=False)
            print(f"💾 {apd_label} ranking saved to: {apd_file}")
            
            # Print top 5 for this APD label
            print(f"\n🏅 Top 5 Models for {apd_label}:")
            top_5 = apd_specific.head(5)
            for idx, row in top_5.iterrows():
                rank = int(row['Rank']) 
                model = f"{row['Model_Type']}-{row['Model_Variant']}"
                rmse_formatted = row['Test_RMSE_Formatted']
                print(f"   {rank}. {model:<25} ({rmse_formatted})")
    
    # Print overall statistics
    print(f"\n📊 OVERALL APD ANALYSIS STATISTICS")
    print("=" * 40)
    print(f"Total experiments analyzed: {len(comparison_df)}")
    print(f"Unique models: {len(comparison_df.groupby(['Model_Type', 'Model_Variant']))}")
    print(f"APD labels: {sorted(comparison_df['APD_Label'].unique())}")
    
    # Model type breakdown
    print(f"\nModel type breakdown:")
    type_counts = comparison_df['Model_Type'].value_counts()
    for model_type, count in type_counts.items():
        print(f"   {model_type}: {count} experiments")
    
    print(f"\n✅ APD results aggregation completed!")
    print(f"📂 Output files in: {args.output_dir}")
    print(f"   - apd_detailed_comparison.xlsx")
    print(f"   - apd_model_summary.xlsx") 
    print(f"   - apd_apd30_ranking.xlsx")
    print(f"   - apd_apd60_ranking.xlsx")
    print(f"   - apd_apd90_ranking.xlsx")


if __name__ == "__main__":
    main()
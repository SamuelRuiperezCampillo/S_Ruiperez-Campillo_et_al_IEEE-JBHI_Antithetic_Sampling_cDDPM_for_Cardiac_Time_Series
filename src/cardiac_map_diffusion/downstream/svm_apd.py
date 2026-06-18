#!/usr/bin/env python3
"""
Baseline APD Analysis Script - Noisy vs Clean Signals
====================================================

Computes APD prediction using noisy signals as input and clean signals as reference.
This provides baseline comparison for evaluating denoised signal performance.

Faithful migration of ``MAP_VAE/baseline_svm_apd_noisy_clean.py``. Only mechanical
edits were applied: the ``MAP_functions_metrics``/``data`` imports were rewritten to
the ``cardiac_map_diffusion`` package layout, the hard-coded cluster CSV literal
(``/cluster/work/vogtlab/Group/pblasco/data/MAP_vent_complete_pandas.csv``) is now
resolved relative to :func:`cardiac_map_diffusion.paths.data_root`, and this module
docstring was expanded. The SVM model, normalisation, metrics and saving logic are
byte-for-byte unchanged.

Usage:
    python baseline_svm_apd_noisy_clean.py [--noise_type allmixed] [--seed_split 29] [--num_folds 4]
"""

import os
import numpy as np
import random
import pandas as pd
import argparse
from pathlib import Path

from cardiac_map_diffusion import paths

import cardiac_map_diffusion.metrics.map_functions_metrics as mapf
from cardiac_map_diffusion.data.data_baselines import get_MAP_vent_data, get_train_test_kfolds
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, median_absolute_error,
    mean_absolute_percentage_error, r2_score, explained_variance_score,
    max_error
)


def generate_noisy_signals(clean_signals, noise_type='allmixed', random_seed=17, arrays=None):
    """
    Generate noisy signals from clean signals using MAP noise generation

    Args:
        clean_signals: Clean MAP signals
        noise_type: Type of noise to add ('allmixed', 'gaussian', etc.)
        random_seed: Random seed for reproducibility
        arrays: Noise arrays needed for electrophysiological noise (from mapf.get_np_noisearrays)

    Returns:
        noisy_signals: Signals with added noise
    """
    print(f"  Generating {noise_type} noise...")

    # Set random seed
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Use MAP_functions to generate noise (same as training data)
    if noise_type == 'allmixed':
        # Get noise parameters for allmixed noise (same as in training)
        noise_params = mapf.find_noise_params('allmixed')
        # noise_params = [noise_ids, min_number_noises, max_number_noises]
        noisy_signals = mapf.introduce_several_noises(clean_signals,
                                                     noise_ids=noise_params[0],
                                                     min_number_noises=noise_params[1],
                                                     max_number_noises=noise_params[2],
                                                     arrays=arrays if arrays is not None else [])
    elif noise_type == 'gaussian':
        # Add Gaussian noise
        noisy_signals = mapf.introduce_gaussian_noise(loc=0, scale=0.05, MAP_array=clean_signals)
    else:
        print(f"Warning: Unknown noise type {noise_type}, using allmixed")
        noise_params = mapf.find_noise_params('allmixed')
        noisy_signals = mapf.introduce_several_noises(clean_signals,
                                                     noise_ids=noise_params[0],
                                                     min_number_noises=noise_params[1],
                                                     max_number_noises=noise_params[2],
                                                     arrays=arrays if arrays is not None else [])

    return noisy_signals


def run_baseline_svm_apd_analysis(noise_type='allmixed', seed_split=29, num_folds=4,
                                 random_seed=17, cluster=True, output_dir=None):
    """
    Run baseline SVM APD analysis using noisy signals as input

    METHODOLOGY:
    - Train SVM on CLEAN signals (ground truth model)
    - Evaluate performance on NOISY signals (both train and test sets)
    - This approach measures how well noisy signals perform compared to a clean signal baseline

    Args:
        noise_type: Type of noise to add to clean signals
        seed_split: Random seed for splits (default: 29)
        num_folds: Number of folds (default: 4)
        random_seed: Random seed for noise generation (default: 17)
        cluster: Whether running on cluster (default: True)
        output_dir: Directory to save results (default: current directory)
    """
    if output_dir is None:
        output_dir = Path('.')
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(exist_ok=True)

    print(f"[INFO] Running Baseline SVM APD Analysis")
    print(f"   Noise type: {noise_type}")
    print(f"   Seed split: {seed_split}")
    print(f"   Random seed: {random_seed}")
    print(f"   Output directory: {output_dir}")

    # Load MAP data
    print("[INFO] Loading MAP data...")
    try:
        if cluster:
            csv_path = os.path.join(str(paths.data_root()), 'MAP_vent_complete_pandas.csv')
            if os.path.exists(csv_path):
                MAP_vent_complete = pd.read_csv(csv_path)
                MAP_vent_complete['MAP_segments'] = MAP_vent_complete['MAP_segments'].apply(
                    lambda x: np.fromstring(x.strip('[]'), sep=' ') if isinstance(x, str) else x
                )
            else:
                MAP_vent_complete = get_MAP_vent_data(CLUSTER=True)
        else:
            MAP_vent_complete = get_MAP_vent_data(CLUSTER=False)

        # Get noise arrays for electrophysiological noise (needed for allmixed noise)
        print("[INFO] Generating noise arrays for electrophysiological noise...")
        arrays = mapf.get_np_noisearrays(MAP_vent_complete)

    except Exception as e:
        print(f"[ERROR] Failed to load MAP data: {e}")
        return

    # APD labels to analyze
    apd_labels = ['APD30_gs', 'APD60_gs', 'APD90_gs']

    # SVM parameters (default from MAP_functions_metrics)
    svm_params = {'kernel': 'rbf', 'C': 1, 'degree': 3}

    results = {}

    for apd_label in apd_labels:
        print(f"\n{'='*50}")
        print(f"[ANALYSIS] Analyzing {apd_label}")
        print(f"{'='*50}")

        fold_results = []
        all_metrics = []

        for fold_idx in range(num_folds):
            print(f"\n[FOLD] Processing fold {fold_idx}...")

            # Get clean signals and APD labels using same splits as denoising experiments
            X_train_clean, X_test_clean, y_train, y_test = get_train_test_kfolds(
                MAP_vent_complete,
                num_folds=num_folds,
                split_number=fold_idx,
                r_seed=seed_split,
                apd_label=apd_label
            )

            # Generate noisy signals from clean signals
            X_train_noisy = generate_noisy_signals(X_train_clean, noise_type=noise_type,
                                                 random_seed=random_seed + fold_idx,
                                                 arrays=arrays)
            X_test_noisy = generate_noisy_signals(X_test_clean, noise_type=noise_type,
                                                random_seed=random_seed + fold_idx + 1000,
                                                arrays=arrays)

            print(f"  Shapes: X_train_clean={X_train_clean.shape}, X_test_clean={X_test_clean.shape}")
            print(f"  Shapes: X_train_noisy={X_train_noisy.shape}, X_test_noisy={X_test_noisy.shape}")
            print(f"  APD range: [{y_test.min():.1f}, {y_test.max():.1f}]")

            # Normalize CLEAN signals for training (ground truth model)
            X_train_clean_std = mapf.normalize_EGM_array(X_train_clean)
            X_test_clean_std = mapf.normalize_EGM_array(X_test_clean)

            # Normalize NOISY signals for evaluation
            X_train_noisy_std = mapf.normalize_EGM_array(X_train_noisy)
            X_test_noisy_std = mapf.normalize_EGM_array(X_test_noisy)

            # Normalize APD labels
            y_train_std = mapf.normalize_y(y_train, perc_min=0.5, perc_max=99.5)
            y_test_std = mapf.normalize_y(y_test, perc_min=0.5, perc_max=99.5)

            # Train SVM on CLEAN signals (ground truth model)
            from sklearn import svm
            regr = svm.SVR(**svm_params)
            regr.fit(X_train_clean_std, y_train_std)
            print(f"  [SUCCESS] SVM trained on CLEAN signals")

            # Evaluate on NOISY signals
            y_train_pred_std = regr.predict(X_train_noisy_std)  # Train noisy evaluation
            y_test_pred_std = regr.predict(X_test_noisy_std)    # Test noisy evaluation

            # Unnormalize predictions
            y_train_pred = mapf.un_normalize_ystd(y_train, y_train_pred_std, perc_min=0.5, perc_max=99.5)
            y_test_pred = mapf.un_normalize_ystd(y_test, y_test_pred_std, perc_min=0.5, perc_max=99.5)

            # Compute comprehensive metrics for TEST set
            test_metrics = {}
            try:
                test_metrics['mean_absolute_percentage_error'] = mean_absolute_percentage_error(y_test, y_test_pred)
            except:
                test_metrics['mean_absolute_percentage_error'] = np.inf

            test_metrics['median_absolute_error'] = median_absolute_error(y_test, y_test_pred)
            test_metrics['mean_absolute_error'] = mean_absolute_error(y_test, y_test_pred)
            test_metrics['root_mean_squared_error'] = mean_squared_error(y_test, y_test_pred, squared=False)
            test_metrics['r2_score'] = r2_score(y_test, y_test_pred)
            test_metrics['explained_variance_score'] = explained_variance_score(y_test, y_test_pred)
            test_metrics['max_error'] = max_error(y_test, y_test_pred)
            test_metrics['mean_squared_error'] = mean_squared_error(y_test, y_test_pred)

            # Compute comprehensive metrics for TRAIN set
            train_metrics = {}
            try:
                train_metrics['mean_absolute_percentage_error'] = mean_absolute_percentage_error(y_train, y_train_pred)
            except:
                train_metrics['mean_absolute_percentage_error'] = np.inf

            train_metrics['median_absolute_error'] = median_absolute_error(y_train, y_train_pred)
            train_metrics['mean_absolute_error'] = mean_absolute_error(y_train, y_train_pred)
            train_metrics['root_mean_squared_error'] = mean_squared_error(y_train, y_train_pred, squared=False)
            train_metrics['r2_score'] = r2_score(y_train, y_train_pred)
            train_metrics['explained_variance_score'] = explained_variance_score(y_train, y_train_pred)
            train_metrics['max_error'] = max_error(y_train, y_train_pred)
            train_metrics['mean_squared_error'] = mean_squared_error(y_train, y_train_pred)

            fold_results.append({
                'fold': fold_idx,
                'test_metrics': test_metrics,
                'train_metrics': train_metrics,
                'test_predictions': y_test_pred,
                'train_predictions': y_train_pred,
                'true_test_values': y_test,
                'true_train_values': y_train,
                'noise_characteristics': {
                    'train_snr_db': 10 * np.log10(np.var(X_train_clean) / np.var(X_train_noisy - X_train_clean)),
                    'test_snr_db': 10 * np.log10(np.var(X_test_clean) / np.var(X_test_noisy - X_test_clean))
                }
            })
            all_metrics.append(test_metrics)  # Store test metrics for summary

            # Print key metrics for both train and test
            print(f"  [TRAIN] Results (Noisy): R2={train_metrics['r2_score']:.4f}, RMSE={train_metrics['root_mean_squared_error']:.2f}")
            print(f"  [TEST] Results (Noisy):  R2={test_metrics['r2_score']:.4f}, RMSE={test_metrics['root_mean_squared_error']:.2f}")
            print(f"  [SNR] Train={fold_results[-1]['noise_characteristics']['train_snr_db']:.1f}dB, Test={fold_results[-1]['noise_characteristics']['test_snr_db']:.1f}dB")
            print(f"  [NOTE] SVM trained on CLEAN signals, evaluated on NOISY signals")

        # Calculate summary statistics
        if all_metrics:
            print(f"\n[SUMMARY] Summary for {apd_label}:")

            # Prepare summary data with separate train and test metrics
            summary_rows = []

            # Add fold rows with both train and test metrics
            for i, result in enumerate(fold_results):
                test_metrics = result['test_metrics']
                train_metrics = result['train_metrics']

                # Test metrics row
                test_row = {'Split': f'split{i}_test', 'Set': 'test'}
                test_row.update(test_metrics)
                test_row['train_snr_db'] = result['noise_characteristics']['train_snr_db']
                test_row['test_snr_db'] = result['noise_characteristics']['test_snr_db']
                summary_rows.append(test_row)

                # Train metrics row
                train_row = {'Split': f'split{i}_train', 'Set': 'train'}
                train_row.update(train_metrics)
                train_row['train_snr_db'] = result['noise_characteristics']['train_snr_db']
                train_row['test_snr_db'] = result['noise_characteristics']['test_snr_db']
                summary_rows.append(train_row)

            # Calculate averages and std devs for test and train separately
            test_metrics_list = [r['test_metrics'] for r in fold_results]
            train_metrics_list = [r['train_metrics'] for r in fold_results]

            metric_names = list(test_metrics_list[0].keys())

            # Test set averages
            test_avg_metrics = {}
            test_std_metrics = {}
            for metric_name in metric_names:
                values = [m[metric_name] for m in test_metrics_list if np.isfinite(m[metric_name])]
                test_avg_metrics[metric_name] = np.mean(values) if values else np.nan
                test_std_metrics[metric_name] = np.std(values) if values else np.nan

            # Train set averages
            train_avg_metrics = {}
            train_std_metrics = {}
            for metric_name in metric_names:
                values = [m[metric_name] for m in train_metrics_list if np.isfinite(m[metric_name])]
                train_avg_metrics[metric_name] = np.mean(values) if values else np.nan
                train_std_metrics[metric_name] = np.std(values) if values else np.nan

            # Add SNR averages
            snr_train_values = [r['noise_characteristics']['train_snr_db'] for r in fold_results]
            snr_test_values = [r['noise_characteristics']['test_snr_db'] for r in fold_results]
            avg_snr_train = np.mean(snr_train_values)
            avg_snr_test = np.mean(snr_test_values)
            std_snr_train = np.std(snr_train_values)
            std_snr_test = np.std(snr_test_values)

            # Add summary rows for test
            test_avg_row = {'Split': 'average_test', 'Set': 'test'}
            test_avg_row.update(test_avg_metrics)
            test_avg_row['train_snr_db'] = avg_snr_train
            test_avg_row['test_snr_db'] = avg_snr_test
            summary_rows.append(test_avg_row)

            test_std_row = {'Split': 'st_dev_test', 'Set': 'test'}
            test_std_row.update(test_std_metrics)
            test_std_row['train_snr_db'] = std_snr_train
            test_std_row['test_snr_db'] = std_snr_test
            summary_rows.append(test_std_row)

            # Add summary rows for train
            train_avg_row = {'Split': 'average_train', 'Set': 'train'}
            train_avg_row.update(train_avg_metrics)
            train_avg_row['train_snr_db'] = avg_snr_train
            train_avg_row['test_snr_db'] = avg_snr_test
            summary_rows.append(train_avg_row)

            train_std_row = {'Split': 'st_dev_train', 'Set': 'train'}
            train_std_row.update(train_std_metrics)
            train_std_row['train_snr_db'] = std_snr_train
            train_std_row['test_snr_db'] = std_snr_test
            summary_rows.append(train_std_row)

            # Create DataFrame and save
            summary_df = pd.DataFrame(summary_rows)

            # Round to 5 decimal places
            numeric_cols = summary_df.select_dtypes(include=[np.number]).columns
            summary_df[numeric_cols] = summary_df[numeric_cols].round(5)

            # Save to Excel
            output_file = output_dir / f'baseline_noisy_{noise_type}_{apd_label.lower()}.xlsx'
            summary_df.to_excel(output_file, index=False)

            print(f"[SAVED] Saved to: {output_file}")
            print(f"   [TEST]  - Average R2: {test_avg_metrics['r2_score']:.4f} +/- {test_std_metrics['r2_score']:.4f}")
            print(f"   [TEST]  - Average RMSE: {test_avg_metrics['root_mean_squared_error']:.2f} +/- {test_std_metrics['root_mean_squared_error']:.2f}")
            print(f"   [TRAIN] - Average R2: {train_avg_metrics['r2_score']:.4f} +/- {train_std_metrics['r2_score']:.4f}")
            print(f"   [TRAIN] - Average RMSE: {train_avg_metrics['root_mean_squared_error']:.2f} +/- {train_std_metrics['root_mean_squared_error']:.2f}")
            print(f"   [SNR] Average SNR: {avg_snr_train:.1f}dB +/- {std_snr_train:.1f}dB")
            print(f"   [NOTE] SVM trained on CLEAN signals, metrics computed on NOISY signals")

            results[apd_label] = {
                'fold_results': fold_results,
                'summary_df': summary_df,
                'test_avg_metrics': test_avg_metrics,
                'test_std_metrics': test_std_metrics,
                'train_avg_metrics': train_avg_metrics,
                'train_std_metrics': train_std_metrics
            }

    print(f"\n[SUCCESS] Baseline analysis completed!")
    print(f"[INFO] Results saved in: {output_dir}")
    print(f"[INFO] Compare these results with denoised signal results to evaluate denoising performance")

    return results


def main():
    parser = argparse.ArgumentParser(description='Baseline SVM APD Analysis - Noisy vs Clean Signals')
    parser.add_argument('--noise_type', type=str, default='allmixed',
                       help='Type of noise to add (default: allmixed)')
    parser.add_argument('--seed_split', type=int, default=29,
                       help='Random seed for data splits (default: 29)')
    parser.add_argument('--num_folds', type=int, default=4,
                       help='Number of cross-validation folds (default: 4)')
    parser.add_argument('--random_seed', type=int, default=17,
                       help='Random seed for noise generation (default: 17)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results (default: current directory)')

    args = parser.parse_args()

    # Detect if running on cluster
    cluster = os.path.exists('/cluster/')

    try:
        results = run_baseline_svm_apd_analysis(
            noise_type=args.noise_type,
            seed_split=args.seed_split,
            num_folds=args.num_folds,
            random_seed=args.random_seed,
            cluster=cluster,
            output_dir=args.output_dir
        )
        print("\n[SUCCESS] Baseline SVM APD analysis completed successfully!")
    except Exception as e:
        print(f"\n[ERROR] Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()

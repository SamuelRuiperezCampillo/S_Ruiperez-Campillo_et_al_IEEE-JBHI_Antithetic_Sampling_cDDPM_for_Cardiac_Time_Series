#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Traditional Filter Baselines with Signal Saving
===============================================

This script applies traditional signal processing filters (Butterworth, etc.)
as baselines for comparison with deep learning models. It computes metrics and
saves denoised signals in the same format as VAE/DAE/LUNet models for fair comparison.

Faithful migration of ``MAP_VAE/baseline_filters.py``. Only mechanical edits were
applied: the ``MAP_functions_metrics``/``data``/``utils`` imports were rewritten to
the ``cardiac_map_diffusion`` package layout, the hard-coded cluster CSV literal
(``/cluster/work/vogtlab/Group/pblasco/data/MAP_vent_complete_pandas.csv``) is now
resolved relative to :func:`cardiac_map_diffusion.paths.data_root`, and this module
docstring was expanded. All filtering, metric and saving logic is byte-for-byte
unchanged.

This module provides the classical filter baselines selected via ``--filter_type``
(butterworth | median | savgol | gaussian | wavelet | wiener | hybrid_savgol_median
| tv_l1 | adaptive_notch | none). It exposes reusable functions and an argparse
``main()`` entry point (see ``scripts/run_filters.py``).

Usage:
    python baseline_filters.py --filter_type butterworth --noise_type allmixed
    python baseline_filters.py --filter_type butterworth --working_dir /path/to/output
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import argparse
from pathlib import Path

from cardiac_map_diffusion import paths

import cardiac_map_diffusion.metrics.map_functions_metrics as mapf
from cardiac_map_diffusion.data.data_baselines import get_MAP_vent_data, get_train_test_kfolds
from cardiac_map_diffusion.data.retrieve_dataset import retrieveDataSet


def butterworth_notch_filter(noisy_signal: np.ndarray, fs: int = 1000, lowcut: float = 0.01,
                           highcut: float = 400, order: int = 5, f0: float = 60, Q: float = 30.0) -> np.ndarray:
    """
    Apply Butterworth + Notch filter (exact implementation from MAP_functions_metrics)

    Args:
        noisy_signal: Input noisy signal
        fs: Sampling frequency
        lowcut: Low cutoff frequency
        highcut: High cutoff frequency
        order: Filter order
        f0: Notch frequency
        Q: Quality factor for notch filter

    Returns:
        Filtered signal
    """
    from scipy.signal import butter, lfilter, iirnotch

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    w0 = f0 / nyq

    # Butterworth bandpass filter
    b_butter, a_butter = butter(order, [low, high], btype='band', analog=False)
    butter_filtered = lfilter(b_butter, a_butter, noisy_signal)

    # Notch filter
    b_notch, a_notch = iirnotch(f0, Q, fs=fs)
    filtered = lfilter(b_notch, a_notch, butter_filtered)

    return filtered


def median_filter(noisy_signal: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply median filter for baseline denoising

    Args:
        noisy_signal: Input noisy signal
        kernel_size: Size of median filter kernel

    Returns:
        Filtered signal
    """
    from scipy.signal import medfilt
    return medfilt(noisy_signal, kernel_size=kernel_size)


def savitzky_golay_filter(noisy_signal: np.ndarray, window_length: int = 51, polyorder: int = 3) -> np.ndarray:
    """
    Apply Savitzky-Golay filter for baseline denoising

    Args:
        noisy_signal: Input noisy signal
        window_length: Length of filter window (must be odd)
        polyorder: Order of polynomial used to fit samples

    Returns:
        Filtered signal
    """
    from scipy.signal import savgol_filter
    # Ensure window_length is odd and smaller than signal length
    if window_length % 2 == 0:
        window_length += 1
    if window_length > len(noisy_signal):
        window_length = min(len(noisy_signal) - 1, 51)
        if window_length % 2 == 0:
            window_length -= 1
    if window_length < polyorder + 1:
        window_length = polyorder + 2
        if window_length % 2 == 0:
            window_length += 1

    return savgol_filter(noisy_signal, window_length, polyorder)


def gaussian_filter(noisy_signal: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Apply Gaussian filter for baseline denoising

    Args:
        noisy_signal: Input noisy signal
        sigma: Standard deviation for Gaussian kernel

    Returns:
        Filtered signal
    """
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(noisy_signal, sigma=sigma)


def wavelet_shrinkage_filter(noisy_signal: np.ndarray, wavelet: str = 'db4', levels: int = 5,
                           threshold_mode: str = 'soft', threshold_method: str = 'sure') -> np.ndarray:
    """
    Apply Wavelet shrinkage denoising (DWT with soft-thresholding)
    Standard ECG denoising approach using PyWavelets

    Args:
        noisy_signal: Input noisy signal
        wavelet: Wavelet type ('db4', 'sym4', etc.)
        levels: Number of decomposition levels (default 5)
        threshold_mode: 'soft' or 'hard' thresholding
        threshold_method: 'sure' (SURE threshold) or 'visushrink' (universal threshold)

    Returns:
        Denoised signal
    """
    import pywt

    # Perform wavelet decomposition
    coeffs = pywt.wavedec(noisy_signal, wavelet, level=levels)

    # Estimate noise standard deviation from finest detail coefficients
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # Robust noise estimation

    # Apply thresholding to detail coefficients
    coeffs_thresh = coeffs.copy()

    for i in range(1, len(coeffs)):  # Skip approximation coeffs (index 0)
        if threshold_method == 'sure':
            # SURE threshold (adaptive) - use coefficient-specific threshold
            coeff_sigma = np.median(np.abs(coeffs[i])) / 0.6745  # Robust estimation per level
            threshold_val = coeff_sigma * np.sqrt(2 * np.log(len(coeffs[i])))
            coeffs_thresh[i] = pywt.threshold(coeffs[i], threshold_val, mode=threshold_mode)
        else:  # visushrink (universal threshold)
            threshold_val = sigma * np.sqrt(2 * np.log(len(noisy_signal)))
            coeffs_thresh[i] = pywt.threshold(coeffs[i], threshold_val, mode=threshold_mode)

    # Reconstruct signal
    denoised = pywt.waverec(coeffs_thresh, wavelet)

    # Handle length mismatch due to padding
    if len(denoised) != len(noisy_signal):
        denoised = denoised[:len(noisy_signal)]

    return denoised


def wiener_filter_adaptive(noisy_signal: np.ndarray, mysize: int = 15, noise: float = None) -> np.ndarray:
    """
    Apply adaptive Wiener filter for colored noise handling
    Textbook adaptive baseline approach

    Args:
        noisy_signal: Input noisy signal
        mysize: Kernel size for local statistics (7-31, default 15)
        noise: Noise variance (if None, estimated automatically)

    Returns:
        Filtered signal
    """
    from scipy.signal import wiener

    # Apply Wiener filter
    filtered = wiener(noisy_signal, mysize=mysize, noise=noise)

    return filtered


def savgol_median_hybrid_filter(noisy_signal: np.ndarray, median_kernel: int = 5,
                               savgol_window: int = 21, savgol_order: int = 3) -> np.ndarray:
    """
    Apply Savitzky-Golay + Median hybrid filter
    Common in physiology - removes high-freq noise while preserving morphology

    Args:
        noisy_signal: Input noisy signal
        median_kernel: Median filter kernel size (3-7, default 5)
        savgol_window: Savitzky-Golay window length (11-31, default 21)
        savgol_order: Savitzky-Golay polynomial order (2-3, default 3)

    Returns:
        Filtered signal
    """
    from scipy.signal import medfilt, savgol_filter

    # Step 1: Median filter to remove spikes
    median_filtered = medfilt(noisy_signal, kernel_size=median_kernel)

    # Step 2: Savitzky-Golay filter for smoothing while preserving morphology
    # Ensure window_length is odd and valid
    if savgol_window % 2 == 0:
        savgol_window += 1
    if savgol_window > len(median_filtered):
        savgol_window = min(len(median_filtered) - 1, 21)
        if savgol_window % 2 == 0:
            savgol_window -= 1
    if savgol_window < savgol_order + 1:
        savgol_window = savgol_order + 2
        if savgol_window % 2 == 0:
            savgol_window += 1

    hybrid_filtered = savgol_filter(median_filtered, savgol_window, savgol_order)

    return hybrid_filtered


def tv_l1_denoising_filter(noisy_signal: np.ndarray, weight: float = 0.01) -> np.ndarray:
    """
    Apply Total Variation (TV-L1) denoising for piecewise-smooth signals
    Strong on signals with sharp transitions, robust to spikes

    Args:
        noisy_signal: Input noisy signal
        weight: Regularization weight (λ ∈ {0.01, 0.02, 0.05, 0.1})

    Returns:
        Denoised signal
    """
    try:
        from skimage.restoration import denoise_tv_chambolle
        # Use 1D TV denoising
        denoised = denoise_tv_chambolle(noisy_signal, weight=weight, channel_axis=None)
        return denoised
    except ImportError:
        # Fallback: Simple TV-L1 implementation using proximal gradient
        return _tv_l1_proximal(noisy_signal, weight)


def _tv_l1_proximal(signal: np.ndarray, lam: float, max_iter: int = 100) -> np.ndarray:
    """
    Simple 1D TV-L1 denoising using proximal gradient method
    Fallback implementation when scikit-image is not available
    """
    n = len(signal)
    x = signal.copy()

    # TV-L1 proximal operator
    for _ in range(max_iter):
        # Gradient step
        grad = np.zeros(n)
        grad[:-1] += x[:-1] - x[1:]  # Forward difference
        grad[1:] += x[1:] - x[:-1]   # Backward difference

        # Proximal step
        x_new = x - 0.01 * grad

        # Soft thresholding (L1 proximal operator)
        x_new = np.sign(x_new) * np.maximum(np.abs(x_new) - lam * 0.01, 0)

        # Check convergence
        if np.linalg.norm(x_new - x) < 1e-6:
            break
        x = x_new

    return x


def adaptive_notch_lms_filter(noisy_signal: np.ndarray, fs: int = 1000, f0: float = 60,
                            mu: float = 0.01, include_harmonics: bool = True) -> np.ndarray:
    """
    Apply Adaptive Notch/LMS line enhancer for powerline interference
    Smart notch that adapts and preserves morphology better than fixed notch

    Args:
        noisy_signal: Input noisy signal
        fs: Sampling frequency (default 1000 Hz)
        f0: Fundamental frequency to remove (50/60 Hz)
        mu: LMS adaptation step size (0.001-0.1)
        include_harmonics: Whether to include 2nd and 3rd harmonics

    Returns:
        Filtered signal
    """
    n = len(noisy_signal)
    filtered = noisy_signal.copy()

    # Frequencies to remove
    freqs_to_remove = [f0]
    if include_harmonics:
        freqs_to_remove.extend([2*f0, 3*f0])

    for freq in freqs_to_remove:
        if freq > fs/2:  # Skip frequencies above Nyquist
            continue

        # Generate reference sinusoids for LMS
        t = np.arange(n) / fs
        ref_cos = np.cos(2 * np.pi * freq * t)
        ref_sin = np.sin(2 * np.pi * freq * t)

        # LMS adaptive filter
        w_cos = 0.0  # Cosine weight
        w_sin = 0.0  # Sine weight

        for i in range(n):
            # Predict interference
            interference = w_cos * ref_cos[i] + w_sin * ref_sin[i]

            # Error signal
            error = filtered[i] - interference

            # Update weights
            w_cos += mu * error * ref_cos[i]
            w_sin += mu * error * ref_sin[i]

            # Update filtered signal
            filtered[i] = error

    return filtered


def no_filter(noisy_signal: np.ndarray) -> np.ndarray:
    """
    No filtering - returns the noisy signal unchanged.
    This provides a baseline for evaluating filter effectiveness.

    Args:
        noisy_signal: Input noisy signal

    Returns:
        Unchanged noisy signal
    """
    return noisy_signal.copy()


FILTER_FUNCTIONS = {
    'butterworth': butterworth_notch_filter,
    'median': median_filter,
    'savgol': savitzky_golay_filter,
    'gaussian': gaussian_filter,
    'wavelet': wavelet_shrinkage_filter,
    'wiener': wiener_filter_adaptive,
    'hybrid_savgol_median': savgol_median_hybrid_filter,
    'tv_l1': tv_l1_denoising_filter,
    'adaptive_notch': adaptive_notch_lms_filter,
    'none': no_filter
}

FILTER_PARAMS = {
    'butterworth': {'fs': 1000, 'lowcut': 0.01, 'highcut': 400, 'order': 5, 'f0': 60, 'Q': 30.0},
    'median': {'kernel_size': 5},
    'savgol': {'window_length': 51, 'polyorder': 3},
    'gaussian': {'sigma': 1.0},
    'wavelet': {'wavelet': 'db4', 'levels': 5, 'threshold_mode': 'soft', 'threshold_method': 'sure'},
    'wiener': {'mysize': 15, 'noise': None},
    'hybrid_savgol_median': {'median_kernel': 5, 'savgol_window': 21, 'savgol_order': 3},
    'tv_l1': {'weight': 0.01},
    'adaptive_notch': {'fs': 1000, 'f0': 60, 'mu': 0.01, 'include_harmonics': True},
    'none': {}  # No parameters needed for no filtering
}


def _fallback_dtw(array1: np.ndarray, array2: np.ndarray) -> float:
    """
    Fallback DTW implementation when mapf.compute_dtw is not available
    """
    try:
        from dtaidistance import dtw
        dtw_distances = []
        for i in range(array1.shape[0]):
            distance = dtw.distance(array1[i, :], array2[i, :])
            dtw_distances.append(distance)
        return np.mean(dtw_distances)
    except ImportError:
        # Simple DTW fallback
        dtw_distances = []
        for i in range(array1.shape[0]):
            distance = _simple_dtw_fallback(array1[i, :], array2[i, :])
            dtw_distances.append(distance)
        return np.mean(dtw_distances)


def _simple_dtw_fallback(x: np.ndarray, y: np.ndarray) -> float:
    """
    Simple DTW implementation fallback
    """
    n, m = len(x), len(y)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(x[i-1] - y[j-1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],      # insertion
                                         dtw_matrix[i, j-1],      # deletion
                                         dtw_matrix[i-1, j-1])    # match
    return dtw_matrix[n, m]


def _fallback_lsd(array1: np.ndarray, array2: np.ndarray) -> float:
    """
    Fallback LSD implementation when mapf.compute_lsd is not available
    """
    from scipy import signal as scipy_signal

    lsd_distances = []
    for i in range(array1.shape[0]):
        signal1 = array1[i, :]
        signal2 = array2[i, :]

        # Compute power spectral densities
        nperseg = min(256, len(signal1)//4)
        f1, psd1 = scipy_signal.welch(signal1, window='hann', nperseg=nperseg)
        f2, psd2 = scipy_signal.welch(signal2, window='hann', nperseg=nperseg)

        # Avoid log of zero by adding small epsilon
        eps = 1e-10
        psd1 = np.maximum(psd1, eps)
        psd2 = np.maximum(psd2, eps)

        # Compute log-spectral distance
        lsd = np.sqrt(np.mean((10 * np.log10(psd1) - 10 * np.log10(psd2)) ** 2))
        lsd_distances.append(lsd)

    return np.mean(lsd_distances)


def compute_all_metrics_robust(array1: np.ndarray, array2: np.ndarray, mode: str = 'total',
                              skip_slow_metrics: bool = False) -> Dict[str, float]:
    """
    Compute EXACTLY the same metrics as deep learning models (DAE/VAE/LUNet/DRRN)
    INCLUDING loss and nmae metrics to match output format

    Args:
        array1: Original/clean signals
        array2: Processed signals (filtered/noisy)
        mode: 'total' for average across batch
        skip_slow_metrics: If True, skip DTW and LSD computations (default: False)

    Returns:
        Dictionary with ALL metric values matching deep learning models
    """
    import time

    print(f"    🔍 Computing metrics for {array1.shape[0]} signals...")
    start_time = time.time()

    metrics = {}

    # Core 6 metrics (same as VAE/DAE/LUNet/DRRN models)
    metric_start = time.time()
    try:
        metrics['pcorr'] = mapf.compute_pearson_corr(array1, array2, mode=mode)
    except Exception as e:
        print(f"    WARNING: PCC failed: {e}")
        metrics['pcorr'] = 0.0
    print(f"    ✓ Pearson correlation: {time.time() - metric_start:.3f}s")

    metric_start = time.time()
    try:
        metrics['rmse'] = mapf.compute_rmse(array1, array2, mode=mode)
    except Exception as e:
        print(f"    WARNING: RMSE failed: {e}")
        metrics['rmse'] = 0.0
    print(f"    ✓ RMSE: {time.time() - metric_start:.3f}s")

    metric_start = time.time()
    try:
        metrics['psnr'] = mapf.compute_psnr(array1, array2, mode=mode)
    except Exception as e:
        print(f"    WARNING: PSNR failed: {e}")
        metrics['psnr'] = 0.0
    print(f"    ✓ PSNR: {time.time() - metric_start:.3f}s")

    metric_start = time.time()
    try:
        metrics['mse'] = mapf.compute_mse(array1, array2, mode=mode)
    except Exception as e:
        print(f"    WARNING: MSE failed: {e}")
        metrics['mse'] = 0.0
    print(f"    ✓ MSE: {time.time() - metric_start:.3f}s")

    metric_start = time.time()
    try:
        metrics['spearman'] = mapf.compute_spearman_corr(array1, array2, mode=mode)
    except Exception as e:
        print(f"    WARNING: Spearman failed: {e}")
        metrics['spearman'] = 0.0
    print(f"    ✓ Spearman: {time.time() - metric_start:.3f}s")

    metric_start = time.time()
    try:
        metrics['snr'] = mapf.compute_snr(array1, array2, mode=mode)
    except Exception as e:
        print(f"    WARNING: SNR failed: {e}")
        metrics['snr'] = 0.0
    print(f"    ✓ SNR: {time.time() - metric_start:.3f}s")

    # Additional metrics to match DAE output format
    # Loss is typically MSE (same as mse metric)
    metrics['loss'] = metrics['mse']

    # NMAE metrics - Normalized Mean Absolute Error variations
    metric_start = time.time()
    try:
        # Check if MAP_functions has nmae computation functions
        if hasattr(mapf, 'compute_nmae_range'):
            metrics['nmae_range'] = mapf.compute_nmae_range(array1, array2, mode=mode)
        else:
            # Fallback: NMAE with range normalization
            mae = np.mean(np.abs(array1 - array2))
            signal_range = np.max(array1) - np.min(array1)
            metrics['nmae_range'] = mae / signal_range if signal_range > 0 else 0.0

        if hasattr(mapf, 'compute_nmae_l1'):
            metrics['nmae_l1'] = mapf.compute_nmae_l1(array1, array2, mode=mode)
        else:
            # Fallback: NMAE with L1 norm normalization
            mae = np.mean(np.abs(array1 - array2))
            l1_norm = np.mean(np.abs(array1))
            metrics['nmae_l1'] = mae / l1_norm if l1_norm > 0 else 0.0

        if hasattr(mapf, 'compute_nmae_mean'):
            metrics['nmae_mean'] = mapf.compute_nmae_mean(array1, array2, mode=mode)
        else:
            # Fallback: NMAE with mean normalization (same as range for consistency)
            metrics['nmae_mean'] = metrics['nmae_range']

    except Exception as e:
        print(f"    WARNING: NMAE metrics failed: {e}")
        metrics['nmae_range'] = 0.0
        metrics['nmae_l1'] = 0.0
        metrics['nmae_mean'] = 0.0
    print(f"    ✓ NMAE metrics: {time.time() - metric_start:.3f}s")

    # DTW and LSD metrics (POTENTIALLY SLOW - these are the bottlenecks!)
    if skip_slow_metrics:
        print(f"    ⏩ SKIPPING slow DTW/LSD metrics for faster execution")
        metrics['dtw'] = 0.0
        metrics['lsd'] = 0.0
    else:
        metric_start = time.time()
        try:
            print(f"    ⏳ Computing DTW for {array1.shape[0]} signals (this may be slow)...")
            if hasattr(mapf, 'compute_dtw'):
                metrics['dtw'] = mapf.compute_dtw(array1, array2, mode=mode)
            else:
                print(f"    WARNING: compute_dtw not available in mapf, using fallback")
                metrics['dtw'] = _fallback_dtw(array1, array2)
        except Exception as e:
            print(f"    WARNING: DTW failed: {e}")
            metrics['dtw'] = 0.0
        print(f"    ✓ DTW: {time.time() - metric_start:.3f}s (BOTTLENECK!)")

        metric_start = time.time()
        try:
            print(f"    ⏳ Computing LSD for {array1.shape[0]} signals (this may be slow)...")
            if hasattr(mapf, 'compute_lsd'):
                metrics['lsd'] = mapf.compute_lsd(array1, array2, mode=mode)
            else:
                print(f"    WARNING: compute_lsd not available in mapf, using fallback")
                metrics['lsd'] = _fallback_lsd(array1, array2)
        except Exception as e:
            print(f"    WARNING: LSD failed: {e}")
            metrics['lsd'] = 0.0
        print(f"    ✓ LSD: {time.time() - metric_start:.3f}s (BOTTLENECK!)")

    total_time = time.time() - start_time
    print(f"    📊 Total metrics computation: {total_time:.3f}s")
    if total_time > 10:
        print(f"    ⚠️ PERFORMANCE WARNING: Metrics took {total_time:.1f}s - DTW/LSD are slow!")

    return metrics


def create_dae_format_summary(all_fold_results: List, num_folds: int) -> pd.DataFrame:
    """
    Create summary table in EXACT same format as DAE output

    Args:
        all_fold_results: Results from all folds
        num_folds: Number of folds

    Returns:
        DataFrame with DAE-format summary
    """
    # Extract metrics from fold results
    train_filtered_metrics = [result['train_filtered'] for result in all_fold_results]
    test_filtered_metrics = [result['test_filtered'] for result in all_fold_results]

    # Check for hidden metrics
    has_hidden = all(result.get('hidden_filtered') is not None for result in all_fold_results)
    hidden_filtered_metrics = []
    if has_hidden:
        hidden_filtered_metrics = [result['hidden_filtered'] for result in all_fold_results]

    # Metric order matching DAE output exactly, plus DTW and LSD for comprehensive comparison
    metric_order = [
        'loss', 'pcorr', 'psnr', 'rmse', 'mse', 'spearman', 'snr',
        'nmae_range', 'nmae_l1', 'nmae_mean', 'dtw', 'lsd'
    ]

    summary_rows = []

    # Create fold rows (split0, split1, etc.)
    for fold_idx in range(num_folds):
        row = {'Split': f'split{fold_idx}'}

        train_metrics = train_filtered_metrics[fold_idx]
        test_metrics = test_filtered_metrics[fold_idx]

        # Add metrics in exact DAE order
        for metric in metric_order:
            row[f'{metric}_train'] = train_metrics.get(metric, 0.0)
            row[f'{metric}_test'] = test_metrics.get(metric, 0.0)

            if has_hidden:
                hidden_metrics = hidden_filtered_metrics[fold_idx]
                row[f'{metric}_hidden'] = hidden_metrics.get(metric, 0.0)

        summary_rows.append(row)

    # Calculate average row
    avg_row = {'Split': 'average'}
    for metric in metric_order:
        # Calculate means
        train_values = [train_filtered_metrics[i].get(metric, 0.0) for i in range(num_folds)]
        test_values = [test_filtered_metrics[i].get(metric, 0.0) for i in range(num_folds)]

        avg_row[f'{metric}_train'] = np.mean(train_values)
        avg_row[f'{metric}_test'] = np.mean(test_values)

        if has_hidden:
            hidden_values = [hidden_filtered_metrics[i].get(metric, 0.0) for i in range(num_folds)]
            avg_row[f'{metric}_hidden'] = np.mean(hidden_values)

    summary_rows.append(avg_row)

    # Calculate std dev row
    std_row = {'Split': 'st. dev.'}
    for metric in metric_order:
        # Calculate standard deviations
        train_values = [train_filtered_metrics[i].get(metric, 0.0) for i in range(num_folds)]
        test_values = [test_filtered_metrics[i].get(metric, 0.0) for i in range(num_folds)]

        std_row[f'{metric}_train'] = np.std(train_values)
        std_row[f'{metric}_test'] = np.std(test_values)

        if has_hidden:
            hidden_values = [hidden_filtered_metrics[i].get(metric, 0.0) for i in range(num_folds)]
            std_row[f'{metric}_hidden'] = np.std(hidden_values)

    summary_rows.append(std_row)

    # Create DataFrame with exact column ordering
    df = pd.DataFrame(summary_rows)

    # Reorder columns to match Desired format exactly
    ordered_columns = ['Split']

    # Loss only has train/test usually, but we check
    ordered_columns.extend(['loss_train', 'loss_test'])
    if has_hidden and 'loss_hidden' in df.columns: # Rare for baseline filters but safe check
        ordered_columns.append('loss_hidden')

    # Metrics with train/test/hidden pattern
    metrics_with_hidden = [
        'pcorr', 'psnr', 'rmse', 'mse', 'spearman', 'snr',
        'dtw', 'lsd',
        'nmae_range', 'nmae_l1', 'nmae_mean'
    ]

    for metric in metrics_with_hidden:
        ordered_columns.append(f'{metric}_train')
        ordered_columns.append(f'{metric}_test')
        if has_hidden:
            ordered_columns.append(f'{metric}_hidden')

    # Only include columns that exist in the DataFrame
    final_columns = [col for col in ordered_columns if col in df.columns]
    df = df[final_columns]

    return df


def save_fold_results(fold_idx: int, train_clean: np.ndarray, train_filtered: np.ndarray, train_noisy: np.ndarray,
                     test_clean: np.ndarray, test_filtered: np.ndarray, test_noisy: np.ndarray,
                     train_metrics: Dict, test_metrics: Dict, filter_name: str, working_dir: str,
                     hidden_metrics: Dict = None):
    """
    Save fold results in the same format as deep learning models

    Args:
        fold_idx: Fold index
        train_clean: Clean training signals
        train_filtered: Filtered training signals
        train_noisy: Noisy training signals
        test_clean: Clean test signals
        test_filtered: Filtered test signals
        test_noisy: Noisy test signals
        train_metrics: Training metrics
        test_metrics: Test metrics
        filter_name: Name of filter used
        working_dir: Output directory
        hidden_metrics: Hidden test set metrics (optional)
    """

    # Create directories
    signals_dir = os.path.join(working_dir, 'denoised_signals')
    metrics_dir = os.path.join(working_dir, 'metrics')
    os.makedirs(signals_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Save signals (same format as VAE/DAE/LUNet)
    train_file = os.path.join(signals_dir, f'fold{fold_idx}_train_signals.npz')
    test_file = os.path.join(signals_dir, f'fold{fold_idx}_test_signals.npz')

    np.savez_compressed(train_file,
                       original_clean=train_clean,
                       denoised_output=train_filtered,
                       noisy_input=train_noisy)

    np.savez_compressed(test_file,
                       original_clean=test_clean,
                       denoised_output=test_filtered,
                       noisy_input=test_noisy)

    print(f"    💾 Saved signals: {train_file}")
    print(f"    💾 Saved signals: {test_file}")

    # Save metrics (same format as deep learning models)
    metrics_file = os.path.join(metrics_dir, f'fold{fold_idx}_metrics.json')

    fold_metrics = {
        'fold': fold_idx,
        'filter': filter_name,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'train_samples': len(train_clean),
        'test_samples': len(test_clean)
    }

    if hidden_metrics:
        fold_metrics['hidden_metrics'] = hidden_metrics

    import json
    with open(metrics_file, 'w') as f:
        json.dump(fold_metrics, f, indent=2)

    print(f"    💾 Saved metrics: {metrics_file}")


def process_single_fold_with_custom_filter(fold_idx: int, filter_name: str, filter_params: dict, noise_type: str = 'allmixed',
                                         seed_split: int = 29, num_folds: int = 4, working_dir: str = None,
                                         skip_slow_metrics: bool = False, exclude_patients_file: str = None) -> Dict:
    """
    Process a single fold with specified filter and custom parameters

    Args:
        fold_idx: Fold index (0-3)
        filter_name: Name of filter to apply
        filter_params: Custom filter parameters dict
        noise_type: Type of noise
        seed_split: Cross-validation seed
        num_folds: Number of folds
        working_dir: Output directory

    Returns:
        Dictionary with metrics for train and test sets
    """
    print(f"📁 Processing Fold {fold_idx} with {filter_name} filter (custom params)...")

    # Load data (same as VAE training)
    try:
        csv_path = os.path.join(str(paths.data_root()), 'MAP_vent_complete_pandas.csv')
        if os.path.exists(csv_path):
            print(f"  📊 Loading preprocessed data from: {csv_path}")
            MAP_vent_complete = pd.read_csv(csv_path)
            MAP_vent_complete['MAP_segments'] = MAP_vent_complete['MAP_segments'].apply(
                lambda x: np.fromstring(x.strip('[]'), sep=' ') if isinstance(x, str) else x
            )
        else:
            raise FileNotFoundError("Preprocessed CSV not found, trying raw data loading")
    except Exception as e:
        print(f"  ⚠️ CSV loading failed: {e}")
        print(f"  📊 Trying raw data loading...")
        MAP_vent_complete = get_MAP_vent_data(CLUSTER=True)

    # -------------------------------------------------------------
    # Exclude patients (Hidden Test Set Logic)
    # -------------------------------------------------------------
    excluded_patients = []
    if exclude_patients_file:
        exclusion_file_path = exclude_patients_file
        # Handle relative paths
        if not os.path.isabs(exclusion_file_path):
            possible_paths = [
                os.path.join(os.getcwd(), exclusion_file_path),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), exclusion_file_path)
            ]
            for p in possible_paths:
                if os.path.exists(p):
                    exclusion_file_path = p
                    break

        if os.path.exists(exclusion_file_path):
            try:
                exclusion_df = pd.read_csv(exclusion_file_path)
                if 'pat_ID' in exclusion_df.columns:
                    excluded_patients = exclusion_df['pat_ID'].astype(str).tolist()
                    print(f"  🛑 Excluding {len(excluded_patients)} patients from {exclusion_file_path}")

                    # Capture hidden set
                    df_hidden = MAP_vent_complete[MAP_vent_complete['pat_ID'].isin(excluded_patients)].copy()

                    # Remove from dataset
                    original_len = len(MAP_vent_complete)
                    MAP_vent_complete = MAP_vent_complete[~MAP_vent_complete['pat_ID'].isin(excluded_patients)].reset_index(drop=True)
                    print(f"  📉 Dataset reduced from {original_len} to {len(MAP_vent_complete)} samples (Hidden set: {len(df_hidden)})")
                else:
                    print(f"  ⚠️ Warning: Exclusion file missing 'pat_ID' column")
            except Exception as e:
                print(f"  ⚠️ Error reading exclusion file: {e}")
        else:
            print(f"  ⚠️ Warning: Exclusion file not found at {exclusion_file_path}")
    # -------------------------------------------------------------

    # Get noise parameters and arrays
    noise_params = mapf.find_noise_params(noise_type)
    if noise_type in ['ep', 'allmixed']:
        arrays = mapf.get_np_noisearrays(MAP_vent_complete)
    else:
        arrays = []

    # Get train/test split (EXACT same as VAE training)
    X_train, X_test, y_train, y_test = get_train_test_kfolds(
        MAP_vent_complete,
        num_folds=num_folds,
        split_number=fold_idx,
        r_seed=seed_split,
        apd_label='APD30_gs'
    )

    print(f"  📊 Data: {len(X_train)} train, {len(X_test)} test samples")

    # Normalize (same as VAE training)
    X_std_train, X_std_test = mapf.normalize_EGM_input(X_train, X_test)

    # Generate noisy data (same as VAE training)
    train, test, _ = retrieveDataSet(noise_type, noise_params, X_train, X_test,
                                    X_std_train, X_std_test, arrays=arrays)

    # Extract clean and noisy signals
    print(f"  🔊 Extracting signals...")

    train_data = []
    for i in range(len(train)):
        clean, noisy = train[i]
        train_data.append((clean, noisy))

    test_data = []
    for i in range(len(test)):
        clean, noisy = test[i]
        test_data.append((clean, noisy))

    # Convert to numpy arrays
    X_clean_train = np.array([item[0] for item in train_data])
    X_noisy_train = np.array([item[1] for item in train_data])
    X_clean_test = np.array([item[0] for item in test_data])
    X_noisy_test = np.array([item[1] for item in test_data])

    print(f"      📋 Shapes: train_clean={X_clean_train.shape}, test_clean={X_clean_test.shape}")

    # Apply filter with custom parameters
    print(f"  🔧 Applying {filter_name} filter with custom params: {filter_params}...")
    filter_func = FILTER_FUNCTIONS[filter_name]

    # Filter training signals
    X_filtered_train = []
    for i, signal in enumerate(X_noisy_train):
        try:
            if hasattr(signal, 'shape') and signal.ndim > 1:
                signal = signal.flatten()
            elif not hasattr(signal, 'shape'):
                signal = np.array(signal).flatten()

            filtered_signal = filter_func(signal, **filter_params)

            if np.any(np.isnan(filtered_signal)) or np.any(np.isinf(filtered_signal)):
                print(f"        ⚠️ Filtering produced NaN/inf for train signal {i}, using original")
                X_filtered_train.append(signal)
            else:
                X_filtered_train.append(filtered_signal)

        except Exception as e:
            print(f"        ❌ Error filtering train signal {i}: {e}, using original")
            X_filtered_train.append(signal)

    X_filtered_train = np.array(X_filtered_train)

    # Filter test signals
    X_filtered_test = []
    for i, signal in enumerate(X_noisy_test):
        try:
            if hasattr(signal, 'shape') and signal.ndim > 1:
                signal = signal.flatten()
            elif not hasattr(signal, 'shape'):
                signal = np.array(signal).flatten()

            filtered_signal = filter_func(signal, **filter_params)

            if np.any(np.isnan(filtered_signal)) or np.any(np.isinf(filtered_signal)):
                print(f"        ⚠️ Filtering produced NaN/inf for test signal {i}, using original")
                X_filtered_test.append(signal)
            else:
                X_filtered_test.append(filtered_signal)

        except Exception as e:
            print(f"        ❌ Error filtering test signal {i}: {e}, using original")
            X_filtered_test.append(signal)

    X_filtered_test = np.array(X_filtered_test)

    print(f"      📋 Filtered shapes: train={X_filtered_train.shape}, test={X_filtered_test.shape}")

    # Ensure proper shapes for metrics computation (same as before)
    for arr_name, arr in [('X_clean_train', X_clean_train), ('X_noisy_train', X_noisy_train),
                          ('X_filtered_train', X_filtered_train), ('X_clean_test', X_clean_test),
                          ('X_noisy_test', X_noisy_test), ('X_filtered_test', X_filtered_test)]:
        if arr.ndim != 2:
            if arr.ndim == 3:
                arr = arr.squeeze()
            arr = np.atleast_2d(arr)
            if arr.shape[0] == 1 and arr.shape[1] > 1:
                arr = arr.T
            # Update the variable
            if arr_name == 'X_clean_train':
                X_clean_train = arr
            elif arr_name == 'X_noisy_train':
                X_noisy_train = arr
            elif arr_name == 'X_filtered_train':
                X_filtered_train = arr
            elif arr_name == 'X_clean_test':
                X_clean_test = arr
            elif arr_name == 'X_noisy_test':
                X_noisy_test = arr
            elif arr_name == 'X_filtered_test':
                X_filtered_test = arr

    # Compute metrics
    print(f"  📈 Computing metrics...")
    train_filtered_metrics = compute_all_metrics_robust(X_clean_train, X_filtered_train, mode='total')
    test_filtered_metrics = compute_all_metrics_robust(X_clean_test, X_filtered_test, mode='total')

    # Also compute noisy baseline for comparison
    train_noisy_metrics = compute_all_metrics_robust(X_clean_train, X_noisy_train, mode='total')
    test_noisy_metrics = compute_all_metrics_robust(X_clean_test, X_noisy_test, mode='total')

    print(f"    📊 Results - Train PCC: {train_filtered_metrics['pcorr']:.4f}, Test PCC: {test_filtered_metrics['pcorr']:.4f}")

    # Save results if working directory specified
    if working_dir:
        save_fold_results(fold_idx, X_clean_train, X_filtered_train, X_noisy_train,
                         X_clean_test, X_filtered_test, X_noisy_test,
                         train_filtered_metrics, test_filtered_metrics, filter_name, working_dir)

        # -------------------------------------------------------------
        # Process Hidden Test Set (if available) - Added for Hidden Test Support
        # -------------------------------------------------------------
        if 'df_hidden' in locals() and not df_hidden.empty:
            print(f"  🕵️ Processing Hidden Test Set ({len(df_hidden)} samples)...")
            try:
                # Prepare data
                X_hidden = np.stack(df_hidden['MAP_segments'].values)

                # Get separate noise arrays for hidden set (reuse existing arrays if suitable, or generate)
                # Note: 'arrays' variable from earlier is valid
                if noise_type in ['ep', 'allmixed'] and len(arrays) == 0:
                     arrays_hidden = mapf.get_np_noisearrays(df_hidden)
                else:
                     arrays_hidden = arrays

                # Normalize (independent)
                X_std_hidden, _ = mapf.normalize_EGM_input(X_hidden, X_hidden)

                # Generate noisy data
                # We reuse retrieveDataSet by passing hidden as both train and test to get standard processing
                # (Ignoring the 'train' output)
                _, hidden_out, _ = retrieveDataSet(noise_type, noise_params, X_hidden, X_hidden,
                                                X_std_hidden, X_std_hidden, arrays=arrays_hidden)

                # Extract signals
                hidden_data_list = []
                for i in range(len(hidden_out)):
                    clean, noisy = hidden_out[i]
                    hidden_data_list.append((clean, noisy))

                X_clean_hidden = np.array([item[0] for item in hidden_data_list])
                X_noisy_hidden = np.array([item[1] for item in hidden_data_list])

                # Filter Hidden Signals
                X_filtered_hidden = []
                for i, signal in enumerate(X_noisy_hidden):
                    try:
                        if hasattr(signal, 'shape') and signal.ndim > 1:
                            signal = signal.flatten()
                        elif not hasattr(signal, 'shape'):
                            signal = np.array(signal).flatten()

                        filtered_signal = filter_func(signal, **filter_params)

                        if np.any(np.isnan(filtered_signal)) or np.any(np.isinf(filtered_signal)):
                            X_filtered_hidden.append(signal)
                        else:
                            X_filtered_hidden.append(filtered_signal)
                    except:
                        X_filtered_hidden.append(signal)

                X_filtered_hidden = np.array(X_filtered_hidden)

                # Squeeze dimensions if needed
                if X_clean_hidden.ndim == 3: X_clean_hidden = X_clean_hidden.squeeze()
                if X_noisy_hidden.ndim == 3: X_noisy_hidden = X_noisy_hidden.squeeze()
                if X_filtered_hidden.ndim == 3: X_filtered_hidden = X_filtered_hidden.squeeze()

                # Save Hidden Signals
                signals_dir = os.path.join(working_dir, 'denoised_signals')
                if not os.path.exists(signals_dir): os.makedirs(signals_dir)

                hidden_file = os.path.join(signals_dir, f'fold{fold_idx}_hidden_test_signals.npz')
                np.savez_compressed(hidden_file,
                                original_clean=X_clean_hidden,
                                denoised_output=X_filtered_hidden,
                                noisy_input=X_noisy_hidden,
                                data_type='hidden_test_filtered')
                print(f"    💾 Saved Hidden Test signals: {hidden_file}")

                # Compute hidden metrics
                hidden_filtered_metrics = compute_all_metrics_robust(X_clean_hidden, X_filtered_hidden, mode='total')
                print(f"    📊 Hidden Results - PCC: {hidden_filtered_metrics['pcorr']:.4f}")

            except Exception as e:
                print(f"    ⚠️ Failed to process hidden test set: {e}")
                import traceback
                traceback.print_exc()

    return {
        'train_filtered': train_filtered_metrics,
        'test_filtered': test_filtered_metrics,
        'train_noisy': train_noisy_metrics,
        'test_noisy': test_noisy_metrics,
        'hidden_filtered': locals().get('hidden_filtered_metrics', None)
    }


def process_single_fold_with_filter(fold_idx: int, filter_name: str, noise_type: str = 'allmixed',
                                  seed_split: int = 29, num_folds: int = 4, working_dir: str = None,
                                  skip_slow_metrics: bool = False, exclude_patients_file: str = None) -> Dict:
    """
    Process a single fold with specified filter and save results

    Args:
        fold_idx: Fold index (0-3)
        filter_name: Name of filter to apply
        noise_type: Type of noise
        seed_split: Cross-validation seed
        num_folds: Number of folds
        working_dir: Output directory

    Returns:
        Dictionary with metrics for train and test sets
    """
    print(f"📁 Processing Fold {fold_idx} with {filter_name} filter...")

    # Load data (same as VAE training)
    try:
        csv_path = os.path.join(str(paths.data_root()), 'MAP_vent_complete_pandas.csv')
        if os.path.exists(csv_path):
            print(f"  📊 Loading preprocessed data from: {csv_path}")
            MAP_vent_complete = pd.read_csv(csv_path)
            MAP_vent_complete['MAP_segments'] = MAP_vent_complete['MAP_segments'].apply(
                lambda x: np.fromstring(x.strip('[]'), sep=' ') if isinstance(x, str) else x
            )
        else:
            raise FileNotFoundError("Preprocessed CSV not found, trying raw data loading")
    except Exception as e:
        print(f"  ⚠️ CSV loading failed: {e}")
        print(f"  📊 Trying raw data loading...")
        MAP_vent_complete = get_MAP_vent_data(CLUSTER=True)

    # -------------------------------------------------------------
    # Exclude patients (Hidden Test Set Logic)
    # -------------------------------------------------------------
    excluded_patients = []
    if exclude_patients_file:
        exclusion_file_path = exclude_patients_file
        # Handle relative paths
        if not os.path.isabs(exclusion_file_path):
            possible_paths = [
                os.path.join(os.getcwd(), exclusion_file_path),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), exclusion_file_path)
            ]
            for p in possible_paths:
                if os.path.exists(p):
                    exclusion_file_path = p
                    break

        if os.path.exists(exclusion_file_path):
            try:
                exclusion_df = pd.read_csv(exclusion_file_path)
                if 'pat_ID' in exclusion_df.columns:
                    excluded_patients = exclusion_df['pat_ID'].astype(str).tolist()
                    print(f"  🛑 Excluding {len(excluded_patients)} patients from {exclusion_file_path}")

                    # Capture hidden set
                    df_hidden = MAP_vent_complete[MAP_vent_complete['pat_ID'].isin(excluded_patients)].copy()

                    # Remove from dataset
                    original_len = len(MAP_vent_complete)
                    MAP_vent_complete = MAP_vent_complete[~MAP_vent_complete['pat_ID'].isin(excluded_patients)].reset_index(drop=True)
                    print(f"  📉 Dataset reduced from {original_len} to {len(MAP_vent_complete)} samples (Hidden set: {len(df_hidden)})")
                else:
                    print(f"  ⚠️ Warning: Exclusion file missing 'pat_ID' column")
            except Exception as e:
                print(f"  ⚠️ Error reading exclusion file: {e}")
        else:
            print(f"  ⚠️ Warning: Exclusion file not found at {exclusion_file_path}")
    # -------------------------------------------------------------

    # Get noise parameters and arrays
    noise_params = mapf.find_noise_params(noise_type)
    if noise_type in ['ep', 'allmixed']:
        arrays = mapf.get_np_noisearrays(MAP_vent_complete)
    else:
        arrays = []

    # Get train/test split (EXACT same as VAE training)
    X_train, X_test, y_train, y_test = get_train_test_kfolds(
        MAP_vent_complete,
        num_folds=num_folds,
        split_number=fold_idx,
        r_seed=seed_split,
        apd_label='APD30_gs'
    )

    print(f"  📊 Data: {len(X_train)} train, {len(X_test)} test samples")

    # Normalize (same as VAE training)
    X_std_train, X_std_test = mapf.normalize_EGM_input(X_train, X_test)

    # Generate noisy data (same as VAE training)
    train, test, _ = retrieveDataSet(noise_type, noise_params, X_train, X_test,
                                    X_std_train, X_std_test, arrays=arrays)

    # Extract clean and noisy signals
    print(f"  🔊 Extracting signals...")

    train_data = []
    for i in range(len(train)):
        clean, noisy = train[i]
        train_data.append((clean, noisy))

    test_data = []
    for i in range(len(test)):
        clean, noisy = test[i]
        test_data.append((clean, noisy))

    # Convert to numpy arrays
    X_clean_train = np.array([item[0] for item in train_data])
    X_noisy_train = np.array([item[1] for item in train_data])
    X_clean_test = np.array([item[0] for item in test_data])
    X_noisy_test = np.array([item[1] for item in test_data])

    print(f"      📋 Shapes: train_clean={X_clean_train.shape}, test_clean={X_clean_test.shape}")

    # Apply filter
    print(f"  🔧 Applying {filter_name} filter...")
    filter_func = FILTER_FUNCTIONS[filter_name]
    filter_params = FILTER_PARAMS[filter_name]

    # Filter training signals
    X_filtered_train = []
    for i, signal in enumerate(X_noisy_train):
        try:
            if hasattr(signal, 'shape') and signal.ndim > 1:
                signal = signal.flatten()
            elif not hasattr(signal, 'shape'):
                signal = np.array(signal).flatten()

            filtered_signal = filter_func(signal, **filter_params)

            if np.any(np.isnan(filtered_signal)) or np.any(np.isinf(filtered_signal)):
                print(f"        ⚠️ Filtering produced NaN/inf for train signal {i}, using original")
                X_filtered_train.append(signal)
            else:
                X_filtered_train.append(filtered_signal)

        except Exception as e:
            print(f"        ❌ Error filtering train signal {i}: {e}, using original")
            X_filtered_train.append(signal)

    X_filtered_train = np.array(X_filtered_train)

    # Filter test signals
    X_filtered_test = []
    for i, signal in enumerate(X_noisy_test):
        try:
            if hasattr(signal, 'shape') and signal.ndim > 1:
                signal = signal.flatten()
            elif not hasattr(signal, 'shape'):
                signal = np.array(signal).flatten()

            filtered_signal = filter_func(signal, **filter_params)

            if np.any(np.isnan(filtered_signal)) or np.any(np.isinf(filtered_signal)):
                print(f"        ⚠️ Filtering produced NaN/inf for test signal {i}, using original")
                X_filtered_test.append(signal)
            else:
                X_filtered_test.append(filtered_signal)

        except Exception as e:
            print(f"        ❌ Error filtering test signal {i}: {e}, using original")
            X_filtered_test.append(signal)

    X_filtered_test = np.array(X_filtered_test)

    print(f"      📋 Filtered shapes: train={X_filtered_train.shape}, test={X_filtered_test.shape}")

    # Ensure proper shapes for metrics computation
    for arr_name, arr in [('X_clean_train', X_clean_train), ('X_noisy_train', X_noisy_train),
                          ('X_filtered_train', X_filtered_train), ('X_clean_test', X_clean_test),
                          ('X_noisy_test', X_noisy_test), ('X_filtered_test', X_filtered_test)]:
        if arr.ndim != 2:
            if arr.ndim == 3:
                arr = arr.squeeze()
            arr = np.atleast_2d(arr)
            if arr.shape[0] == 1 and arr.shape[1] > 1:
                arr = arr.T
            # Update the variable
            if arr_name == 'X_clean_train':
                X_clean_train = arr
            elif arr_name == 'X_noisy_train':
                X_noisy_train = arr
            elif arr_name == 'X_filtered_train':
                X_filtered_train = arr
            elif arr_name == 'X_clean_test':
                X_clean_test = arr
            elif arr_name == 'X_noisy_test':
                X_noisy_test = arr
            elif arr_name == 'X_filtered_test':
                X_filtered_test = arr

    # Compute metrics
    print(f"  📈 Computing metrics...")
    train_filtered_metrics = compute_all_metrics_robust(X_clean_train, X_filtered_train, mode='total')
    test_filtered_metrics = compute_all_metrics_robust(X_clean_test, X_filtered_test, mode='total')

    # Also compute noisy baseline for comparison
    train_noisy_metrics = compute_all_metrics_robust(X_clean_train, X_noisy_train, mode='total')
    test_noisy_metrics = compute_all_metrics_robust(X_clean_test, X_noisy_test, mode='total')

    print(f"    📊 Results - Train PCC: {train_filtered_metrics['pcorr']:.4f}, Test PCC: {test_filtered_metrics['pcorr']:.4f}")

    # Save results if working directory specified
    if working_dir:
        save_fold_results(fold_idx, X_clean_train, X_filtered_train, X_noisy_train,
                         X_clean_test, X_filtered_test, X_noisy_test,
                         train_filtered_metrics, test_filtered_metrics, filter_name, working_dir)

        # -------------------------------------------------------------
        # Process Hidden Test Set (if available)
        # -------------------------------------------------------------
        if 'df_hidden' in locals() and len(df_hidden) > 0:
            print(f"  🕵️ Processing Hidden Test Set ({len(df_hidden)} samples)...")
            try:
                # Prepare data
                X_hidden = np.stack(df_hidden['MAP_segments'].values)

                # Get separate noise arrays for hidden set
                if noise_type in ['ep', 'allmixed']:
                    arrays_hidden = mapf.get_np_noisearrays(df_hidden)
                else:
                    arrays_hidden = []

                # Normalize (independent of train/test fold limits)
                X_std_hidden, _ = mapf.normalize_EGM_input(X_hidden, X_hidden)

                # Generate noisy data
                hidden_set, _, _ = retrieveDataSet(noise_type, noise_params, X_hidden, X_hidden,
                                                X_std_hidden, X_std_hidden, arrays=arrays_hidden)

                # Extract signals
                hidden_data = []
                for i in range(len(hidden_set)):
                    clean, noisy = hidden_set[i]
                    hidden_data.append((clean, noisy))

                X_clean_hidden = np.array([item[0] for item in hidden_data])
                X_noisy_hidden = np.array([item[1] for item in hidden_data])

                # Filter Hidden Signals
                X_filtered_hidden = []
                for i, signal in enumerate(X_noisy_hidden):
                    try:
                        if hasattr(signal, 'shape') and signal.ndim > 1:
                            signal = signal.flatten()
                        elif not hasattr(signal, 'shape'):
                            signal = np.array(signal).flatten()

                        filtered_signal = filter_func(signal, **filter_params)

                        if np.any(np.isnan(filtered_signal)) or np.any(np.isinf(filtered_signal)):
                            X_filtered_hidden.append(signal)
                        else:
                            X_filtered_hidden.append(filtered_signal)
                    except:
                        X_filtered_hidden.append(signal)

                X_filtered_hidden = np.array(X_filtered_hidden)

                # Squeeze if needed
                if X_clean_hidden.ndim == 3: X_clean_hidden = X_clean_hidden.squeeze()
                if X_noisy_hidden.ndim == 3: X_noisy_hidden = X_noisy_hidden.squeeze()
                if X_filtered_hidden.ndim == 3: X_filtered_hidden = X_filtered_hidden.squeeze()

                # Save Hidden Signals
                signals_dir = os.path.join(working_dir, 'denoised_signals')
                hidden_file = os.path.join(signals_dir, f'fold{fold_idx}_hidden_test_signals.npz')
                np.savez_compressed(hidden_file,
                                original_clean=X_clean_hidden,
                                denoised_output=X_filtered_hidden,
                                noisy_input=X_noisy_hidden,
                                data_type='hidden_test_filtered')
                print(f"    💾 Saved Hidden Test signals: {hidden_file}")

                # Compute hidden metrics (ADDED)
                hidden_filtered_metrics = compute_all_metrics_robust(X_clean_hidden, X_filtered_hidden, mode='total')
                print(f"    📊 Hidden Results - PCC: {hidden_filtered_metrics['pcorr']:.4f}")

            except Exception as e:
                print(f"    ⚠️ Failed to process hidden test set: {e}")

    # Prepare hidden metrics for return
    hidden_metrics_to_return = None
    if 'hidden_filtered_metrics' in locals():
         hidden_metrics_to_return = hidden_filtered_metrics

    return {
        'train_filtered': train_filtered_metrics,
        'test_filtered': test_filtered_metrics,
        'train_noisy': train_noisy_metrics,
        'test_noisy': test_noisy_metrics,
        'hidden_filtered': hidden_metrics_to_return
    }


def run_filter_baseline_with_params(filter_name: str, filter_params: dict, noise_type: str = 'allmixed', seed_split: int = 29,
                                   num_folds: int = 4, working_dir: str = None, save_results: bool = True,
                                   skip_slow_metrics: bool = False, exclude_patients_file: str = None):
    """
    Run complete filter baseline analysis across all folds with custom parameters

    Args:
        filter_name: Name of filter to apply
        filter_params: Custom filter parameters dict
        noise_type: Type of noise
        seed_split: Cross-validation seed
        num_folds: Number of folds
        working_dir: Output directory
        save_results: Whether to save results
    """
    print(f"🚀 Starting {filter_name.upper()} Filter Baseline")
    print("=" * 60)
    print(f"Filter: {filter_name}")
    print(f"Parameters: {filter_params}")
    print(f"Noise type: {noise_type}")
    print(f"Seed split: {seed_split}")
    print(f"Folds: {num_folds}")
    print(f"Working directory: {working_dir}")
    if exclude_patients_file:
        print(f"Excluding patients from: {exclude_patients_file}")

    if filter_name not in FILTER_FUNCTIONS:
        raise ValueError(f"Unknown filter: {filter_name}. Available: {list(FILTER_FUNCTIONS.keys())}")

    # Create working directory if specified
    if working_dir:
        os.makedirs(working_dir, exist_ok=True)
        print(f"📁 Created working directory: {working_dir}")

    # Process all folds with custom parameters
    all_fold_results = []

    for fold_idx in range(num_folds):
        fold_results = process_single_fold_with_custom_filter(
            fold_idx, filter_name, filter_params, noise_type, seed_split, num_folds, working_dir,
            skip_slow_metrics, exclude_patients_file
        )
        all_fold_results.append(fold_results)

    # Calculate summary statistics (same as before)
    print(f"\n📊 Calculating summary statistics...")

    # Collect metrics across folds
    train_filtered_metrics = [result['train_filtered'] for result in all_fold_results]
    test_filtered_metrics = [result['test_filtered'] for result in all_fold_results]
    train_noisy_metrics = [result['train_noisy'] for result in all_fold_results]
    test_noisy_metrics = [result['test_noisy'] for result in all_fold_results]

    # Check for hidden metrics
    has_hidden = all(result.get('hidden_filtered') is not None for result in all_fold_results)
    if has_hidden:
        hidden_filtered_metrics = [result['hidden_filtered'] for result in all_fold_results]

    # Calculate means and stds
    summary_data = {}

    # Get all metric names
    metric_names = list(train_filtered_metrics[0].keys())

    for metric in metric_names:
        # Filtered results
        train_values = [m[metric] for m in train_filtered_metrics]
        test_values = [m[metric] for m in test_filtered_metrics]

        summary_data[f'{metric}_train_filtered_mean'] = np.mean(train_values)
        summary_data[f'{metric}_train_filtered_std'] = np.std(train_values)
        summary_data[f'{metric}_test_filtered_mean'] = np.mean(test_values)
        summary_data[f'{metric}_test_filtered_std'] = np.std(test_values)

        if has_hidden:
            hidden_values = [m[metric] for m in hidden_filtered_metrics]
            summary_data[f'{metric}_hidden_filtered_mean'] = np.mean(hidden_values)
            summary_data[f'{metric}_hidden_filtered_std'] = np.std(hidden_values)

        # Noisy baseline results
        train_noisy_values = [m[metric] for m in train_noisy_metrics]
        test_noisy_values = [m[metric] for m in test_noisy_metrics]

        summary_data[f'{metric}_train_noisy_mean'] = np.mean(train_noisy_values)
        summary_data[f'{metric}_train_noisy_std'] = np.std(train_noisy_values)
        summary_data[f'{metric}_test_noisy_mean'] = np.mean(test_noisy_values)
        summary_data[f'{metric}_test_noisy_std'] = np.std(test_noisy_values)

        if has_hidden:
            # Assume noisy baseline for hidden is roughly same stats or we can't easily track it unless we returned it
             # For now, let's just create placeholders if needed, or skip noisy comparison for hidden
             pass

    # Print summary table
    print(f"\n🎯 {filter_name.upper()} FILTER BASELINE SUMMARY")
    print("=" * 80)

    key_metrics = ['pcorr', 'rmse', 'psnr', 'mse', 'spearman', 'snr', 'dtw', 'lsd']

    splits_to_show = [('train', 'TRAIN'), ('test', 'VALIDATION')]
    if has_hidden:
        splits_to_show.append(('hidden', 'TEST (HIDDEN)'))

    for split_key, split_label in splits_to_show:
        print(f"\n📊 {split_label} METRICS:")
        print(f"{'Metric':<15} {'Filtered (Mean±Std)':<25} {'Noisy (Mean±Std)':<25} {'Improvement':<12}")
        print("-" * 77)

        for metric in key_metrics:
            filtered_mean = summary_data.get(f'{metric}_{split_key}_filtered_mean', 0.0)
            filtered_std = summary_data.get(f'{metric}_{split_key}_filtered_std', 0.0)

            # For hidden, we might not have noisy baseline computed above in this loop structure easily
            # Fallback to test noisy mean for comparison context or 0 if missing
            if split_key == 'hidden':
                 noisy_mean = summary_data.get(f'{metric}_test_noisy_mean', 0.0) # Approximation/Context
                 noisy_std = summary_data.get(f'{metric}_test_noisy_std', 0.0)
            else:
                 noisy_mean = summary_data.get(f'{metric}_{split_key}_noisy_mean', 0.0)
                 noisy_std = summary_data.get(f'{metric}_{split_key}_noisy_std', 0.0)

            # Calculate improvement
            if metric in ['pcorr', 'psnr', 'spearman', 'snr']:  # Higher is better
                improvement = ((filtered_mean - noisy_mean) / noisy_mean * 100) if noisy_mean != 0 else 0
            else:  # Lower is better (rmse, mse, dtw, lsd)
                improvement = ((noisy_mean - filtered_mean) / noisy_mean * 100) if noisy_mean != 0 else 0

            improvement_str = f"+{improvement:.1f}%" if improvement > 0 else f"{improvement:.1f}%"

            print(f"{metric:<15} {filtered_mean:.4f}±{filtered_std:.4f}      {noisy_mean:.4f}±{noisy_std:.4f}      {improvement_str:<12}")

    # Save comprehensive results matching DAE format
    if save_results and working_dir:
        # Create DAE-format summary (main results file)
        summary_file = os.path.join(working_dir, f'{filter_name}_baseline_summary.xlsx')
        dae_format_df = create_dae_format_summary(all_fold_results, num_folds)
        dae_format_df.to_excel(summary_file, index=False)

        # Also save as CSV for easier reading
        csv_file = os.path.join(working_dir, f'{filter_name}_summary_values.csv')
        dae_format_df.to_csv(csv_file, index=False)

        print(f"\n💾 DAE-format results saved to: {summary_file}")
        print(f"💾 CSV summary saved to: {csv_file}")

        # Create detailed DataFrame (legacy format for additional analysis)
        detailed_file = os.path.join(working_dir, f'{filter_name}_detailed_results.xlsx')
        detailed_results = []
        for fold_idx in range(num_folds):
            fold_result = all_fold_results[fold_idx]

            # Add filtered results
            row_filtered = {'fold': fold_idx, 'condition': 'filtered'}
            for metric in metric_names:
                row_filtered[f'{metric}_train'] = fold_result['train_filtered'][metric]
                row_filtered[f'{metric}_test'] = fold_result['test_filtered'][metric]
                if fold_result.get('hidden_filtered'):
                     row_filtered[f'{metric}_hidden'] = fold_result['hidden_filtered'][metric]
            detailed_results.append(row_filtered)

            # Add noisy results
            row_noisy = {'fold': fold_idx, 'condition': 'noisy'}
            for metric in metric_names:
                row_noisy[f'{metric}_train'] = fold_result['train_noisy'][metric]
                row_noisy[f'{metric}_test'] = fold_result['test_noisy'][metric]
                # Noisy hidden baseline not currently tracked/passed, so skipping or could add if available
            detailed_results.append(row_noisy)

        # Add summary rows
        summary_filtered = {'fold': 'average', 'condition': 'filtered'}
        summary_noisy = {'fold': 'average', 'condition': 'noisy'}
        std_filtered = {'fold': 'std', 'condition': 'filtered'}
        std_noisy = {'fold': 'std', 'condition': 'noisy'}

        for metric in metric_names:
            summary_filtered[f'{metric}_train'] = summary_data[f'{metric}_train_filtered_mean']
            summary_filtered[f'{metric}_test'] = summary_data[f'{metric}_test_filtered_mean']
            std_filtered[f'{metric}_train'] = summary_data[f'{metric}_train_filtered_std']
            std_filtered[f'{metric}_test'] = summary_data[f'{metric}_test_filtered_std']

            summary_noisy[f'{metric}_train'] = summary_data[f'{metric}_train_noisy_mean']
            summary_noisy[f'{metric}_test'] = summary_data[f'{metric}_test_noisy_mean']
            std_noisy[f'{metric}_train'] = summary_data[f'{metric}_train_noisy_std']
            std_noisy[f'{metric}_test'] = summary_data[f'{metric}_test_noisy_std']

            if has_hidden:
                 summary_filtered[f'{metric}_hidden'] = summary_data.get(f'{metric}_hidden_filtered_mean', 0.0)
                 std_filtered[f'{metric}_hidden'] = summary_data.get(f'{metric}_hidden_filtered_std', 0.0)

        detailed_results.extend([summary_filtered, std_filtered, summary_noisy, std_noisy])

        detailed_df = pd.DataFrame(detailed_results)
        detailed_df.to_excel(detailed_file, index=False)

        print(f"\n💾 Detailed results saved to: {summary_file}")

        # Save filter parameters (with custom values)
        params_file = os.path.join(working_dir, f'{filter_name}_parameters.json')
        import json
        with open(params_file, 'w') as f:
            json.dump({
                'filter_name': filter_name,
                'filter_params': filter_params,  # Use custom parameters
                'noise_type': noise_type,
                'seed_split': seed_split,
                'num_folds': num_folds
            }, f, indent=2)

        print(f"💾 Filter parameters saved to: {params_file}")

    print(f"\n✅ {filter_name.upper()} filter baseline completed!")
    return all_fold_results, summary_data


def run_filter_baseline(filter_name: str, noise_type: str = 'allmixed', seed_split: int = 29,
                       num_folds: int = 4, working_dir: str = None, save_results: bool = True,
                       skip_slow_metrics: bool = False, exclude_patients_file: str = None):
    """
    Run complete filter baseline analysis across all folds

    Args:
        filter_name: Name of filter to apply
        noise_type: Type of noise
        seed_split: Cross-validation seed
        num_folds: Number of folds
        working_dir: Output directory
        save_results: Whether to save results
    """
    print(f"🚀 Starting {filter_name.upper()} Filter Baseline")
    print("=" * 60)
    print(f"Filter: {filter_name}")
    print(f"Noise type: {noise_type}")
    print(f"Seed split: {seed_split}")
    print(f"Folds: {num_folds}")
    print(f"Working directory: {working_dir}")
    if exclude_patients_file:
        print(f"Excluding patients from: {exclude_patients_file}")

    if filter_name not in FILTER_FUNCTIONS:
        raise ValueError(f"Unknown filter: {filter_name}. Available: {list(FILTER_FUNCTIONS.keys())}")

    # Create working directory if specified
    if working_dir:
        os.makedirs(working_dir, exist_ok=True)
        print(f"📁 Created working directory: {working_dir}")

    # Process all folds
    all_fold_results = []

    for fold_idx in range(num_folds):
        fold_results = process_single_fold_with_filter(
            fold_idx, filter_name, noise_type, seed_split, num_folds, working_dir,
            skip_slow_metrics, exclude_patients_file
        )
        all_fold_results.append(fold_results)

    # Calculate summary statistics
    print(f"\n📊 Calculating summary statistics...")

    # Collect metrics across folds
    train_filtered_metrics = [result['train_filtered'] for result in all_fold_results]
    test_filtered_metrics = [result['test_filtered'] for result in all_fold_results]
    train_noisy_metrics = [result['train_noisy'] for result in all_fold_results]
    test_noisy_metrics = [result['test_noisy'] for result in all_fold_results]

    # Calculate means and stds
    summary_data = {}

    # Get all metric names
    metric_names = list(train_filtered_metrics[0].keys())

    # Check for hidden metrics
    has_hidden = all(result.get('hidden_filtered') is not None for result in all_fold_results)
    if has_hidden:
        hidden_filtered_metrics = [result['hidden_filtered'] for result in all_fold_results]

    for metric in metric_names:
        # Filtered results
        train_values = [m[metric] for m in train_filtered_metrics]
        test_values = [m[metric] for m in test_filtered_metrics]

        summary_data[f'{metric}_train_filtered_mean'] = np.mean(train_values)
        summary_data[f'{metric}_train_filtered_std'] = np.std(train_values)
        summary_data[f'{metric}_test_filtered_mean'] = np.mean(test_values)
        summary_data[f'{metric}_test_filtered_std'] = np.std(test_values)

        if has_hidden:
            hidden_values = [m[metric] for m in hidden_filtered_metrics]
            summary_data[f'{metric}_hidden_filtered_mean'] = np.mean(hidden_values)
            summary_data[f'{metric}_hidden_filtered_std'] = np.std(hidden_values)

        # Noisy baseline results
        train_noisy_values = [m[metric] for m in train_noisy_metrics]
        test_noisy_values = [m[metric] for m in test_noisy_metrics]

        summary_data[f'{metric}_train_noisy_mean'] = np.mean(train_noisy_values)
        summary_data[f'{metric}_train_noisy_std'] = np.std(train_noisy_values)
        summary_data[f'{metric}_test_noisy_mean'] = np.mean(test_noisy_values)
        summary_data[f'{metric}_test_noisy_std'] = np.std(test_noisy_values)

    # Print summary table
    print(f"\n🎯 {filter_name.upper()} FILTER BASELINE SUMMARY")
    print("=" * 80)

    key_metrics = ['pcorr', 'rmse', 'psnr', 'mse', 'spearman', 'snr', 'dtw', 'lsd']

    # Define splits to print: Train, Val, Test (Hidden)
    splits_to_show = [('train', 'TRAIN'), ('test', 'VALIDATION')] # Map internal 'test' to 'VALIDATION' label
    if has_hidden:
        splits_to_show.append(('hidden', 'TEST (HIDDEN)'))

    for split_key, split_label in splits_to_show:
        print(f"\n📊 {split_label} METRICS:")
        print(f"{'Metric':<15} {'Filtered (Mean±Std)':<25} {'Noisy (Mean±Std)':<25} {'Improvement':<12}")
        print("-" * 77)

        for metric in key_metrics:
            filtered_mean = summary_data.get(f'{metric}_{split_key}_filtered_mean', 0.0)
            filtered_std = summary_data.get(f'{metric}_{split_key}_filtered_std', 0.0)

            # For hidden, we might not have noisy baseline computed above in this loop structure easily
            # Fallback to test noisy mean for comparison context or 0 if missing
            if split_key == 'hidden':
                 noisy_mean = summary_data.get(f'{metric}_test_noisy_mean', 0.0) # Approximation/Context
                 noisy_std = summary_data.get(f'{metric}_test_noisy_std', 0.0)
            else:
                 noisy_mean = summary_data.get(f'{metric}_{split_key}_noisy_mean', 0.0)
                 noisy_std = summary_data.get(f'{metric}_{split_key}_noisy_std', 0.0)

            # Calculate improvement
            if metric in ['pcorr', 'psnr', 'spearman', 'snr']:  # Higher is better
                improvement = ((filtered_mean - noisy_mean) / noisy_mean * 100) if noisy_mean != 0 else 0
            else:  # Lower is better (rmse, mse, dtw, lsd)
                improvement = ((noisy_mean - filtered_mean) / noisy_mean * 100) if noisy_mean != 0 else 0

            improvement_str = f"+{improvement:.1f}%" if improvement > 0 else f"{improvement:.1f}%"

            print(f"{metric:<15} {filtered_mean:.4f}±{filtered_std:.4f}      {noisy_mean:.4f}±{noisy_std:.4f}      {improvement_str:<12}")

    # Save comprehensive results in DAE format
    if save_results and working_dir:
        # Use new DAE-format output function
        summary_df = create_dae_format_summary(all_fold_results, summary_data, filter_name)
        summary_file = os.path.join(working_dir, f'{filter_name}_baseline_summary.xlsx')
        summary_df.to_excel(summary_file, index=False)
        print(f"\n💾 DAE-format results saved to: {summary_file}")

        # Save filter parameters
        params_file = os.path.join(working_dir, f'{filter_name}_parameters.json')
        import json
        with open(params_file, 'w') as f:
            json.dump({
                'filter_name': filter_name,
                'filter_params': FILTER_PARAMS[filter_name],
                'noise_type': noise_type,
                'seed_split': seed_split,
                'num_folds': num_folds
            }, f, indent=2)

        print(f"💾 Filter parameters saved to: {params_file}")

    print(f"\n✅ {filter_name.upper()} filter baseline completed!")
    return all_fold_results, summary_data


def main():
    parser = argparse.ArgumentParser(description='Traditional Filter Baselines with Signal Saving')

    parser.add_argument('--filter_type', required=True,
                       choices=['butterworth', 'median', 'savgol', 'gaussian', 'wavelet',
                               'wiener', 'hybrid_savgol_median', 'tv_l1', 'adaptive_notch', 'none'],
                       help='Type of filter to apply (use "none" for unfiltered noisy signal baseline)')
    parser.add_argument('--noise_type', default='allmixed',
                       choices=['gaussian', 'spike', 'bwander', 'powerline', 'allmixed'],
                       help='Type of noise to filter')
    parser.add_argument('--seed_split', type=int, default=29,
                       help='Cross-validation seed (same as deep learning models)')
    parser.add_argument('--num_folds', type=int, default=4,
                       help='Number of cross-validation folds')
    parser.add_argument('--working_dir', type=str,
                       help='Working directory for outputs (e.g., /path/to/filter_baseline_butterworth)')
    parser.add_argument('--save_results', action='store_true', default=True,
                       help='Save results and signals')
    parser.add_argument('--skip_slow_metrics', action='store_true', default=False,
                       help='Skip slow DTW and LSD metrics for faster execution')
    parser.add_argument('--exclude_patients_file', type=str, default=None,
                       help='CSV file containing list of patients to exclude')

    # Filter-specific parameter overrides
    parser.add_argument('--tv_weight', type=float, default=None,
                       help='TV-L1 regularization weight (overrides default 0.05)')
    parser.add_argument('--wavelet_type', type=str, default=None,
                       help='Wavelet type (overrides default db4)')
    parser.add_argument('--wavelet_levels', type=int, default=None,
                       help='Wavelet decomposition levels (overrides default 5)')
    parser.add_argument('--wavelet_threshold_mode', type=str, default=None,
                       choices=['soft', 'hard'],
                       help='Wavelet threshold mode (overrides default soft)')
    parser.add_argument('--wavelet_threshold_method', type=str, default=None,
                       choices=['sure', 'visushrink'],
                       help='Wavelet threshold method (overrides default sure)')
    parser.add_argument('--wiener_kernel', type=int, default=None,
                       help='Wiener filter kernel size (overrides default 15)')

    args = parser.parse_args()

    # Create default working directory if not specified
    if not args.working_dir:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.working_dir = os.path.join(script_dir, f'filter_baseline_{args.filter_type}_{args.noise_type}')

    print(f"🎯 Filter Baseline Analysis")
    print(f"Filter: {args.filter_type}")
    print(f"Noise: {args.noise_type}")
    print(f"Output: {args.working_dir}")

    # Apply parameter overrides if provided
    filter_params = FILTER_PARAMS[args.filter_type].copy()

    if args.filter_type == 'tv_l1' and args.tv_weight is not None:
        filter_params['weight'] = args.tv_weight
        print(f"🔧 TV-L1 weight override: {args.tv_weight}")

    if args.filter_type == 'wavelet':
        if args.wavelet_type is not None:
            filter_params['wavelet'] = args.wavelet_type
            print(f"🔧 Wavelet type override: {args.wavelet_type}")
        if args.wavelet_levels is not None:
            filter_params['levels'] = args.wavelet_levels
            print(f"🔧 Wavelet levels override: {args.wavelet_levels}")
        if args.wavelet_threshold_mode is not None:
            filter_params['threshold_mode'] = args.wavelet_threshold_mode
            print(f"🔧 Wavelet threshold mode override: {args.wavelet_threshold_mode}")
        if args.wavelet_threshold_method is not None:
            filter_params['threshold_method'] = args.wavelet_threshold_method
            print(f"🔧 Wavelet threshold method override: {args.wavelet_threshold_method}")

    if args.filter_type == 'wiener' and args.wiener_kernel is not None:
        filter_params['mysize'] = args.wiener_kernel
        print(f"🔧 Wiener kernel override: {args.wiener_kernel}")

    print(f"📋 Using parameters: {filter_params}")

    # Run baseline analysis
    fold_results, summary = run_filter_baseline_with_params(
        filter_name=args.filter_type,
        filter_params=filter_params,
        noise_type=args.noise_type,
        seed_split=args.seed_split,
        num_folds=args.num_folds,
        working_dir=args.working_dir,
        save_results=args.save_results,
        skip_slow_metrics=args.skip_slow_metrics,
        exclude_patients_file=args.exclude_patients_file
    )

    print(f"\n🎉 Analysis complete!")
    print(f"📂 Results saved to: {args.working_dir}")
    print(f"🔍 Denoised signals saved in: {os.path.join(args.working_dir, 'denoised_signals')}")
    print(f"📊 Use these signals for comparison with VAE/DAE/LUNet/DRRN models!")


if __name__ == "__main__":
    main()

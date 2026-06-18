"""MAP signal data-prep, noise-injection, APD and metric helpers for the BASELINE
(VAE) track -- the ``mapf`` module used by the baseline loaders and downstream APD.

Faithful migration of ``MAP_VAE/MAP_functions.py``. The function bodies are copied
verbatim. The only mechanical edits are: (1) this module docstring; (2) inside
``get_MAP_vent_data`` the in-function imports were rewritten to package paths
(``retrieve_ventMAP`` -> :mod:`cardiac_map_diffusion.data.retrieve_ventMAP`, and the
``import MAP_functions as mapf`` self-reference -> a self-import of this very module
``cardiac_map_diffusion.metrics.map_functions_baselines``); and (3) the cohort and
APD-annotation file paths there were de-hard-coded through
:mod:`cardiac_map_diffusion.paths`. The ``os.chdir``/``MAP_autoencoder`` module
directory literals have no ``paths`` accessor and are kept with a ``# TODO(paths)``
marker. No array math, hyperparameters, RNG seeding, thresholds or metric formulas
have been changed (note: ``compute_rmse`` returns the pooled MSE without a square
root, exactly as in the source).
"""

# -*- coding: utf-8 -*-
"""
MAP FUNCTIONS
Created on Mon Mar  6 11:01:09 2023

@author: sruip
"""
import os
import numpy             as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas            as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import random

from cardiac_map_diffusion import paths  # de-hardcoded cohort paths (migration)

# Fix random seeds
random.seed(17)  # Random module
np.random.seed(17)  # NumPy
from sklearn import svm # model SVM
from sklearn.ensemble import GradientBoostingRegressor # model GB

def dict_to_dataF_MAPv(MAP_vent, CORRECT_10064):
    MAP_patID = list(MAP_vent.keys())
    for ID in MAP_patID:
        if ID[-1] == 'B':
            MAP_vent[ID[:-2]] = np.concatenate((MAP_vent[ID[:-2]],
                                                MAP_vent[ID]))
            del MAP_vent[ID]
        if CORRECT_10064: # correct inverted MAPs in this patient
            if ID == '10064':
                map_v10064 = MAP_vent['10064']
                map_v10064[-37:,:] = -map_v10064[-37:,:] # correct inverted MAPs
                MAP_vent['10064'] = map_v10064
    del MAP_vent['10012']
    
    # % Transform into a DataFrame
    pat_vent_IDs = list(MAP_vent.keys())
    MAP_vent = pd.DataFrame({'pat_ID':pat_vent_IDs,
                       'EGM':list(MAP_vent.values())})
    #list_EGMs = list(MAP_vent['EGM'])
    #array_EGMs = np.concatenate(list_EGMs, axis=0)
    #list_EGMs_single = list(array_EGMs)
    
    return MAP_vent, pat_vent_IDs

def dataF_to_dataFsingle(MAP_vent):
    listIDs = MAP_vent['pat_ID']
    list_numMAP = [MAP_mat.shape[0] for MAP_mat in MAP_vent['EGM']]
    # Index the numbers in each patient
    idx_pp = []
    for num in list_numMAP:
        idx_pp.append(list(range(num)))
    idx_pp_fl = [item for sublist in idx_pp for item in sublist]
    
    MAP_vent_complete = pd.DataFrame({
        'pat_ID' : listIDs
    })
    MAP_vent_complete = MAP_vent_complete.\
        loc[MAP_vent_complete.index.repeat(list_numMAP)].reset_index(drop=True)
    list_EGMs = list(MAP_vent['EGM'])
    array_EGMs = np.concatenate(list_EGMs, axis=0)
    list_EGMs_single = list(array_EGMs)
    MAP_vent_complete['idx_pp'] = idx_pp_fl
    MAP_vent_complete['MAP_segments'] = list_EGMs_single
    return MAP_vent_complete

def introduce_gaussian_noise(loc=0, scale=0.1, MAP_array=[]):
    if MAP_array.ndim == 1:
        MAP_array = MAP_array.reshape(-1, 1)
    noise_mat = np.random.normal(loc=loc, scale=scale, size=[MAP_array.shape[0],
                                                             MAP_array.shape[1]])
    noisy_MAPs = MAP_array + noise_mat
    return noisy_MAPs

def introduce_truncation_noise(array, option='end', percent=0.8, var=0.05):
    # Check that the array has at least two dimensions - e.g. not (370,)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    # Each of the rows should be an MAP, thus create one percentage for each
    # MAP signal.
    percentages = np.random.uniform(percent-var, percent+var, array.shape[0])
    square_func = np.zeros((array.shape[0], array.shape[1]), dtype=int)
    
    for i in range(array.shape[0]):
        n_ones = int(percentages[i]*array.shape[1])
        n_zeros = array.shape[1] - n_ones
        if option == "end":
            square_func[i, :] = np.concatenate((np.ones(n_ones, dtype=int), 
                                                 np.zeros(n_zeros, dtype=int)))
        elif option == "beginning":
            square_func[i, :] = np.concatenate((np.zeros(n_zeros, dtype=int),
                                                np.ones(n_ones, dtype=int)))
    array_trunc = array * square_func
    
    return array_trunc
    
def generate_random_freqs(min_freq, max_freq, min_sins, max_sins):
    n = random.randint(min_sins, max_sins)
    random_list = [round(random.uniform(min_freq, max_freq), 4) for _ in range(n)]
    return random_list


def find_noise_params(noise_type):
    if noise_type == 'truncation':
        noise_params = [0.7, 0.05, 'end']
    elif noise_type == 'gaussian':
        noise_params = [0.05]
    elif noise_type == 'spike':
        noise_params = []
    elif noise_type == 'bwander':
        noise_params = [0.01, 0.3, 1, 4, 1]
    elif noise_type == 'powerline':
        noise_params = [60, 60, 1, 1, 0.1]
    elif noise_type == 'ep':
        noise_params = [0.25]
    elif noise_type == 'allmixed':
        noise_ids = [1, 2, 3, 4, 5, 6]
        min_number_noises = 1
        max_number_noises = 6
        noise_params = [noise_ids, min_number_noises, max_number_noises]
    return noise_params

def simulate_baseline_wander_single(signal, sample_frequency=1000, min_freq=0.01,
                             max_freq=0.3, min_sins=1, max_sins=4):
    """
    Simulates baseline wander and adds it to the original signal.
    
    Args:
        A: Amplitude of the noise
        signal (numpy.ndarray): The original signal.
        sample_frequency (int): The sample frequency in Hz.
        frequencies (list): List of frequencies for baseline wander simulation.
        min_freq = 0.01  # minimum frequency
        max_freq = 0.3  # maximum frequency
        min_sins = 1  # Minimum number of sinusoidal frequencies
        max_sins = 4  # Maximum number of sinusoidal frequencies
    
    Returns:
        numpy.ndarray: The signal with baseline wander added.
    """
    num_samples = len(signal)
    time = np.arange(num_samples) / sample_frequency
    
    baseline_wander = np.zeros(num_samples)
    frequencies = generate_random_freqs(min_freq, max_freq, min_sins, max_sins)
    for freq in frequencies:
        A = random.uniform(0.5, 1)
        phase = np.random.uniform(0, 2 * np.pi)  # Random phase between 0 and 2*pi
        wander = A*np.sin(2 * np.pi * freq * time + phase)
        baseline_wander += wander
    baseline_wander = baseline_wander - np.mean(baseline_wander)
    signal_with_wander = signal + baseline_wander
    
    return signal_with_wander, baseline_wander

def introduce_baseline_wander(matrix, sample_frequency=1000, min_freq=0.01,
                             max_freq=0.3, min_sins=1, max_sins=4,
                             max_amplitude=1):
    """
    Simulates baseline wander and adds it to each row of the matrix.
    
    Args:
        matrix (numpy.ndarray): The original matrix with rows representing different signals.
        sample_frequency (int): The sample frequency in Hz.
        min_freq (float): Minimum frequency for baseline wander simulation.
        max_freq (float): Maximum frequency for baseline wander simulation.
        min_sins (int): Minimum number of sinusoidal frequencies.
        max_sins (int): Maximum number of sinusoidal frequencies.
    
    Returns:
        numpy.ndarray: The matrix with baseline wander added to each row.
        numpy.ndarray: The baseline wander added to each row separately.
    """
    # Check that the array has at least two dimensions - e.g. not (370,)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
        
    num_rows, num_samples = matrix.shape
    time = np.arange(num_samples) / sample_frequency
    
    baseline_wander_matrix = np.zeros_like(matrix)
    signal_with_wander_matrix = np.zeros_like(matrix)
    
    for i in range(num_rows):
        signal = matrix[i]
        baseline_wander = np.zeros(num_samples)
        frequencies = generate_random_freqs(min_freq, max_freq, min_sins, max_sins)
        
        for freq in frequencies:
            A = random.uniform(0.3*max_amplitude, 1*max_amplitude)
            phase = np.random.uniform(0, 2 * np.pi)  # Random phase between 0 and 2*pi
            wander = A * np.sin(2 * np.pi * freq * time + phase)
            baseline_wander += wander
        
        baseline_wander = baseline_wander - np.mean(baseline_wander)
        signal_with_wander = signal + baseline_wander
        
        baseline_wander_matrix[i] = baseline_wander
        signal_with_wander_matrix[i] = signal_with_wander
    
    return signal_with_wander_matrix


def introduce_spike_noise(signals):
    """
    Introduces spike noise to each signal in the input matrix.
    
    Args:
        signals (array-like): The matrix of signals, where each row represents a separate signal.
        
    Returns:
        tuple: A tuple containing the noisy signals matrix and the noise matrix.
    """
    # Check that the array has at least two dimensions - e.g. not (370,)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)
    # Each of the rows should be an MAP, thus create one percentage for each
    # MAP signal.
    n, m = signals.shape  # Get the dimensions of the matrix (n: number of signals, m: signal length)
    noisy_signals = np.copy(signals)
    noise = np.zeros((n, m))  # Initialize the noise matrix
    
    for i in range(n):
        spike_index = np.random.randint(2, 40)  # Random index between 2 and m-1 for each signal
        
        # Define A for each signal
        initial_value = noisy_signals[i, 0]
        maximum_value = np.max(noisy_signals[i])
        value_range = maximum_value - initial_value
        lower_bound = initial_value + 0.1 * value_range
        upper_bound = initial_value + 0.8 * value_range
        
        A = np.random.uniform(lower_bound, upper_bound)  # Generate random A for each signal
        
        noisy_signals[i, spike_index] = A  # Set spike noise for each signal
        noise[i, spike_index] = A
    
    return noisy_signals


def introduce_several_noises_singleMAP(list_noises, MAP_array, arrays):
    # MAP_array is NOT normalized

    # 1: Truncation
    # 2: Baseline Wander
    # 3: Powerline Interference
    # 4: Gaussian Noise
    # 5: Spike noise
    # 6: Electrophysiological Noise

    if MAP_array.ndim == 1:
        array = MAP_array.reshape(1, -1)
    # Each of the rows should be an MAP, thus create one percentage for each
    # MAP signal.

    MAP_noisy = MAP_array

    if list_noises[0] == 1:
        MAP_noisy = introduce_truncation_noise(MAP_noisy, option='end',
                                               percent=0.8, var=0.05)
        MAP_noisy = normalize_EGM_array(MAP_noisy)
        num_noises_left = len(list_noises) - 1
        correction = 1
    else:
        MAP_noisy = normalize_EGM_array(MAP_noisy)
        num_noises_left = len(list_noises)
        correction = 0

    for i in range(num_noises_left):
        if list_noises[i + correction] == 2:
            MAP_noisy = introduce_baseline_wander(MAP_noisy)
        elif list_noises[i + correction] == 3:
            MAP_noisy = introduce_baseline_wander(MAP_noisy, min_freq=60,
                                                  max_freq=60, min_sins=1, max_sins=1,
                                                  max_amplitude=0.1)
        elif list_noises[i + correction] == 4:
            MAP_noisy = introduce_gaussian_noise(loc=0, scale=0.1, MAP_array=MAP_noisy)
        elif list_noises[i + correction] == 5:
            MAP_noisy = introduce_spike_noise(MAP_noisy)
        elif list_noises[i + correction] == 6:
            MAP_noisy = introduce_epnoise(arrays, MAP_noisy, noise_boundary=0.25)

    return MAP_noisy

def generate_unique_list_noises(noise_ids, min_number_noises, max_number_noises):
    """
    Generate a list of unique random noises from the given list of noise_ids,
    with a random length between min_number_noises and max_number_noises.

    Args:
        noise_ids (list): The list of possible noise values.
        min_number_noises (int): The minimum number of noises to generate.
        max_number_noises (int): The maximum number of noises to generate.

    Returns:
        list: A list of unique random noises.

    Raises:
        ValueError: If the provided min and max values are invalid.
    """
    # Check if the provided min and max values are valid
    if not (1 <= min_number_noises <= max_number_noises <= len(noise_ids)):
        raise ValueError("Invalid min_number_noises or max_number_noises values.")

    # Generate a random length between min_number_noises and max_number_noises
    total_possible_noises = random.randint(min_number_noises, max_number_noises)

    # Create a set to store the generated numbers
    list_noises = set()

    # Generate random numbers until the desired length is reached
    while len(list_noises) < total_possible_noises:
        # Generate a random index within the range of the noise_ids list
        random_index = random.randint(0, len(noise_ids) - 1)

        # Add the value at the random index to the set if it hasn't been generated before
        list_noises.add(noise_ids[random_index])

    # Convert the set to a list and return the result
    return list(list_noises)
def introduce_several_noises(MAP_matrix, noise_ids=[1, 2, 3, 4, 5, 6],
                             min_number_noises=1, max_number_noises=6, arrays=[]):
    # MAP_array is NOT normalized

    # 1: Truncation
    # 2: Baseline Wander
    # 3: Powerline Interference
    # 4: Gaussian Noise
    # 5: Spike noise
    # 6: Electrophysiological Noise
    MAP_matrix = np.array(MAP_matrix)
    if MAP_matrix.ndim == 1:
        MAP_matrix = MAP_matrix.reshape(1, -1)
    num_rows, num_samples = MAP_matrix.shape
    MAP_matrix_noisy = np.zeros_like(MAP_matrix)

    for i in range(MAP_matrix.shape[0]):
        list_noises = generate_unique_list_noises(noise_ids, min_number_noises,
                                                  max_number_noises)
        MAP_matrix_noisy[i] = introduce_several_noises_singleMAP(list_noises,
                                                          MAP_matrix[i], arrays)
    return MAP_matrix_noisy

def compute_pearson_corr(array1, array2, mode='total'):
    pcorr_list = []
    for i in range(array1.shape[0]):
        pcorr, _ = pearsonr(array1[i, :], array2[i, :])
        pcorr_list.append(pcorr)
    if mode=='individual':
        return pcorr_list
    elif mode=='total':
        return np.mean(pcorr_list)

def compute_rmse(array1, array2, mode='total'):
    mse_list = []
    for i in range(array1.shape[0]):
        mse_list.append(mean_squared_error(array1, array2))
    if mode=='individual':
        return mse_list
    elif mode=='total':
        return np.mean(mse_list)
    
def signaltonoise(array, axis=1, ddof=0, mode='total'):
    array = np.asanyarray(array)
    m = array.mean(axis)
    sd = array.std(axis=axis, ddof=ddof)
    if mode == 'individual':
        return np.where(sd == 0, 0, m/sd)
    elif mode == 'total':
        return np.mean(np.where(sd == 0, 0, m/sd))
    
def compute_psnr(array1, array2, mode='total'):
    pnsr_list = []
    for i in range(array1.shape[0]):
        signal1 = array1[i, :]
        signal2 = array2[i, :]
        mse = np.mean((signal1 - signal2) ** 2)
        max_value = np.max([signal1.max(), signal2.max()])
        pnsr = 20 * np.log10(max_value / np.sqrt(mse))
        pnsr_list.append(pnsr)
    if mode=='individual':
        return pnsr_list
    elif mode=='total':
        return np.mean(pnsr_list)

# ===================================================================
# ADDITIONAL METRICS FUNCTIONS - Added by GitHub Copilot for enhanced analysis
# ===================================================================

def compute_mse(array1, array2, mode='total'):
    """
    Compute Mean Squared Error (MSE) between two arrays.
    Added by GitHub Copilot to complement existing metrics.
    
    Args:
        array1: Original signal array
        array2: Reconstructed/predicted signal array  
        mode: 'individual' returns list of MSE per signal, 'total' returns average
    
    Returns:
        MSE value(s)
    """
    mse_list = []
    for i in range(array1.shape[0]):
        mse = np.mean((array1[i, :] - array2[i, :]) ** 2)
        mse_list.append(mse)
    if mode=='individual':
        return mse_list
    elif mode=='total':
        return np.mean(mse_list)

def compute_spearman_corr(array1, array2, mode='total'):
    """
    Compute Spearman's rank correlation coefficient (rho) between two arrays.
    Added by GitHub Copilot to complement Pearson correlation.
    
    Args:
        array1: Original signal array
        array2: Reconstructed/predicted signal array
        mode: 'individual' returns list of rho per signal, 'total' returns average
        
    Returns:
        Spearman correlation coefficient(s)
    """
    from scipy.stats import spearmanr
    spearman_list = []
    for i in range(array1.shape[0]):
        rho, _ = spearmanr(array1[i, :], array2[i, :])
        spearman_list.append(rho)
    if mode=='individual':
        return spearman_list
    elif mode=='total':
        return np.mean(spearman_list)

def compute_snr(array1, array2, mode='total'):
    """
    Compute Signal-to-Noise Ratio (SNR) between original and reconstructed signals.
    Added by GitHub Copilot for signal quality assessment.
    
    Args:
        array1: Original signal array (treated as signal)
        array2: Reconstructed signal array
        mode: 'individual' returns list of SNR per signal, 'total' returns average
        
    Returns:
        SNR value(s) in dB
    """
    snr_list = []
    for i in range(array1.shape[0]):
        signal_power = np.mean(array1[i, :] ** 2)
        noise_power = np.mean((array1[i, :] - array2[i, :]) ** 2)
        # Avoid division by zero
        if noise_power == 0:
            snr_db = float('inf')
        else:
            snr_db = 10 * np.log10(signal_power / noise_power)
        snr_list.append(snr_db)
    if mode=='individual':
        return snr_list
    elif mode=='total':
        return np.mean(snr_list)

def compute_dtw(array1, array2, mode='total'):
    """
    Compute Dynamic Time Warping (DTW) distance between two arrays.
    Added by GitHub Copilot for temporal alignment-aware comparison.
    
    Args:
        array1: Original signal array
        array2: Reconstructed/predicted signal array
        mode: 'individual' returns list of DTW distances, 'total' returns average
        
    Returns:
        DTW distance(s)
    """
    try:
        from dtaidistance import dtw
        dtw_available = True
    except ImportError:
        # Fallback to simple implementation if dtaidistance not available
        dtw_available = False
    
    dtw_list = []
    for i in range(array1.shape[0]):
        if dtw_available:
            distance = dtw.distance(array1[i, :], array2[i, :])
        else:
            # Simple DTW implementation fallback
            distance = _simple_dtw(array1[i, :], array2[i, :])
        dtw_list.append(distance)
    
    if mode=='individual':
        return dtw_list
    elif mode=='total':
        return np.mean(dtw_list)

def _simple_dtw(x, y):
    """
    Simple DTW implementation fallback.
    Added by GitHub Copilot as fallback when dtaidistance is not available.
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

def compute_lsd(array1, array2, mode='total', window='hann', nperseg=256):
    """
    Compute Log-Spectral Distance (LSD) with Hanning window between two arrays.
    Added by GitHub Copilot for frequency domain comparison.
    
    Args:
        array1: Original signal array
        array2: Reconstructed/predicted signal array
        mode: 'individual' returns list of LSD per signal, 'total' returns average
        window: Window function for spectral analysis (default: 'hann')
        nperseg: Length of each segment for spectral analysis
        
    Returns:
        LSD value(s) in dB
    """
    from scipy import signal as scipy_signal
    
    lsd_list = []
    for i in range(array1.shape[0]):
        signal1 = array1[i, :]
        signal2 = array2[i, :]
        
        # Compute power spectral densities
        f1, psd1 = scipy_signal.welch(signal1, window=window, nperseg=min(nperseg, len(signal1)//4))
        f2, psd2 = scipy_signal.welch(signal2, window=window, nperseg=min(nperseg, len(signal2)//4))
        
        # Avoid log of zero by adding small epsilon
        eps = 1e-10
        psd1 = np.maximum(psd1, eps)
        psd2 = np.maximum(psd2, eps)
        
        # Compute log-spectral distance
        lsd = np.sqrt(np.mean((10 * np.log10(psd1) - 10 * np.log10(psd2)) ** 2))
        lsd_list.append(lsd)
    
    if mode=='individual':
        return lsd_list
    elif mode=='total':
        return np.mean(lsd_list)

def compute_nmae(array1, array2, norm='mean', mode='total'):
    """
    Compute Normalized Mean Absolute Error (NMAE) between two arrays.
    Added by GitHub Copilot with multiple normalization options.
    
    Args:
        array1: Original signal array
        array2: Reconstructed/predicted signal array
        norm: Normalization method - 'range', 'l1', or 'mean'
              'range': (1 / (N * range(x))) * ||x - x_hat||_1
              'l1':    (1 / ||x||_1) * ||x - x_hat||_1  
              'mean':  (1 / N) * ||x - x_hat||_1 (i.e., MAE)
        mode: 'individual' returns list of NMAE per signal, 'total' returns average
        
    Returns:
        NMAE value(s)
    """
    nmae_list = []
    for i in range(array1.shape[0]):
        signal1 = array1[i, :]
        signal2 = array2[i, :]
        
        # Compute absolute error
        abs_error = np.abs(signal1 - signal2)
        mae = np.sum(abs_error)  # Sum of absolute errors
        N = len(signal1)
        
        if norm == 'range':
            # Normalize by range
            signal_range = np.max(signal1) - np.min(signal1)
            # Avoid division by zero
            if signal_range == 0:
                nmae = 0.0
            else:
                nmae = mae / (N * signal_range)
        elif norm == 'l1':
            # Normalize by L1 norm of original signal
            l1_norm = np.sum(np.abs(signal1))
            # Avoid division by zero
            if l1_norm == 0:
                nmae = 0.0
            else:
                nmae = mae / l1_norm
        elif norm == 'mean':
            # Normalize by number of samples (standard MAE)
            nmae = mae / N
        else:
            raise ValueError(f"Unknown normalization method: {norm}. Use 'range', 'l1', or 'mean'.")
        
        nmae_list.append(nmae)
    
    if mode=='individual':
        return nmae_list
    elif mode=='total':
        return np.mean(nmae_list)

def acquire_APD_annotations(file_APD_name, file_APD_path, BEAT=False, 
                            MERGE=True):
    path_APD = os.path.join(file_APD_path, file_APD_name) # complete path
    APD_xl = pd.ExcelFile(path_APD)
    pat_ids = APD_xl.sheet_names # retrieve sheet names in xlsx
    
    pat_ids_listpp = [] # initialise lists
    beats_listpp = []
    APD30_listpp = []
    APD60_listpp = []
    APD90_listpp = []
    
    for pat_id in pat_ids: # for each sheet (there may be differences among them)
        APD_singlep = pd.read_excel(path_APD, sheet_name=pat_id)
        pat_ids_listpp.append([pat_id] * (len(APD_singlep)-2)) # pat_id repeated
        APD30_listpp.append(list(APD_singlep['APD30'])[1:-1])
        APD60_listpp.append(list(APD_singlep['APD60'])[1:-1])
        APD90_listpp.append(list(APD_singlep['APD90'])[1:-1])
        if BEAT:
            beats_listpp.append(list(APD_singlep['Beat #'])[1:-1]) # number of beats
        
    # flatten the lists
    APD30_list = [item for sublist in APD30_listpp for item in sublist]
    APD60_list = [item for sublist in APD60_listpp for item in sublist]
    APD90_list = [item for sublist in APD90_listpp for item in sublist]
    pat_ids_list = [item for sublist in pat_ids_listpp for item in sublist]
    if BEAT:
        beats_list = [item for sublist in beat_listpp for item in sublist]
    
    # construct the dataframe
    APD_df = pd.DataFrame() # initalise dataframe
    APD_df['pat_ID'] = pat_ids_list
    APD_df['APD30'] = APD30_list
    APD_df['APD60'] = APD60_list
    APD_df['APD90'] = APD90_list
    if BEAT: # if we want to include the beat count, for reference to original data
        APD_df['beat_num'] = beats_list
    if MERGE: # if we want to merge the '_B' recordings for each patient
        APD_df['pat_ID'] = APD_df['pat_ID'].str.replace('_B','') 
        
    pat_IDs = list(APD_df['pat_ID'].unique())
    APD30pp = []
    APD60pp = []
    APD90pp = []
    for ID in pat_IDs:
        APD30pp.append(list(APD_df[APD_df['pat_ID']==ID]['APD30']))
        APD60pp.append(list(APD_df[APD_df['pat_ID']==ID]['APD60']))
        APD90pp.append(list(APD_df[APD_df['pat_ID']==ID]['APD90']))
    APD_df_pp = pd.DataFrame({
        "pat_ID": pat_IDs,
        "APD30": APD30pp,
        "APD60": APD60pp,
        "APD90": APD90pp
        })
    return APD_df, APD_df_pp

def plot_distribution_MAPpp(MAP_vent, mode='latex', dpi=1200):
    if mode=='normal':
        mpl.rcParams.update(mpl.rcParamsDefault)
    elif mode=='latex':
        mpl.rcParams.update({
                    "text.usetex": True,
                    "font.family": "sans-serif",
                })
    list_EGMs = list(MAP_vent['EGM'])
    MAP_vent_pp = [list_EGMs[i].shape[0] for i in range(len(list_EGMs))]
    fig, ax = plt.subplots(figsize=(6, 5), dpi=dpi)
    bins = np.arange(0, 500, 20) # fixed bin size
    plt.xlim([min(MAP_vent_pp)-50, max(MAP_vent_pp)])
    plt.hist(MAP_vent_pp, bins=bins, alpha=0.5, edgecolor='black')
    plt.title('Ventricular MAP segments per patient')
    plt.xlabel('Number of MAP segments', fontsize=14)
    plt.ylabel('Patients', fontsize=14)
    plt.grid(b=bool, which='major', axis='y', color='gray', linestyle='--', 
             linewidth=0.5)
    plt.show()
    
def plot_single_MAP(MAP_single, mode='latex', dpi=600):
    if mode=='normal':
        mpl.rcParams.update(mpl.rcParamsDefault)
    elif mode=='latex':
        mpl.rcParams.update({
                    "text.usetex": True,
                    "font.family": "sans-serif",
                })
    fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
    plt.plot(MAP_single, color='k', linewidth=1.7)
    plt.ylabel('Voltage Amplitude [mV]', fontfamily = 'Times New Roman', size=15)
    plt.xlabel('Time [msec]', fontfamily = 'Times New Roman', size=15)
    plt.grid(b=bool, which='major', axis='both', color='gray', linestyle='--', 
             linewidth=0.6)
    plt.minorticks_on()
    plt.grid(b=bool, which='minor', axis='both', color='gray', linestyle='--', 
             linewidth=0.4)
    
def compute_corr_intrapat(MAP_vent, pat_vent_IDs, PLOT=True, form='intra-patient', 
                          mode='latex', dpi=600):
    list_EGMs = list(MAP_vent['EGM'])
    corr_intra = []
    for k in range(len(list_EGMs)):
        r = []
        for i in range(list_EGMs[k].shape[0]):
            for j in range(list_EGMs[k].shape[0]):
                if i<j:
                    r.append(np.corrcoef(list_EGMs[k][i,:], list_EGMs[k][j,:])[0,1])
        corr_intra.append(r)
    if PLOT:
        plot_corr(corr_intra, pat_vent_IDs, form='intra-patient', mode='latex', 
                  dpi=600)
    return corr_intra

def plot_corr(corr_intra_MAPv, pat_vent_IDs, form='intra-patient', 
              mode='latex', dpi=600):
    if mode=='normal':
        mpl.rcParams.update(mpl.rcParamsDefault)
    elif mode=='latex':
        mpl.rcParams.update({
                    "text.usetex": True,
                    "font.family": "sans-serif",
                })
    fig, ax = plt.subplots(figsize=(13, 6), dpi=dpi)
    plt.boxplot(corr_intra_MAPv)
    
    if form=='intra-patient':
        plt.xticks(ticks=list(np.array(list(range(len(pat_vent_IDs))))+1), 
                      labels=pat_vent_IDs, rotation=90)
    
    plt.xlabel('Patient ID', size=18)
    plt.ylabel('Pearson Correlation', size=18)
    plt.title(f'Average Pear. Correlation {form}', size=18)

def compute_corr_interpat(MAP_vent):
    list_EGMs = list(MAP_vent['EGM'])
    corr_inter_MAPv = []
    corr_inter_pp = []
    corr_inter_pp_av = []
    for k1 in range(len(list_EGMs)):
        r = []
        corr_inter_pp_ = []
        corr_inter_pp_av_ = []
        for k2 in range(len(list_EGMs)):
            r_pp = []
            for i in range(list_EGMs[k1].shape[0]):
                for j in range(list_EGMs[k2].shape[0]):
                    if k1 != k2:
                        r.append(np.corrcoef(list_EGMs[k1][i,:], 
                                             list_EGMs[k2][j,:])[0,1])
                        r_pp.append(np.corrcoef(list_EGMs[k1][i,:], 
                                                list_EGMs[k2][j,:])[0,1])
                    if k1 == k2:
                        if i<j:
                            r_pp.append(np.corrcoef(list_EGMs[k1][i,:], 
                                                    list_EGMs[k2][j,:])[0,1])
            corr_inter_pp_.append(r_pp)
            corr_inter_pp_av_.append(np.mean(r_pp))
        corr_inter_MAPv.append(r)
        corr_inter_pp.append(corr_inter_pp_)
        corr_inter_pp_av.append(corr_inter_pp_av_)
    return corr_inter_MAPv, corr_inter_pp, corr_inter_pp_av

def get_APD_singleMAP(MAP_single, delay_depol, EXC1, PLOT, mode='latex', 
                      dpi=1200):
    peak_idx = np.argmax(MAP_single[:300]) # peak of the MAP
    if EXC1: # Exception1: higher plateau than depolarization peak
        peak_idx = np.argmax(MAP_single[:62]) # peak of the MAP
    APD_init = peak_idx + delay_depol # beginning of the depolarisation
    plateau = np.max(MAP_single[APD_init:300])
    plateau_idx = np.argmax(MAP_single[APD_init:]) + APD_init
    depolar_end_idx = np.argmin(MAP_single[plateau_idx:]) + plateau_idx
    depolar_end = MAP_single[depolar_end_idx]
    APD30_volt = 0.7 * (plateau - depolar_end) + depolar_end
    APD60_volt = 0.4 * (plateau - depolar_end) + depolar_end
    APD90_volt = 0.1 * (plateau - depolar_end) + depolar_end
    APD30_endpoint = (np.abs(MAP_single[plateau_idx:depolar_end_idx] -
                             APD30_volt)).argmin() + plateau_idx
    APD60_endpoint = (np.abs(MAP_single[plateau_idx:depolar_end_idx] - 
                             APD60_volt)).argmin() + plateau_idx
    APD90_endpoint = (np.abs(MAP_single[plateau_idx:depolar_end_idx] - 
                             APD90_volt)).argmin() + plateau_idx
    APD30 = APD30_endpoint - APD_init
    APD60 = APD60_endpoint - APD_init
    APD90 = APD90_endpoint - APD_init
    
    if PLOT:
        plot_MAP_APD(MAP_single, APD_init, APD30_volt, APD30_endpoint, APD60_volt, 
                 APD60_endpoint, APD90_volt, APD90_endpoint, plateau,
                 depolar_end, depolar_end_idx, mode=mode, dpi=dpi)
        
    return [APD30, APD60, APD90], [APD30_volt, APD60_volt, APD90_volt],\
           [APD30_endpoint, APD60_endpoint, APD90_endpoint],\
           [plateau, plateau_idx], [depolar_end, depolar_end_idx] 
           
           
def plot_MAP_APD(MAP_single, APD_init, APD30_volt, APD30_endpoint, APD60_volt, 
                 APD60_endpoint, APD90_volt, APD90_endpoint, plateau,
                 depolar_end, depolar_end_idx, mode='latex', dpi=1200):
    if mode=='normal':
        mpl.rcParams.update(mpl.rcParamsDefault)
    elif mode=='latex':
        mpl.rcParams.update({
                    "text.usetex": True,
                    "font.family": "sans-serif",
                    })
    fig, ax = plt.subplots(figsize=(5, 4), dpi=dpi)
    plt.plot(MAP_single, color='k', linewidth=1.7)
    plt.ylabel('Voltage Amplitude [mV]', fontsize=15)
    plt.xlabel('Time [msec]', fontsize=15)
    plt.grid(b=bool, which='major', axis='both', color='gray', 
             linestyle='--', linewidth=0.6)
    plt.minorticks_on()
    plt.grid(b=bool, which='minor', axis='both', color='gray', 
             linestyle='--', linewidth=0.4)
    plt.axvline(x=APD30_endpoint, ls='--', color='blue', alpha=0.7)
    plt.axhline(y=APD30_volt, ls='--', color='blue', alpha=0.5)
    plt.axvline(x=APD60_endpoint, ls='--', color='green', alpha=0.7)
    plt.axhline(y=APD60_volt, ls='--', color='green', alpha=0.5)
    plt.axvline(x=APD90_endpoint, ls='--', color='red', alpha=0.7)
    plt.axhline(y=APD90_volt, ls='--', color='red', alpha=0.5)
    plt.axvline(x=APD_init, color='k', alpha=0.4)
    plt.axhline(y=plateau, color='k', alpha=0.4)
    plt.axvline(x=depolar_end_idx, color='k', alpha=0.4)
    plt.axhline(y=depolar_end, color='k', alpha=0.4)

def get_APD_multipleMAP(MAP_matrix, delay_depol, EXC1):

    peak_idx = np.argmax(MAP_matrix[:,:300], axis=1) # peak of the MAP
                                                     # avoid second det. peak
    APD_init = peak_idx + delay_depol # beginning of the depolarisation
    plateau = []
    plateau_idx = []
    depolar_end_idx = []
    depolar_end = []
    
    for i in range(MAP_matrix.shape[0]):
        plateau.append(np.max(MAP_matrix[i, APD_init[i]:300])) # avoid 2nd peak
        plateau_idx.append(np.argmax(MAP_matrix[i, APD_init[i]:]) + APD_init[i])
        depolar_end_idx.append(np.argmin(MAP_matrix[i, plateau_idx[i]:]) +
                               plateau_idx[i])
        depolar_end.append(MAP_matrix[i, depolar_end_idx[i]])
    plateau = np.array(plateau)
    plateau_idx = np.array(plateau_idx)
    depolar_end_idx = np.array(depolar_end_idx)
    depolar_end = np.array(depolar_end)
    APD30_volt = 0.7 * (plateau - depolar_end) + depolar_end
    APD60_volt = 0.4 * (plateau - depolar_end) + depolar_end
    APD90_volt = 0.1 * (plateau - depolar_end) + depolar_end
    APD30_endpoint = []
    APD60_endpoint = []
    APD90_endpoint = []
    
    for i in range(MAP_matrix.shape[0]):
        APD30_endpoint.append((np.abs(MAP_matrix[i, plateau_idx[i]:depolar_end_idx[i]] - 
                                      APD30_volt[i])).argmin() + plateau_idx[i])
        APD60_endpoint.append((np.abs(MAP_matrix[i, plateau_idx[i]:depolar_end_idx[i]] - 
                                      APD60_volt[i])).argmin() + plateau_idx[i])
        APD90_endpoint.append((np.abs(MAP_matrix[i, plateau_idx[i]:depolar_end_idx[i]] - 
                                      APD90_volt[i])).argmin() + plateau_idx[i])
    APD30_endpoint = np.array(APD30_endpoint)
    APD60_endpoint = np.array(APD60_endpoint)
    APD90_endpoint = np.array(APD90_endpoint)
    APD30 = APD30_endpoint - APD_init
    APD60 = APD60_endpoint - APD_init
    APD90 = APD90_endpoint - APD_init
    
    return [APD30, APD60, APD90], [APD30_volt, APD60_volt, APD90_volt],\
           [APD30_endpoint, APD60_endpoint, APD90_endpoint],\
           [plateau, plateau_idx], [depolar_end, depolar_end_idx], APD_init
           
def plot_APDpp(MAP_vent_complete, label30='APD30', label60='APD60', 
               label90='APD_90'):
    mpl.rcParams.update(mpl.rcParamsDefault)
    fig, ax = plt.subplots(figsize=(18,8), dpi=1200)
    plt.suptitle('APD30 per patient')
    MAP_vent_complete.boxplot(column=label30, by='pat_ID', ax=ax, color='b')
    plt.xticks(rotation=90)
    plt.ylim((0, 370))
    plt.ylabel('APD (msec)')
    plt.xlabel('Patient ID')
    
    fig, ax = plt.subplots(figsize=(18,8), dpi=1200)
    plt.suptitle('APD30 per patient')
    MAP_vent_complete.boxplot(column=label60,by='pat_ID', ax=ax, color='g')
    plt.xticks(rotation=90)
    plt.ylim((0, 370))
    plt.ylabel('APD (msec)')
    plt.xlabel('Patient ID')
    
    fig, ax = plt.subplots(figsize=(18,8), dpi=1200)
    plt.suptitle('APD30 per patient')
    MAP_vent_complete.boxplot(column=label90, by='pat_ID', ax=ax, color ='r')
    plt.xticks(rotation=90)
    plt.ylim((0, 370))
    plt.ylabel('APD (msec)')
    plt.xlabel('Patient ID')
    plt.show()
 
def divide_train_test_pp(MAP_vent, cut_pat=33, label_outcome='APD30_gs'):
    MAP_vent = MAP_vent.sort_values(by='pat_ID')
    # Divide Training and test
    Train = MAP_vent_complete.loc[(MAP_vent_complete['pat_ID'].\
                                   isin(list(MAP_vent['pat_ID'])[:cut_pat]))]
    Test = MAP_vent_complete.loc[(MAP_vent_complete['pat_ID'].\
                                  isin(list(MAP_vent['pat_ID'])[cut_pat:]))]
    
    X_train = list(Train['MAP_segments'])
    X_test = list(Test['MAP_segments'])
    y_train = list(Train[label_outcome])
    y_test = list(Test[label_outcome])
    return X_train, y_train, X_test, y_test

def normalize_EGM_input(X_train, X_test):
    X_train_ = np.array(X_train)
    a = (X_train_.T - X_train_.min(axis=1)).T
    b = (X_train_.max(axis=1) - X_train_.min(axis=1))
    X_std_train = (a.T/b).T
    
    X_test_ = np.array(X_test)
    a = (X_test_.T - X_test_.min(axis=1)).T
    b = (X_test_.max(axis=1) - X_test_.min(axis=1))
    X_std_test = (a.T/b).T
    return X_std_train, X_std_test

def normalize_EGM_array(array):
    array_ = np.array(array)
    if array_.ndim == 1:
        array_ = array_.reshape(1, -1)
    a = (array_.T - array_.min(axis=1)).T
    b = (array_.max(axis=1) - array_.min(axis=1))
    array_std = (a.T/b).T
    
    return array_std

def normalize_y(y, perc_min=0.5, perc_max=99.5):
    y_std = (y - np.percentile(y, perc_min))/(np.percentile(y, perc_max) - np.percentile(y, 0.5))
    return y_std

def un_normalize_ystd(y, y_std, perc_min=0.5, perc_max=99.5):
    y_unstd = (np.percentile(y, perc_max) - np.percentile(y, 0.5)) * y_std + np.percentile(y, perc_min)
    return y_unstd
    

    
def normalize_outcome(MAP_vent_complete,  label_outcome, cut_segm=4540,
                      perc_min=0.5, perc_max=99.5):
    y = np.array(MAP_vent_complete[label_outcome])
    #y_std = (y - y.min())/(y.max() - y.min())
    y_std = (y - np.percentile(y, perc_min))/(np.percentile(y, perc_max) - np.percentile(y, 0.5))
    y_std_train = y_std[:cut_segm]
    y_std_test = y_std[cut_segm:]
    return y_std_train, y_std_test

def GBR_MAP(X_std_train, y_std_train, X_std_test, y_std_test, learning_rate=0.1,
            n_estimators=100, min_samples_split=2):
     regr = GradientBoostingRegressor(random_state=12, learning_rate=learning_rate,
            n_estimators=n_estimators, min_samples_split=min_samples_split)
     regr.fit(X_std_train, y_std_train)
     y_pred = regr.predict(X_std_test)
     RMSE = mean_squared_error(y_std_test, y_pred)
     return y_pred, RMSE
 
def SVM_MAP(X_std_train, y_std_train, X_std_test, y_std_test, kernel='rbf', 
            C=1, degree=3):
    regr = svm.SVR(C=C, kernel=kernel, degree=degree)
    regr.fit(X_std_train, y_std_train)
    y_pred = regr.predict(X_std_test)
    RMSE = mean_squared_error(y_std_test, y_pred)
    return y_pred, RMSE

def plot_rmse_vs_param(RMSE_test, RMSE_train, param, name_param, l_o, 
                       mode='latex', dpi=1200):
    if mode=='normal':
        mpl.rcParams.update(mpl.rcParamsDefault)
    elif mode=='latex':
        mpl.rcParams.update({
                    "text.usetex": True,
                    "font.family": "sans-serif",
                    })
    
    if l_o[:5] == 'APD30':
        color = 'b'
    elif l_o[:5] == 'APD60':
        color = 'g'
    elif l_o[:5] == 'APD90':
        color = 'r'
    else:
        color = 'black'
    
    fig, ax = plt.subplots(figsize=(10,6), dpi=dpi)
    
    plt.plot(param, RMSE_test, color=color)
    plt.plot(param, RMSE_train, color=color, linestyle='--')
    
    #ax.set_xscale('log')
    plt.ylabel('RMSE', fontsize=18)
    #plt.xlabel('Regularization parameter (l2 penalty)', fontsize=18)
    plt.xlabel(f'{name_param}', fontsize=18)
    plt.grid(b=bool, which='major', axis='both', color='gray', 
             linestyle='--', linewidth=0.6)
    plt.minorticks_on()
    plt.grid(b=bool, which='minor', axis='both', color='gray', 
             linestyle='--', linewidth=0.4)
    plt.title(f'APD Prediction {l_o}', fontsize=18)
    ax.legend(['Test', 'Train'])
    plt.show()
    

def plot_gs_vs_pred(y_pred, y_test, l_o='APD30', alpha=0.3, alg_mode='svm', c=1, ne=100,
                    lr=0.1, ms=2, LIM=True, save_path='scatter_plot.png'):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=1024)
    if l_o[:5] == 'APD30':
        color = 'b'
    elif l_o[:5] == 'APD60':
        color = 'g'
    elif l_o[:5] == 'APD90':
        color = 'r'
    else:
        color = 'black'
    plt.scatter(y_pred, y_test, c=color, alpha=alpha)
    ax.axline((0.56, 0.56), slope=1, c='k', linestyle='dashed', alpha=0.8)
    plt.xlabel(f'Prediction of {l_o}')
    plt.ylabel(f'Gold Standard: {l_o}')
    plt.title(f'2D scatter plot {alg_mode} model, {l_o}, C={c}')
    plt.title(f'2D scatter plot {alg_mode} model, {l_o}, NE={ne}, LR={lr}, MS={ms}')
    plt.minorticks_on()
    if LIM:
        plt.xlim((0, 1))
        plt.ylim((0, 1))
    plt.savefig(save_path, format='png')

def get_MAP_vent_data(CLUSTER=True):
    # TODO(paths): `path_modules`/`os.chdir` point at a `MAP_autoencoder` module
    # directory that has no `paths` accessor; the literals are kept verbatim. With the
    # package-absolute imports below the chdir is no longer needed for resolution, but
    # it is preserved to keep behaviour identical to the original migration source.
    if CLUSTER:
        path_modules = os.path.join(r"/cluster/work/vogtlab/Group/sruiperez",
                                    "MAP_autoencoder")  # cluster
    else:
        path_modules = os.path.join(r'C:/Users/sruip/Desktop/Universities/ETH',
                                    r'Research_Projects/Master_Thesis/MAP_autoencoder')
    os.chdir(path_modules)  # Directory of functions
    # Migration: in-function imports rewritten to package paths. NOTE (self-import):
    # the original did `import MAP_functions as mapf` from *within* MAP_functions
    # itself; the faithful package equivalent imports this very module as `mapf`.
    from cardiac_map_diffusion.data.retrieve_ventMAP import retrieve_ventMAP
    import cardiac_map_diffusion.metrics.map_functions_baselines as mapf

    # % Define function to acquire ventricular MAPs from raw data files
    root_path = str(paths.ventmap_root())
    MAP_vent_dict = retrieve_ventMAP(root_path)

    # % Group MAP sets by patient
    MAP_vent, pat_vent_IDs = mapf.dict_to_dataF_MAPv(MAP_vent_dict,
                                                     CORRECT_10064=True)
    # % Order the ventricular MAP set by patient ID
    MAP_vent = MAP_vent.sort_values(by='pat_ID').reset_index(drop=True)

    # % Complete dataframe of ventricular MAPs (one MAP per row)
    MAP_vent_complete = mapf.dataF_to_dataFsingle(MAP_vent)

    # % Acquire APD annotations
    file_APD_name = paths.apd_annotations_filename()
    file_APD_path = str(paths.apd_annotations_dir())

    APD_df, APD_df_pp = mapf.acquire_APD_annotations(file_APD_name, file_APD_path,
                                                     BEAT=False, MERGE=True)
    # % Include ventricular annotations in the complete dataframe
    MAP_vent_complete['APD30_gs'] = APD_df['APD30']
    MAP_vent_complete['APD60_gs'] = APD_df['APD60']
    MAP_vent_complete['APD90_gs'] = APD_df['APD90']

    # % Synthetic APD 30, 60, 90 Points - Compute for multiple MAPs
    # % APD 30, 60, 90 Points
    MAP_matrix = np.array(list(MAP_vent_complete['MAP_segments']))
    (APD, APD_volt, APD_endpoint, plateau, depolar_end,
     APD_init) = mapf.get_APD_multipleMAP(MAP_matrix, delay_depol=15, EXC1=False)

    # % Add to the dataframe the corresponding columns
    MAP_vent_complete['APD_init_synth'] = APD_init
    MAP_vent_complete['APD30_endpoint_synth'] = APD_endpoint[0]
    MAP_vent_complete['APD60_endpoint_synth'] = APD_endpoint[1]
    MAP_vent_complete['APD90_endpoint_synth'] = APD_endpoint[2]
    MAP_vent_complete['APD30_synth'] = APD[0]
    MAP_vent_complete['APD60_synth'] = APD[1]
    MAP_vent_complete['APD90_synth'] = APD[2]

    # % Correct mislabels of APD90
    for idx in range(len(MAP_vent_complete)):
        if MAP_vent_complete.iloc[idx]['APD90_gs']>=360:
            MAP_vent_complete.loc[idx, ('APD90_gs')] = MAP_vent_complete.loc[idx, 'APD90_synth']
        if MAP_vent_complete.iloc[idx]['APD60_gs']>=360:
            MAP_vent_complete.loc[idx, ('APD60_gs')] = MAP_vent_complete.loc[idx, 'APD60_synth']
    return MAP_vent_complete


# Define a function to calculate Pearson correlation coefficient
def calculate_corr_to_av(row, MAP_vent_av):
    pat_ID = row['pat_ID']
    segment = row['MAP_segments']
    average_segment = MAP_vent_av[MAP_vent_av['pat_ID'] == pat_ID]['MAP_segments_av'].iloc[0]
    correlation, _ = pearsonr(average_segment, segment)
    return correlation

# Define a function to calculate the difference from the average
def calculate_difference_from_av(row, MAP_vent_av):
    pat_ID = row['pat_ID']
    segment = row['MAP_segments']
    average_segment = MAP_vent_av[MAP_vent_av['pat_ID'] == pat_ID]['MAP_segments_av'].iloc[0]
    normalized_segment = normalize_EGM_array(segment)
    normalized_average_segment = normalize_EGM_array(average_segment)
    difference = normalized_segment[0] - normalized_average_segment[0]
    return difference



# Define a function to calculate the difference from the average
def plot_spectrum(signal, sampling_rate):
    # Apply FFT to the signal
    fft_result = np.fft.fft(signal)

    # Calculate the magnitude spectrum
    magnitude_spectrum = np.abs(fft_result)

    # Generate the frequency axis
    frequency = np.fft.fftfreq(len(signal), d=1 / sampling_rate)

    # Plot the spectrum
    plt.figure()
    plt.plot(frequency, magnitude_spectrum)
    plt.title("Magnitude Spectrum")
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.show()


def get_np_noisearrays(MAP_vent_complete):
    # Group by 'pat_ID' and calculate the average of 'MAP_segments' for each group
    MAP_vent_av = MAP_vent_complete.groupby('pat_ID')['MAP_segments'].apply(np.mean).reset_index()

    # Rename the column to 'MAP_segments_av'
    MAP_vent_av = MAP_vent_av.rename(columns={'MAP_segments': 'MAP_segments_av'})

    # Iterate over each row in MAP_vent_av dataframe
    for index, row in MAP_vent_av.iterrows():
        pat_ID = row['pat_ID']
        segments_av = row['MAP_segments_av']

    # Apply the function to calculate correlation and add it as a new column
    MAP_vent_complete['corr_to_av'] = MAP_vent_complete.apply(calculate_corr_to_av, args=(MAP_vent_av,), axis=1)

    # Group by 'pat_ID' and find the 10% lowest corr_to_av values
    lowest_corr = MAP_vent_complete.groupby('pat_ID')['corr_to_av'].apply(
        lambda x: x.nsmallest(int(len(x) * 0.1))).reset_index()

    # Check if the lowest_corr values are below 0.98
    lowest_corr['low_corr_to_av'] = lowest_corr['corr_to_av'] < 1

    # Merge the lowest_corr values back into the MAP_vent_complete dataframe
    MAP_vent_complete = pd.merge(MAP_vent_complete, lowest_corr[['pat_ID', 'corr_to_av', 'low_corr_to_av']],
                                 on=['pat_ID', 'corr_to_av'], how='left')
    MAP_vent_complete['low_corr_to_av'] = MAP_vent_complete['low_corr_to_av'].fillna(False).astype(int)

    # Apply the function to calculate the difference and add it as a new column
    MAP_vent_complete['difference_from_av'] = MAP_vent_complete.apply(calculate_difference_from_av, args=(MAP_vent_av,),
                                                                      axis=1)

    # %
    # Select rows where 'low_corr_to_av' is 1
    MAP_ep_noise = MAP_vent_complete.loc[
        MAP_vent_complete['low_corr_to_av'] == 1, ['pat_ID', 'difference_from_av']].copy()

    # Get the arrays from the 'difference_from_av' column
    arrays = MAP_ep_noise['difference_from_av'].tolist()

    return arrays


def generate_epnoise(arrays, n, noise_boundary=0.2):
    # arrays = get_np_noisearrays(MAP_vent_complete)
    ep_noise_mat = []
    for _ in range(n):
        # Randomly select the number of elements to average
        num_elements = random.randint(1, 5)

        # Randomly select elements from the list and calculate the average
        selected_arrays = random.sample(arrays, num_elements)
        ep_noise = np.mean(selected_arrays, axis=0)

        ep_noise_modified = ep_noise.copy()

        if np.any(ep_noise > noise_boundary):
            factor = noise_boundary / np.max(ep_noise)
            ep_noise_modified *= factor

        if np.any(ep_noise < -noise_boundary):
            factor = -noise_boundary / np.min(ep_noise)
            ep_noise_modified *= factor

        ep_noise_mat.append(ep_noise_modified)

    return np.asarray(ep_noise_mat)


def introduce_epnoise(arrays, MAP_array, noise_boundary=0.2):
    if MAP_array.ndim == 1:
        MAP_array = MAP_array.reshape(1, -1)
    ep_noise_mat = generate_epnoise(arrays, n=MAP_array.shape[0],
                                    noise_boundary=noise_boundary)
    noisy_MAP = ep_noise_mat + MAP_array
    return noisy_MAP


def butterworth_notch_filter(noisy_signal, fs=1000, lowcut=0.01, highcut=400, order=5, f0=60, Q=30.0):
    """
    Apply a Butterworth filter to a noisy signal, apply a notch filter
    to the Butterworth-filtered signal afterward.
    
    This function implements the exact same filtering technique as used in the Diffusion_MAP pipeline
    for creating a baseline filtered signal comparison.
    
    Parameters:
    -----------
    noisy_signal : numpy.ndarray
        Input noisy signal to be filtered
    fs : float, default=1000
        Sampling frequency in Hz
    lowcut : float, default=0.01
        Low cutoff frequency for bandpass filter in Hz
    highcut : float, default=400
        High cutoff frequency for bandpass filter in Hz
    order : int, default=5
        Order of the Butterworth filter
    f0 : float, default=60
        Frequency to be removed by notch filter in Hz (typically power line frequency)
    Q : float, default=30.0
        Quality factor for the notch filter
        
    Returns:
    --------
    numpy.ndarray
        Filtered signal with the same shape as input
    """
    from scipy.signal import butter, lfilter, iirnotch
    
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    w0 = f0 / nyq

    # Apply Butterworth bandpass filter
    b_butter, a_butter = butter(order, [low, high], btype='band', analog=False)
    butter_filtered = lfilter(b_butter, a_butter, noisy_signal)

    # Apply notch filter to remove power line frequency
    b_notch, a_notch = iirnotch(f0, Q, fs=fs)
    filtered = lfilter(b_notch, a_notch, butter_filtered)

    return filtered


"""Electrophysiological (EP) noise generation for the diffusion pipeline.

Faithful copy of the original ``ep_noise_sam.py`` (no cluster paths, no local
imports). It derives a bank of low-correlation residual templates per patient
and adds bounded combinations of them to clean beats. ``generate_epnoise``
returns the noise as a torch tensor (the only line adapted from the source).
"""

import numpy as np
import pandas as pd
import random
import torch
from scipy.stats import pearsonr

def normalize_EGM_array(array):
    array_ = np.array(array)
    if array_.ndim == 1:
        array_ = array_.reshape(1, -1)
    a = (array_.T - array_.min(axis=1)).T
    b = (array_.max(axis=1) - array_.min(axis=1))
    array_std = (a.T/b).T

    return array_std

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


def generate_epnoise(arrays, n, noise_boundary=0.25):
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

    return torch.tensor(np.asarray(ep_noise_mat)) # This is the only line that was adjusted


def introduce_epnoise(arrays, MAP_array, noise_boundary=0.25):
    if MAP_array.ndim == 1:
        MAP_array = MAP_array.reshape(1, -1)
    ep_noise_mat = generate_epnoise(arrays, n=MAP_array.shape[0],
                                    noise_boundary=noise_boundary)
    noisy_MAP = ep_noise_mat + MAP_array
    return noisy_MAP

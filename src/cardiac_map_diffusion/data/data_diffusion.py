"""Dataset class and data utilities for the DDPM diffusion track.

Faithful migration of ``Diffusion_MAP_fullpipeline_final/data.py``. The body is
copied verbatim; only this module docstring has been expanded to note the
migration. No logic, splitting, normalisation or plotting behaviour has changed.

    - This file contains the functions to create the dataset class used for the dataloader.

    - The create_split function is used to create the train and test splits for the dataset.

 """


import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
import matplotlib.pyplot as plt
import matplotlib as mpl


class TimeSeriesDataset(Dataset):
    '''
    Define a custom dataset class for time series data. This class is used to create a dataloader.
    '''
    def __init__(self, dataframe, series_column):
        self.data = []
        for item in dataframe[series_column]:
            self.data.append(torch.tensor(item, dtype=torch.float))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        series = self.data[idx]
        return series

    # in the final pipeline the data is normalised beforehand the function is not used.
    def normalise_signal(self, signal):
        signal = np.array(signal)
        signal_max = np.max(signal)
        signal_min = np.min(signal)
        signal_normal = (signal - signal_min) / (signal_max - signal_min)
        return signal_normal


def create_split(df, kgroups=True, n_splits=5, test_size=0.2, random_state=42):
    """
    Parameters:
    kgroups: If True creates exclusive splits, meaning each pat exactly once in train. (leads to imbalanced splits)
    df (pandas.DataFrame): Input dataframe with a 'pat_ID' column.
    n_splits (int): Number of folds for the k-fold cross-validation.

    Returns:
    Generator yielding train and test data splits.
    """
    if kgroups:
        gkf = GroupKFold(n_splits=n_splits)
        for train_idx, test_idx in gkf.split(df, groups=df['pat_ID']):
            train_data = df.iloc[train_idx]
            test_data = df.iloc[test_idx]
            
            # Ensure no patient appears in both train and test sets
            assert len(set(train_data['pat_ID']).intersection(set(test_data['pat_ID']))) == 0, "Overlap in train and test sets"
            
            yield train_data, test_data

    else:
        gss = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
        for train_idx, test_idx in gss.split(df, groups=df['pat_ID']):
            train_data = df.iloc[train_idx]
            test_data = df.iloc[test_idx]
            
            # Ensure no patient appears in both train and test sets
            assert len(set(train_data['pat_ID']).intersection(set(test_data['pat_ID']))) == 0, "Overlap in train and test sets"
            
            yield train_data, test_data


def save_patient_ids(df_train, df_test, log_file_path):
    """
    Store the patient ids of the splits in a json file.
    """
    patients_path = os.path.join(log_file_path, "patients.json")
    os.makedirs(os.path.dirname(patients_path), exist_ok=True)

    patients_train = df_train['pat_ID'].unique()
    patients_test = df_test['pat_ID'].unique()

    overlap = set(patients_train) & set(patients_test)
    assert len(overlap) == 0, f"Overlap found in patient IDs between training and testing sets: {overlap}"

    patients_data = {
        "train": list(patients_train),
        "test": list(patients_test)
    }

    with open(patients_path, 'w') as file:
        json.dump(patients_data, file, indent=4)  

    print(f"Patient IDs have been saved successfully to {patients_path}")


def plot_and_log_signals(writer, denoised, noisy, clean, n_prints, steps, dpi=300):
    """
    Store the plots of a few signals in the tensorboard.
    """
    mpl.rcParams.update(mpl.rcParamsDefault)  
    mpl.rcParams.update({
        "text.usetex": False,  
        "font.family": "STIXGeneral",  
        "mathtext.fontset": "stix",
        "axes.labelsize": 11,  
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    # In case the last batch has very few samples 
    if n_prints > denoised.shape[0]:
        n_prints = denoised.shape[0]

    for i in range(n_prints):
        fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
        
        ax.plot(clean[i, 0].cpu().numpy(), color='k', linewidth=2, label='Clean')
        ax.plot(noisy[i, 0].cpu().numpy(), color='r', linewidth=1, label='Noisy')
        ax.plot(denoised[i, 0].cpu().numpy(), color='b', linewidth=2, label='Denoised')
        
        ax.set_ylabel('Normalised Voltage Amplitude', fontsize=15)
        ax.set_xlabel('Time [msec]', fontsize=15)
        ax.set_title(f'Signal {i+1}', fontsize=18)
        
        ax.grid(True, which='major', axis='both', color='gray', linestyle='--', linewidth=0.6)
        ax.minorticks_on()
        ax.grid(True, which='minor', axis='both', color='gray', linestyle='--', linewidth=0.4)
        
        ax.legend()
        plt.tight_layout()
        
        writer.add_figure(f'Steps_{steps}_Signal_{i+1}', fig)
        plt.close(fig)




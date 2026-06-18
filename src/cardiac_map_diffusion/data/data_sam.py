"""Ventricular MAP data loading for the **diffusion** pipeline.

Faithful migration of the original ``data_sam.py`` ("copied from ... MAP_VAE/
MAP_functions, and merely serves to load the data. The code is unchanged.").

Only path handling was de-hard-coded: the cluster literals
(``/cluster/dataset/.../src_vae/...``) are replaced by
:mod:`cardiac_map_diffusion.paths`, and the vestigial ``os.chdir`` into a
``MAP_autoencoder`` directory was removed (the modules it used to expose --
``retrieve_ventMAP`` and ``MAP_functions`` -- are now ordinary package imports,
so the chdir had no effect on the returned data). The numerical logic, fold
construction, and APD post-processing are unchanged.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os

from cardiac_map_diffusion import paths
from cardiac_map_diffusion.data.retrieve_ventMAP import retrieve_ventMAP
from cardiac_map_diffusion.data.splits import get_train_test_kfolds
from cardiac_map_diffusion.metrics import map_functions as mapf


def get_train_test():
    # %% Input Data
    # % Acquire ventricular MAP EGMs, patient IDs and APD annotations
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

    kfold_6train_vent = []
    kfold_6test_vent = []

    size_folds = 7
    if size_folds == 7:
        for i in range(6):
            # 2nd fold
            kfold_6train_vent.append(MAP_vent.iloc[np.r_[:i * size_folds,
                                                   ((i + 1) * size_folds):len(MAP_vent)]])
            kfold_6test_vent.append(MAP_vent.iloc[i * size_folds:(i + 1) * size_folds])
    elif size_folds == 6:
        for i in range(7):
            # 2nd fold
            kfold_6train_vent.append(MAP_vent.iloc[np.r_[:i * size_folds,
                                                   ((i + 1) * size_folds):len(MAP_vent)]])
            kfold_6test_vent.append(MAP_vent.iloc[i * size_folds:(i + 1) * size_folds])
    else:
        print("incorrect number of size for complete folds")

    # %%
    j = 1
    X_train = list(np.concatenate(list(kfold_6train_vent[j]['EGM']), axis=0))
    X_test = list(np.concatenate(list(kfold_6test_vent[j]['EGM']), axis=0))
    return X_train, X_test

def get_train_test_truncated(option='end', percent=0.7, var=0.05):

    # %% Input Data
    # % Acquire ventricular MAP EGMs, patient IDs and APD annotations
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

    kfold_6train_vent = []
    kfold_6test_vent = []

    size_folds = 7
    if size_folds == 7:
        for i in range(6):
            # 2nd fold
            kfold_6train_vent.append(MAP_vent.iloc[np.r_[:i * size_folds,
                                                   ((i + 1) * size_folds):len(MAP_vent)]])
            kfold_6test_vent.append(MAP_vent.iloc[i * size_folds:(i + 1) * size_folds])
    elif size_folds == 6:
        for i in range(7):
            # 2nd fold
            kfold_6train_vent.append(MAP_vent.iloc[np.r_[:i * size_folds,
                                                   ((i + 1) * size_folds):len(MAP_vent)]])
            kfold_6test_vent.append(MAP_vent.iloc[i * size_folds:(i + 1) * size_folds])
    else:
        print("incorrect number of size for complete folds")

    # %%
    j = 1
    X_train = list(np.concatenate(list(kfold_6train_vent[j]['EGM']), axis=0))
    X_test = list(np.concatenate(list(kfold_6test_vent[j]['EGM']), axis=0))
    X_std_train, X_std_test = mapf.normalize_EGM_input(X_train, X_test)

    # %%
    X_train_arr = np.array(X_train)
    X_test_arr = np.array(X_test)

    X_train_trunc = mapf.introduce_truncation_noise(X_train_arr, option=option, percent=percent, var=var)
    X_test_trunc = mapf.introduce_truncation_noise(X_test_arr, option=option, percent=percent, var=var)
    # %%

    X_std_train, X_std_test = mapf.normalize_EGM_input(X_train, X_test)
    X_std_train_trunc, X_std_test_trunc = mapf.normalize_EGM_input(X_train_trunc, X_test_trunc)

    return X_std_train_trunc, X_std_test_trunc

def get_MAP_vent_data():
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

# get_train_test_kfolds is centralised in cardiac_map_diffusion.data.splits
# (it was byte-identical across the diffusion loader, the baseline loader, and the
# trainer); imported above.



def get_MAP_and_label_data(CLUSTER=True):
    MAP_vent_complete = get_MAP_vent_data()
    X = np.array(MAP_vent_complete['MAP_segments'].tolist())
    y_30 = np.array(MAP_vent_complete['APD30_gs'])
    y_60 = np.array(MAP_vent_complete['APD60_gs'])
    y_90 = np.array(MAP_vent_complete['APD90_gs'])
    return X, y_30, y_60, y_90

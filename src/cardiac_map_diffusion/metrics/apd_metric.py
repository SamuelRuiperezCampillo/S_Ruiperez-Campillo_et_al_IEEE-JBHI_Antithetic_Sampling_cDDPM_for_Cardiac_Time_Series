"""APD-prediction downstream evaluation for the baseline (VAE) track.

Faithful migration of ``MAP_VAE/apd_metric.py``. Runs a k-fold SVR
hyper-parameter sweep over the synthetic-APD regression task on the ventricular
MAP cohort and writes the resulting RMSE statistics to ``results_apd.json``. The
body is copied verbatim; only this module docstring and the two import statements
below have been rewritten for the package layout (``data`` -> the baseline loader
``cardiac_map_diffusion.data.data_baselines`` and ``MAP_functions`` -> the baseline
track's ``cardiac_map_diffusion.metrics.map_functions_baselines``). No logic, math,
hyperparameters, RNG seeding or metric formulas have been changed.
"""

from cardiac_map_diffusion.data.data_baselines import get_MAP_vent_data
import torch
import json
import logging
import os
from ml_collections import config_flags
from absl import app
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import cardiac_map_diffusion.metrics.map_functions_baselines as mapf
from sklearn import svm
import random
from itertools import product
from math import sqrt


def main(_):
    config_apd = _CONFIG.value
    device = "cuda" if torch.cuda.is_available() else "cpu"

    working_dir = config_apd.working_dir

    # setup directories
    config_apd.experiment_dir = os.path.join(working_dir, config_apd.experiment_name)
    if not os.path.isdir(config_apd.experiment_dir):
        os.mkdir(config_apd.experiment_dir)

    # set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(
        os.path.join(config_apd.experiment_dir, f"{config_apd.experiment_name}.log"),
        mode="a")
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.info(f"Starting {config_apd.experiment_name} on {device}")
    with open(os.path.join(config_apd.experiment_dir, "config_apd.json"), "w") as f:
        f.write(config_apd.to_json(indent=4))


    # Value for random seed
    r_seed = config_apd.random_seed
    CLUSTER=config_apd.cluster
    # Values of C, t, k, g, and d

    c_values = [0.01, 0.03, 0.05, 0.07,  0.09,
                0.1, 0.3, 0.5, 0.7, 0.9,
                1, 3, 5, 7, 9, 10, 50, 100]
    t_values = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4,
                1e-3, 5e-3, 1e-2,5e-2, 1e-1, 5e-1]
    k_values = ['linear', 'poly', 'rbf']
    g_values = ['scale', 'auto']
    d_values = [1, 2, 3, 4, 5]

    # Label of apd
    apd_label = config_apd.label_apd

    # Percentiles for standardizatoin
    perc_minimum = config_apd.perc_minimum
    perc_maximum = config_apd.perc_maximum
    # Number of folds
    num_folds = config_apd.num_folds
    random.seed(r_seed)

    # Acquire Data
    MAP_vent_complete = get_MAP_vent_data(CLUSTER=CLUSTER)

    # Get unique patient IDs
    unique_patients = MAP_vent_complete['pat_ID'].unique()

    # Shuffle the list of unique patient IDs
    random.shuffle(unique_patients)

    # Preprocess the data
    X = np.array(MAP_vent_complete['MAP_segments'].tolist())
    X_std = mapf.normalize_EGM_array(X)
    y = np.array(MAP_vent_complete[apd_label])
    y_std = mapf.normalize_y(y, perc_min=perc_minimum, perc_max=perc_maximum)

    # Create a list of all possible combinations of c, t, k, g, and d
    param_combinations = list(product(c_values, t_values, k_values, g_values,
                                      d_values))

    # Lists to store results
    mean_mse_train = []
    std_mse_train = []
    mean_mse_train_std = []
    std_mse_train_std = []
    mean_mse_test = []
    std_mse_test = []
    mean_mse_test_std = []
    std_mse_test_std = []

    #  Train the regressor using k-fold cross-validation for each combination
    #  of c, t, k, g, and d
    kf = KFold(n_splits=num_folds)

    for c, t, k, g, d in param_combinations:
        logger.info(f"c={c}, t={t}, k={k}, g={g}, d={d}")
        mse_scores_train = []
        mse_scores_train_std = []
        mse_scores_test = []
        mse_scores_test_std = []

        for train_index, test_index in kf.split(unique_patients):
            train_patients = unique_patients[train_index]
            test_patients = unique_patients[test_index]

            train = MAP_vent_complete[MAP_vent_complete['pat_ID'].isin(train_patients)]
            test = MAP_vent_complete[MAP_vent_complete['pat_ID'].isin(test_patients)]

            X_train = np.array(train['MAP_segments'].tolist())
            X_train_std = mapf.normalize_EGM_array(X_train)
            y_train = np.array(train[apd_label])
            y_train_std = mapf.normalize_y(y_train, perc_min=perc_minimum,
                                           perc_max=perc_maximum)

            X_test = np.array(test['MAP_segments'].tolist())
            X_test_std = mapf.normalize_EGM_array(X_test)
            y_test = np.array(test[apd_label])
            y_test_std = mapf.normalize_y(y_test, perc_min=perc_minimum,
                                          perc_max=perc_maximum)

            regressor = svm.SVR(C=c, kernel=k, degree=d, gamma=g, tol=t)
            regressor.fit(X_train_std, y_train_std)

            y_pred_train = regressor.predict(X_train_std)
            y_pred_train_unstd = mapf.un_normalize_ystd(y_train, y_pred_train,
                                                        perc_min=perc_minimum,
                                                        perc_max=perc_maximum)
            y_pred_test = regressor.predict(X_test_std)
            y_pred_test_unstd = mapf.un_normalize_ystd(y_test, y_pred_test,
                                                       perc_min=perc_minimum,
                                                       perc_max=perc_maximum)

            mse_scores_train.append(sqrt(mean_squared_error(y_train, y_pred_train_unstd)))
            mse_scores_train_std.append(sqrt(mean_squared_error(y_train_std, y_pred_train)))
            mse_scores_test.append(sqrt(mean_squared_error(y_test, y_pred_test_unstd)))
            mse_scores_test_std.append(sqrt(mean_squared_error(y_test_std, y_pred_test)))

        mean_mse_train.append(np.mean(mse_scores_train))
        std_mse_train.append(np.std(mse_scores_train))
        mean_mse_train_std.append(np.mean(mse_scores_train_std))
        std_mse_train_std.append(np.std(mse_scores_train_std))
        mean_mse_test.append(np.mean(mse_scores_test))
        std_mse_test.append(np.std(mse_scores_test))
        mean_mse_test_std.append(np.mean(mse_scores_test_std))
        std_mse_test_std.append(np.std(mse_scores_test_std))

    # Save the useful information in a result_dictionary
    result_dict = {
        'r_seed': r_seed,
        'c_values': c_values,
        't_values': t_values,
        'k_values': k_values,
        'g_values': g_values,
        'd_values': d_values,
        'param_combinations': param_combinations,
        'apd_label': apd_label,
        'perc_minimum': perc_minimum,
        'perc_maximum': perc_maximum,
        'num_folds': num_folds,
        'mean_mse_train': mean_mse_train,
        'std_mse_train': std_mse_train,
        'mean_mse_train_std': mean_mse_train_std,
        'std_mse_train_std': std_mse_train_std,
        'mean_mse_test': mean_mse_test,
        'std_mse_test': std_mse_test,
        'mean_mse_test_std': mean_mse_test_std,
        'std_mse_test_std': std_mse_test_std
    }

    with open(os.path.join(config_apd.experiment_dir,
                           "results_apd.json"), "w") as fp:
        json.dump(result_dict, fp, indent=4)  # encode dict into JSON
    logger.info("Done writing dict into .json file: results_dict")

_CONFIG = config_flags.DEFINE_config_file("config_apd", lock_config=False)

if __name__ == "__main__":
    app.run(main)
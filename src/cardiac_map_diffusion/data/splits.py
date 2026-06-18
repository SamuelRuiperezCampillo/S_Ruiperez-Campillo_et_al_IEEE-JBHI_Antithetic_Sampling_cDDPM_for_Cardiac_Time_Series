"""Patient-grouped K-fold splitting, shared by the diffusion and baseline pipelines.

``get_train_test_kfolds`` was byte-identical in the diffusion loader
(``data.data_sam``), the baseline loader (``data.data_baselines``), and inlined in
the DDPM trainer; it is centralised here to remove that duplication. The logic is
unchanged: a patient-grouped ``KFold`` shuffled with ``r_seed`` (so no patient
appears in both train and test of a fold), returning the arrays for ``split_number``.
"""

import random

import numpy as np
from sklearn.model_selection import KFold


def get_train_test_kfolds(MAP_vent_complete, num_folds=4, split_number=0, r_seed=5, apd_label='APD30_gs'):
    random.seed(r_seed)
    unique_patients = MAP_vent_complete['pat_ID'].unique()
    random.shuffle(unique_patients)
    kf = KFold(n_splits=num_folds)

    X_train_allfolds = []
    y_train_allfolds = []
    X_test_allfolds = []
    y_test_allfolds = []
    for train_index, test_index in kf.split(unique_patients):
        train_patients = unique_patients[train_index]
        test_patients = unique_patients[test_index]

        train = MAP_vent_complete[MAP_vent_complete['pat_ID'].isin(train_patients)]
        test = MAP_vent_complete[MAP_vent_complete['pat_ID'].isin(test_patients)]

        X_train = np.array(train['MAP_segments'].tolist())
        y_train = np.array(train[apd_label])
        X_test = np.array(test['MAP_segments'].tolist())
        y_test = np.array(test[apd_label])

        X_train_allfolds.append(X_train)
        y_train_allfolds.append(y_train)
        X_test_allfolds.append(X_test)
        y_test_allfolds.append(y_test)

    return X_train_allfolds[split_number], X_test_allfolds[split_number], y_train_allfolds[split_number], y_test_allfolds[split_number]

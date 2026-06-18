"""Death-within-3-years downstream task (unused; kept for completeness).

Faithful migration of ``Diffusion_MAP_fullpipeline_final/downstreamDEATH.py``. The
body is copied verbatim; only this module docstring has been expanded to note the
migration. No model, hyper-parameters or logic has changed.

    - This class is used to predict the Death within 3 years labels.

    - Since it turned out that this task is unfeasible, it is not used in the thesis.

    - It is kept here for completeness and possible future use. 
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

class DownstreamDeath:
    ''' 
    Class for predicting Death within 3 years labels. Instantiate with training and testing dataframes and a logger.
    Signals should be normalized to [0,1], whereas labels are input in raw form.
    '''
    def __init__(self, df_train, df_test, logger, max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=50):
        self.x_train = np.array(df_train['MAP_segments'].to_list())
        self.x_test = np.array(df_test['MAP_segments'].to_list())
        self.y_train = np.array(df_train['Death in 3Y'].to_list())
        self.y_test = np.array(df_test['Death in 3Y'].to_list())
        self.logger = logger
        self.model = RandomForestClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, n_estimators=n_estimators)

    def fit(self):
        self.model.fit(self.x_train, self.y_train)

        y_pred_train = self.model.predict(self.x_train)
        y_pred_test = self.model.predict(self.x_test)

        acc_train = accuracy_score(self.y_train, y_pred_train)
        acc_test = accuracy_score(self.y_test, y_pred_test)

        self.logger.add_scalar(f"Train ACC Death Predictions", acc_train)
        self.logger.add_scalar(f"Test ACC Death Predictions", acc_test)

        # Calculate sensitivity and specificity for training data
        tn, fp, fn, tp = confusion_matrix(self.y_train, y_pred_train).ravel()
        sensitivity_train = tp / (tp + fn) if (tp + fn) != 0 else 0
        specificity_train = tn / (tn + fp) if (tn + fp) != 0 else 0
        
        self.logger.add_scalar(f"Train Sensitivity Death Predictions", sensitivity_train)
        self.logger.add_scalar(f"Train Specificity Death Predictions", specificity_train)

        # Calculate sensitivity and specificity for test data
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred_test).ravel()
        sensitivity_test = tp / (tp + fn) if (tp + fn) != 0 else 0
        specificity_test = tn / (tn + fp) if (tn + fp) != 0 else 0
        
        self.logger.add_scalar(f"Test Sensitivity Death Predictions", sensitivity_test)
        self.logger.add_scalar(f"Test Specificity Death Predictions", specificity_test)
    
    def predict(self, x_noisy, x_denoised, global_step):
        y_pred_noisy = self.model.predict(x_noisy)
        y_pred_denoised = self.model.predict(x_denoised)

        acc_noisy = accuracy_score(self.y_test, y_pred_noisy)
        acc_denoised = accuracy_score(self.y_test, y_pred_denoised)

        self.logger.add_scalar(f"Test ACC Death Predictions Noisy", acc_noisy, global_step=global_step)
        self.logger.add_scalar(f"Test ACC Death Predictions Denoised", acc_denoised, global_step=global_step)

        # Calculate sensitivity and specificity for noisy data
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred_noisy).ravel()
        sensitivity_noisy = tp / (tp + fn) if (tp + fn) != 0 else 0
        specificity_noisy = tn / (tn + fp) if (tn + fp) != 0 else 0
        
        self.logger.add_scalar(f"Test Sensitivity Death Predictions Noisy", sensitivity_noisy, global_step=global_step)
        self.logger.add_scalar(f"Test Specificity Death Predictions Noisy", specificity_noisy, global_step=global_step)

        # Calculate sensitivity and specificity for denoised data
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred_denoised).ravel()
        sensitivity_denoised = tp / (tp + fn) if (tp + fn) != 0 else 0
        specificity_denoised = tn / (tn + fp) if (tn + fp) != 0 else 0
        
        self.logger.add_scalar(f"Test Sensitivity Death Predictions Denoised", sensitivity_denoised, global_step=global_step)
        self.logger.add_scalar(f"Test Specificity Death Predictions Denoised", specificity_denoised, global_step=global_step)



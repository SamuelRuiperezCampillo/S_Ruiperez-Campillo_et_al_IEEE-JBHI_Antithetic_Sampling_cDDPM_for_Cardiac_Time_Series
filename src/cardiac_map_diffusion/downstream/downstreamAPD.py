"""APD-regression downstream task.

Faithful migration of ``Diffusion_MAP_fullpipeline_final/downstreamAPD.py``. The
body is copied verbatim; only this module docstring has been expanded to note the
migration. No model, hyper-parameters, normalisation or logic has changed.

    - This class is used to predict the APD labels of the signals.

    - The regressor is a Support Vector Machine with a Radial Basis Function kernel (identical as proposed by Ruiperez-Campillo).

    - Since it turned out that in prior work the normalisation was done including the test set, leading to data leakage, results may differ.
"""

import numpy as np
from sklearn import svm
from sklearn.metrics import root_mean_squared_error

class DownstreamAPD:
    ''' 
    Class for predicting APD labels. Instantiate with training and testing dataframes and a logger.
    Signals should be normalized to [0,1], whereas labels are input in raw form.
    '''
    def __init__(self, df_train, df_test, logger, perc_min=0.5, perc_max=99.5, kernel='rbf', C=1, degree=3):
        self.x_train = np.array(df_train['MAP_segments'].to_list())
        self.x_test = np.array(df_test['MAP_segments'].to_list())
        self.logger = logger
        self.models = {}
        self.perc = {}
        for label in [30, 60, 90]:
            setattr(self, f'y_train_{label}', np.array(df_train[f'APD{label}_gs'].to_list()))
            setattr(self, f'y_test_{label}', np.array(df_test[f'APD{label}_gs'].to_list()))
            self.models[label] = svm.SVR(C=C, kernel=kernel, degree=degree)
            self.perc[label] = {
                'min': np.percentile(df_train[f'APD{label}_gs'], perc_min),
                'max': np.percentile(df_train[f'APD{label}_gs'], perc_max)
            }
    
    def normalise_y(self, y, label):
        perc = self.perc[label]
        return (y - perc['min']) / (perc['max'] - perc['min'])

    def denormalise_y(self, y_std, label):
        perc = self.perc[label]
        return y_std * (perc['max'] - perc['min']) + perc['min']

    def fit(self, APD_labels=[30, 60, 90]):
        '''
        Function for fitting the models and logging the errors of the clean signals.
        '''
        for label in APD_labels:
            model = self.models[label]
            y_train = getattr(self, f'y_train_{label}')
            y_train_norm = self.normalise_y(y_train, label)
            model.fit(self.x_train, y_train_norm)
            y_pred_train = model.predict(self.x_train)
            y_pred_train_denorm = self.denormalise_y(y_pred_train, label)

            mse_train = root_mean_squared_error(y_train, y_pred_train_denorm)
            self.logger.add_scalar(f"Train RMSE APD {label} prediction", mse_train)

            y_test = getattr(self, f'y_test_{label}')
            y_pred_test = model.predict(self.x_test)
            y_pred_test_denorm = self.denormalise_y(y_pred_test, label)
            mse_test = root_mean_squared_error(y_test, y_pred_test_denorm)
            self.logger.add_scalar(f"Test RMSE APD {label} prediction", mse_test)
    
    def predict(self, x_noisy, x_filtered, x_denoised, global_step, name='', APD_labels=[30, 60, 90]):
        '''
        Function for predicting and logging the errors of the noisy/filtered/denoised signals.
        '''
        for label in APD_labels:
            model = self.models[label]
            
            y = getattr(self, f'y_{name}_{label}')

            y_pred_noisy = model.predict(x_noisy)
            y_pred_filtered = model.predict(x_filtered)
            y_pred_denoised = model.predict(x_denoised)

            mse_noisy = root_mean_squared_error(y, self.denormalise_y(y_pred_noisy, label))
            mse_filtered = root_mean_squared_error(y, self.denormalise_y(y_pred_filtered, label))
            mse_denoised = root_mean_squared_error(y, self.denormalise_y(y_pred_denoised, label))

            self.logger.add_scalar(f"{name} RMSE APD {label} prediction noisy", mse_noisy, global_step=global_step)
            self.logger.add_scalar(f"{name} RMSE APD {label} prediction filtered", mse_filtered, global_step=global_step)
            self.logger.add_scalar(f"{name} RMSE APD {label} prediction denoised", mse_denoised, global_step=global_step)


    
    
    




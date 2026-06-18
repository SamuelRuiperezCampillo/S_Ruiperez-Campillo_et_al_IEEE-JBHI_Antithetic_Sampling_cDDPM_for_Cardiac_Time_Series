"""Torch ``Dataset`` wrappers and the ``retrieveDataSet`` factory (baseline pipeline).

Faithful migration of the original ``utils.py``. The only change is the import
of the noise helpers: ``import MAP_functions as mapf`` becomes a package import
of :mod:`cardiac_map_diffusion.metrics.map_functions_baselines`. Each dataset
class injects a noise type on-the-fly so that every training batch (and epoch)
sees freshly sampled corruption; the test set is corrupted once for a fixed
evaluation. Logic is unchanged.
"""

import torch
import numpy as np


from cardiac_map_diffusion.metrics import map_functions_baselines as mapf
class NumpyDataSet(torch.utils.data.Dataset):
    def __init__(self, array):
        self.array = array
    def __len__(self):
        return len(self.array)

    def __getitem__(self, i):
        clean = self.array[i]
        return clean

class NumpyDataSet_gaussian(torch.utils.data.Dataset):
    def __init__(self, array, noise=0.2):
        self.array = array
        self.noise = noise
    def __len__(self):
        return len(self.array)

    def __getitem__(self, i):
        clean = self.array[i]
        noisy = mapf.introduce_gaussian_noise(0, self.noise, self.array[i])
        return clean, noisy

class NumpyDataSet_spike(torch.utils.data.Dataset):
    def __init__(self, array):
        self.array = array

    def __len__(self):
        return len(self.array)

    def __getitem__(self, i):
        clean = self.array[i]
        noisy = mapf.introduce_spike_noise(self.array[i])
        return clean, noisy

class NumpyDataSet_bwander(torch.utils.data.Dataset):
    def __init__(self, array, min_freq=0.01, max_freq=0.3, min_sins=1, max_sins=4, max_amplitude=1):
        self.array = array
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.min_sins = min_sins
        self.max_sins = max_sins
        self.max_amplitude = max_amplitude
    def __len__(self):
        return len(self.array)

    def __getitem__(self, i):
        clean = self.array[i]
        noisy = mapf.introduce_baseline_wander(self.array[i], min_freq=self.min_freq, max_freq=self.max_freq,
                                               min_sins=self.min_sins, max_sins=self.max_sins,
                                               max_amplitude=self.max_amplitude)
        return clean, noisy


class NumpyDataSet_truncation(torch.utils.data.Dataset):
    def __init__(self, array, option, percent, var):
        self.array = array
        self.option = option
        self.percent = percent
        self.var = var
        self.trunc_func = mapf.introduce_truncation_noise
        self.normalize_func = mapf.normalize_EGM_array
    def __len__(self):
        return len(self.array)

    def __getitem__(self, i):
        clean = self.array[i]
        clean_std = self.normalize_func(clean)
        truncated = self.trunc_func(self.array[i], self.option, self.percent, self.var)
        truncated_std = self.normalize_func(truncated)
        return clean_std, truncated_std

class NumpyDataSet_allmixed(torch.utils.data.Dataset):
    def __init__(self, array, noise_ids, min_number_noises, max_number_noises, arrays):
        self.array = array
        self.all_noises_func = mapf.introduce_several_noises
        self.noise_ids = noise_ids
        self.min_number_noises = min_number_noises
        self.max_number_noises = max_number_noises
        self.normalize_func = mapf.normalize_EGM_array
        self.arrays = arrays

    def __len__(self):
        return len(self.array)

    def __getitem__(self, i):
        clean = self.array[i]
        clean_std = self.normalize_func(clean)
        noisy = self.all_noises_func(self.array[i], noise_ids=self.noise_ids,
                                     min_number_noises=self.min_number_noises,
                                     max_number_noises=self.max_number_noises,
                                     arrays=self.arrays)

        # Clip the input between 0 and 1 using numpy.clip
        clean_std = np.clip(clean_std, 0, 1)
        noisy = np.clip(noisy, 0, 1)
        return clean_std, noisy
class NumpyDataSet_epnoise(torch.utils.data.Dataset):
    def __init__(self, array, arrays, noise=0.25):
        self.array = array
        self.noise = noise
        self.arrays = arrays
    def __len__(self):
        return len(self.array)

    def __getitem__(self, i):
        clean = self.array[i]
        #print(f'shape clean is {clean.shape}')
        #clean = clean.reshape(1, -1)
        #print(f'shape clean is {clean.shape}')
        noisy = mapf.introduce_epnoise(self.arrays, clean, noise_boundary=self.noise)
        #clean = clean.reshape(-1, 1)
        #clean = np.squeeze(clean)
        #print(f'shape clean is {clean.shape}')
        #noisy = noisy.reshape(-1, 1)
        #print(f'shape is {noisy.shape}')
        return clean, noisy
class NumpyDataSet_test_noisy(torch.utils.data.Dataset):
    def __init__(self, array, array_noise):
        self.array = array
        self.array_noise = array_noise

    def __len__(self):
        return len(self.array)

    def __getitem__(self, i):
        clean = self.array[i]
        noisy = self.array_noise[i]

        # Clip the input between 0 and 1 using numpy.clip
        clean = np.clip(clean, 0, 1)
        noisy = np.clip(noisy, 0, 1)
        return clean, noisy

def retrieveDataSet(noise_type, noise_params, X_train, X_test, X_std_train, X_std_test, arrays=[]):
    if noise_type == 'none':
        train = NumpyDataSet(X_std_train)  # Returns original and noisy set of signals
        test = NumpyDataSet(X_std_test)  # Returns original and noisy set of signals
    elif noise_type == 'gaussian':
        train = NumpyDataSet_gaussian(X_std_train,
                                      noise=noise_params[0])  # Returns original and noisy set of signals
        X_std_test_noisy = mapf.introduce_gaussian_noise(0, noise_params[0], X_std_test)
        test = NumpyDataSet_test_noisy(X_std_test, X_std_test_noisy)  # Returns original and noisy set of signals
        # Does not change noise every batch
    elif noise_type == 'spike':
        train = NumpyDataSet_spike(X_std_train)  # Returns original and noisy set of signals
        X_std_test_noisy = mapf.introduce_spike_noise(X_std_test)
        test = NumpyDataSet_test_noisy(X_std_test, X_std_test_noisy)  # Returns original and noisy set of signals
        # Does not change noise every batch
    elif noise_type == 'bwander' or noise_type == 'powerline':
        train = NumpyDataSet_bwander(X_std_train, min_freq=noise_params[0], max_freq=noise_params[1],
                                     min_sins=noise_params[2], max_sins=noise_params[3],
                                     max_amplitude=noise_params[4])
        # Returns original and noisy set of signals
        X_std_test_noisy = mapf.introduce_baseline_wander(X_std_test, min_freq=noise_params[0],
                                                          max_freq=noise_params[1],
                                                          min_sins=noise_params[2],
                                                          max_sins=noise_params[3],
                                                          max_amplitude=noise_params[4])
        test = NumpyDataSet_test_noisy(X_std_test, X_std_test_noisy)  # Returns original and noisy set of signals
        # Does not change noise every batch
    elif noise_type == 'truncation':  # Introduce the truncated function in the NumpyDataSet function
        train = NumpyDataSet_truncation(X_train, option=noise_params[2], percent=noise_params[0],
                                        var=noise_params[1])  # Returns original and noisy set of signals
        # Changes noise every batch
        X_std_test_noisy = mapf.introduce_truncation_noise(np.array(X_test), option=option, percent=percent,
                                                           var=var)
        # _, X_std_test_noisy = get_train_test_truncated(option=noise_params[2], percent=noise_params[0],
        #                                               var=noise_params[1])  # Truncates test set only once
        test = NumpyDataSet_test_noisy(X_std_test, X_std_test_noisy)  # Returns original and noisy set of signals
        # Does not change noise every batch
        # If we wanted the test set to get different noise for each batch. Yet, it is not common practice.
        # test = NumpyDataSet_truncation(X_std_test, option=noise_params[2], percent=noise_params[0],
        #                                var=noise_params[1])  # Returns original and noisy set of signals
    elif noise_type == 'ep':  # Electrophysiological noise
        train = NumpyDataSet_epnoise(X_std_train, arrays,
                                     noise=noise_params[0])  # Returns original and noisy set of signals
        X_std_test_noisy = mapf.introduce_epnoise(arrays, X_std_test, noise_boundary=noise_params[0])
        test = NumpyDataSet_test_noisy(X_std_test, X_std_test_noisy)  # Returns original and noisy set of signals

        # Does not change noise every batch
    elif noise_type == 'allmixed':
        train = NumpyDataSet_allmixed(X_train, noise_ids=noise_params[0], min_number_noises=noise_params[1],
                                      max_number_noises=noise_params[2], arrays=arrays)
        X_std_test_noisy = mapf.introduce_several_noises(X_test, noise_ids=noise_params[0],
                                                         min_number_noises=noise_params[1],
                                                         max_number_noises=noise_params[2],
                                                         arrays=arrays)
        test = NumpyDataSet_test_noisy(X_std_test, X_std_test_noisy)
    return train, test, X_std_test_noisy

        # Every time I access an element in the data loader, the function is called, and therefore for each batch the
        # noise is computed again, from a different random seed, different random params, and for each epoch, different
        # noise parameters are created, allowing a further generalisability. Yet, there is a trade off in terms of
        # computational cost.

        # speed up version, change the NumpyDataSet_truncation function to be similar to the gaussian one, and use this
        # line above the if: #X_std_train_trunc, X_std_test_trunc = get_train_test_truncated(option=noise_params[2],
        # percent=noise_params[0], var=noise_params[1]). Then, just introduce the train and test both together in the
        # NumpyDataSet function and go.

"""Noise-synthesis utilities for the DDPM diffusion track.

Faithful migration of ``Diffusion_MAP_fullpipeline_final/generate_noise.py``. The
body is copied verbatim; only this module docstring and the import rewrites
required by the package layout have been changed. No noise models, parameters or
logic have been altered.

    - This file contains the functions to introduce different types of noise to the signals, using the noise library.

    - The main function introduce_several_noises() introduces a random number of noise types to a batch of signals.
"""


import torch
from cardiac_map_diffusion.data import utils_noise as nl
from cardiac_map_diffusion.data import ep_noise_sam as ep

def generate_unique_list_noises(noise_ids, min_number_noises, max_number_noises):
    ''' - Generates a unique list of noise ids based on the input noise_ids.
    '''
    if not isinstance(noise_ids, torch.Tensor):
        noise_ids = torch.tensor(noise_ids)
    num_noises = torch.randint(min_number_noises, max_number_noises + 1, (1,)).item()
    indices = torch.randperm(len(noise_ids))[:num_noises]
    unique_noises = noise_ids[indices]
    return unique_noises


def several_noises(noise_ids, signals_squeezed, device, ep_noise_arrays):
    ''' - Introduces several types of noise to the signals.
        - The index corrsepoding to each noise type is as follows:
            1: Gaussian noise
            2: Spike noise
            3: Baseline wander noise
            4: Powerline noise
            5: Electrophysiological noise
            6: Truncation noise
    '''
    noise = torch.zeros_like(signals_squeezed).to(device)
    assert noise.device == signals_squeezed.device  

    for noise_id in noise_ids:
        if noise_id == 1:
            noise += nl.introduce_gaussian_noise(signals_squeezed)

        elif noise_id == 2:
            noise += nl.introduce_spike_noise(signals_squeezed)

        elif noise_id == 3:
            noise += nl.introduce_baseline_wander(signals_squeezed)

        elif noise_id == 4:
            noise += nl.introduce_baseline_wander(signals_squeezed, min_freq=60,
                             max_freq=60, min_sins=1, max_sins=1,
                             max_amplitude=0.1)
        
        elif noise_id == 5:
            noise += ep.generate_epnoise(ep_noise_arrays, signals_squeezed.shape[0])
            
    noisy_signals = signals_squeezed + noise

    if 6 in noise_ids:
        noisy_signals = nl.introduce_truncation_noise(noisy_signals)

    return noisy_signals


def introduce_several_noises(signals, ep_noise_arrays, noise_batchwise=True, noise_ids=[1, 2, 3, 4, 5, 6],
                             min_number_noises=1, max_number_noises=6):
    ''' - The main function to introduce a random number of noise types to a batch of signals.
        - If noise_batchwise: The same combination of noise types will be introduced to the whole batch, however,
            the parameters of each noise type will be different for each signal.
        - If noise_batchwise false: Each signal in the batch has different noise types and parameters.
    '''
    noise_ids = torch.tensor(noise_ids)
    device = signals.device
    signals_squeezed = signals.clone().squeeze(1)

    if noise_batchwise:
        list_noises = generate_unique_list_noises(noise_ids=noise_ids, min_number_noises=min_number_noises, max_number_noises=max_number_noises).to(device)
        noisy_signals = several_noises(list_noises, signals_squeezed, device, ep_noise_arrays).unsqueeze(1).float()
    
    else:
        noisy_signals = torch.zeros_like(signals_squeezed)
        for i in range(signals_squeezed.shape[0]):        
            list_noises = generate_unique_list_noises(noise_ids=noise_ids, min_number_noises=min_number_noises, max_number_noises=max_number_noises).to(device)
            noisy_signals[i] = several_noises(list_noises, signals_squeezed[i].unsqueeze(0), device, ep_noise_arrays).float()
        noisy_signals = noisy_signals.unsqueeze(1)

    return noisy_signals

  
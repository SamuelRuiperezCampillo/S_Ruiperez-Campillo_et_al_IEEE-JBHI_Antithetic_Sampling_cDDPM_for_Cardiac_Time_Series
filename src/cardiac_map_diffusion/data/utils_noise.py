"""Batchwise, pure-PyTorch noise generators used by the diffusion data pipeline.

Faithful migration of ``utils_noise.py`` (logic unchanged; no paths, no
first-party imports). Each function returns the noise component only (to be
superposed on the clean signals), except the truncation function, which
modifies the signals directly. Based on Ruiperez-Campillo et al. (2024), with
batchwise handling in PyTorch. Consumed by
:mod:`cardiac_map_diffusion.data.generate_noise`.
"""

import torch

def introduce_gaussian_noise(signals_squeezed, loc=0, scale=0.1):
    ''' - Generate white noise.
    '''
    batch_size, signal_length = signals_squeezed.shape
    gaussian_noise = torch.normal(mean=loc, std=scale, size=[batch_size, signal_length], device=signals_squeezed.device)
    return gaussian_noise


def introduce_spike_noise(signals_squeezed, max_index=40):
    ''' - Creates spike noise at a random index between 2 and max_index in each signal.
        - The spike noise is a random value between 10% and 80% of the "amplitude" of the signal.
    '''
    dtype = signals_squeezed.dtype
    batch_size = signals_squeezed.size(0)
    spike_noise = torch.zeros(signals_squeezed.size(), dtype=dtype)
    spike_index = torch.randint(2, max_index, (batch_size,))

    max_value = torch.max(signals_squeezed, dim=-1).values
    initial_value = signals_squeezed[:, 0]
    amplitude = max_value - initial_value
    lower_bound = initial_value + 0.1 * amplitude
    upper_bound = initial_value + 0.8 * amplitude

    random_uniform = torch.rand(batch_size, dtype=dtype)
    spike = lower_bound + random_uniform * (upper_bound - lower_bound)
    spike_noise[range(batch_size), spike_index] = spike
    return spike_noise



def generate_random_freqs(min_freq, max_freq, min_sins, max_sins):
    ''' - Helper function to generate a random number of sinusoids with random frequencies
            between min_freq and max_freq.
    '''
    n = torch.randint(min_sins, max_sins + 1, (1,)).item()
    random_list = (torch.rand(n) * (max_freq - min_freq) + min_freq).round(decimals=4)
    return random_list

def introduce_baseline_wander(signals_squeezed, sample_frequency=1000, min_freq=0.01,
                              max_freq=0.3, min_sins=1, max_sins=4,
                              max_amplitude=1):
    ''' - Introduces baseline wander noise to the signals.
        - The baseline wander noise is a sum of sinusoids with random frequencies between min_freq and max_freq.
        - The function can also be used to introduce powerline noise by adjusting min_freq and max_freq.
    '''
    batch_size, signal_length = signals_squeezed.shape

    time = torch.arange(signal_length).float() / sample_frequency

    baseline_wander_tensor = torch.zeros(signals_squeezed.size())

    for i in range(batch_size):
        baseline_wander = torch.zeros(signal_length)
        frequencies = generate_random_freqs(min_freq, max_freq, min_sins, max_sins)

        for freq in frequencies:
            A = torch.rand(1).uniform_(0.3 * max_amplitude, 1 * max_amplitude)
            phase = torch.rand(1).uniform_(0, 2 * torch.pi)
            wander = A * torch.sin(2 * torch.pi * freq * time + phase)
            baseline_wander += wander.squeeze()

        baseline_wander = baseline_wander - baseline_wander.mean()
        baseline_wander_tensor[i,:] = baseline_wander

    return baseline_wander_tensor


def introduce_truncation_noise(signals_squeezed, percent=0.8, var=0.05, beginning=False):
    ''' - Introduces truncation noise to the signals.
        - The truncation noise is a random mask that sets a random percentage of the signal to zero.
        - The percentage is between (percent - var) and (percent + var).
    '''
    batch_size, signal_length = signals_squeezed.shape
    percentages = (percent - var) + torch.rand(batch_size) * (2 * var)

    zero_indices = (percentages * signal_length).long()
    range_tensor = torch.arange(signal_length).expand(batch_size, -1)

    if beginning:
        mask = range_tensor < zero_indices.unsqueeze(1)
    else:
        mask = range_tensor >= zero_indices.unsqueeze(1)

    signals_squeezed[mask] = 0

    return signals_squeezed

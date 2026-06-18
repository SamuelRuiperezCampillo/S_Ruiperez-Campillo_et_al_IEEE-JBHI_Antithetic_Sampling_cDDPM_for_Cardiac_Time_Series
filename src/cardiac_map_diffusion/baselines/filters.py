"""Classical filtering baseline (Butterworth band-pass + notch).

Faithful migration of ``Diffusion_MAP_fullpipeline_final/filters.py``. The body
is copied verbatim; only this module docstring has been expanded to note the
migration. No filter design, cut-off frequencies or logic has changed.

    - This file contains the functions to apply filters to the noisy signal for creating a baseline.
"""

import numpy as np
from scipy.signal import butter, lfilter, iirnotch

def butterworth_notch(noisy_signal, fs=1000, lowcut=0.01, highcut=400, order=5, f0=60, Q=30.0):
    """
    Apply a Butterworth filter to a noisy signal, apply a notch filter
    to the Butterworth-filtered signal afterward.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    w0 = f0 / nyq

    b_butter, a_butter = butter(order, [low, high], btype='band', analog=False)
    butter_filtered = lfilter(b_butter, a_butter, noisy_signal)

    b_notch, a_notch = iirnotch(f0, Q, fs=fs)
    filtered = lfilter(b_notch, a_notch, butter_filtered)

    return filtered


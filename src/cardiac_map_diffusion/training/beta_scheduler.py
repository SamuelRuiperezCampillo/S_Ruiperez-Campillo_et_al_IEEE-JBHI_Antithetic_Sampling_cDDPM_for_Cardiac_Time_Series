"""KL/beta annealing schedules used by the baseline (VAE) training track.

Provides cyclical and monotonic schedules for the KL-divergence weight (``beta``)
during VAE training: linear, sigmoid and cosine cyclical schedules
(``frange_cycle_*``), plus helpers to build a square-wave schedule
(``create_square_function``) and to prepend a warm-up of zeros
(``add_zeros_to_schedule``). This is a faithful migration of
``MAP_VAE/beta_scheduler.py``: the body is copied verbatim and only this module
docstring has been added for the package layout. No logic, math, or
hyperparameters have been changed.
"""

import numpy as np
import math

def frange_cycle_linear(n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio=0.5):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L

def frange_cycle_sigmoid(n_iter, start=0.0, stop=1.0, n_cycle=4, ratio=0.5):
    L = np.ones(n_iter)
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio) # step is in [0,1]

    # transform into [-6, 6] for plots: v*12.-6.

    for c in range(n_cycle):

        v , i = start , 0
        while v <= stop:
            L[int(i+c*period)] = 1.0/(1.0+ np.exp(- (v*12.-6.)))
            v += step
            i += 1
    return L

def frange_cycle_cosine(n_iter, start=0, stop=1, n_cycle=4, ratio=0.5):
    L = np.ones(n_iter)
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio) # step is in [0,1]

    # transform into [0, pi] for plots:

    for c in range(n_cycle):

        v , i = start , 0
        while v <= stop:
            L[int(i+c*period)] = 0.5-.5*math.cos(v*math.pi)
            v += step
            i += 1
    return L

def create_square_function(n_iter, ratio_th=4):
    quarter = n_iter // ratio_th
    zeros = np.zeros(quarter)
    ones = np.ones(3 * quarter)
    sq_function = np.concatenate((zeros, ones))
    if len(sq_function)<n_iter:
        extra_ones = np.ones(n_iter-len(sq_function))
        sq_function = np.concatenate((sq_function, extra_ones))
    return sq_function

def add_zeros_to_schedule(schedule, ratio):
    return np.concatenate((np.zeros(len(schedule)-(len(schedule) - len(schedule)//ratio)),
                           schedule[:-len(schedule)//ratio]))
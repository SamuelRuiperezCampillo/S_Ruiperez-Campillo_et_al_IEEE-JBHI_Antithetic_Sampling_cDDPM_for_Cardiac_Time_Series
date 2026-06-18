"""Learning-rate scheduler used by the baseline (VAE) training track.

Provides a cosine-annealing learning-rate schedule with a linear warm-up phase
(``CosineAnnealingLRWarmup``). This is a faithful migration of
``MAP_VAE/lr_scheduler.py``: the body is copied verbatim and only this module
docstring has been added for the package layout. No logic, math, or
hyperparameters have been changed.
"""

import torch
import math
import functools

def _cosine_decay_warmup(iteration, warmup_iterations, total_iterations):
    """
    Linear warmup from 0 --> 1.0, then decay using cosine decay to 0.0
    """
    if iteration <= warmup_iterations:
        multiplier = iteration / warmup_iterations
    else:
        multiplier = (iteration - warmup_iterations) / (total_iterations - warmup_iterations)
        multiplier = 0.5 * (1 + math.cos(math.pi * multiplier))
    return multiplier

def CosineAnnealingLRWarmup(optimizer, T_max, T_warmup):
    _decay_func = functools.partial(
        _cosine_decay_warmup,
        warmup_iterations=T_warmup, total_iterations=T_max
    )
    scheduler   = torch.optim.lr_scheduler.LambdaLR(optimizer, _decay_func)
    return scheduler
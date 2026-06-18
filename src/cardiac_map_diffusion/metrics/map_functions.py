"""Diffusion-track MAP helpers -- thin re-export of the shared canonical module.

The repo historically carried three near-identical ``MAP_functions`` copies
(diffusion / baseline / filter tracks). A line-level diff confirmed that every
function the pipelines actually call is byte-identical across them (the only
differences were docstrings/comments, commented-out unused sklearn regressors,
the presence of ``butterworth_notch_filter``, and an inactive ``get_MAP_vent_data``
path literal). The single implementation now lives in
:mod:`cardiac_map_diffusion.metrics.map_functions_baselines`; this module re-exports
it under the historical ``map_functions`` name used by the diffusion pipeline.
(``compute_rmse`` still returns the pooled MSE without a square root, as in the paper.)
"""

from cardiac_map_diffusion.metrics.map_functions_baselines import *  # noqa: F401,F403
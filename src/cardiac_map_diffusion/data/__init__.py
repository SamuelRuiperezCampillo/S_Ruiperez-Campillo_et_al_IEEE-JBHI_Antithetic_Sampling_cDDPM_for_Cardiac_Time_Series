"""Data loading and noise generation for the ventricular MAP cohort.

Two faithful loaders are preserved side by side (they read the *same* cohort
files and produce identical patient-grouped splits at ``seed_split=29``):

* :mod:`~cardiac_map_diffusion.data.data_sam` -- used by the diffusion pipeline.
* :mod:`~cardiac_map_diffusion.data.data_baselines` -- used by the DL/classical
  baselines and the downstream APD task.

See :mod:`cardiac_map_diffusion.paths` for how the private cohort is located.
"""

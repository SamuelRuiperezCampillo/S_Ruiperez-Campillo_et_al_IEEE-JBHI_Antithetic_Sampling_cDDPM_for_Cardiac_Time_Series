"""cardiac_map_diffusion -- antithetic-sampling diffusion denoising for cardiac MAP signals.

Reference implementation accompanying the paper
*"Antithetic Sampling Enhanced Probabilistic Diffusion for Denoising Cardiac
Time Series"*.

The package is organised into two faithful pipelines that share the same
patient-grouped cross-validation splits:

* the conditional DDPM (``diffusion`` + ``models`` + ``data`` + ``metrics``), and
* the deep-learning / classical baselines (``models`` + ``baselines`` + ``data`` +
  ``metrics``), plus the downstream APD-prediction task (``downstream``).

See the top-level ``README.md`` and ``docs/`` for usage and reproducibility, and
``cardiac_map_diffusion.paths`` for how the (private) cohort is located.
"""

__version__ = "0.1.0"

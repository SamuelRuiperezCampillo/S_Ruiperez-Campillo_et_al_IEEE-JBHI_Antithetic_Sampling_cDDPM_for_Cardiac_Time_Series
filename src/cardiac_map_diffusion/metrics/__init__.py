"""Reconstruction metrics and MAP signal utilities.

Two metric modules are preserved to keep each pipeline byte-faithful to the paper:

* :mod:`~cardiac_map_diffusion.metrics.map_functions` -- data-prep + metric helpers
  used by the diffusion loader/trainer (the ``mapf`` of the diffusion track).
* :mod:`~cardiac_map_diffusion.metrics.map_functions_metrics` -- the baseline track's
  metric helpers.

.. warning::
   ``compute_rmse(..., mode='total')`` returns the **pooled MSE (no square root)**.
   The "RMSE" column reported in the paper is therefore an MSE; the true RMSE is
   its square root. This behaviour is intentional and preserved -- see
   ``docs/reproducibility.md``.
"""

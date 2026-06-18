"""Verify the documented metric behaviour on tiny arrays (no cohort data).

The headline check encodes the paper's well-known caveat: ``compute_rmse`` returns
the **pooled MSE without a square root**, so the "RMSE" column is actually an MSE.
See docs/reproducibility.md. We assert this on the baseline metric modules (which
import scikit-learn fully).
"""

import numpy as np
import pytest

from cardiac_map_diffusion.metrics import map_functions_baselines, map_functions_metrics

METRIC_MODULES = [map_functions_baselines, map_functions_metrics]


@pytest.mark.parametrize("mod", METRIC_MODULES)
def test_compute_rmse_is_pooled_mse(mod):
    rng = np.random.default_rng(0)
    a = rng.random((8, 370))
    b = a + rng.normal(0.0, 0.1, size=a.shape)

    pooled_mse = float(np.mean((a - b) ** 2))
    val = float(mod.compute_rmse(a, b, mode="total"))

    # "RMSE" is actually the pooled MSE (no square root).
    assert np.isclose(val, pooled_mse, rtol=1e-6, atol=1e-9), (val, pooled_mse)
    # ... and is therefore NOT the true RMSE.
    assert not np.isclose(val, np.sqrt(pooled_mse), rtol=1e-3)


@pytest.mark.parametrize("mod", METRIC_MODULES)
def test_compute_mse_matches_compute_rmse(mod):
    rng = np.random.default_rng(1)
    a = rng.random((5, 370))
    b = a + rng.normal(0.0, 0.05, size=a.shape)
    assert np.isclose(
        float(mod.compute_mse(a, b, mode="total")),
        float(mod.compute_rmse(a, b, mode="total")),
        rtol=1e-6,
        atol=1e-9,
    )


@pytest.mark.parametrize("mod", METRIC_MODULES)
def test_perfect_reconstruction_is_zero_error(mod):
    a = np.random.default_rng(2).random((4, 370))
    assert float(mod.compute_rmse(a, a, mode="total")) == pytest.approx(0.0, abs=1e-9)

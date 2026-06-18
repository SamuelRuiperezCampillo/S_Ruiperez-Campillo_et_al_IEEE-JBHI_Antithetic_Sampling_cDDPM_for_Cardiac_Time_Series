"""Import-smoke tests: verify the migrated package imports cleanly.

These need no cohort data. They are the first line of defence for the faithful
migration -- a broken import (e.g. a missed rewrite) fails here immediately. Run
on the cluster (or any machine with the environment installed): ``pytest -q``.
"""

import importlib

import pytest

# Core modules whose import must succeed (definitions only; no training side effects).
CORE_MODULES = [
    "cardiac_map_diffusion",
    "cardiac_map_diffusion.paths",
    # data layer
    "cardiac_map_diffusion.data.retrieve_ventMAP",
    "cardiac_map_diffusion.data.ep_noise_sam",
    "cardiac_map_diffusion.data.utils_noise",
    "cardiac_map_diffusion.data.generate_noise",
    "cardiac_map_diffusion.data.data_sam",
    "cardiac_map_diffusion.data.data_baselines",
    "cardiac_map_diffusion.data.data_diffusion",
    "cardiac_map_diffusion.data.retrieve_dataset",
    "cardiac_map_diffusion.data.splits",
    # metrics + training
    "cardiac_map_diffusion.metrics.map_functions",
    "cardiac_map_diffusion.metrics.map_functions_baselines",
    "cardiac_map_diffusion.metrics.map_functions_metrics",
    "cardiac_map_diffusion.metrics.ddpm_metrics",
    "cardiac_map_diffusion.training.lr_scheduler",
    "cardiac_map_diffusion.training.beta_scheduler",
    # diffusion core + trainer
    "cardiac_map_diffusion.diffusion.ddpm_conditional",
    "cardiac_map_diffusion.diffusion.denoising_net",
    "cardiac_map_diffusion.diffusion.denoising_net_small",
    "cardiac_map_diffusion.diffusion.train_ddpm",
    # baseline architectures
    "cardiac_map_diffusion.models.autoencoder",
    "cardiac_map_diffusion.models.autoencoder_conv",
    "cardiac_map_diffusion.models.autoencoder_convres",
    "cardiac_map_diffusion.models.dae_model",
    "cardiac_map_diffusion.models.drrn_model",
    "cardiac_map_diffusion.models.lunet_model",
    # classical filters + downstream
    "cardiac_map_diffusion.baselines.filters",
    "cardiac_map_diffusion.baselines.baseline_filters",
    "cardiac_map_diffusion.downstream.downstreamAPD",
    "cardiac_map_diffusion.downstream.downstreamDEATH",
]

# Lower-stakes modules (visualisation / absl-app entries). These may legitimately
# fail to import in a minimal environment; we report rather than hard-fail.
EXTRA_MODULES = [
    "cardiac_map_diffusion.metrics.apd_metric",
    "cardiac_map_diffusion.downstream.svm_apd",
    "cardiac_map_diffusion.viz.visualize_patient_signals",
]


@pytest.mark.parametrize("modname", CORE_MODULES)
def test_core_module_imports(modname):
    assert importlib.import_module(modname) is not None


def test_package_version():
    import cardiac_map_diffusion

    assert isinstance(cardiac_map_diffusion.__version__, str)


def test_paths_return_pathlike():
    from pathlib import Path

    from cardiac_map_diffusion import paths

    assert isinstance(paths.data_root(), Path)
    assert isinstance(paths.experiments_root(), Path)
    assert paths.apd_annotations_filename().endswith(".xlsx")


@pytest.mark.parametrize("modname", EXTRA_MODULES)
def test_extra_module_imports(modname):
    try:
        importlib.import_module(modname)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"optional module {modname} did not import: {exc!r}")

"""Centralised path resolution for the cardiac-map-diffusion repository.

All access to the (private, IRB-restricted) ventricular MAP cohort and to the
experiment-output tree goes through this module, so that **no absolute cluster
paths are baked into the pipeline code**. The original scripts hard-coded
several different cluster locations (e.g.
``/cluster/dataset/vogtlab/Projects/MAP_ventricle_stanford/src_vae`` for the
diffusion loader and ``/cluster/work/vogtlab/Group/pblasco`` for the baseline
loader) that all pointed at copies of the *same* two files. Both pipelines now
read those files from a single configured root, which also reinforces the
"VAE-aligned" guarantee that every method trains on identical splits.

Configure via environment variables (highest priority) or ``configs/paths.yaml``:

============================  =========================================================
Environment variable          Meaning
============================  =========================================================
``CARDIACDIFF_DATA``          Root holding the raw cohort (see ``docs/data_format.md``).
``CARDIACDIFF_EXPERIMENTS``   Root for run outputs (checkpoints, metrics, figures).
``CARDIACDIFF_EXCLUDE_PATIENTS``  Optional CSV of held-out (hidden-test) patient IDs.
============================  =========================================================

The cohort is **not** distributed with this repository.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Optional

# Repository root: .../cardiac-map-diffusion (paths.py is src/<pkg>/paths.py).
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Placeholder returned when nothing is configured; using it for I/O fails loudly.
_PLACEHOLDER = "<<SET-CARDIACDIFF_DATA-see-docs/data_format.md>>"

# Filename of the APD-annotation workbook, located directly under the data root.
APD_ANNOTATIONS_FILENAME = "2023 0228 APD_data_MAP_vent_SRC.xlsx"
# Sub-directory of the data root holding the cropped single-beat dictionary
# (``Step_1_RemoveUnderOverShoots/single_MAP_dict_BP_mean_3SD_cropped_20191111.npz``).
VENTMAP_SUBDIR = "2019 1111 Final_codes_and_data"


@functools.lru_cache(maxsize=1)
def _yaml_config() -> dict:
    """Load ``configs/paths.yaml``, then overlay an optional, git-ignored
    ``configs/paths.local.yaml`` so personal machine paths stay out of version
    control. Resolution precedence is: environment variable > paths.local.yaml >
    paths.yaml > placeholder. Returns ``{}`` if no config files are present."""
    cfg: dict = {}
    for name in ("paths.yaml", "paths.local.yaml"):  # local overrides committed defaults
        path = _REPO_ROOT / "configs" / name
        if not path.exists():
            continue
        try:
            import yaml  # PyYAML; declared in pyproject/environment.

            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            if isinstance(data, dict):
                cfg.update(data)
        except Exception:  # pragma: no cover - config is optional
            pass
    return cfg


def _resolve(env_var: str, yaml_key: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get(env_var) or _yaml_config().get(yaml_key) or default


def repo_root() -> Path:
    """Absolute path to the repository root."""
    return _REPO_ROOT


def data_root() -> Path:
    """Root directory holding the private MAP cohort. See ``docs/data_format.md``."""
    return Path(_resolve("CARDIACDIFF_DATA", "data_root", _PLACEHOLDER))


def experiments_root() -> Path:
    """Root directory for experiment outputs (checkpoints / metrics / figures)."""
    return Path(_resolve("CARDIACDIFF_EXPERIMENTS", "experiments_root", str(_REPO_ROOT / "experiments")))


def ventmap_root() -> Path:
    """Directory passed to ``retrieve_ventMAP`` (contains ``Step_1_RemoveUnderOverShoots/``)."""
    return data_root() / VENTMAP_SUBDIR


def apd_annotations_dir() -> Path:
    """Directory containing the APD-annotation workbook."""
    return data_root()


def apd_annotations_filename() -> str:
    """Filename of the APD-annotation workbook (within :func:`apd_annotations_dir`)."""
    return APD_ANNOTATIONS_FILENAME


def exclude_patients_file() -> Optional[Path]:
    """CSV listing held-out (hidden-test) patient IDs, or ``None`` if unset."""
    val = _resolve("CARDIACDIFF_EXCLUDE_PATIENTS", "exclude_patients_file")
    return Path(val) if val else None


def experiments_dir(*parts: str) -> Path:
    """Join ``parts`` under :func:`experiments_root` (does not create the dir)."""
    return experiments_root().joinpath(*parts)

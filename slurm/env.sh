#!/bin/bash
# slurm/env.sh -- shared environment for all SLURM jobs in this repo.
# Sourced by every *.sbatch script. Edit the defaults below to match your
# cluster, or `export` these variables before calling sbatch.
# ---------------------------------------------------------------------------

# --- Conda -----------------------------------------------------------------
# Point CONDA_SH at your conda profile and CONDA_ENV at the env (name or path).
# On the original cluster this was the `gpu_env2` environment.
CONDA_SH="${CONDA_SH:-$HOME/anaconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-cardiac-map-diffusion}"

# --- Data + outputs (see docs/data_format.md) ------------------------------
# CARDIACDIFF_DATA: the private MAP cohort root (NOT shipped). Either set it
# here, OR leave it unset and use a git-ignored configs/paths.local.yaml
# (paths.py reads the env var FIRST, so only set this if you are NOT relying on
# paths.local.yaml -- otherwise a placeholder would shadow your YAML value).
# export CARDIACDIFF_DATA=/path/to/MAP_cohort_root

# Where run outputs go. Exported with a real default because the SLURM scripts
# below interpolate $CARDIACDIFF_EXPERIMENTS directly in shell, so it must be a
# genuine environment variable here (not only a paths.yaml entry).
export CARDIACDIFF_EXPERIMENTS="${CARDIACDIFF_EXPERIMENTS:-$PWD/experiments}"

# Optional: CSV of held-out (hidden test set) patient IDs.
# export CARDIACDIFF_EXCLUDE_PATIENTS=/path/to/hidden_test_set_selected.csv

# Repo root, so `cd "$REPO_ROOT"` makes scripts/ and configs/ resolve.
export REPO_ROOT="${REPO_ROOT:-$PWD}"

export PYTHONUNBUFFERED=1

# Activate the environment.
if [ -f "$CONDA_SH" ]; then
    # shellcheck disable=SC1090
    source "$CONDA_SH" && conda activate "$CONDA_ENV"
else
    echo "WARNING: conda profile not found at $CONDA_SH; assuming the environment is already active." >&2
fi

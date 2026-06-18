#!/bin/bash
#SBATCH --job-name=filt_butter
#SBATCH --output=filt_butter_%j.out
#SBATCH --time=12:00:00
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000
# ---------------------------------------------------------------------------
# Butterworth + notch filter baseline (all folds). CPU job.
# Faithful parameterisation of MAP_VAE/submit_jobs_butterworth.sh
# (fs=1000, lowcut=0.01, highcut=400, order=5, f0=60, Q=30.0 -- the defaults
# inside baseline_filters.py).
# ---------------------------------------------------------------------------
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$REPO_ROOT"

python -u scripts/run_filters.py --filter_type=butterworth --noise_type=allmixed \
  --seed_split=29 --num_folds=4 \
  --working_dir="$CARDIACDIFF_EXPERIMENTS/butterworth" --save_results

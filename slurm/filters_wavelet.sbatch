#!/bin/bash
#SBATCH --job-name=filt_wave
#SBATCH --output=filt_wave_%j.out
#SBATCH --time=12:00:00
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000
# ---------------------------------------------------------------------------
# Wavelet-shrinkage denoising baseline (all folds). CPU job.
# Faithful parameterisation of MAP_VAE/submit_jobs_wavelet.sh, which sweeps
# wavelet in {db4, sym4}, levels in {3, 5}, mode in {soft, hard}, method in
# {sure, visushrink}. Override via env vars (defaults below = one sweep point).
# ---------------------------------------------------------------------------
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$REPO_ROOT"

WAVELET="${WAVELET:-db4}"; LEVELS="${LEVELS:-3}"
THRESH_MODE="${THRESH_MODE:-soft}"; THRESH_METHOD="${THRESH_METHOD:-sure}"

python -u scripts/run_filters.py --filter_type=wavelet --noise_type=allmixed \
  --seed_split=29 --num_folds=4 \
  --working_dir="$CARDIACDIFF_EXPERIMENTS/wavelet_${WAVELET}_L${LEVELS}_${THRESH_MODE}_${THRESH_METHOD}" \
  --save_results --wavelet_type="$WAVELET" --wavelet_levels="$LEVELS" \
  --wavelet_threshold_mode="$THRESH_MODE" --wavelet_threshold_method="$THRESH_METHOD"

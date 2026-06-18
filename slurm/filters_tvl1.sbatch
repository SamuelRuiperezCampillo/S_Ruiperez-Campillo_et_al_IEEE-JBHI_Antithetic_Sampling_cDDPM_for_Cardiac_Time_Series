#!/bin/bash
#SBATCH --job-name=filt_tvl1
#SBATCH --output=filt_tvl1_%j.out
#SBATCH --time=18:00:00
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000
# ---------------------------------------------------------------------------
# Total-Variation (TV-L1) denoising baseline (all folds). CPU job.
# Faithful parameterisation of MAP_VAE/submit_jobs_tvl1.sh, which sweeps the
# regularisation weight over {0.01, 0.02, 0.05, 0.1}. Set TV_WEIGHT to pick one,
# e.g.  TV_WEIGHT=0.05 sbatch slurm/filters_tvl1.sbatch
# ---------------------------------------------------------------------------
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$REPO_ROOT"

TV_WEIGHT="${TV_WEIGHT:-0.01}"

python -u scripts/run_filters.py --filter_type=tv_l1 --tv_weight="$TV_WEIGHT" \
  --noise_type=allmixed --seed_split=29 --num_folds=4 \
  --working_dir="$CARDIACDIFF_EXPERIMENTS/tvl1_weight_${TV_WEIGHT}" --save_results

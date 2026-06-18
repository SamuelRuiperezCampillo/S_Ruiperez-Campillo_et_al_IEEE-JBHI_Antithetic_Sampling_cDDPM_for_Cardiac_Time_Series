#!/bin/bash
#SBATCH --job-name=apd_baseline
#SBATCH --output=apd_baseline_%j.out
#SBATCH --time=02:00:00
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6000
# ---------------------------------------------------------------------------
# Downstream APD30/60/90 SVR baseline on noisy signals (all folds). CPU job.
# Faithful parameterisation of MAP_VAE/submit_jobs_baseline_apd.sh (noisy branch).
# ---------------------------------------------------------------------------
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$REPO_ROOT"

python -u scripts/downstream_apd.py --noise_type=allmixed --seed_split=29 \
  --num_folds=4 --random_seed=17 \
  --output_dir="$CARDIACDIFF_EXPERIMENTS/baseline_apd_noisy"

#!/bin/bash
#SBATCH --job-name=denoise_drrn
#SBATCH --output=denoise_drrn_%j.out
#SBATCH --time=80:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000
# ---------------------------------------------------------------------------
# DRRN baseline (all folds). Faithful parameterisation of MAP_VAE/submit_jobs_drrn.sh.
# ---------------------------------------------------------------------------
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$REPO_ROOT"

python -u scripts/denoise_drrn.py --config=configs/config_drrn.py \
  --config.working_dir="$CARDIACDIFF_EXPERIMENTS/drrn" \
  --config.cluster=True --config.batch_size=32 --config.optimizer=adam \
  --config.num_workers=4 --config.prefetch_factor=2 --config.learning_rate=5e-5 \
  --config.seed_split=29 --config.num_epochs=1000 --config.noise_type=allmixed \
  --config.seed=17 --config.hidden_size=64 --config.early_stopping=True \
  --config.early_stopping_patience=10 --config.experiment_name=DRRN_baseline \
  --config.num_folds=4 --config.split_number=100 \
  ${CARDIACDIFF_EXCLUDE_PATIENTS:+--config.exclude_patients_file="$CARDIACDIFF_EXCLUDE_PATIENTS"}

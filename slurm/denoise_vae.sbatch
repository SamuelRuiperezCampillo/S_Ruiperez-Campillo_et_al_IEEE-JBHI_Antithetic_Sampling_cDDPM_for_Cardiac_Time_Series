#!/bin/bash
#SBATCH --job-name=denoise_vae
#SBATCH --output=denoise_vae_%j.out
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000
# ---------------------------------------------------------------------------
# beta-VAE baseline (CNN, beta=0.005), all folds. Faithful parameterisation of
# MAP_VAE/submit_jobs_test.sh.
# NOTE: submit_jobs_test.sh set num_epochs=1 (a smoke-test value). The released
# checkpoints are VAE_CNN_beta_0.005_rs_17_fold[0-3].pth; set NUM_EPOCHS to your
# production value, e.g.  NUM_EPOCHS=1000 sbatch slurm/denoise_vae.sbatch
# ---------------------------------------------------------------------------
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$REPO_ROOT"

NUM_EPOCHS="${NUM_EPOCHS:-1}"   # 1 == the smoke value in submit_jobs_test.sh; override for a real run

python -u scripts/denoise_vae.py --config=configs/config.py \
  --config.working_dir="$CARDIACDIFF_EXPERIMENTS/vae" \
  --config.cluster=True --config.batch_size=32 --config.optimizer=adam \
  --config.num_workers=4 --config.prefetch_factor=2 --config.learning_rate=5e-3 \
  --config.beta_schedule_mode=linear --config.seed_split=29 \
  --config.beta_schedule_cycles=4 --config.beta_schedule_ratio=0.5 \
  --config.num_epochs="$NUM_EPOCHS" --config.noise_type=allmixed --config.beta=0.005 \
  --config.seed=17 --config.model=CNN --config.weight_constant=1 \
  --config.experiment_name=allmixed_beta_0.005 \
  --config.num_folds=4 --config.split_number=100 \
  ${CARDIACDIFF_EXCLUDE_PATIENTS:+--config.exclude_patients_file="$CARDIACDIFF_EXCLUDE_PATIENTS"}

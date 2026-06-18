#!/bin/bash
#SBATCH --job-name=ddpm_train
#SBATCH --output=ddpm_train_%j.out
#SBATCH --time=80:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8000
# ---------------------------------------------------------------------------
# Conditional DDPM training (experiment "final_big_80"), all folds.
# Faithful parameterisation of Diffusion_MAP_fullpipeline_final/submit_jobs_cluster.sh.
# NOTE: that script passed --n_splits 5; the paper reports 4-fold CV, so the
# default here is 4 (see docs/reproducibility.md / docs/decisions.md). Override
# any value with an environment variable, e.g.  N_SPLITS=5 sbatch slurm/train_ddpm.sbatch
# ---------------------------------------------------------------------------
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$REPO_ROOT"

EXPERIMENT="${EXPERIMENT:-final_big_80}"
SEED="${SEED:-17}"; SEED_SPLIT="${SEED_SPLIT:-29}"; N_SPLITS="${N_SPLITS:-4}"
NOISE_STEPS="${NOISE_STEPS:-5000}"; FEATS="${FEATS:-80}"; NUM_EPOCHS="${NUM_EPOCHS:-400}"

python -u scripts/train_ddpm.py \
  --experiment "$EXPERIMENT" --device cuda \
  --seed "$SEED" --seed_split "$SEED_SPLIT" --signal_length 370 --n_prints 5 --lr 0.001 \
  --noise_batchwise True --noise_steps "$NOISE_STEPS" --noise_schedule quadratic \
  --beta_start 0.0001 --beta_end 0.05 --use_pretrained False \
  --model_small False --feats "$FEATS" --num_epochs "$NUM_EPOCHS" --batch_size 96 \
  --batch_size_test 192 --normalise True --test_size 0.2 --n_splits "$N_SPLITS" \
  --step_size 100 --gamma 0.5 \
  ${CARDIACDIFF_EXCLUDE_PATIENTS:+--exclude_patients_file "$CARDIACDIFF_EXCLUDE_PATIENTS"}

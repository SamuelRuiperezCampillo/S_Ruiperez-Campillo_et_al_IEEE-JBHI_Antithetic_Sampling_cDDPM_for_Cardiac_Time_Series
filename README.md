# S_Ruiperez-Campillo_et_al_IEEE-JBHI_Antithetic_Sampling_cDDPM_for_Cardiac_Time_Series
Code associated to Journal Article: S. Ruiperez-Campillo et al. “Antithetic Sampling Enhanced Probabilistic Diffusion for Denoising Cardiac Time Series.” IEEE Journal of Biomedical and Health Informatics, accepted June 2026.

Reference implementation for **"Antithetic Sampling Enhanced Probabilistic
Diffusion for Denoising Cardiac Time Series."**

A conditional denoising diffusion probabilistic model (cDDPM) removes mixed
physiological and acquisition noise from single-beat ventricular **monophasic
action potential (MAP)** signals. The key contribution is an **antithetic-variates
(AV) sampler** that runs the reverse chain as sign-flipped, negatively-correlated
pairs, sharply reducing the variance of the reconstruction at fixed compute, and
yielding a usable predictive-uncertainty signal. The repository reproduces cDDPM
training, all baselines (β-VAE, DAE, DRRN, LU-Net; Butterworth+notch, TV-L1,
wavelet shrinkage, and other classical filters), and the downstream APD30/60/90
prediction task.

> **Data availability.** The ventricular MAP cohort is private and IRB-restricted
> and is **not** included here. The code reads it from a configurable location;
> see [docs/data_format.md](docs/data_format.md).

## Installation

```bash
conda env create -f environment.yml      # cluster gpu_env2: PyTorch 2.5.1 / CUDA 12.1
conda activate cardiac-map-diffusion
pip install -e .
# (pip-only alternative: pip install -r requirements.txt && pip install -e .)
```

## Point at the data

```bash
export CARDIACDIFF_DATA=/path/to/MAP_cohort_root        # see docs/data_format.md
export CARDIACDIFF_EXPERIMENTS=$PWD/experiments         # where outputs are written
# optional: export CARDIACDIFF_EXCLUDE_PATIENTS=/path/to/hidden_test_set_selected.csv
```

(Or set the same values in [`configs/paths.yaml`](configs/paths.yaml).)

## Quickstart

```bash
# Train the conditional DDPM (paper config, 4 folds) — GPU
python scripts/train_ddpm.py --experiment final_big_80 --device cuda \
  --seed 17 --seed_split 29 --signal_length 370 --n_prints 5 --lr 0.001 \
  --noise_batchwise True --noise_steps 5000 --noise_schedule quadratic \
  --beta_start 0.0001 --beta_end 0.05 --use_pretrained False --model_small False \
  --feats 80 --num_epochs 400 --batch_size 96 --batch_size_test 192 \
  --normalise True --test_size 0.2 --n_splits 4 --step_size 100 --gamma 0.5

# A classical baseline (CPU)
python scripts/run_filters.py --filter_type butterworth --noise_type allmixed \
  --seed_split 29 --num_folds 4 --working_dir "$CARDIACDIFF_EXPERIMENTS/butterworth" --save_results
```

On a SLURM cluster, prefer the ready-made jobs (defaults reproduce the paper):

```bash
sbatch slurm/train_ddpm.sbatch       # + denoise_{vae,dae,drrn,lunet}, filters_*, downstream_apd
```

See [slurm/README.md](slurm/README.md) and [docs/reproducibility.md](docs/reproducibility.md).

## Repository layout

```
cardiac-map-diffusion/
├── src/cardiac_map_diffusion/
│   ├── paths.py            # single point of (de-hard-coded) data/output path resolution
│   ├── data/               # cohort loaders (diffusion + baseline tracks), noise generators
│   ├── diffusion/          # DDPM process, denoising networks, trainer
│   ├── models/             # β-VAE / DAE / DRRN / LU-Net architectures
│   ├── baselines/          # classical filters (butterworth, TV-L1, wavelet, ...)
│   ├── metrics/            # MAP utilities + reconstruction metrics (two faithful tracks)
│   ├── training/           # LR / β schedulers
│   ├── downstream/         # APD30/60/90 prediction + SVR baseline
│   └── viz/                # figure + LaTeX-table generation
├── scripts/                # thin CLIs (train_ddpm, denoise_*, run_filters, downstream_apd, evaluate_*)
├── configs/                # paths.yaml, ddpm.yaml, ml_collections configs (config*.py)
├── slurm/                  # env.sh + parameterised *.sbatch jobs
├── docs/                   # data_format, reproducibility, methods, results, decisions
└── tests/                  # data-free unit/smoke tests
```

## Reproducing the paper

The migration is faithful (only imports/paths/docstrings changed), so the
published numbers reproduce from the trained checkpoints or by re-running the
SLURM jobs. Read [docs/reproducibility.md](docs/reproducibility.md) first — it
documents the shared seeds (17 / split 29), the 4-fold split, the `T = 5000`
diffusion-steps setting, and an important metric caveat: the
column reported as **"RMSE" is actually a pooled MSE** (`compute_rmse` returns MSE
without the square root).

## Documentation

- [docs/data_format.md](docs/data_format.md) — expected cohort layout (no data shipped)
- [docs/reproducibility.md](docs/reproducibility.md) — environment, seeds, exact commands, caveats
- [docs/methods.md](docs/methods.md) — AV sampling, NFE accounting, architectures, noise model
- [docs/results.md](docs/results.md) — headline results and how to regenerate tables/figures
- [docs/decisions.md](docs/decisions.md) — migration decisions and reconciliations

## License & citation

Source code is released under the [MIT License](LICENSE) (the cohort is **not**
licensed by this repo). If you use this software, please cite the paper — see
[CITATION.cff](CITATION.cff).

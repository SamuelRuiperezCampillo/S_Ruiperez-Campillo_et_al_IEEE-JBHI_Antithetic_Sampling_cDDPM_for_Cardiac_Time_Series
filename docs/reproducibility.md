# Reproducibility

This repository is a **faithful** migration of the exact code that produced the
paper's results: only imports, hard-coded paths, and module docstrings were
changed (see [decisions.md](decisions.md)). Computational logic, seeds,
hyper-parameters, the noise schedule, and the sampling loops are byte-identical
to the originals.

## Environment

```bash
conda env create -f environment.yml      # reconstructs the cluster gpu_env2 (torch 2.5.1 / CUDA 12.1)
conda activate cardiac-map-diffusion
pip install -e .                          # editable install of the package
```

For an exact lock from the machine that produced the results, run on that machine:
`conda env export --no-builds > environment.lock.yml`.

## Fixed settings (shared across methods for a fair comparison)

| Setting | Value |
|---|---|
| Model / training seed | `seed = 17` |
| Patient-grouped K-fold split seed | `seed_split = 29` |
| Number of folds | **4** |
| Signal length | 370 samples @ 1 kHz |
| Normalisation | min-max `[0, 1]` |
| Noise type (training corruption) | `allmixed` |

## DDPM specifics

| Setting | Value |
|---|---|
| Diffusion steps `T` (`noise_steps`) | **5000** |
| `noise_schedule` | `quadratic`, `beta_start = 1e-4`, `beta_end = 0.05` |
| Denoising net | HNF backbone, `feats = 80`, 5 blocks (`model_small = False`) |
| Optimiser | AdamW, `lr = 1e-3`, StepLR (`step_size = 100`, `gamma = 0.5`) |
| Epochs / batch | `num_epochs = 400`, `batch_size = 96` |
| Deployed sampler | **2-shot antithetic variates** (`inference_antithetic`, then `0.5*(crude+anti)`) |

> **Diffusion steps `T = 5000`.** This is the value used in all experiments and
> the repository default (`noise_steps = 5000`).

> **`n_splits` 4 vs 5.** `Diffusion_MAP_fullpipeline_final/submit_jobs_cluster.sh`
> passed `--n_splits 5`; the paper uses **4-fold** CV (matching the baselines and
> `MAP_VAE/test/submit_jobs_cluster.sh`). The repository default is **4**; override
> with `N_SPLITS=5`.

## Metric caveat (important)

`compute_rmse(..., mode='total')` (in every `map_functions*` module) returns the
**pooled MSE without a square root**. Consequently the column labelled "RMSE" in
the paper tables is actually an **MSE**; the true RMSE is its square root. This
behaviour is preserved verbatim. The diffusion trainer reports this quantity as
`rmse` in `summary_values.xlsx`. Representative values: AV 2-shot MSE ≈ 3.2–3.3e-3.

## How to run (cluster)

The defaults of the SLURM scripts reproduce the paper configuration. After
`export CARDIACDIFF_DATA=...` and `export CARDIACDIFF_EXPERIMENTS=...`:

```bash
sbatch slurm/train_ddpm.sbatch            # conditional DDPM, 4 folds
sbatch slurm/denoise_dae.sbatch           # DAE   baseline
sbatch slurm/denoise_drrn.sbatch          # DRRN  baseline
sbatch slurm/denoise_lunet.sbatch         # LU-Net baseline
NUM_EPOCHS=1000 sbatch slurm/denoise_vae.sbatch   # beta-VAE (set a real epoch count)
sbatch slurm/filters_butterworth.sbatch   # classical filters ...
sbatch slurm/filters_tvl1.sbatch
sbatch slurm/filters_wavelet.sbatch
sbatch slurm/downstream_apd.sbatch        # APD30/60/90 SVR on noisy beats
```

Each writes checkpoints, per-fold `denoised_signals/*.npz`, metrics JSON, and a
`summary_values.xlsx` under `$CARDIACDIFF_EXPERIMENTS/<method>/...`.

## Checkpoints

The DDPM trainer saves **whole-pickled models** (`torch.save(model, ...)`),
not state dicts — loading therefore requires the package to be importable
(`pip install -e .`) so that `ConditionalModel` resolves. Released checkpoints
follow `model_fold{0-3}.pth` (DDPM) and `VAE_CNN_beta_0.005_rs_17_fold{0-3}.pth`
(VAE).

## Without local Python

The repository was assembled on a machine without a Python interpreter; it has
**not** been executed locally. Validate on the cluster with `pytest` (data-free
tests) and a short 1-fold / few-epoch dry run before launching full jobs.

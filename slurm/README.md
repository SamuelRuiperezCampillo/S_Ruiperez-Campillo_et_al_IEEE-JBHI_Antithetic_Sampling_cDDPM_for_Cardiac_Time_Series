# SLURM job scripts

Parameterised, de-hard-coded rewrites of the original `submit_jobs_*.sh`. Every
job sources [`env.sh`](env.sh), which activates the conda environment and exports
the cohort/output locations. **Edit `env.sh` (or `export` the variables) before
submitting** — in particular `CARDIACDIFF_DATA` (the private cohort, see
[../docs/data_format.md](../docs/data_format.md)) and, optionally,
`CARDIACDIFF_EXCLUDE_PATIENTS`.

| Script | Original | Partition | What it runs |
|---|---|---|---|
| `train_ddpm.sbatch` | `submit_jobs_cluster.sh` | gpu | Conditional DDPM (`final_big_80`), 4 folds |
| `denoise_vae.sbatch` | `submit_jobs_test.sh` | gpu | β-VAE (CNN, β=0.005) |
| `denoise_dae.sbatch` | `submit_jobs_dae.sh` | gpu | Denoising autoencoder |
| `denoise_drrn.sbatch` | `submit_jobs_drrn.sh` | gpu | DRRN |
| `denoise_lunet.sbatch` | `submit_jobs_lunet.sh` | gpu | LU-Net |
| `filters_butterworth.sbatch` | `submit_jobs_butterworth.sh` | compute | Butterworth + notch |
| `filters_tvl1.sbatch` | `submit_jobs_tvl1.sh` | compute | TV-L1 (sweep `TV_WEIGHT`) |
| `filters_wavelet.sbatch` | `submit_jobs_wavelet.sh` | compute | Wavelet shrinkage (sweep via env vars) |
| `downstream_apd.sbatch` | `submit_jobs_baseline_apd.sh` | compute | APD30/60/90 SVR on noisy beats |

Submit, e.g.:

```bash
export CARDIACDIFF_DATA=/path/to/MAP_cohort_root
export CARDIACDIFF_EXPERIMENTS=$PWD/experiments
sbatch slurm/train_ddpm.sbatch
N_SPLITS=5 sbatch slurm/train_ddpm.sbatch          # override any default via env var
TV_WEIGHT=0.05 sbatch slurm/filters_tvl1.sbatch
```

Defaults reproduce the paper run (seed 17, split seed 29, 4 folds, `noise_steps=5000`,
`feats=80`). The original scripts swept some hyper-parameters (wavelet/TV-L1 grids,
VAE β); those are exposed as overridable environment variables. `submit_jobs_test.sh`
used `num_epochs=1` (a smoke value) for the VAE — set `NUM_EPOCHS` for a real run.

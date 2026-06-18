# Results

This page explains **where the numbers come from** and **how to regenerate** the
paper's tables and figures from this repository. It does not restate the full
results; see the paper.

## What each run produces

Every training/denoising job writes, per fold, under
`$CARDIACDIFF_EXPERIMENTS/<method>/...`:

- `models/model_fold{0-3}.pth` — trained model (DDPM: whole-pickled; see reproducibility).
- `denoised_signals/fold{k}_{train,test}_signals.npz` — arrays
  `original_clean`, `noisy_input`, `denoised_output`.
- `denoised_signals/fold{k}_metadata.json` — per-fold metrics.
- `summary_values.xlsx` — per-fold metrics plus an `average` and `st. dev.` row
  (the table format used throughout the paper).

The reconstruction metrics computed for every method are: PCC, "RMSE" (**pooled
MSE** — see the caveat below), PSNR, MSE, Spearman, SNR, DTW, LSD, and NMAE
(range / l1 / mean).

## Metric caveat (read this before comparing to the paper)

`compute_rmse(..., mode='total')` returns the **pooled MSE without a square root**.
So the column printed as `rmse` / "RMSE" is an **MSE**. The deployed 2-shot
antithetic DDPM reaches an "RMSE" (= MSE) of roughly **3.2–3.3e-3** on the
held-out folds; the true RMSE is the square root of that. This is intentional and
preserved verbatim from the original code (`docs/reproducibility.md`).

## Headline comparison

The paper compares the conditional DDPM (deployed: **2-shot antithetic variates**)
against:

- **Deep-learning baselines:** β-VAE (CNN), DAE, DRRN, LU-Net.
- **Classical baselines:** Butterworth+notch, TV-L1, wavelet shrinkage, and others.
- **Downstream:** APD30/60/90 prediction (SVR) on denoised vs noisy beats.

The antithetic sampler attains lower reconstruction error than crude
Monte-Carlo at equal compute and a markedly lower run-to-run variance (the basis
of the paper's variance-reduction and uncertainty analyses).

## Regenerating tables and figures

After the runs complete, the aggregation/plotting utilities under
[`src/cardiac_map_diffusion/viz/`](../src/cardiac_map_diffusion/viz/) build the
comparison tables and figures from the per-method `summary_values.xlsx` /
`denoised_signals/*.npz`. Point them at `$CARDIACDIFF_EXPERIMENTS`. Typical
outputs:

- model-comparison metric tables (`generate_latex_table`, `aggregate_model_comparison`),
- APD tables (`generate_apd_latex_table`, `aggregate_apd_results`),
- sampling-strategy tables (`generate_sampling_table`, `generate_local_sampling_table*`),
- per-signal / per-patient comparison plots (`plot_single_*`, `plot_mechanism_comparison`),
- sampling analysis (`ddpm_sampling_analysis`).

> Some viz scripts carry hard-coded experiment sub-paths from the original runs;
> adjust the input path to your `$CARDIACDIFF_EXPERIMENTS` layout. Any such spot is
> flagged with a `# TODO(paths):` comment.

## Not included

The reviewer-response tooling (inference-time benchmark, uncertainty-vs-error
calibration, metric-reproduction harness) is **out of scope** for this repository
by design — see [decisions.md](decisions.md).

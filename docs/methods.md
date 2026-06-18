# Methods overview

A concise map from the paper's method to the code.

## Conditional DDPM

A conditional denoising diffusion probabilistic model removes mixed
physiological/acquisition noise from single-beat ventricular MAP signals. The
reverse process is conditioned on the noisy observation; the network predicts the
added Gaussian noise at each step.

- **Process** — [`diffusion/ddpm_conditional.py`](../src/cardiac_map_diffusion/diffusion/ddpm_conditional.py),
  class `Diffusion`: `prepare_noise_schedule`, `noise_signal`, `sample_timesteps`,
  `inference` (crude Monte-Carlo) and `inference_antithetic`.
- **Network** — [`diffusion/denoising_net.py`](../src/cardiac_map_diffusion/diffusion/denoising_net.py)
  `ConditionalModel` (HNF backbone, 5 blocks, `feats=80`) and the 3-block
  `ConditionalModelSmall`.
- **Training** — [`diffusion/train_ddpm.py`](../src/cardiac_map_diffusion/diffusion/train_ddpm.py).

### Antithetic-variates (AV) sampling

The deployed estimator runs the reverse chain **twice** with sign-flipped
transition noise, producing a negatively-correlated pair (r(τ) ≈ −0.96), and
averages them: `x_hat = 0.5 * (crude + anti)`. At equal compute this sharply
reduces the variance of the MMSE estimate versus independent Monte-Carlo draws.

### NFE accounting

- One crude trajectory ≈ `noise_steps − 1` network forward evaluations (NFEs).
- The antithetic pass costs **2 model calls per step** → one AV pair ≈ 2
  trajectories. The deployed "2-shot AV" therefore costs the same as 2 crude
  shots while matching the quality of substantially more (~10–14) crude shots —
  i.e. an effective speed-up of roughly K/2 at equal quality.

> **Device note.** `Diffusion.__init__` accepts a `device` argument but then
> overrides it with `'cuda' if torch.cuda.is_available() else 'cpu'`. This is
> preserved verbatim: on a CUDA machine the run uses the GPU; on a CPU-only
> machine it falls back to CPU automatically, but you cannot force CPU on a CUDA
> host.

## Noise model

Training corruption (`noise_type='allmixed'`) composes a random subset of:
baseline wander, powerline interference, Gaussian noise, truncation, spike, and
electrophysiological (EP) residual noise. See
[`data/utils_noise.py`](../src/cardiac_map_diffusion/data/utils_noise.py),
[`data/generate_noise.py`](../src/cardiac_map_diffusion/data/generate_noise.py)
(diffusion track) and the `introduce_*` functions in `metrics/map_functions*`
(baseline track), plus EP-noise templates from
[`data/ep_noise_sam.py`](../src/cardiac_map_diffusion/data/ep_noise_sam.py).

## Baselines

- **Deep learning** — β-VAE (CNN/FC/CNNres), DAE, DRRN, LU-Net
  ([`models/`](../src/cardiac_map_diffusion/models/), entry scripts
  `scripts/denoise_*.py`).
- **Classical** — Butterworth+notch, TV-L1, wavelet shrinkage, and others, applied
  through [`baselines/baseline_filters.py`](../src/cardiac_map_diffusion/baselines/baseline_filters.py)
  (`scripts/run_filters.py --filter_type ...`).

## Downstream clinical task

APD30/60/90 prediction from denoised beats with an SVR
([`downstream/`](../src/cardiac_map_diffusion/downstream/),
`scripts/downstream_apd.py`), quantifying whether denoising preserves clinically
relevant repolarisation timing.

## Two faithful pipelines

The diffusion and baseline pipelines were developed with separate but
split-compatible data loaders and metric modules. They are preserved side by side
rather than merged (to avoid numeric drift); see [decisions.md](decisions.md).
Both read the **same** cohort files through `cardiac_map_diffusion.paths`, which
guarantees identical patient-grouped folds at `seed_split=29`.

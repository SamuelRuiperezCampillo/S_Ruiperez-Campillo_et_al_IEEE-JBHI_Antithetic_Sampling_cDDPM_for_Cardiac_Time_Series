# Migration decisions & reconciliations

This repository is a **faithful migration** of the research code that produced
the paper, assembled from several overlapping working trees
(`Diffusion_MAP_fullpipeline_final/`, `MAP_VAE/`, `MAP_VAE/test/`). This log
records every non-obvious choice so the mapping back to the originals is auditable.

## Faithfulness contract

For each migrated `.py`, the only changes were: (1) import statements rewritten
to the package layout; (2) hard-coded `/cluster/...` (and local) paths routed
through [`cardiac_map_diffusion.paths`](../src/cardiac_map_diffusion/paths.py);
(3) an expanded module docstring. **No** array math, hyper-parameters, RNG
seeding, loss/metric formulas, schedule construction, or sampling loops were
altered. Bugs were preserved and reported, not fixed.

## Source-of-truth choices

- **Conditional DDPM** — migrated from `Diffusion_MAP_fullpipeline_final/`
  (`ddpm_main_vae_aligned.py` → `diffusion/train_ddpm.py`, plus
  `ddpm_conditional.py`, `denoising_net*.py`, `metrics.py`, `downstream*.py`,
  `data.py`, `generate_noise.py`, `utils_noise.py`, `filters.py`, and
  `scripts_from_sam/*`). This tree has the complete co-located dependency graph;
  the `MAP_VAE/test/ddpm_main_vae_aligned.py` copy imported modules absent beside
  it.
- **Baselines / VAE / downstream** — migrated from `MAP_VAE/` (root). The
  `MAP_VAE/test/` copies of the `*_denoising_allfolds.py` entry scripts are
  **older** (Oct 2025) than the root copies (Jan 2026) that the `submit_jobs_*.sh`
  actually invoke, so the **root** copies are canonical and the `test/` copies
  were not migrated.

## Path de-hard-coding

- All cohort access goes through `paths.ventmap_root()` / `paths.apd_annotations_dir()`
  / `paths.apd_annotations_filename()`; outputs through `paths.experiments_root()`;
  the optional held-out CSV through `paths.exclude_patients_file()`. The original
  diffusion and baseline loaders pointed at *different* cluster roots that held
  copies of the same files — routing both through one root also reinforces the
  "VAE-aligned" identical-splits guarantee.
- The vestigial `os.chdir(.../MAP_autoencoder)` in the **data loaders**
  (`data_sam`, `data_baselines`) was removed (the modules it exposed are now
  package imports; it had no effect on the returned data). The analogous
  `os.chdir` inside the *unused* `get_MAP_vent_data` copies that live in the
  `map_functions*` modules was **kept** (flagged `# TODO(paths)`), since those
  functions are not on the active code path.
- `baseline_filters.py` / `svm_apd.py` reference an **optional** cohort cache
  `MAP_vent_complete_pandas.csv`; this was routed to
  `paths.data_root()/MAP_vent_complete_pandas.csv`. If absent, the code falls back
  to building the dataframe from the cohort `.npz` (`get_MAP_vent_data`).

## Two pipelines preserved (not unified)

To avoid numeric drift, the diffusion and baseline tracks keep separate modules:

| Concern | Diffusion track | Baseline track |
|---|---|---|
| Cohort loader | `data/data_sam.py` | `data/data_baselines.py` |
| MAP utils + metrics | `metrics/map_functions.py` | `metrics/map_functions_baselines.py` |
| Filter metrics | — | `metrics/map_functions_metrics.py` |
| DDPM eval metrics | `metrics/ddpm_metrics.py` | — |
| Config style | argparse | absl + `ml_collections` config files |

`map_functions.py` (diffusion) vs `map_functions_baselines.py` (baseline) differ
only in: the baseline copy imports scikit-learn fully and adds
`butterworth_notch_filter`; the diffusion copy comments out the sklearn regressor
imports and omits that filter. The metric functions are otherwise the same.

## "RMSE" is a pooled MSE

`compute_rmse(..., mode='total')` returns the **pooled MSE without a square root**
in all three `map_functions*` modules (it also ignores its per-row loop index and
evaluates `mean_squared_error` on the whole arrays). The paper's "RMSE" column is
therefore an MSE. Preserved verbatim; covered by a unit test and
`docs/reproducibility.md`.

## Defaults reconciled to the paper

- **`n_splits` = 4.** `Diffusion_MAP_fullpipeline_final/submit_jobs_cluster.sh`
  passed `--n_splits 5`, but the paper (and the baselines) use 4-fold CV. Repo
  default = 4; override with `N_SPLITS=5`.
- **`noise_steps` (T) = 5000.** The diffusion-step count used in all experiments
  and the repository default.

## Excluded by design

- **`functions_several_noise_types.py`** — broken draft (syntax errors, undefined
  names, top-level side effects); not on any live code path. Dropped.
- **`Diffusion_MAP_fullpipeline_score_final/`** — the score-based SDE variant; not
  part of the paper's reported pipeline.
- **Reviewer-response tooling** — inference-time benchmark, uncertainty-vs-error
  calibration, metric-reproduction harness (per the project decision).
- **`main.py`** (the pre-"aligned" DDPM trainer, superseded by
  `ddpm_main_vae_aligned.py`) and the various `_backup`/`_old`/`_` duplicates.
- **`models/vae.py`** is the VAE *training driver* (not a model class); the VAE
  architecture is `AutoEncoder` in `models/autoencoder*.py`. It was migrated for
  completeness but the canonical VAE entry point is `scripts/denoise_vae.py`.

## Preserved behaviours worth knowing

- **DDPM device override** — `Diffusion.__init__` accepts `device` but forces
  `'cuda' if torch.cuda.is_available() else 'cpu'`. Kept verbatim (see methods.md).
- **Whole-pickled checkpoints** — the DDPM trainer does `torch.save(model, ...)`,
  so loading requires the package importable.
- **scikit-learn pin** — `svm_apd.py` uses `mean_squared_error(..., squared=False)`,
  removed in scikit-learn ≥ 1.6; `environment.yml` pins 1.5.1 where it still works.
- Minor preserved oddities flagged during migration (dead `sys.path.append`s,
  a 3-arg call to a 2-arg helper on an inactive `baseline_filters` path, a couple
  of help-string/default mismatches) — none on the active CLI paths.

## Visualization & evaluation scripts (`viz/`, `scripts/evaluate_*`)

These secondary scripts (plots, LaTeX-table generators, sampling/hidden-test
evaluation) were migrated faithfully. Two corrections and a few documented
limitations apply:

- `scripts/select_hidden_test_set.py` imported `get_MAP_vent_data` from the
  mechanically-mapped `data.data_diffusion`, which does not define it; repointed
  to `data.data_sam` (the diffusion-track loader, matching the sibling
  `evaluate_*` scripts).
- `viz/compute_comprehensive_metrics.py` had a dead
  `sys.path.append('/cluster/.../MAP_VAE')` supporting an optional
  `from metrics import compute_lsd, compute_dwt_distance`; the `sys.path` hack was
  removed. That import has self-contained `try/except` fallbacks (and
  `compute_dwt_distance` exists nowhere in the package), so the module runs anyway.
- `viz/generate_local_sampling_table*.py` previously pointed at a personal
  `C:\Users\pablo\Downloads\sample ddpm` folder; the base directory is now the
  `CARDIACDIFF_SAMPLING_DIR` env var (default `./sample_ddpm`).
- Several plot/aggregate scripts keep `/cluster/...` (or local) **argparse
  defaults** for experiment/output locations, flagged `# TODO(paths)`. They are
  overridable on the command line; adjust them to your `$CARDIACDIFF_EXPERIMENTS`
  layout. A couple of preserved pre-existing bugs in inactive fallback branches
  (`ddpm_sampling_analysis.normalize_EGM_input_with_stats`, a duplicated
  `ROW_FILES` assignment in `generate_latex_table`) were left untouched.

## Code quality

A behaviour-preserving cleanup pass was applied (no numeric/logic/hyperparameter
changes; the import-smoke tests backstop it):

- `viz/` and `scripts/` path defaults that were hard-coded to `/cluster/...` (or a
  personal local folder) now route through `cardiac_map_diffusion.paths`
  (experiment roots) or sensible relative defaults (`figures/`,
  `hidden_test_set_selected.csv`).
- Dead `sys.path.append(...)` shims (left over from the pre-package layout) were
  removed across `scripts/` and `viz/`, plus the now-unused `import sys`.
- Provably-unused imports were removed (e.g. `SummaryWriter` in
  `diffusion/ddpm_conditional.py`; `torch` in the `map_functions*` modules; six
  unused sklearn regressors in `metrics/apd_metric.py`; stray `plt`/`F` in the data
  loaders).
- `ruff` is configured in `pyproject.toml`; run `ruff check --fix` on a machine with
  the environment to sweep any remaining lint, then `pytest -q`.

### Duplication addressed (on the `refactor` branch -- validate before merging to main)

A line-level diff (ignoring whitespace/order) proved the shared functions were
byte-identical, so on the `refactor` branch the duplication was removed:

- **`metrics/map_functions{,_baselines,_metrics}.py`** were collapsed to one
  implementation: `map_functions_baselines.py` is canonical (the superset, including
  `butterworth_notch_filter`), and `map_functions.py` / `map_functions_metrics.py`
  are now thin re-export shims. The verified diff showed the only differences were
  docstrings/comments, commented-out unused sklearn regressors, the extra filter,
  one never-hit `compute_nmae` error string, and an inactive `get_MAP_vent_data`
  path literal -- i.e. **no change to any function the pipelines call**.
- **`get_train_test_kfolds`** (byte-identical across the two loaders and the trainer)
  was centralised into `data/splits.py`; the three sites import it (and their now-
  unused `random`/`KFold` imports were dropped).

Still **not** merged (intentional):

- **`NumpyDataSet*` + `retrieveDataSet`** in `data/retrieve_dataset.py` vs the copy
  inlined in `diffusion/train_ddpm.py` **differ in computation** (the `truncation`
  and `allmixed` branches source parameters differently), so they are kept separate.
- Minor smells (bare `except:`, mutable default args `arrays=[]`) -- behaviour-
  touching, deferred to a verified pass.

Because the repo can't be executed here, **validate on the cluster before merging
`refactor` into `main`**: `pip install -e . && pytest -q`, then run one DDPM fold and
one baseline and confirm the metrics match `main`.

## Verification limitation

The repository was assembled on a machine **without a Python interpreter** and has
not been executed locally. Faithfulness was ensured by byte-level copy/diff and
import tracing; runtime verification (import smoke tests, then short dry runs)
must be done on the cluster — see `docs/reproducibility.md` and `tests/`.

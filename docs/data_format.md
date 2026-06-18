# Data format

The ventricular monophasic action potential (MAP) cohort used in the paper is a
**private, IRB-restricted** dataset (Stanford ventricular MAP cohort). It is
**not** distributed with this repository and cannot be redistributed. This page
documents the layout the code expects so that authorised holders of the data can
run the pipeline.

## Pointing the code at the data

All cohort access goes through [`cardiac_map_diffusion.paths`](../src/cardiac_map_diffusion/paths.py).
Set **one** root. Resolution precedence is: environment variable (preferred) >
a git-ignored `configs/paths.local.yaml` (copy it from
`configs/paths.local.yaml.example` for persistent, uncommitted machine-specific
paths) > the committed `configs/paths.yaml`. For example:

```bash
export CARDIACDIFF_DATA=/path/to/MAP_cohort_root
```

Your personal cluster paths therefore live in your environment (or your
git-ignored `paths.local.yaml`) and never get committed to the public repo,
while other users supply their own — the code is unchanged either way.

## Expected directory layout

`CARDIACDIFF_DATA` must contain exactly these two items (names matter):

```
$CARDIACDIFF_DATA/
├── 2019 1111 Final_codes_and_data/
│   └── Step_1_RemoveUnderOverShoots/
│       └── single_MAP_dict_BP_mean_3SD_cropped_20191111.npz
└── 2023 0228 APD_data_MAP_vent_SRC.xlsx
```

- **`single_MAP_dict_BP_mean_3SD_cropped_20191111.npz`** — loaded with
  `np.load(path, allow_pickle=True)['arr_0'].item()`, yielding a `dict` keyed by
  patient ID (string; some keys carry a `_B` suffix). Each value is an array of
  cropped single beats, each **370 samples** long, sampled at **1 kHz**.
- **`2023 0228 APD_data_MAP_vent_SRC.xlsx`** — gold-standard APD30/60/90
  annotations, read by `acquire_APD_annotations`.

After loading, beats are organised one-per-row, min-max normalised to `[0, 1]`,
and split by **patient-grouped K-fold** (so no patient appears in both train and
test of a fold). The reported cohort is **5706 beats / 42 patients** (53 enrolled,
11 excluded).

### Optional cohort cache

The classical-filter and downstream-APD baselines look for an optional cached
dataframe `MAP_vent_complete_pandas.csv` directly under `CARDIACDIFF_DATA`. If it
is absent the code rebuilds the dataframe from the `.npz` above, so the cache is a
convenience only and need not be present.

## Hidden test set (optional)

To hold patients out of training (the paper's "hidden test set"), provide a CSV
with a `pat_ID` column and point the code at it:

```bash
export CARDIACDIFF_EXCLUDE_PATIENTS=/path/to/hidden_test_set_selected.csv
```

The DDPM trainer reads it via `--exclude_patients_file`; the SLURM scripts wire
this automatically when the variable is set.

## No data? 

Per the project decision, this repository ships **no synthetic data generator**.
Without the cohort the pipeline cannot run end-to-end; the data-free unit tests
(`pytest`) still exercise the metric and model code on random arrays.

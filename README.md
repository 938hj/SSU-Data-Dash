# SSU Data Dash: final H1/H2

The final model choice is fixed. `configs.FINAL` selects the laboratory generation:

| Track | Configuration | Members and geometric weights |
|---|---|---|
| Off-market | H1_om_hedge | HGB with TF-IDF, LightGBM and CatBoost, each with a 3-month and a 6-month market level; six equal weights |
| List-aware | H2_la_hedge | HGB with 3/6-month levels (0.2/0.2), sold/list-ratio HGB (0.4), enhanced Ridge with 3/6-month levels (0.1/0.1) |

H1 excludes list price. H2 uses it. Learning implementations and numerical
configurations match the supplied Claude source. Development used Claude and
ChatGPT/Codex for code generation, debugging, tests and documentation. The local
integration preserves validation guards and isolates test model files.

## Reproduce the submitted predictions

Use Python 3.13 (lab training: 3.13.15; Windows reproduction: 3.13.5).
Install dependencies once; do not assume that a fresh machine already has them.
Run the following from the repository root, alongside `run.py` and `requirements.txt`.

Windows PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py predict-saved
.\.venv\Scripts\python.exe run.py check
.\.venv\Scripts\python.exe run.py compare
```

Linux:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run.py predict-saved
.venv/bin/python run.py check
.venv/bin/python run.py compare
```

The `.venv/` directory is created on each machine, isolates the installed packages
and is excluded from Git. It is not part of the submission. If a compatible Python
environment is already selected, use `python -m pip install -r requirements.txt`
followed by the same `python run.py ...` commands. No activation script is needed
when using the explicit interpreter paths above. On the current Windows project,
PyCharm should select `.venv/Scripts/python.exe`.

`requirements.txt` is the installation entry point. It includes the complete
`requirements.lock.txt`, so one pip command installs all pinned dependencies.
Keep both files in the repository root. Python and an Internet connection are
needed for the first package/data download; installed packages and locally
available data can be reused afterward.

`run.py` expects a command argument. In a PyCharm Run configuration, set the
script to `run.py` and Parameters to `predict-saved`; running it without arguments
prints usage. The [official starter README](https://github.com/erwinzhang7/SSU-Data-Dash/blob/main/README.md)
also documents dependency installation followed by commands such as
`python run.py predict`, rather than a parameter-free run.

### Data: automatic download or a project-local folder

The official source is [Baliyang/SSU_data_dash](https://huggingface.co/datasets/Baliyang/SSU_data_dash).
`data.py` looks for each requested file in this order:

1. A directory explicitly selected with the `DATA_DIR` environment variable.
2. The repository's own `data/` directory, next to `data.py`; Parquet or CSV.
3. Hugging Face download/cache, following the official starter's approach.

The project-local directory is anchored to `Path(__file__).resolve().parent`,
so it does not depend on the terminal's current working directory. No path to
a team member's Windows profile is hard-coded. Automatic downloads go into the
Hugging Face cache; they do not have to appear in the repository's `data/` folder.

For manual/offline data placement, create `data/` in the repository root:

```text
data/listings.parquet           # required for a fresh fit or validation
data/test_off_market.parquet    # required for saved-model predictions/checks
data/test_list_aware.parquet    # required for saved-model predictions/checks
```

Matching `.csv` filenames are also supported. `predict-saved` uses only the two
test input files, because the fitted models already contain their historical
reference state. The hidden test sale prices are not supplied or required.
Keep `data/` out of Git. If `DATA_DIR` is explicitly set, it must point to valid
files; an invalid override raises a clear error instead of silently using
different data. When reusing data from another local folder, use its path through
`DATA_DIR`; judges do not need that team-specific setting with automatic downloads.

### Models and outputs

The two complete `.joblib` files in `trained_models/`, CSVs in `submissions/`,
reference copies and generation records come from the same lab run on `b2240-01`
(Linux, RTX 5070). Include them in the repository. Transfer parts and
`join_models.py` are not needed here. `predict-saved` reloads the models and writes:

```text
submissions/submission_off_market.csv
submissions/submission_list_aware.csv
```

`check` validates file format and test IDs. `compare` displays price differences
from `submissions/reference/`. Local reproduction previously matched all 3,974
off-market and 4,064 list-aware reference prices exactly. Detailed hashes and
checks are in `reports/validation_evidence.json`.

For a fresh fit of the same fixed models, retain the official-style command
`python run.py predict`. It trains H1/H2, then replaces saved model/submission
files; it is a different operation from reloading already fitted models.
CatBoost requests GPU with CPU fallback. CPU/GPU retraining can produce different
prices. Measured `predict` times:

| Machine | CatBoost | Time | Prices vs `submissions/reference/` |
|---|---|---|---|
| Lab `b2240-01`: Linux, 10 threads, RTX 5070 (produced the submission) | GPU | 17.8 min | the reference itself |
| Cloud VM: Linux, 2 CPU cores, no GPU | CPU | 112.5 min | off-market median 0.84%, 99th pct 4.2%, max 8.0%; list-aware median 0.005%, max 0.03% |

The off-market differences come from CatBoost, which trains a different model on
a GPU than on a CPU. The list-aware track has no CatBoost member. Seeds are fixed
at 0, so reruns on the same hardware repeat. A machine with more cores should be
faster than the two-core time, which was not measured. For the exact submitted
prices on any machine, use `predict-saved` (about 9 minutes on two CPU cores).

## Tests

```powershell
python -m pytest -q tests/
```

Tests use synthetic/temporary data and reduced CPU fits. They exercise the
selected H1/H2 structures, saved-model round trips and confirmation protections.
Model-directory paths follow `run.ROOT` at call time so synthetic tests cannot
overwrite the real saved models when using an isolated ROOT.

## Confirmation is already complete

The lab evaluated April-May 2026 once before transfer. Saved overall results:

| Track | MdAPE | MAPE | Homes |
|---|---|---|---|
| H1 off-market | 7.115% | 9.448% | 4,515 |
| H2 list-aware | 4.269% | 5.222% | 4,515 |

This import copies those metrics; it does not rerun or retune using confirmation.
`outputs/final_confirmation.json` records the imported completed campaign and
blocks another freeze/confirmation. Preserve it when moving machines. The
original lab freeze/source/data fingerprint manifest was not transferred, so
the local record does not claim to reconstruct that historical freeze.

### Official six-month validation (December 2025-May 2026)

The handout asks for walk-forward validation MdAPE on December 2025-May 2026.
Here it is assembled from two parts, without scoring April-May again:

1. December 2025-March 2026, run with this source and data:

   ```
   python experiments.py run H1_om_hedge --split dev
   python experiments.py run H2_la_hedge --split dev
   ```

2. April-May 2026, the per-home predictions from the one lab confirmation,
   copied into `outputs/experiments/H1_om_hedge/confirm/` and
   `outputs/experiments/H2_la_hedge/confirm/` (`predictions.csv` next to the
   imported `metrics.json` and `spec.json`).

Then:

```
python run.py collect-validation
python run.py report
```

`collect-validation` trains nothing. It accepts the December-March files only if
their recorded source and listings fingerprints match the current files, and the
April-May files only if `metrics.json` and `spec.json` match the hashes in
`outputs/final_confirmation.json` and the predictions reproduce the recorded
MdAPE and home count. Both parts must use the same model settings. `report` then
writes the six-month table with bootstrap intervals. Default `validate` stays
development-only (December 2025-March 2026), so it never scores April-May again.

## Evidence and report

`reports/validation_evidence.json` consolidates H1/H2
summer23/24/25, rise21 and confirmation metrics/specifications. Summer23 informed
tuning; summer24/25 and rise21 also informed selection. Rise21 uses a different
season and only nine months of training, so it is a directional stress check.

The market hedge was chosen for its average/worst-case behavior, despite failing
the original requirement to improve each summer by at least 0.05 percentage
points. Explain that choice in the report. Hidden-test scores remain unknown.
Method details and unresolved audit limitations are in `docs/methodology.md`.
Earlier HGB market-adjustment control metrics remain in the evidence JSON for context.

`handout.md`, `rubric.md` and `report/report.tex` retain the supplied requirements
and report template. Full work logs, research scripts, tuning records and the
previous F3/F4 release were archived outside the project to keep it compact.

Latest setup/data verification: 48 synthetic tests passed (6 warnings). Public test inputs loaded successfully without a personal DATA_DIR and matched the original files by SHA-256. The earlier saved-model reproduction matched every reference price for both tracks exactly; its detailed results remain in `reports/validation_evidence.json`.

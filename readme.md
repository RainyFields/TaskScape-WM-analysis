# TaskScape-WM-analysis

Analysis code, results and figures for **TaskScape-WM**, the CNeuroMod multi-task working-memory fMRI dataset
(5 subjects × 16 sessions, tasks `1back`, `ctxdm`, `dms`, `interdms`). The dataset and this repository are separate:

| What | Where |
|---|---|
| Dataset (shared, read-only, BIDS + derivatives) | `/project/def-pbellec/xuan/TaskScape-WM` on Rorqual |
| This repository | `https://github.com/RainyFields/TaskScape-WM-analysis` — branch **`taskscape-migration`** (current code); `taskscape-migration-nc` (clean noise ceiling); `main`, `spatial_smoothed`, `noise-ceiling-regen` and tags `pre-migration-2026-10*` are the frozen pre-migration history |
| Owner's checkout | `/project/def-pbellec/xuan/TaskScape-WM-analysis` (do not write there; clone your own) |

The previous README (legacy `fmri_dataset_project` layout) is preserved at tag `pre-migration-2026-10`.

## 1. Access requirements on Rorqual

- Membership in **`def-pbellec`** (dataset tree, surface/GLM/clean-delay files) **and `rrg-pbellec`** (raw and
  fMRIPrep imaging: these files are symlinks into the CNeuroMod git-annex store
  `/lustre09/project/6003287/ria-rorqual`, group `rrg-pbellec`, mode 640). Today both groups have the same members.
- The path `/project/def-pbellec/xuan` is traverse-only for `def-pbellec` (you cannot list it; use full paths).
- Check your access (read-only, standard-library Python, seconds; `--full` scans every file, ~1–2 min):

```bash
python3 /project/def-pbellec/xuan/TaskScape-WM-analysis/scripts/check_dataset_access.py --full
# or, from your clone: python3 scripts/check_dataset_access.py --full
```

It prints PASS/FAIL for: group membership, traversal, `participants.tsv`, BIDS events, raw BOLD NIfTI header (RIA),
fMRIPrep preprocessed BOLD header (RIA) and confounds, surface CIFTI header, full GLM betas H5 (+ a 3×3 slice if
`h5py` is importable), design matrix, trial table, clean-delay H5, and (with `--full`) read permission of all files.

## 2. Clone and set up (any directory you own)

```bash
mkdir -p ~/taskscape && cd ~/taskscape                       # or a directory under your own /project space
git clone --branch taskscape-migration https://github.com/RainyFields/TaskScape-WM-analysis.git
ln -s /project/def-pbellec/xuan/figure_style figure_style    # figure scripts import figure_style from the clone's parent dir

module load StdEnv/2023 python/3.11.5 scipy-stack/2025a
# either use the shared environment read-only ...
source /project/def-pbellec/xuan/nilearn/bin/activate
# ... or build your own from the pinned list (Alliance wheelhouse):
#   virtualenv --no-download ~/taskscape/venv && source ~/taskscape/venv/bin/activate
#   pip install --no-index -r TaskScape-WM-analysis/requirements.txt
```

## 3. Configure the dataset root and the output directory

All paths come from `config/paths.py`:

| Setting | Default | Override |
|---|---|---|
| Dataset root (read-only input) | `/project/def-pbellec/xuan/TaskScape-WM` | `export TASKSCAPE_DATASET_ROOT=...` or `--dataset_root` |
| Output root (must be writable) | `<your clone>/outputs` (gitignored) | `export TASKSCAPE_OUTPUT_ROOT=...` or `--output_root` |
| Task-stimuli repo (run-label verification only) | `/project/def-pbellec/xuan/task_stimuli/data/multfs` | `export TASKSCAPE_TASK_STIMULI=...` |

`config.paths` refuses an output root inside the dataset and any path inside the frozen legacy project.

```bash
cd ~/taskscape/TaskScape-WM-analysis
export TASKSCAPE_OUTPUT_ROOT=$PWD/outputs           # or e.g. /project/<your-group>/<you>/taskscape_outputs
python scripts/accuracy_level.py --task 1back        # -> $TASKSCAPE_OUTPUT_ROOT/behavior/1back_accuracy_results.csv
```

Heavier jobs go through SLURM (`--account=def-pbellec`); a job must reload the modules and re-activate the
environment, and should `export TASKSCAPE_OUTPUT_ROOT` explicitly:

```bash
sbatch -A def-pbellec -t 00:30:00 -c 4 --mem=16G --wrap "module load StdEnv/2023 python/3.11.5 scipy-stack/2025a; \
  source /project/def-pbellec/xuan/nilearn/bin/activate; export MPLBACKEND=Agg TASKSCAPE_OUTPUT_ROOT=$HOME/taskscape/out; \
  cd $HOME/taskscape/TaskScape-WM-analysis && python scripts/sanity_check/glm_evaluation.py --subj sub-01 --ses ses-004 \
  --task ctxdm --run run-01 --max_vertices 500 --max_vertices_r2 1000 --compute_all_r2 --seed 123"
```

Verified 2026-10-09: a fresh clone in another directory ran `accuracy_level.py --task 1back` (with
`TASKSCAPE_OUTPUT_ROOT` and with the default) and `glm_evaluation.py` (above); all 5 deterministic outputs are
byte-identical to the owner's 2026-10-07 smoke run, with 0 opens of the owner's checkout or the legacy project and
0 writes inside the dataset.

## 4. Where things are in the dataset

| Content | Path (relative to the dataset root) |
|---|---|
| Participants (id, age, sex, group; demographics `n/a`, unverified upstream) | `participants.tsv`, `participants.json` |
| BIDS events (one row per stimulus × phase) | `sub-XX/ses-YYY/func/*_events.tsv` (+ `events.json`) |
| Raw behaviour (verbatim, key presses) | `sourcedata/events-raw/sub-XX/ses-YYY/func/*_events.tsv` |
| Corrected / reformatted behaviour | `sourcedata/behavior-corrected/...`, `sourcedata/behavior-reformatted/...` |
| Run → detailed condition (12 conditions) | `sourcedata/study-designs/sub-XX_design_design_with_converted.tsv` |
| Raw BOLD / fMRIPrep | `sub-XX/ses-YYY/func/*_bold.nii.gz`, `derivatives/fmriprep/sub-XX/ses-YYY/func/` |
| Surface BOLD (Glasser64k, fsLR 32k; raw + 4 mm smoothed) | `derivatives/taskscape-surface/sub-XX/ses-YYY/func/*_space-fsLR_den-32k[_desc-sm4]_bold.dtseries.nii` |
| Atlas, valid-vertex mask | `derivatives/taskscape-surface/atlas/` |
| Complete trial-level GLM betas (all datasets + attributes) | `derivatives/glm/sub-XX/ses-YYY/func/*_betas.h5` |
| Design matrices / trial tables | `derivatives/glm/.../*_designmatrix.tsv`, `*_trials.tsv` |
| Clean delay betas (one record per delay regressor) | `derivatives/glm-clean-delay/sub-XX/sub-XX_delay_betas.h5` (+ provenance, counts, QC) |
| Run-level exceptions (2 runs without events/GLM) | `exceptions.tsv` |

Mapping (verified on all 558 GLM runs): events row *i* = `trials.tsv` row *i* = `betas[:, i]` = design column
`task_col_start + i`. 560 BOLD runs; 558 modelled (`sub-05/ses-012` dms run-01 and interdms run-01 have header-only
placeholder events, preserved in `sourcedata/events-unconverted/`).

## 5. Status of the data and code

- Validation: bids-validator 3.0.2 with NIfTI headers, `--recursive`: raw dataset 0 errors; derivative findings are
  unsupported custom formats (H5/NPZ/TSV), the schema's missing rule for functional CIFTI, and verbatim fMRIPrep 20.2.6
  conventions. Custom checks (checksums, H5 schema, design matrices, trial tables, CIFTI, events↔betas) all pass.
- Code: see `AGENTS.md`. The migrated scripts reproduce the pre-migration outputs exactly
  (`results/restructure/repro_report.md`). Scripts marked blocked in `results/restructure/path_migration_review.tsv`
  still read the invalid legacy grouped pickles and must not be used for results.
- `synthetic_data/` is a GLM-validation fixture kept for history; it is not part of the dataset.
- The dataset is a shared working copy on Rorqual (raw and fMRIPrep payloads live in the CNeuroMod store); a
  self-contained public release export is a future task.

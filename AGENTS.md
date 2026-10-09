# AGENTS.md — TaskScape-WM-analysis

> **Migrated repository (2026-10-07).** This is the analysis repository for the TaskScape-WM BIDS dataset
> (`/project/def-pbellec/xuan/TaskScape-WM`). Inputs come ONLY from that dataset through `config/paths.py`
> (`TASKSCAPE_DATASET_ROOT`, `--dataset_root`); outputs go ONLY under `TASKSCAPE_OUTPUT_ROOT` (default `outputs/`,
> untracked). `config.paths` rejects any path inside the frozen legacy project and installs an audit hook that
> blocks legacy reads. The text below this banner is the legacy project's AGENTS.md kept for reference: its paths
> (`data/...`, `/project/def-pbellec/xuan/fmri_dataset_project/...`) are the LEGACY equivalents of the dataset paths
> listed in `config/paths.py`.
>
> Migrated entry points (fidelity-verified, `results/restructure/repro_report.md`): `scripts/glm_analysis.py`,
> `scripts/verify_run_labels_from_blockfiles.py`, `scripts/build_clean_delay_betas.py`,
> `scripts/validate_clean_delay_betas.py`, `scripts/similarity_analysis_clean.py`,
> `scripts/figure_making/plot_fig8_regenerated_clean.py`, and on branch `taskscape-migration-nc`
> `scripts/noise_ceiling_clean/{prepare_nc_bundle.py,noise_ceiling_clean.py,job_clean_primary_taskscape.sh}`.
> Migrated active scripts (smoke-tested 2026-10-07): `accuracy_level.py`, `export_beta_activation_session_maps.py`,
> `aggregate_subject_r2_vertex_maps.py`, `verify_encoding_delay_beta_compliance.py`, `figure_making/{plot_fd_distribution,
> plot_glm_sanity_check,plot_response_rate,plot_behavioral_performance,plot_accuracy}.py`,
> `sanity_check/{glm_evaluation,plot_unadjusted_r2_distribution_all_tasks,plot_adjusted_r2_distribution_all_tasks,
> plot_subject_region_average_r2}.py`. Archived (not deleted) scripts: `scripts/legacy/README.md`. Per-file
> dispositions: `results/restructure/path_migration_review.tsv`. Harness: `scripts/repro/`.
>
> Legacy project doc kept below. Migration workspace: `/project/def-pbellec/xuan/TaskScape-WM-migration` (read its `HANDOFF.md`).
>
> **Shared use on Rorqual (2026-10-09).** Dataset stays at `/project/def-pbellec/xuan/TaskScape-WM` (read-only for
> everyone; never write into it). Collaborators clone this repo (`--branch taskscape-migration`) into their own
> directory; `DEFAULT_OUTPUT_ROOT` is `<clone>/outputs`; set `TASKSCAPE_OUTPUT_ROOT` in SLURM jobs. Figure scripts
> need `figure_style` next to the clone (symlink `/project/def-pbellec/xuan/figure_style`). Access needs groups
> `def-pbellec` + `rrg-pbellec` (raw/fMRIPrep are symlinks into the CNeuroMod RIA store); check with
> `python3 scripts/check_dataset_access.py --full`. Environment: shared venv `/project/def-pbellec/xuan/nilearn`
> (read-only) or `requirements.txt`. GitHub: `https://github.com/RainyFields/TaskScape-WM-analysis` (public; default
> branch on GitHub is still the legacy `main`). Workflow and dataset locations: `readme.md`.

---

# (legacy) AGENTS.md — fMRI Dataset Project

Context for coding agents working on the CNeuroMod multi-task working-memory fMRI analysis pipeline on Alliance Canada (Rorqual).

Human-oriented overview: see `readme.md`.

---

## Project overview

Analysis pipeline for the **CNeuroMod multi-task fMRI dataset** (working memory tasks). Data are preprocessed with fMRIPrep, resampled to Glasser64k CIFTI space, smoothed, and modeled with trial-level GLMs. Downstream work covers noise ceiling, task/condition similarity, behavioral QC, and publication figures.

**Related repo:** publication figures and shared styling live in `/project/def-pbellec/xuan/figures4papers` and `/project/def-pbellec/xuan/figure_style`.

---

## Subjects and tasks

| Level | Values |
|---|---|
| Subjects | `sub-01`, `sub-02`, `sub-03`, `sub-05`, `sub-06` (5 total) |
| BOLD task names | `1back`, `ctxdm`, `dms`, `interdms` |
| Detailed task conditions (12) | `nback_loc`, `nback_ctg`, `nback_obj`, `dms_loc`, `ctxdm_col`, `ctxdm_lco`, `interdms_loc_ABAB`, `interdms_loc_ABBA`, `interdms_obj_ABAB`, `interdms_obj_ABBA`, `interdms_ctg_ABAB`, `interdms_ctg_ABBA` |

Task naming in raw BOLD files uses short names only. Full condition mappings are in `data/study_designs/`.

---

## Key paths

| Resource | Path |
|---|---|
| Project root | `/project/def-pbellec/xuan/fmri_dataset_project` |
| fMRIPrep preprocessed data | `/project/def-pbellec/xuan/cneuromod.multfs.fmriprep` |
| Raw behavioral (inside fMRIPrep) | `.../cneuromod.multfs.fmriprep/sourcedata/cneuromod.multfs.raw` |
| Glasser-resampled BOLD | `data/glasser_resampled/{subj}/ses-{NNN}/` |
| Reformatted events | `data/reformated_behavior/` |
| GLM outputs (encoding+delay) | `data/encoding_delay/trial_level_betas/` |
| Grouped betas (legacy condition pickles; **invalid for Figure 8**) | `data/encoding_delay/grouped_betas/` |
| Clean delay betas (one record per physical delay column) | `data/encoding_delay/clean_delay_betas/` |
| Figure 8 / Section 4.5 similarity (canonical) | `results/similarity_analysis/regenerated_clean/` |
| Figure 8 figures (canonical) | `figures/fig8_regenerated_clean/` |
| Beta activation (cross-session avg, uncorrected) | `data/encoding_delay/beta_activation_glasser64k/task_averaged/` |
| Beta activation (session-level maps) | `data/encoding_delay/beta_activation_glasser64k/session_level/` |
| Beta activation (Level 1 DMS-demean task avg) | `data/encoding_delay/beta_activation_glasser64k/task_averaged_level1_dms_demean/` |
| Valid vertex mask | `data/valid_vertex_mask.npy` (64,984 vertices total; 58,396 valid, 6,588 excluded = medial wall + low-variance, std ≤ 0.1) |
| Glasser atlas | `data/Glasser_LR_Dense64k.dlabel.nii` |
| Analysis results | `results/` |
| SLURM logs | `logs/` |
| Synthetic test data | `synthetic_data/` |

---

## Directory layout

```
fmri_dataset_project/
├── AGENTS.md          # this file
├── readme.md
├── data/              # gitignored — large derived data stay local
├── scripts/           # all analysis code
│   ├── glm_analysis.py           # main trial-level GLM
│   ├── utils.py                  # shared helpers (vertex mask, confounds)
│   ├── figure_making/            # publication plots
│   ├── sanity_check/             # GLM / design-matrix diagnostics
│   └── decoding_analysis/
├── results/           # tracked in git (CSVs, reports, small outputs)
├── synthetic_data/    # ground-truth data for GLM validation
└── logs/              # SLURM job logs
```

---

## Environment setup

Run on **Rorqual** (SLURM account: `def-pbellec`). Use `virtualenv` + `pip`, not conda.

```bash
module load StdEnv/2023
module load python/3.11.5
module load scipy-stack/2025a
source /project/def-pbellec/xuan/nilearn/bin/activate
```

For CIFTI / Connectome Workbench operations:

```bash
module load connectomeworkbench
```

For Datalad data access:

```bash
source /project/def-pbellec/shared/venvs/datalad/bin/activate
```

**Job scripts must reload modules and reactivate the venv** — login-node state does not carry to compute nodes.

Install packages on login nodes only; use `pip install --no-index` when possible.

---

## Pipeline stages

### 1. Resample to Glasser space

```bash
cd scripts
bash cifti_resample.sh   # edit subj inside script before running
```

Uses `wb_command -cifti-resample` from fMRIPrep fsLR → Glasser64k.

### 2. Spatial smoothing (4 mm FWHM)

```bash
bash smooth_glasser_sm4.sh          # single subject
sbatch smooth_glasser_sm4_array.sh  # array over subjects
```

Output suffix: `_bold_sm4.dtseries.nii`.

### 3. Trial-level GLM

Primary script: `scripts/glm_analysis.py`

```bash
python scripts/glm_analysis.py \
    --subj sub-01 \
    --tasks ctxdm \
    --estimator ols \
    --fmri_root data/glasser_resampled \
    --out_root data/encoding_delay \
    --overwrite
```

Array submission (5 subjects × 4 tasks):

```bash
sbatch scripts/submit_glm_array.sh
```

Key flags: `--correct_only`, `--include_types encoding delay`, `--estimator wls_ar1|ols`.

Alternative GLM implementations: `glm_analysis_nilearn.py`, `glm_analysis_condition_based.py`, `glm_per_task_concat.py`.

### 4. Beta activation export (session maps + Level 1 DMS demean)

Rebuild session-level vertex maps from trial-level H5 betas, then subtract per-session DMS global mean before cross-session averaging (removes session baseline drift for surface figures).

```bash
python scripts/export_beta_activation_session_maps.py --subj sub-03
python scripts/export_beta_activation_session_maps.py   # all subjects
```

Aggregation: mean trials within run → mean runs within session → Level 1 correction (`corrected = map - mean(DMS_session)`) → mean across sessions.

Legacy uncorrected cross-session averages: `scripts/aggregate_subject_beta_activation_vertex.py` → `task_averaged/`. Prefer `task_averaged_level1_dms_demean/` for publication figures.

### 5. Downstream analysis

| Goal | Script |
|---|---|
| Noise ceiling | `noise_ceiling.py`, `noise_ceiling_tradition.py` |
| Behavioral accuracy | `accuracy_sanity.py`, `accuracy_level.py` |
| Task condition counts | `task_condition_identification.py` |
| Cross-subject similarity (Figure 8) | `similarity_analysis_clean.py` + `build_clean_delay_betas.py` → `results/similarity_analysis/regenerated_clean/` |
| Cross-subject similarity (other) | `subject_within_vs_cross_similarity.py`, `condition_level_activity_cross_subj_corr.py` |
| Region-level correlations | `region_level_task_correlation.py` |
| R² aggregation | `aggregate_subject_r2_vertex_maps.py` |
| Nilearn GLM evaluation | `evaluate_nilearn_glm.py`, `quick_nilearn_evaluation.py` |

### 6. Figures

Scripts in `scripts/figure_making/`. For publication styling, import from the shared style repo:

```python
from figure_style.figure_utils import get_figsize, save_figure, apply_clean_spines
plt.style.use("/project/def-pbellec/xuan/figure_style/nature_style.mplstyle")
```

See `.cursor/rules/nature-viz.md` for Nature Scientific Data figure specs.

---

## Code conventions

- **Shared utilities:** import from `scripts/utils.py` — especially `load_valid_vertex_mask()` and `apply_vertex_mask()`. Do not duplicate vertex-mask logic.
- **Paths:** use absolute paths under `/project/def-pbellec/xuan/...` in scripts and SLURM jobs (cluster layout is fixed).
- **Python style:** type hints where already used; argparse CLI for runnable scripts; `matplotlib.use("Agg")` before pyplot import in batch scripts.
- **Output organization:** keep GLM betas under `data/encoding_delay/`, analysis CSVs/reports under `results/{analysis_name}/`. Avoid overwriting without `--overwrite`.
- **Naming:** BOLD files follow BIDS-like pattern `{subj}_{ses}_task-{task}_{run}_space-Glasser64k_bold_sm4.dtseries.nii`.

---

## Git boundaries

| Path | Status |
|---|---|
| `data/` | **gitignored** — never commit large derived data |
| `results/` | tracked (CSVs, markdown reports, small outputs) |
| `scripts/` | tracked |
| `.pydeps/` | local pip target — do not commit |
| `logs/` | untracked job output |

Do not commit secrets, credentials, or raw fMRI data.

---

## SLURM template

```bash
#!/bin/bash
#SBATCH --account=def-pbellec
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/project/def-pbellec/xuan/fmri_dataset_project/logs/%x_%A_%a.log
#SBATCH --error=/project/def-pbellec/xuan/fmri_dataset_project/logs/%x_%A_%a.err

module load StdEnv/2023
module load python/3.11.5
source /project/def-pbellec/xuan/nilearn/bin/activate

python /project/def-pbellec/xuan/fmri_dataset_project/scripts/<script>.py ...
```

---

## Maintaining this file

You do **not** need to update AGENTS.md after every agent session. Update only when durable project knowledge changes.

**Update when:**
- A pipeline stage is finished or replaced (new default script, new output path)
- Environment or cluster setup changes (modules, venv path, SLURM account)
- A recurring gotcha is discovered
- Project status shifts (milestone done, new active workstream)

**Skip when:**
- One-off debugging, exploratory edits, or small localized fixes
- Changes already self-evident from code or git history

**What to edit:**
- **Project status** — most often (Completed / Active / Known gotchas + date)
- **Key paths / Pipeline stages / Code conventions** — only when those sections are stale

Agents should **offer** to update this file at the end of milestone-level work; apply edits only if the user agrees (or explicitly asks).

---

## Project status (update this section)

Last updated: **2026-09-26**

### Completed

- Glasser64k resampling and 4 mm smoothing pipeline
- Trial-level GLM (`glm_analysis.py`) with encoding + delay trial types
- Valid vertex mask generation (58,396 valid of 64,984 vertices; 6,588 / 10.14% excluded medial-wall + low-variance)
- Nilearn GLM implementation validated (`results/final_nilearn_assessment.md`)
- Behavioral accuracy / response-rate summaries for all 4 tasks
- Noise ceiling (voxel-wise and traditional split-half)
- Task/condition similarity and cross-subject correlation analyses
- Encoding-delay beta cleanup and grouped-beta regeneration
- Session-level beta activation export + Level 1 DMS demean (`export_beta_activation_session_maps.py`; all 5 subjects on cluster)
- BIDS release build (`scripts/build_bids_release.py` + SLURM array `scripts/submit_build_bids_release.sh`): assembled at `$SCRATCH/cneuromod_multfs_bids` (~544 GB, ~22.6k files). Physically copies (dereferences git-annex) raw imaging + fMRIPrep into `derivatives/fmriprep/`, and single-trial betas into `derivatives/glm/` (`*_betas.h5` + `*_designmatrix.tsv` exported from each H5). Trial-level `func/*_events.tsv` rebuilt from `reformated_behavior` with BIDS columns + raw key-press/RT merged from the wide raw events. `eyetrack/` and participants metadata omitted; `sub-02/ses-000` pilot and all `*__dup*` files excluded.
- **Figure 8 / Section 4.5 regenerated (2026-09-26).** Indexing + duplicate-delay bugs fixed. Canonical inputs: `data/encoding_delay/clean_delay_betas/`. Canonical tables: `results/similarity_analysis/regenerated_clean/` (primary variant `clean_verified`). Canonical figures: `figures/fig8_regenerated_clean/`. Report: `results/similarity_analysis/regenerated_clean/FINAL_REPORT.md`. Proposed Methods/Results/caption are in the same directory; the manuscript itself was not edited.

### Active / in progress

- Encoding-delay beta outputs under `data/encoding_delay/`
- Figure scripts in `scripts/figure_making/` (wire surface figures to `task_averaged_level1_dms_demean/`)
- Region-level and hierarchical clustering analyses
- Distributing the BIDS release: copy to `/project/def-bashivan/shared` blocked by inode quota (def-bashivan ~497k/500k files); considering tar-per-subject (~5 archives) to fit.

### Known gotchas

- Run labels may be zero-padded (`run-01` vs `run-1`); `glm_analysis.py` handles aliases.
- Task names in BOLD filenames differ from detailed condition names — always check `data/study_designs/`.
- Medial-wall vertices must be masked; use `utils.load_valid_vertex_mask()`.
- `task_averaged/` cross-session maps bake in session baseline drift; use `task_averaged_level1_dms_demean/` for corrected surface figures.
- BIDS release: raw + fMRIPrep are git-annex datasets (symlinks into `.git/annex/objects/`); always copy with `rsync -L` (dereference) and writable `--chmod`. Raw `anat/` is localizer/scout only; the preprocessed T1w lives in `derivatives/fmriprep`. In the release, per-stimulus event rows are not 1:1 with betas — the authoritative beta ordering is the GLM design matrix (`derivatives/glm/.../*_designmatrix.tsv`), not `func/*_events.tsv`.
- `readme.md` references older script names (`accuracy_eval.py`, `glm_analysis.py` paths) — prefer this file and actual script names.
- **Figure 8 / Section 4.5 — legacy inputs and manuscript numbers are INVALID.** `scripts/task_condition_identification.py` used to (1) re-number `col_index` after dropping incorrect trials, so 26–65% of stored vectors were the wrong GLM column (often encoding), and (2) store the same delay beta under both pair-rule and single-frame keys (~42% duplicates), leaking across split halves. Do not use `data/encoding_delay/trial_level_betas/grouped_betas/task_relevant_only/*_task_condition_betas.pkl`, `results/similarity_analysis/with_dms/`, or `results/similarity_analysis/similarity_bootstrap_results_*.csv` for the paper. Do not restore the paired *t*-test over 50 split iterations (they are Monte Carlo partitions, not independent samples). The February 2026 family-pooled / no-DMS text and the June 2026 12-task figure are different estimands and both used the broken pickles. Use `data/encoding_delay/clean_delay_betas/` + `results/similarity_analysis/regenerated_clean/` + `figures/fig8_regenerated_clean/` (`clean_verified`). `task_condition_identification.py` now maps `col_index` from `trial_index` before filtering, but still can duplicate pair+single-frame keys; task-level similarity must go through `build_clean_delay_betas.py`.

---

## Verification checklist

Before considering a change done:

1. Run the script on one subject/session as a smoke test.
2. Confirm output paths match existing conventions.
3. For GLM changes, check a sanity script in `scripts/sanity_check/`.
4. For figure changes, export both PDF and PNG at 600 dpi per Nature specs.
5. Do not commit anything under `data/`.

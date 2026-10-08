# Migration fidelity report: TaskScape-WM-analysis on TaskScape-WM inputs

Date: 2026-10-07. Run directory (untracked): `outputs/fidelity_2026-10-07/`
(`code/` immutable snapshots, `reference/` git-archive of the two pre-migration tags, `out/`, `compare/`, `logs/`, `strace/`, `jobs/`, `env_snapshot.txt`).
Submitted with `scripts/repro/submit_fidelity.sh` (commit `83883d3`); comparisons by `scripts/repro/compare_fidelity.py`,
fallback audit by `scripts/repro/fallback_audit.py`. 28 SLURM jobs (`22631421`-`22631448`), all `COMPLETED 0:0`.

## 1. Verdict

**PASS.** Every pipeline stage re-run from the migrated code on TaskScape-WM inputs reproduces the pre-migration
outputs with max |delta| = 0 (bit-identical arrays and tables), and no job opened any path inside the frozen legacy project.

| Stage | Compared | Identical | max abs delta | Note |
|---|---|---|---|---|
| GLM (`glm`) | 19 H5 | 19 | 0.0 | 3 sessions, all task families |
| Run labels (`labels`) | 2 CSV | 2 (on the 560 dataset runs) | 0.0 | 4 reference rows = excluded pilot session (section 5.2) |
| Clean delay betas (`clean`) | 30 files | 30 | 0.0 | all 5 subjects |
| Similarity tables (`similarity_tables`) | 45 files | 45 | 0.0 | 12 excluded: prose `.md`, label CSVs compared separately, `legacy_pickle` matrix |
| Fig. 8 source data (`fig8_source_data`) | 5 CSV | 5 | 0.0 | |
| Clean noise ceiling, primary (`nc_primary`) | 55 files | 50 + 5 JSON with only new provenance keys | 0.0 | section 5.5 |
| Fallback audit | 27 strace traces + 33 audit logs | legacy opens 0, BLOCKED 0 | | dataset opens 4,421 |

Target was max |delta| = 0 for every numeric value; the plan's 1e-6 / 1e-4 BLAS tolerances were not needed. The only
exclusions are run-specific provenance fields, listed in section 5.6.

## 2. Code and references

| Role | Ref | Commit |
|---|---|---|
| Code under test (GLM, run labels, clean delay betas, similarity, Fig. 8) | branch `taskscape-migration` | `83883d3d4201f4ce55cd94fc53904df6ab9c7e7c` |
| Code under test (clean noise ceiling) | branch `taskscape-migration-nc` | `f17be088f0cdfeb3b514a8f493bb1dccb0eb8d85` |
| Reference tables / figures (Fig. 8 chain, run labels) | tag `pre-migration-2026-10` (= legacy `spatial_smoothed` HEAD) | `d7b266de3929f0c04c755719761a43586800c6ca` |
| Reference tables (clean noise ceiling) | tag `pre-migration-2026-10-ncregen` (= legacy `noise-ceiling-regen` HEAD) | `4fa4121d4f93fda8423a84a7b56259fd6f62b097` |
| Reference GLM H5 and clean delay betas | TaskScape-WM `derivatives/glm`, `derivatives/glm-clean-delay` (hardlinks of the pre-migration files; sha256 == 2026-10-04 source baseline, see `TaskScape-WM-migration/reports/build_full_2026-10-07/validation/`) | dataset build job `22631170` |

Reference outputs were taken with `git archive <tag> results figures`; nothing was read from the frozen legacy project (section 6).

## 3. Environment

Compute jobs (`sbatch -A def-pbellec`, Rorqual, 4 CPUs), each with
`module load StdEnv/2023 python/3.11.5 scipy-stack/2025a; source /project/def-pbellec/xuan/nilearn/bin/activate`.

The venv ships its own interpreter, so the effective interpreter is the venv's **Python 3.10.13**, not the loaded
`python/3.11.5` module (verified: `which python` -> `.../nilearn/bin/python`, `python -V` -> 3.10.13; the strace of every
job opens `.../nilearn/lib/python3.10/...` only). The pre-migration references were produced with the same venv
(3.10.13), so the "3.10 vs 3.11" caveat in the 2026-10-07 01:25 handoff does not apply: interpreter and packages are identical.

Key packages (`pip freeze`; full list in `outputs/fidelity_2026-10-07/env_snapshot.txt`):

| package | version |
|---|---|
| numpy | 2.2.2+computecanada |
| scipy | 1.15.1+computecanada |
| pandas | 2.2.3+computecanada |
| h5py | 3.12.0+computecanada |
| nibabel | 5.3.2+computecanada |
| nilearn | 0.12.0 |
| matplotlib | 3.10.0+computecanada |
| statsmodels | 0.14.3+computecanada |

Loaded modules: CCconfig, gentoo/2023, gcccore/.12.3, gcc/12.3, hwloc/2.9.1, ucx/1.14.1, libfabric/1.18.0, pmix/4.2.4,
ucc/1.2.0, openmpi/4.1.5, flexiblas/3.3.1, aocl-blas/5.1, aocl-lapack/5.1, StdEnv/2023, mii/1.1.2, python/3.11.5,
ipykernel/2025a, scipy-stack/2025a.

## 4. Input mapping (legacy -> dataset)

All inputs are resolved through `config/paths.py` (`TASKSCAPE_DATASET_ROOT`, default `/project/def-pbellec/xuan/TaskScape-WM`;
outputs under `TASKSCAPE_OUTPUT_ROOT`). Any root or input resolving into the legacy tree raises `SystemExit`.

| Legacy path (frozen `fmri_dataset_project`) | Dataset path (TaskScape-WM) | Used by |
|---|---|---|
| `data/glasser_resampled/{sub}/{ses}/*_space-Glasser64k_bold_sm4.dtseries.nii` | `derivatives/taskscape-surface/{sub}/{ses}/func/*_space-fsLR_den-32k_desc-sm4_bold.dtseries.nii` | GLM |
| `cneuromod.multfs.fmriprep/{sub}/{ses}/func/*_desc-confounds_timeseries.tsv` | `derivatives/fmriprep/{sub}/{ses}/func/*_desc-confounds_timeseries.tsv` | GLM |
| `data/reformated_behavior/{sub}/{ses}/func/*_events.tsv` | `sourcedata/behavior-reformatted/{sub}/{ses}/func/*_events.tsv` | GLM |
| `data/valid_vertex_mask.npy`, `data/Glasser_LR_Dense64k.dlabel.nii`, network assignments | `derivatives/taskscape-surface/atlas/` | all |
| `data/encoding_delay/trial_level_betas/{sub}/{ses}/func/*_{betas.h5, design.csv, design_matrix.csv}` | `derivatives/glm/{sub}/{ses}/func/*_{betas.h5, trials.tsv, designmatrix.tsv}` | clean delay betas, noise ceiling, GLM reference |
| `data/encoding_delay/clean_delay_betas/` | `derivatives/glm-clean-delay/{sub}/` | noise-ceiling bundle, `clean` reference |
| `cneuromod.multfs.raw/.../*_events.tsv` (raw) | `sourcedata/events-raw/` (+ `sourcedata/events-unconverted/` for the 2 sub-05/ses-012 exception runs) | run-label verification |
| `data/behavior/` (legacy corrected copies) | `sourcedata/behavior-corrected/` | preserved; not read by this run |
| `/project/def-pbellec/xuan/task_stimuli/data/multfs` (EXTERNAL, rev `4b88bce0`) | unchanged, external | run-label verification |

Run labels: legacy files use unpadded `run-N` for fMRIPrep/surface/GLM and `run-0N` for raw; the dataset zero-pads
everything and `config.paths.run_label()` normalises. The dataset excludes `sub-02/ses-000` (pilot) and all `__dup`
raw acquisitions that have no GLM output, which explains the only row-count differences below.

## 5. Comparisons

### 5.1 GLM (`glm_comparison.json`): PASS, bit-identical

Subset: sub-01/ses-004 (session with a `__dup` 1back acquisition), sub-03/ses-002, sub-05/ses-012 (exception
session), all four task families, OLS, same flags as the legacy `submit_glm_array.sh`.

| compared | identical | max abs delta |
|---|---|---|
| 19 H5 files (every dataset and attribute) | 19 | 0.0 |

sub-05/ses-012 yields 5 runs, not 7: `dms run-01` and `interdms run-01` have no usable events (header-only raw
events, preserved in `sourcedata/events-unconverted/`, listed in the dataset `exceptions.tsv`); the dataset's
`derivatives/glm/sub-05/ses-012/func/` holds the same 5 runs. Ignored H5 attribute: `labels_file` (re-rooted path).

### 5.2 Run labels (`labels_comparison.json`): PASS on all 560 dataset runs

`compare_fidelity.py` flags both files as differing because of row counts (`run_labels_verified.csv` 560 vs 564,
`lookup_mismatch_report.csv` 22 vs 26). Row-key reconciliation on (subject, session, run, bold_task)
(`compare/labels_reconciliation.md`):

| file | rows new | rows ref | only in ref | only in new | differing columns on common rows |
|---|---|---|---|---|---|
| run_labels_verified.csv | 560 | 564 | 4 | 0 | none (22/22 columns, NaN-aware) |
| lookup_mismatch_report.csv | 22 | 26 | 4 | 0 | none |

The 4 reference-only rows are `sub-02 ses-000 run-01` x {1backcat, 1backloc, ctxlco, dmsloc}: the pilot session
(`in_glm_outputs=False`, `lookup_label=LOOKUP_FAIL`), which the dataset excludes by design. The sub-06/ses-007
relabellings (4 runs, `relabel_verified`) and the 2 sub-05/ses-012 `unresolved_keep_lookup` runs are reproduced.

### 5.3 Clean delay betas (`clean_comparison.json`): PASS, bit-identical

Rebuilt with `build_clean_delay_betas.py` for all 5 subjects from `derivatives/glm` (`*_betas.h5` + `*_trials.tsv`)
and the re-verified run labels, compared with the dataset's `derivatives/glm-clean-delay` (= pre-migration products).

| compared | identical | max abs delta |
|---|---|---|
| 30 files (per subject: `delay_betas.h5`, `provenance.csv`, `counts_per_run.csv`, `counts_per_task.csv`, `validation.json`, `vertex_qc.npz`) | 30 | 0.0 |

Ignored CSV columns: `source_beta_file`, `source_design_file`, `source_events_file` (re-rooted paths); ignored H5
attribute `labels_file`. `validate_clean_delay_betas.py` on the rebuilt files reproduces the pre-migration
`validation_report.md` table exactly: 0 wrong-column, 0 unresolved runs, retained 6034 / 6227 / 6098 / 5368 / 6036
records (sub-01/02/03/05/06), 2 relabelled runs counted for sub-06, CIFTI axis sha1 `5da42bc458` for every subject.

### 5.4 Similarity tables and Fig. 8 source data: PASS, bit-identical

`similarity_analysis_clean.py --stage halves` (5 subjects) -> `--stage similarity` -> `plot_fig8_regenerated_clean.py`,
compared with the tag's `results/similarity_analysis/regenerated_clean/` and `figures/fig8_regenerated_clean/`.

| comparison | compared | identical | max abs delta | reference rows dropped (`legacy_pickle`) |
|---|---|---|---|---|
| `similarity_tables` | 45 files: 24 `matrices/*.npz`, 15 CSV, 2 JSON, 5 `qc/*_record_qc.csv` | 45 | 0.0 | across 18,000; within 9,000; pair_task 360; subject_task 180; task_level 36; family_level 12; n_trials 60 |
| `fig8_source_data` | 5 `*_source_data.csv` | 5 | 0.0 | `figS_audit_variants` 12 |

The `legacy_pickle` variant (reads the invalid grouped pickles, not a dataset input) was not re-run: its rows/columns
were dropped from the reference and counted per file; `matrices/similarity_matrices_legacy_pickle.npz` was excluded.
All other variants (`clean_verified`, `clean_lookup`, `clean_behaviour_relabel`, `clean_exclude_inconsistent`,
`clean_verified_familypooled`, `clean_verified_sessionscale`, `clean_verified_robust_psc_*`,
`fixed_with_duplicates`) and the 5 control analyses are identical. Excluded with `--exclude .md`: the 8 prose
reports; `run_labels_verified.csv` and `lookup_mismatch_report.csv` are compared in section 5.2.

### 5.5 Clean noise ceiling, primary variant (`nc_primary_comparison.json`): PASS, numerically identical

| compared | identical | max abs delta |
|---|---|---|
| 55 files (per subject: `unit_estimates_{region,network}.csv`, `raw_split_r_long_*.csv`, `raw_r_*.npy`, `loco_raw_r_*.npy`, `splits.npz`, `trial_counts_per_variant.csv`, `config_and_summary.json`) | 50 | 0.0 |

The 5 non-identical files are the per-subject `config_and_summary.json`: the only differences are two **new** keys,
`args/dataset_root` and `args/output_root` (added by the migrated CLI), absent from the reference. Every numeric
value in those JSONs (all summary statistics) is identical (max |delta| = 0.0). Seed 20260926, 50 iterations,
`--require_resolved`, as in the pre-migration `job_clean_subject.sh`. Source validation inside `prepare_nc_bundle.py`
passed for all 5 subjects (0 wrong-column, 0 encoding, 0 duplicate keys, 0 identical vectors, 0 NaN/Inf).

### 5.6 Fields excluded from comparison (run-specific provenance only)

- JSON keys: `elapsed_s`, `elapsed_sec`, `hostname`, `slurm_job_id`, `command`, `python`, `numpy`, `pandas`,
  `git_commit_worktree`, `labels_file`, `input_dir`, `out_dir`, `atlas_path`, `source_h5`, `created`, `timestamp`, `timestamp_utc`.
- CSV columns: `source_beta_file`, `source_design_file`, `source_events_file` (re-rooted provenance paths).
- H5 attributes: `labels_file`.
- Reference variant `legacy_pickle` (`--drop_variant`), see 5.4.
- `.md` reports in the similarity reference directory (prose, not data).

## 6. Fallback audit (no legacy input): PASS

Two independent mechanisms: (1) `config.paths.install_legacy_guard` (Python audit hook) fails any open/listdir/scandir
inside the legacy tree and logs every opened path under `/project/def-pbellec/xuan` to `out/audit/*.opened.tsv`
(33 logs, one per Python process); (2) every job ran under `strace -f -e trace=open,openat,openat2`, which also sees HDF5/C-library opens.

| traces | total opens | opens under TaskScape-WM | opens under legacy project (attempted or successful) | BLOCKED events |
|---|---|---|---|---|
| 27 | 52,693 | 4,421 | 0 | 0 |

Jobs without dataset opens (expected): `fig8`, `labels_compare`, `nc_compare`, `sim_compare` read only this run's
outputs and the git-archive references. `halves_*` open the dataset once (`dataset_description.json` root check);
their inputs are the rebuilt clean betas in `out/`. Per-job counts: `compare/fallback_audit.tsv`.

## 7. Job summary

All 28 jobs `COMPLETED`, exit `0:0`, 4 CPUs each; wall time per step (requested memory):

| step | jobs | wall time | mem |
|---|---|---|---|
| glm (3 sessions) | 22631421-23 | 1m18s-1m35s | 32G |
| labels | 22631425 | 1m10s | 16G |
| clean (5 subjects) | 22631427-31 | 1m11s-2m27s | 48G |
| halves (5 subjects) | 22631434-38 | 5m39s-8m55s | 64G |
| similarity | 22631439 | 3m06s | 96G |
| fig8 | 22631440 | 26s | 16G |
| nc (5 subjects) | 22631442-46 | 23s-25s | 32G |
| compare / validate / audit | 22631424, 26, 32, 33, 41, 47, 48 | 3s-36s | 4-32G |

## 8. Not exercised here / known limits

- GLM re-run covers 3 of 80 sessions (19 of 558 runs); the other GLM H5 in `derivatives/glm` are the pre-migration
  files themselves (hardlinks, sha256-verified against the 2026-10-04 baseline), not re-derived.
- `legacy_pickle` similarity variant not re-run (invalid input, by design).
- Figures (PDF/PNG/SVG) are compared through their `*_source_data.csv`, not pixel-wise.
- The clean noise ceiling `sensitivity` variant was not re-run (only `primary`).
- Scripts outside the migrated core chain still reference legacy paths: see `results/restructure/path_migration_table.tsv`
  and the per-file proposals in `results/restructure/path_migration_review.tsv` (nothing moved or deleted).

## 9. Addendum (2026-10-07, later): smoke tests of the 13 migrated active scripts

Run dir `outputs/smoke_2026-10-07/` (untracked), submitted with `scripts/repro/submit_smoke.sh` (strace per job,
legacy guard installed, `TASKSCAPE_OUTPUT_ROOT` = run dir). Code: commit `92888b7`. 16 jobs (`22697782`-`22697797`),
all `COMPLETED 0:0`; stderr contains only numpy `Mean of empty slice` warnings on masked vertices and nilearn
deprecation warnings.

| script | smoke command (one subject / session) | wall |
|---|---|---|
| `accuracy_level.py` | `--task {1back,dms,ctxdm,interdms}` (4 jobs) | 6-14 s |
| `figure_making/plot_response_rate.py` | (all subjects, behaviour TSVs only) | 13 s |
| `figure_making/plot_behavioral_performance.py` | after the 5 jobs above | 16 s |
| `figure_making/plot_accuracy.py` | after the 4 accuracy jobs | 11 s |
| `export_beta_activation_session_maps.py` | `--subj sub-01` (112 runs, 16 sessions) | 19 s |
| `aggregate_subject_r2_vertex_maps.py` | `--subj sub-01` (112 runs) | 14 s |
| `verify_encoding_delay_beta_compliance.py` | `--beta-root <dataset>/derivatives/glm/sub-01/ses-004` (7 runs, 0 failures) | 7 s |
| `figure_making/plot_fd_distribution.py` | (all subjects, confounds TSVs) | 2 m 57 s |
| `figure_making/plot_glm_sanity_check.py` | (hard-coded sub-03, first run) | 22 s |
| `sanity_check/glm_evaluation.py` | `--subj sub-01 --ses ses-004 --task ctxdm --run run-01 --max_vertices 500 --max_vertices_r2 1000 --compute_all_r2 --seed 123` | 53 s |
| `sanity_check/plot_unadjusted_r2_distribution_all_tasks.py` | `--glm_root <dataset>/derivatives/glm/sub-01/ses-004 --subjects sub-01 --tasks ctxdm` | 37 s |
| `sanity_check/plot_adjusted_r2_distribution_all_tasks.py` | same | 37 s |
| `sanity_check/plot_subject_region_average_r2.py` | `--subj sub-01 --metric r2_full` (112 runs) | 3 m 16 s |

Fallback audit over the 16 strace traces (`compare/fallback_audit.json` in the run dir): legacy opens 0, BLOCKED 0,
dataset opens 3,791, every job opened the dataset. PASS.

Output comparison with pre-migration products (`TaskScape-WM-migration/reports/smoke_2026-10-07/smoke_vs_legacy.{tsv,json}`,
job `22698071`, run outside the analysis repo because the legacy beta-activation and R2 products live only in the frozen
legacy data tree): 
| group | compared | identical | note |
|---|---|---|---|
| behaviour CSVs (accuracy x4, response rate x4) vs tracked `results/*.csv` at the tag | 8 | 8 | identical after sorting on (subject, session, task, run); 4 files (ctxdm, interdms = multi-run sessions) differ in row order only, because the legacy script emitted rows in unsorted `glob()` order of a different directory |
| beta activation export, sub-01 (64 session maps + 4 Level-1 task averages + 2 index arrays) vs legacy `data/encoding_delay/beta_activation_glasser64k` | 70 | 70 | max abs delta 0.0 |
| R2 aggregation, sub-03 (63 maps) vs legacy `data/r2_averaged_glasser64k` | 63 | 0 | the stored legacy maps are STALE: written 2026-05-28 09:40, before the GLM H5 were regenerated on 2026-05-30 07:16 (max abs delta up to 0.35) |
| R2 aggregation, sub-03 vs the **pre-migration script** (tag `pre-migration-2026-10`) re-run on the current legacy H5 (job `22699986`, `reports/smoke_2026-10-07/r2_legacy_rerun_vs_smoke.json`) | 63 | 63 | max abs delta 0.0: the migrated script reproduces the legacy code on identical inputs; sub-01 has no legacy maps (legacy ran sub-03 only) |

Scripts whose outputs are figures only (`plot_*`, `glm_evaluation.py`) were checked for successful completion and output
files, not pixel-compared.

Behaviour-change notes carried over from the migration (documented in each script):
- `verify_encoding_delay_beta_compliance.py` now writes its report under the output root (the dataset is read-only);
  the legacy report ships verbatim at `derivatives/glm/provenance/compliance_report.json`.
- `accuracy_level.py` gained `--task` (default `interdms`, the legacy hard-coded value); `plot_response_rate.py`,
  `plot_behavioral_performance.py` and `plot_accuracy.py` had their module-level code wrapped in `main()`; figure code unchanged.
- `plot_glm_sanity_check.py` still overlays the GLM fit on the UNsmoothed surface BOLD (legacy behaviour kept; flagged).
- `sanity_check/glm_evaluation.py` default `--glm_root` is now `derivatives/glm` (the dataset holds only the encoding+delay GLM).

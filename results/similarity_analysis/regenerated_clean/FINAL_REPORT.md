# Final report — regenerated clean Figure 8 / Section 4.5

Last updated: 2026-09-26. Primary variant: `clean_verified`.  
Production pickles, `results/similarity_analysis/with_dms/`, and `similarity_bootstrap_results_*.csv` were not overwritten. Those legacy Figure 8 inputs and the manuscript numbers derived from them are **invalid**.

## 1. Code changes

| Script | Role |
|---|---|
| `scripts/verify_run_labels_from_blockfiles.py` | Match raw events to PsychoPy block files |
| `scripts/build_clean_delay_betas.py` | One delay record per GLM column + provenance |
| `scripts/validate_clean_delay_betas.py` | Cross-subject assertions; fail the job on violation |
| `scripts/similarity_analysis_clean.py` | 50 split-halves, 3 resolutions, variants, controls |
| `scripts/figure_making/plot_fig8_regenerated_clean.py` | Figure 8 + supplements |

Minimal patch to `scripts/task_condition_identification.py`: assign `col_index` from `trial_index` **before** the correct-only filter; assert design rows = GLM columns. The script still stores pair + single-frame keys if used for condition-level work; task-level analyses must use `build_clean_delay_betas.py`.

## 2. Exact bug fixes

1. **Beta-column indexing.** After `is_correct` filtering, the old code set `col_index = arange(len(df))`, so later rows pointed at the wrong GLM column (26–65% of vectors; many encoding). Fix: `col_index = trial_index` on the unfiltered design. Vector-by-vector equality with the source H5 column: 0 mismatches.
2. **Duplicate delay storage.** Pair-rule `(k-1, k)` and single-frame `(k, k)` both stored the delay after stimulus *k* (~42% duplicates), leaking across split halves. Fix: one record per `(subject, session, bold_task, run, trial_index, phase)`.

## 3. Validation

`validation_report.md`: **PASS**. Shared CIFTI axis `5da42bc458`; 360/360 parcels (≥22 valid vertices); 12 networks; 0 wrong-column vectors; 0 encoding betas; 0 duplicate physical keys; 0 NaN/Inf.

## 4. Counts (correct-only, then dedup)

| subject | GLM columns | delay rows | retained | excluded incorrect | legacy records | duplicates removed |
|---|---:|---:|---:|---:|---:|---:|
| sub-01 | 12,736 | 6,368 | 6,034 | 334 | 10,260 | 4,226 |
| sub-02 | 12,736 | 6,368 | 6,227 | 141 | 10,646 | 4,419 |
| sub-03 | 12,736 | 6,368 | 6,098 | 270 | 10,388 | 4,290 |
| sub-05 | 12,544 | 6,272 | 5,368 | 904 | 8,960 | 3,592 |
| sub-06 | 12,736 | 6,368 | 6,036 | 332 | 10,264 | 4,228 |

All 12 task variants present in every subject. No trial cap.

## 5. The 14 lookup mismatches

Authoritative source: stimulus-sequence match to `task_stimuli/.../blockfiles`.

- **4 runs relabelled (sub-06 ses-007)** to the matched block file (`nback_obj`, `interdms_loc_ABBA`, `interdms_loc_ABBA`, `interdms_loc_ABAB`). These enter `clean_verified`.
- Remaining suspected runs: block file **confirms** the lookup label. Behaviour is not treated as ground truth.
- Six unresolved events files have no GLM outputs and are unused.
- Sensitivity variants (`clean_lookup`, `clean_exclude_inconsistent`, `clean_behaviour_relabel`) do not change the family-level pattern. See `lookup_mismatch_report.md`.

## 6. Task-level and family-level results (`clean_verified`)

Family values = mean of task-level estimates (each already averaged over 50 splits). Not family-pooled trials.

| Family | Vertex within / across | Region within / across | Network within / across |
|---|---|---|---|
| DMS | 0.84 / 0.03 | 0.89 / 0.01 | 0.92 / 0.00 |
| 1-Back | 0.17 / 0.01 | 0.28 / 0.04 | 0.40 / 0.22 |
| InterDMS | 0.21 / 0.02 | 0.33 / 0.05 | 0.47 / 0.18 |
| CTXDM | 0.10 / 0.02 | 0.20 / 0.04 | 0.23 / 0.15 |

Tables: `summary_task_level.csv`, `summary_subject_task.csv`, `summary_pair_task.csv`. `*_mc_sd_splits` are Monte Carlo SDs of the random partition.

## 7. Comparison with old manuscript / production figure

**Invalid — do not cite:**

| Source | Why invalid |
|---|---|
| Feb 2026 text | Family-pooled, no DMS; called “50 bootstraps”; paired *t*-tests over splits; used mis-indexed + duplicated pickles |
| Jun 2026 Figure 8 (`with_dms/`, `similarity_bootstrap_results_*.csv`) | 12 tasks, vertex cap 300; same broken pickles; network across-subject *r* near 0 is an artefact |

This regeneration matches the audit’s fixed+dedup diagnostic (~0.01–0.03). Residual differences: four sub-06 relabels, no 300-trial cap. Legacy pickles inflate within-subject *r* (region 1-Back 0.59 vs 0.28) and suppress network across-subject *r* (1-Back 0.01 vs 0.22).

## 8. Controls

| Control | Result |
|---|---|
| Spatial shuffle of sub-01 | Vertex across pairs involving sub-01 → ~0; vertex within unchanged |
| Task-label derangement of sub-01 | Network across 1-Back 0.22→0.19, InterDMS 0.18→0.18, CTXDM 0.15→0.18 — do **not** claim task-specific shared structure |
| Clone sub-01 as sub-02 | Pair *r* 0.54 / 0.59 / 0.57 (vertex/region/network); adjacent splits can share trials |
| Affine \(3.7x+12\) on sub-03 maps | max \|Δ*r*\| = 7.45×10⁻⁷ |
| Extreme PSC | 162 records with cortical \|β\| p95 > 100; spikes at near-zero baseline vertices. Robust drop of those records + 3,202 vertices (baseline < 1) leaves the family pattern unchanged |
| Lookup sensitivity | `clean_lookup` ≈ `clean_verified` |

## 9. Figure paths and verification

Job `fig8_plot` **21886822** (`def-bashivan`, 49 s, COMPLETED, exit 0:0). Outputs in `figures/fig8_regenerated_clean/` (6 PDF, 6 SVG, 6 PNG at 600 dpi, 5 source-data CSVs):

| Stem | PDF | SVG | PNG | source CSV |
|---|---|---|---|---|
| `fig8_within_across_similarity` | yes | yes | 4322×1228, 600 dpi | yes (36 task + 12 family rows) |
| `figS_network_matrices_family` | yes | yes | 4322×1133, 600 dpi | shared `figS_network_matrices_source_data.csv` (400 cells = 4×25 + 12×25) |
| `figS_network_matrices_tasks` | yes | yes | 4322×3259, 600 dpi | same |
| `figS_audit_variants` | yes | yes | 4322×1370, 600 dpi | yes |
| `figS_sensitivity` | yes | yes | 4322×1370, 600 dpi | yes |
| `figS_task_level` | yes | yes | 4322×2834, 600 dpi | yes (36 rows = 12 tasks × 3 resolutions) |

Checked against `summary_task_level.csv` / `summary_family_level.csv` (`clean_verified`):

- Main-figure family means match to floating-point precision (vertex DMS 0.836453 / 0.028179; region 0.889924 / 0.008422; network 0.922237 / 0.002117; network 1-Back 0.401339 / 0.220153; InterDMS 0.468385 / 0.179967; CTXDM 0.227739 / 0.147569).
- Twelve task points per panel; no split-iteration cloud.
- Shared y-axis includes the negative range (floor −0.1).
- SVG tick labels are `DMS`, `1-Back`, `InterDMS`, `CTXDM` (PNG rasterization of the Nature font can look ligatured; the SVG is the publication source).
- Audit figure: production hollow markers sit on the legacy-pickle bars, not on `clean_verified`.
- Network family matrices show high DMS diagonals and pair heterogeneity off-diagonal (including negative DMS pairs).

## 10. Remaining uncertainties

See `limitations_similarity.md`. Confirm LCI/CIL naming. Do not interpret network-level shared structure as task-specific. Do not call all families highly reliable. None of these block the descriptive headline.

## 11. Recommended manuscript wording

Use `manuscript_methods_similarity.md`, `manuscript_results_similarity.md`, and `figure8_caption.md`. The manuscript file itself was not edited. Do not restore the paired *t*-test. Do not call the 50 iterations bootstraps.

> Within-subject spatial similarity generally exceeded across-subject similarity, although its magnitude varied across task families and spatial resolutions. Across-subject similarity was low at the vertex and parcel levels but remained non-zero at the network level for 1-Back, InterDMS and CTXDM, with substantial heterogeneity across participant pairs.

## 12. Reproduction

See `reproducibility.md`. Split-half maps live on `$SCRATCH` (`/scratch/xuan/fig8_clean/halves/`) and must be recomputed if purged. Clean H5 files, tables, figures, and this report live under the project directory.

Queue after wrap-up: `squeue -u $USER` empty. Job IDs: labels 21873605; build 21873766–21873770; halves 21873939–21873942, 21874057; validate 21874229; similarity 21877024 (21874759 failed at an overly aggressive PSC floor and was superseded); figures 21886822.

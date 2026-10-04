# Execution plan — regenerated clean cross-subject spatial-pattern similarity (Figure 8)

Repository commit at start: `6af305885bea0a2e77f3a59bb607cc6b44f335e5` (working tree had unrelated
modified files under `results/noise_ceiling/`; they are not touched by this work).

All heavy computation runs on Rorqual compute nodes (`def-bashivan`) through
`~/.cursor/skills/alliance-compute-jobs/scripts/run_on_compute.sh -b`. Job IDs, seeds and paths are
recorded in `reproducibility.md` in this directory.

## Data flow (before)

```
glasser_resampled/*_bold_sm4.dtseries.nii  --glm_analysis.py (all trials)-->  {run}_betas.h5 [betas: 64984 x K] + {run}_design.csv [K rows, trial_index = column]
{run}_betas.h5 + design.csv  --task_condition_identification.py --correct_only-->  {subj}_task_condition_betas.pkl   <-- BUG: col_index re-numbered after filtering; duplicates
pkl  --plot_per_task_within_across_subj_similarity.py-->  results/similarity_analysis/with_dms/*   (Figure 8)
pkl  --(task_group, Feb 2026)-->  results/similarity_analysis/similarity_bootstrap_results_task_group.csv   (Section 4.5 text)
```

## Data flow (after)

```
raw events.tsv + task_stimuli blockfiles  --verify_run_labels_from_blockfiles.py-->  run_labels_verified.csv, lookup_mismatch_report.{csv,md}
{run}_betas.h5 + design.csv + run_labels_verified.csv  --build_clean_delay_betas.py-->  data/encoding_delay/clean_delay_betas/{subj}_delay_betas.h5 + {subj}_provenance.csv + {subj}_validation.json
all subjects  --validate_clean_delay_betas.py-->  validation_table.csv, validation_report.md  (job fails on any violation)
clean h5  --similarity_analysis_clean.py-->  results/similarity_analysis/regenerated_clean/{variant}/  (long tables, 5x5 matrices, summaries, controls)
summaries  --figure_making/plot_fig8_regenerated_clean.py-->  figures/fig8_regenerated_clean/  (+ source-data CSVs)
```

## Phases

1. **Phase 2 first (labels)**: match every acquired run's stimulus sequence (raw `*_events.tsv`) to the
   124 block files in `/project/def-pbellec/xuan/task_stimuli/data/multfs/updated_cond_file/blockfiles`
   (the files the PsychoPy session runner `src/sessions/ses-multfs.py` actually loads). The matched block
   is the authoritative label. Compare with the lookup table and the behaviour-inferred label.
2. **Phase 1/3 (clean inputs)**: one record per physical delay beta; column index = `trial_index` of the
   unfiltered design CSV (asserted equal to the GLM column); provenance fields; legacy multiplicity kept
   as a field (for the "fixed indexing, duplicates kept" sensitivity variant) but the vector stored once;
   uniqueness key `subject+session+run+trial_index+phase`. Minimal patch of the legacy script so it can
   no longer produce mis-indexed pickles.
3. **Phase 4–6 (analysis)**: 50 recorded-seed split-half partitions, all unique trials, identical splits
   at vertex / 360-region / 12-network resolutions (region and network maps derived from the same
   vertex half-maps), Pearson across spatial units, within = diagonal, across = 10 unordered pairs
   (symmetrised). Variants: `original` (production pickles), `fixed_dups` (correct indexing, duplicates
   kept), `clean_lookup`, `clean_verified` (primary if block-file matching resolves the labels),
   `clean_exclude_unresolved`. Controls: spatial shuffle, task-label shuffle, clone, affine/session
   scalar invariance, extreme-PSC sensitivity.
4. **Phase 7 (figures)**: main 3-panel Figure 8 from regenerated task-level estimates; network 5x5
   matrices; audit figure; task-level plots. PDF + SVG + PNG (600 dpi) + source CSV.
5. **Phase 8–9 (text/report)**: methods, results, caption, limitations, final report.

## Terminology mapping (manuscript-facing)

| internal | task variant | family |
|---|---|---|
| dms_loc | DMS | DMS |
| nback_loc / nback_ctg / nback_obj | 1-Back location / category / identity | 1-Back |
| interdms_{loc,ctg,obj}_{ABAB,ABBA} | InterDMS location/category/identity ABAB/ABBA | InterDMS |
| ctxdm_lco | CTXDM LCI (location → category → identity) | CTXDM |
| ctxdm_col | CTXDM CIL (category → identity → location) | CTXDM |

(`lco`/`col` → LCI/CIL follows the letter order of the internal names; flagged for author confirmation.)

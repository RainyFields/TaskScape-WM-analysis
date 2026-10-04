# Execution plan — regenerated clean noise ceiling (region / network)

Isolated worktree: `/project/def-pbellec/xuan/fmri_dataset_project/.worktrees/nc_regen`
(branch `noise-ceiling-regen`, forked from `6af305885bea0a2e77f3a59bb607cc6b44f335e5`).
Nothing under the main working tree is modified; the concurrent similarity session's files
(`scripts/task_condition_identification.py` patch, `scripts/verify_run_labels_from_blockfiles.py`,
`results/similarity_analysis/regenerated_clean/`, `data/encoding_delay/clean_delay_betas/`) are read
at most, never written.

All computation runs on Rorqual compute nodes (`def-bashivan`) via
`~/.cursor/skills/alliance-compute-jobs/scripts/run_on_compute.sh`; job IDs / seeds / commits are
recorded in `reproducibility.md`.

## Dependency map of the existing (contaminated) pipeline

```
glasser_resampled/*_bold_sm4.dtseries.nii
  --glm_analysis.py (ALL trials, encoding+delay)-->
data/encoding_delay/trial_level_betas/{subj}/ses-*/func/{run}_betas.h5   [betas: 64984 x K, cols = task_regressor_names 'trialNNN_itemMM_{encoding|delay}']
                                                   + {run}_design.csv    [K rows, trial_index = column, is_correct on responded delay rows]
  --task_condition_identification.py --correct_only --target_selection delay_of_second-->
      run -> task variant via data/study_designs/{subj}_design_design_with_converted.tsv (dms -> dms_loc)
      BUG 1: col_index = arange(len(df)) AFTER dropping is_correct==False rows -> shifted columns (often encoding)
      BUG 2: pair schedule (1,2),(2,3),... + self pairs (1,1),(2,2),... both store the delay after item i2
             -> the same physical delay beta stored under 2 condition keys (~42% duplicates)
data/encoding_delay/trial_level_betas/grouped_betas/task_relevant_only/{subj}_task_condition_betas.pkl
  --noise_ceiling_final_version.py --aggregation task --unit region|network --n_bootstrap 50 --seed 0-->
      group_betas(): substring match of 12 task names over condition keys -> pooled reps (duplicates now in one pool)
      per iteration: permute reps, split at n//2, mean halves, region/network mean, column-wise Pearson across
      the 12 task variants, clip r to [-0.9999, 0.9999], Spearman-Brown per iteration, nanmean over 50 iterations
results/noise_ceiling/final_version/{subj}_agg-task_unit-{region,network}_{stats.json,_signed.npy,.npy}
  --(mean of the five subject 'mean_nc' values; no script)-->  0.657 (region) / 0.711 (network)
results/noise_ceiling/brain_visualization_inputs/*.npy  (renamed copies, older run, no script)
```

The manuscript values are reproduced exactly from the `final_version` stats JSON files
(see `phase1_audit.md`), so no re-run of the legacy script is needed to identify the source.

## Phases

1. Audit (this file + `phase1_audit.md`): reproduce 0.657 / 0.711 from stored outputs; simulate the legacy
   indexing on the design CSVs to count wrong-column / encoding-column / duplicate records per subject.
2. Clean inputs (`scripts/noise_ceiling_clean/build_clean_delay_betas_nc.py`): one record per physical delay
   beta, column = `trial_index` of the unfiltered design CSV, asserted against H5 `task_regressor_names`;
   trial-level correctness (all responses in the trial correct) as primary, item-level as sensitivity;
   provenance table; region (360) and network (12) matrices computed with the shared valid-vertex mask;
   fail-fast validation; per-subject manifest.
3. Estimator (`scripts/noise_ceiling_clean/noise_ceiling_clean.py`): 50 recorded-seed disjoint balanced
   splits per task variant, Pearson across the 12 variants per spatial unit, all raw r saved, Fisher-z
   aggregation, bounded Spearman-Brown; sensitivity aggregations (arithmetic mean, median, legacy
   per-iteration SB as diagnostic).
4. Controls: label permutation (one half / both halves), duplicate injection, old misindexed pickles,
   equalized trial counts, leave-one-condition-out, extreme PSC, lookup-label sensitivity
   (lookup labels vs verified block-file labels vs exclusion of unresolved runs).
5. Summaries (pooled vs subject-balanced), figure, manuscript text, final report.

# Reproducibility — noise ceiling regeneration

- Repository commit at the fork: `6af305885bea0a2e77f3a59bb607cc6b44f335e5` (`spatial_smoothed`).
- Branch / worktree: `noise-ceiling-regen` at `.worktrees/nc_regen`. Analysis scripts live only there and are uncommitted.
- Environment on compute nodes: `StdEnv/2023`, `python/3.11.5`, `scipy-stack/2025a`, then `source /project/def-pbellec/xuan/nilearn/bin/activate`. Account `def-bashivan`.
- Wrapper: `~/.cursor/skills/alliance-compute-jobs/scripts/run_on_compute.sh`.
- Primary seed: `20260926`. Iterations: 50. Odd-count policy: extra trial alternates (half A on even iterations).
- Clean beta input: `data/encoding_delay/clean_delay_betas/` (similarity-session build; validation PASS, CIFTI axis `5da42bc458858e1012633dda07ee397f17bafb0d`).
- Estimator output: `.worktrees/nc_regen/results/noise_ceiling/regenerated_clean/estimates/primary/`.
- Manuscript files and figures: `results/noise_ceiling/regenerated_clean/` and `figures/noise_ceiling/regenerated_clean/`.

## Jobs

| Job | ID | Role |
|---|---|---|
| nc_labels | 21873886 | Block-file label check (copy under the noise-ceiling results tree) |
| nc_legacy_audit | 21873887 | Count wrong columns and duplicates in the old grouping rule |
| nc_legacy_sub-01 … sub-06, sub-05 variants | 21873890–21873896 | Reproduce `noise_ceiling_final_version.py` on the production pickles |
| nc_clean_sub-01 … sub-06 | 21887679–21887683 | Region/network matrices and all estimator variants |
| nc_figures | 21903793 | Summaries and figures |

## Commands

```bash
cd /project/def-pbellec/xuan/fmri_dataset_project/.worktrees/nc_regen
# one subject; the array of five was submitted the same way
~/.cursor/skills/alliance-compute-jobs/scripts/run_on_compute.sh -b -A def-bashivan \
  -t 02:00:00 -c 2 -m 24G -J nc_clean_sub-01 -- \
  bash scripts/noise_ceiling_clean/job_clean_subject.sh sub-01

~/.cursor/skills/alliance-compute-jobs/scripts/run_on_compute.sh -b -A def-bashivan \
  -t 01:00:00 -c 2 -m 16G -J nc_figures -- \
  python scripts/noise_ceiling_clean/summarize_and_plot_nc.py
```

`job_clean_subject.sh` reads the shared H5, writes `data/encoding_delay/clean_delay_betas_nc/{subj}/`, then runs `noise_ceiling_clean.py` for `primary`, `labels_lookup`, `trial_level_correct`, `equalized_counts`, `perm_one_half`, `perm_both`, `seed_check_rerun`, `exclude_extreme_psc`, and (sub-01 only) `inject_duplicates_DIAGNOSTIC`, plus `legacy_misindexed` on the converted production pickle.

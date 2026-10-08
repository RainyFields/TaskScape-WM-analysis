# Regenerated R² products (2026-10-08)

Fresh outputs of the migrated scripts on TaskScape-WM `derivatives/glm` (all 558 current OLS H5), produced in
`outputs/r2_regen_2026-10-08/` (untracked; SLURM jobs 22718120-22718129, strace-audited) and copied here for review.
The previously tracked sub-03 region-average products under `scripts/sanity_check/glm_fitting_check_results/customized_glm/
encoding_delay_ssmoothing_psc/` and the legacy `data/r2_averaged*` maps are STALE (built 2026-05-27/28 from the
pre-cleanup GLM; the current H5 date from 2026-05-30) and were left unchanged. Trace, evidence and quantified
differences: `TaskScape-WM-migration/reports/r2_regen_2026-10-08/R2_STALENESS_REPORT.md`.

| file | content |
|---|---|
| `sub-03_all_sessions_runs_{r2_full,r2_task,r2_task_adj}_region_avg.npy` + `_summary.txt` + `_surface.png` | fresh region averages (112 direct runs each; old: 2 direct + 110 reconstructed / 2 runs) |
| `fig5c_unadjusted_r2_distribution_summary.json` | numbers behind Figure 5c (the tracked Figure 5c PNG is pixel-identical to the fresh one) |
| `all_tasks_adjusted_r2_distribution_paper.{pdf,_summary.json}` | fresh adjusted-R² figure and its numbers |
| `sub-XX_r2_averaged_aggregation_manifest.json` | run lists of the fresh vertex-wise R² averages for all 5 subjects (maps themselves: 63 x 5 npy in `outputs/`) |

No manuscript text was edited; whether to replace the old sub-03 values (0.878 / 0.484 / 0.276 -> 0.927 / 0.415 / 0.256)
wherever they were used is an author decision.

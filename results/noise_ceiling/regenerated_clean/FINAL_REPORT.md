# Final report — regenerated noise ceiling

Worktree: `/project/def-pbellec/xuan/fmri_dataset_project/.worktrees/nc_regen` (branch `noise-ceiling-regen`, forked from `6af305885bea`). Scripts there were not merged into the similarity-analysis edits. Manuscript-facing files are in this directory. Full estimate tables remain in the worktree under `results/noise_ceiling/regenerated_clean/estimates/`.

## 1. Existing code path and estimand

```
GLM H5 + design.csv
  → task_condition_identification.py --correct_only --target_selection delay_of_second
  → grouped_betas/task_relevant_only/{subj}_task_condition_betas.pkl
  → noise_ceiling_final_version.py --aggregation task --unit region|network --n_bootstrap 50 --seed 0
  → results/noise_ceiling/final_version/{subj}_agg-task_unit-{region,network}_signed.npy
```

The estimand was a split-half Pearson correlation across 12 task variants at each region or network, with Spearman–Brown applied to every iteration and then averaged. The manuscript numbers 0.657 (SD 0.298) and 0.711 (SD 0.234) are the pooled means and pooled SDs of those signed maps. They were reproduced exactly from the files on disk (maximum absolute difference 0). For sub-01, 02, 03, and 06 the source pickle was `{subj}_task_condition_betas.pkl`. For sub-05 it was `sub-05_task_condition_betas_badruns_excluded.pkl`, not the unfiltered sub-05 pickle (that pickle reproduces means of 0.07 and −0.31).

## 2. Identified bugs

In the committed grouping script, `col_index` was assigned after incorrect trials were dropped, so later rows read the wrong GLM column. Of 50,518 stored records, 19,657 (39%) pointed at the wrong column and 11,644 (23%) pointed at an encoding column. The pair schedule also stored the same delay beta under a pair key and a single-frame key: 20,755 records (41%) were duplicates of another physical delay beta. Those duplicates could fall in both split halves. The per-iteration Spearman–Brown formula `2r/(1+r)` was applied after clipping *r* to ±0.9999, which still produces values near ±10⁴ when *r* is close to −1. The saved maps were then clipped to [−1, 1].

## 3. Code changes

New scripts, only in the worktree: `scripts/noise_ceiling_clean/prepare_nc_bundle.py`, `noise_ceiling_clean.py`, `summarize_and_plot_nc.py`, plus the legacy-audit and legacy-pickle converters. The shared clean-beta builder and `task_condition_identification.py` were not edited. The production pickles and `results/noise_ceiling/final_version/` were not overwritten.

## 4. Input provenance and validation

Primary input: `data/encoding_delay/clean_delay_betas/{subj}_delay_betas.h5` and `{subj}_provenance.csv`, accepted because `results/similarity_analysis/regenerated_clean/validation_report.md` is PASS and each `{subj}_validation.json` reports `wrong_column = 0`, `n_encoding_in_delay_dataset = 0`, `n_duplicate_keys = 0`, `n_physically_identical_vectors = 0`, and `n_nan_inf = 0`. The CIFTI axis SHA1 is `5da42bc458858e1012633dda07ee397f17bafb0d` for every subject. Region and network matrices for the estimator are in the worktree `data/encoding_delay/clean_delay_betas_nc/`.

## 5. Trial counts

Retained unique delay betas (item-level correct): sub-01 6,034; sub-02 6,227; sub-03 6,098; sub-05 5,368; sub-06 6,036. All 12 variants are present. Per-variant counts are in `clean_inputs/{subj}/{subj}_trial_counts.csv` (worktree). The smallest cells are sub-05 1-Back category (46) and 1-Back identity (68). Odd counts use the alternating-extra-trial rule. sub-05 has 110 GLM runs; the others have 112.

## 6. Duplicates and wrong columns

| | Legacy stored records | Unique physical betas | Duplicate records | Wrong column | Encoding column |
|---|---:|---:|---:|---:|---:|
| All subjects | 50,518 | 29,763 | 20,755 (41%) | 19,657 (39%) | 11,644 (23%) |

After the clean build those three error counts are 0. The unique-beta counts match the retained counts above.

## 7. Lookup mismatches

Block-file matching (`run_labels_verified.csv`): 558/564 acquired runs matched a block file exactly; 4 runs were relabelled, all in sub-06 session 007 (1-Back location→identity, and three InterDMS runs). Those relabels cover 200 retained delay betas. No GLM run was unresolved. The primary analysis uses the block-file labels. Repeating it with lookup labels changed only sub-06 (region mean 0.526→0.541; network 0.652→0.673).

## 8. Negative correlations

Raw *r* is saved for every subject, resolution, unit, and iteration. Fisher averaging is done before the Spearman–Brown step. Units with *r̄* ≤ 0 get NC = 0 (109, 43, 111, 204, and 18 regions; 2, 1, 6, 12, and 1 networks). No noise-ceiling value is outside [0, 1]. The old per-iteration formula, kept only as a diagnostic, has absolute value greater than 1 in up to 185 regions (sub-05).

## 9. Final region and network tables

Subject-balanced summary (mean of subject means), primary estimator:

| | Region | Network |
|---|---:|---:|
| Subject-balanced mean | 0.346 | 0.324 |
| SD of subject means | 0.128 | 0.271 |
| Pooled mean (same, equal *n*) | 0.346 | 0.324 |
| Pooled SD | 0.284 | 0.316 |
| Pooled proportion equal to 0 | 0.269 | 0.367 |

Subject means (region / network): sub-01 0.278 / 0.298; sub-02 0.432 / 0.536; sub-03 0.271 / 0.133; sub-05 0.224 / 0.000; sub-06 0.526 / 0.652. Full distributions are in `summaries/subject_summaries.csv` once the figure job has written them, and in each `estimates/primary/{subj}/unit_estimates_{region,network}.csv`.

## 10. Controls and sensitivity

| Variant | Region, subject-balanced mean | Network |
|---|---:|---:|
| Primary | 0.346 | 0.324 |
| Lookup labels | 0.349 | 0.328 |
| One-half label permutation | 0.033 | 0.036 |
| Same permutation on both halves | 0.346 | 0.324 |
| Equalized trial counts | 0.216 | 0.188 |
| Exclude extreme PSC records | 0.383 | 0.416 |
| Legacy mis-indexed pickles, this estimator | 0.692 | 0.710 |
| Duplicate injection, sub-01 only | 0.614 (was 0.278) | 0.590 (was 0.298) |

Trial-level correctness (drop every delay in a trial that has any incorrect response) could not be run for sub-05: 1-Back category then had 0 trials. In the other four subjects the means stayed within about 0.04 of the primary region means. Seed 20260926 reproduced every primary noise ceiling with a maximum absolute difference of 0. Leave-one-variant-out flags 106–137 regions per subject whose NC falls by more than 0.2 when one variant is removed, so single-variant dominance is common at the region scale and uncommon at the network scale (0–4 networks).

## 11. Comparison with 0.657 / 0.711

Those numbers do not reappear from the clean pipeline. They reappear, at the network level (0.71), when the old pickles are scored with the corrected aggregator. The inflation is in the inputs. Duplicate injection alone, without wrong columns, accounts for a large part of it in sub-01.

## 12. Figure paths

`figures/noise_ceiling/regenerated_clean/noise_ceiling_main.{pdf,svg,png}` and `source_data_noise_ceiling.csv`, `plot_config.json`. Supplementary: `supp_old_vs_new`, `supp_equalized_counts`, `supp_loco`, `supp_region_raw_r`, `supp_region_frac_negative`.

## 13. Proposed manuscript text

`noise_ceiling_methods.md`, `noise_ceiling_results.md`, `noise_ceiling_figure_caption.md`. The sentence that the betas “contain strong signal, particularly at the region and network levels” should not be kept. The replacement is the results note: structure is present in some subjects and some regions, and is not a single upper bound for models.

## 14. Unresolved limitations

See `noise_ceiling_limitations.md`. The main open items are the sub-05 trial-level sensitivity (no fully correct 1-Back category trials), the extreme-PSC records that move sub-05 networks from 0 to 0.33 when removed, and the decision not to report a vertex-level map.

## 15. Reproduction

See `reproducibility.md`. Account `def-bashivan`. Seed `20260926`. Fifty iterations. Clean-data jobs `21887679`–`21887683`. Figure job `21903793`.

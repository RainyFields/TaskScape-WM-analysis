# Phase 1 — audit of the existing noise-ceiling implementation

Repository commit audited: `6af305885bea0a2e77f3a59bb607cc6b44f335e5` (branch `spatial_smoothed`; working tree
of the main checkout additionally contains an uncommitted patch to `scripts/task_condition_identification.py`
made by the concurrent similarity session — not used here).

## Scripts and intermediate files (existing pipeline)

| Step | Script / file | Key logic |
|---|---|---|
| Trial-level GLM | `scripts/glm_analysis.py` | one regressor per (trial, item, phase); H5 `betas` (64984 × K), `task_regressor_names` = `trialNNN_itemMM_{encoding,delay}`; sibling `*_design.csv` has one row per column, `trial_index` = column; `is_correct` filled only on the delay row of items that required a response (nback items 2–6, ctxdm item 3, dms item 2, interdms items 3–4) |
| Trial / condition identification, correct-trial filter, delay-beta selection | `scripts/task_condition_identification.py` (`group_betas_within_trial`, lines 336–504 at 6af3058) | drops rows with `is_correct` ∈ {False, b, 0} **then** sets `col_index = arange(len(df))` (BUG 1); for every pair (i1,i2) in the task schedule stores `betas[:, delay row of i2]`, and again for every self pair (i,i) (BUG 2 → duplicates) |
| Run → task variant | `get_detailed_task_name` + `data/study_designs/{subj}_design_design_with_converted.tsv` | string lookup; DMS not in table → `dms_loc` |
| Grouped pickles | `data/encoding_delay/trial_level_betas/grouped_betas/task_relevant_only/{subj}_task_condition_betas.pkl` (2026-05-30; `submit_regenerate_grouped_betas.sh`, `--correct_only --target_selection delay_of_second`) | `dict[condition_key -> list[64984-vectors]]` |
| Region / network aggregation, split halves, Pearson, Spearman–Brown, iteration averaging | `scripts/noise_ceiling_final_version.py` | `group_betas(mode='task')` pools all keys of a task variant (duplicates pooled); per iteration `rng.permutation`, split at `n//2`, means, `avg_by_region` (nanmean over valid vertices), `avg_by_network` (unweighted mean of regions), column-wise Pearson across the 12 variants, clip r to ±0.9999, `2r/(1+r)` **per iteration**, `nanmean` over 50 iterations, clip to [−1, 1] |
| Summary statistics | `results/noise_ceiling/final_version/{subj}_agg-task_unit-{region,network}_stats.json`, `*_signed.npy`, `*.npy` (positive-only) | mean / median / SD / negative counts per subject |
| Figure inputs | `results/noise_ceiling/brain_visualization_inputs/*.npy` (2026-03; renamed copies of an older run; no generating script in the repo) | |
| Figure | no script in `scripts/figure_making/` or `figures4papers/` produces the manuscript NC figure; per-subject histograms are drawn inside `noise_ceiling_final_version.py` | |

## Where 0.657 / 0.711 come from (reproduced exactly)

Pooling the five subjects' `*_signed.npy` arrays from `results/noise_ceiling/final_version/`
(working-tree version — the sub-05 files were regenerated after the commit, see below):

| unit | pooled mean | pooled SD (ddof=0) | subject means |
|---|---|---|---|
| region (5 × 360) | **0.6574** | **0.2976** | 0.552, 0.761, 0.592, 0.559, 0.823 |
| network (5 × 12) | **0.7106** | **0.2339** | 0.662, 0.816, 0.646, 0.511, 0.918 |

These match the manuscript values (0.657 ± 0.298; 0.711 ± 0.234). Because every subject has the same number
of units, the pooled mean equals the mean of subject means; the SD is the pooled SD across all
1800 / 60 units (it includes between-subject spread). The mean of the subject-level SDs is 0.260 / 0.163.

Ambiguity: the committed `sub-05` stats (mean −0.31 network, 0.07 region) differ from the working-tree
files used for 0.657/0.711 (0.51 / 0.56). Three sub-05 pickles exist (`…betas.pkl`, `…_excluded.pkl`,
`…_badruns_excluded.pkl`); the legacy reproduction runs all three to identify which one produced the reported
values (`legacy_reproduction/`).

Inputs used by those results: pickles generated 2026-05-30 09:05–09:40 (sub-05 variants 21:49/21:56) with the
pre-patch `task_condition_identification.py`, i.e. affected by both bugs. The legacy noise-ceiling script
(`--aggregation task --n_bootstrap 50 --seed 0`, vertex mask on, Spearman–Brown on) is unchanged in git.

## Confirmed defects (quantified in `legacy_audit/`)

1. **Wrong GLM column after correct-trial filtering** — `col_index` re-numbered after dropping incorrect rows;
   every row after the first dropped row in a run is shifted onto a lower column, often an encoding column or
   a different trial.
2. **Duplicate physical betas** — the delay after item i (i > 1) is stored once for the pair (i−1, i) and once
   for the self pair (i, i); `group_betas(task)` pools both → the same beta can fall in both halves.
3. **Row-level, not trial-level, correctness** — only the item carrying the incorrect response was dropped; the
   other delay periods of an incorrect trial were kept.
4. **Unstable Spearman–Brown** — `2r/(1+r)` applied per iteration to r clipped at −0.9999 gives values down to
   −19998 which are then averaged; the stored maps were clipped to [−1, 1] afterwards.
5. **Odd counts** — one record silently dropped from half A on every iteration (`n//2`).

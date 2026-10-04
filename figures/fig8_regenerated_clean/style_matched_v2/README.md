# Figure 8 style-matched v2

Revised layout matching the June 2026 12-task boxplot (three stacked panels, family
colours, subject markers, right-hand legends). Numerical values are the corrected
`clean_verified` analysis only.

## Inputs (canonical, corrected)

- `/project/def-pbellec/xuan/fmri_dataset_project/results/similarity_analysis/regenerated_clean/summary_subject_task.csv` — within-subject *r*, already averaged over 50 splits
- `/project/def-pbellec/xuan/fmri_dataset_project/results/similarity_analysis/regenerated_clean/summary_pair_task.csv` — 10 unordered pair *r*, already averaged over 50 splits
- `/project/def-pbellec/xuan/fmri_dataset_project/results/similarity_analysis/regenerated_clean/summary_task_level.csv` — task-level means (validation only)
- `/project/def-pbellec/xuan/fmri_dataset_project/results/similarity_analysis/regenerated_clean/summary_family_level.csv` — family-level means (validation + summary markers)

Variant: `clean_verified` only. Legacy pickles, `with_dms/`, and `similarity_bootstrap_results_*.csv` are not read.

## Aggregation

1. Split iterations are averaged first, inside the clean analysis tables. This script never opens the 50-iteration long tables for plotting.
2. **Within-subject, one subject × task × resolution:** `summary_subject_task.mean` (mean of 50 split-half diagonals).
3. **Across-subject, one subject × task × resolution:** mean of the four pair estimates that include that subject. Each pair estimate is already `0.5 * [r(i_A, j_B) + r(j_A, i_B)]` averaged over 50 splits. Averaging the five subject values recovers the 10-pair mean (each pair is counted once for each of its two members).
4. Boxplots (median, hinges, whiskers to min/max) are built from those **five subject-level values**. They are not built from split iterations.
5. Family-mean markers are the `summary_family_level.csv` values (mean of the task-level estimates). They are a separate layer and do not replace the 12 task boxplots.

## Visual encoding

| Family | Colour |
|---|---|
| DMS | `#377EB8` |
| 1-Back | `#FFD21F` |
| InterDMS | `#A04A9B` |
| CTXDM | `#4899A5` |

- Within-subject box: family colour, black outline and median (1.6 pt).
- Across-subject box: `#D0D0D0`, black outline and median (1.6 pt).
- Subject markers (white fill, 1.15 pt black edge, ~7 pt): Sub 01 circle, Sub 02 square, Sub 03 up-triangle, Sub 05 diamond, Sub 06 down-triangle. Deterministic horizontal jitter `linspace(-0.045, 0.045, 5)` in subject order.
- Family-mean layer: filled circle (within) / square (across), ~10 pt, white edge, placed at the x-centre of each family group.

## Figure dimensions and type

- `figsize = (7.5, 9.3)` inches (manuscript width unknown; Nature double column is 183 mm ≈ 7.2 in).
- Fonts: titles 16 pt, axis labels 15 pt, y-ticks 13 pt, x-ticks 12 pt, legend titles 14 pt, legend entries 12.5 pt.
- x-axis labels name the variant only (`location`, `loc. ABAB`, `LCI`, …). Family identity is the colour band / Task-family legend, so 12 pt labels do not collide. Full names are in the source CSV.
- Shared y-limits: `-0.891` to `1.050` (includes all subject-level values; 48 negative points).
- PNG saved with `facecolor="white", transparent=False`.

## Not included

No paired *t*-test, significance stars, split-iteration point cloud, or legacy numerical values.

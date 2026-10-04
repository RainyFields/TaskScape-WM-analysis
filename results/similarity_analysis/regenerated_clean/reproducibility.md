# Reproducibility record

Repository HEAD at start of this work: `6af305885bea0a2e77f3a59bb607cc6b44f335e5` (2026-05-30).
Working-tree additions for this analysis are uncommitted scripts (not production outputs).

## Environment (every SLURM job)

```text
module load StdEnv/2023 python/3.11.5 scipy-stack/2025a
source /project/def-pbellec/xuan/nilearn/bin/activate
```

Wrapper: `~/.cursor/skills/alliance-compute-jobs/scripts/run_on_compute.sh`  
Account: `def-bashivan`  
Cluster: Rorqual

## Seeds (`scripts/similarity_analysis_clean.py`)

| Item | Value |
|---|---|
| BASE_SEED | 20260926 |
| Split RNG | `SeedSequence([BASE_SEED, split, subject_index, task_index])`; family-pooled uses `task_index + 100` |
| Session-scale | BASE_SEED + 7 |
| Spatial shuffle | BASE_SEED + 11 |
| Task-label derangement | BASE_SEED + 13 → permutation `[6, 8, 1, 11, 5, 10, 2, 4, 3, 0, 9, 7]` |
| N splits | 50 |
| Trial cap | none (all unique retained delay trials at every resolution) |

## SLURM job IDs

| Job | ID | Role | Outcome |
|---|---|---|---|
| fig8_labels | 21873605 | block-file label verification | success |
| build_sub-0{1,2,3,5,6} | 21873766–21873770 | clean delay-beta H5 + provenance | success |
| halves_sub-0{1,2,5,6} | 21873939–21873942 | split-half maps | success |
| halves_sub-03 | 21874057 | split-half maps | success |
| validate_clean | 21874229 | cross-subject validation | PASS |
| fig8_sim | 21874759 | similarity + controls | failed at robust Aggregator (baseline floor 1000 emptied parcel 42) |
| fig8_sim2 | 21877024 | similarity + controls (floor 1.0) | success, 370 s |
| fig8_plot | 21886822 | Figure 8 + supplements | success, 49 s, exit 0:0 |

## Paths

| What | Path |
|---|---|
| GLM source | `data/encoding_delay/trial_level_betas/{subj}/ses-*/func/*_betas.h5` |
| Design CSVs | sibling `*_design.csv` (`trial_index` = column) |
| Verified labels | `results/similarity_analysis/regenerated_clean/run_labels_verified.csv` |
| Clean delay betas | `data/encoding_delay/clean_delay_betas/{subj}_delay_betas.h5` (durable) |
| Split-half maps (intermediate, scratch) | `/scratch/xuan/fig8_clean/halves/halves_*_{subj}.npz` (~11 GB; not durable) |
| Analysis tables | `results/similarity_analysis/regenerated_clean/` |
| Figures | `figures/fig8_regenerated_clean/` |
| Production (untouched) | `results/similarity_analysis/with_dms/` and `similarity_bootstrap_results_*.csv` |

## Commands (in order)

```bash
cd /project/def-pbellec/xuan/fmri_dataset_project
S=~/.cursor/skills/alliance-compute-jobs/scripts
export AGENT_JOB_LOGDIR=$PWD/results/similarity_analysis/regenerated_clean/logs

$S/run_on_compute.sh -b -t 01:00:00 -c 2 -m 8G -J fig8_labels -- \
  python scripts/verify_run_labels_from_blockfiles.py \
    --out_dir results/similarity_analysis/regenerated_clean

for s in sub-01 sub-02 sub-03 sub-05 sub-06; do
  $S/run_on_compute.sh -b -t 02:00:00 -c 4 -m 32G -J build_${s} -- \
    python scripts/build_clean_delay_betas.py --subj $s \
      --labels results/similarity_analysis/regenerated_clean/run_labels_verified.csv \
      --out_dir data/encoding_delay/clean_delay_betas
done

$S/run_on_compute.sh -b -t 01:00:00 -c 4 -m 16G -J validate_clean -- \
  python scripts/validate_clean_delay_betas.py \
    --data_dir data/encoding_delay/clean_delay_betas \
    --out_dir results/similarity_analysis/regenerated_clean

mkdir -p /scratch/xuan/fig8_clean/halves
for s in sub-01 sub-02 sub-03 sub-05 sub-06; do
  $S/run_on_compute.sh -b -t 02:00:00 -c 4 -m 32G -J halves_${s} -- \
    python scripts/similarity_analysis_clean.py --stage halves --subj $s \
      --work_dir /scratch/xuan/fig8_clean/halves \
      --out_dir results/similarity_analysis/regenerated_clean
done

$S/run_on_compute.sh -b -t 01:00:00 -c 4 -m 32G -J fig8_sim2 -- \
  python scripts/similarity_analysis_clean.py --stage similarity \
    --work_dir /scratch/xuan/fig8_clean/halves \
    --out_dir results/similarity_analysis/regenerated_clean

$S/run_on_compute.sh -b -t 00:30:00 -c 2 -m 8G -J fig8_plot -- \
  python scripts/figure_making/plot_fig8_regenerated_clean.py \
    --results results/similarity_analysis/regenerated_clean \
    --out figures/fig8_regenerated_clean
```

Primary variant: `clean_verified`. Config dump: `analysis_config.json`.

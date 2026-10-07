#!/bin/bash
# TaskScape-WM migration fidelity: clean noise ceiling (primary variant) from the dataset's
# derivatives/glm-clean-delay -> prepare_nc_bundle.py -> noise_ceiling_clean.py, outputs under OUTPUT_ROOT.
# Same seed / iterations / flags as job_clean_subject.sh (pre-migration), so results must match exactly.
# Usage: job_clean_primary_taskscape.sh <subj> [output_root]
set -euo pipefail
SUBJ=$1
REPO=$(cd "$(dirname "$0")/../.." && pwd)
export TASKSCAPE_OUTPUT_ROOT=${2:-${TASKSCAPE_OUTPUT_ROOT:-/project/def-pbellec/xuan/TaskScape-WM-analysis/outputs}}
SC=$REPO/scripts/noise_ceiling_clean
BUNDLE=$TASKSCAPE_OUTPUT_ROOT/noise_ceiling/clean_delay_betas_nc/$SUBJ
RES=$TASKSCAPE_OUTPUT_ROOT/noise_ceiling/regenerated_clean
SEED=20260926
export PYTHONDONTWRITEBYTECODE=1
python $SC/prepare_nc_bundle.py --subj $SUBJ --out_dir $BUNDLE
python $SC/noise_ceiling_clean.py --subj $SUBJ --input_dir $BUNDLE \
  --out_dir $RES/estimates/primary/$SUBJ --n_iter 50 --seed $SEED --variant_name primary \
  --label_col task_variant --retain_col retained --require_resolved
echo "DONE $SUBJ"

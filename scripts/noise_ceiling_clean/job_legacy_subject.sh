#!/bin/bash
# Per-subject legacy reproduction on the OLD production pickles (contaminated inputs):
#  1. run the untouched legacy estimator (noise_ceiling_final_version.py @ 6af3058, seed 0, 50 iterations)
#     for region and network -> results/noise_ceiling/regenerated_clean/legacy_reproduction/<pickle tag>/
#  2. convert the pickle into a bundle -> data/encoding_delay/legacy_bundles_nc/<pickle tag>/
# Usage: job_legacy_subject.sh <subj> <pickle path> <tag>
set -euo pipefail
SUBJ=$1; PKL=$2; TAG=$3
WT=/project/def-pbellec/xuan/fmri_dataset_project/.worktrees/nc_regen
PROJECT=/project/def-pbellec/xuan/fmri_dataset_project
OUT=$WT/results/noise_ceiling/regenerated_clean/legacy_reproduction/$TAG
mkdir -p "$OUT"
cd $WT/scripts
for unit in region network; do
  python noise_ceiling_final_version.py --subject $SUBJ --input_pickle "$PKL" \
    --atlas_path $PROJECT/data/Glasser_LR_Dense64k.dlabel.nii \
    --network_assignments_path $PROJECT/data/cortex_parcel_network_assignments.txt \
    --output_dir "$OUT" --unit $unit --aggregation task --n_bootstrap 50 --seed 0
done
python $WT/scripts/noise_ceiling_clean/legacy_pickle_to_bundle.py --subj $SUBJ --pickle "$PKL" \
  --out_dir $WT/data/encoding_delay/legacy_bundles_nc/$TAG

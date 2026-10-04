#!/bin/bash
# Build region/network matrices from the validated shared clean H5, then run the corrected estimator.
# Usage: job_clean_subject.sh <subj>
set -euo pipefail
SUBJ=$1
WT=/project/def-pbellec/xuan/fmri_dataset_project/.worktrees/nc_regen
SC=$WT/scripts/noise_ceiling_clean
BUNDLE=$WT/data/encoding_delay/clean_delay_betas_nc/$SUBJ
RES=$WT/results/noise_ceiling/regenerated_clean
LEGACY=$WT/data/encoding_delay/legacy_bundles_nc/production_$SUBJ
SEED=20260926
mkdir -p "$RES/clean_inputs/$SUBJ"
python $SC/prepare_nc_bundle.py --subj $SUBJ --out_dir $BUNDLE
cp $BUNDLE/${SUBJ}_delay_provenance.csv $BUNDLE/${SUBJ}_manifest.json \
   $BUNDLE/${SUBJ}_trial_counts.csv $BUNDLE/${SUBJ}_extreme_records.csv \
   $RES/clean_inputs/$SUBJ/

FAILS=$RES/estimates/failures.txt
run() { # name extra-args...
  local name=$1; shift
  if ! python $SC/noise_ceiling_clean.py --subj $SUBJ --input_dir $BUNDLE \
      --out_dir $RES/estimates/$name/$SUBJ --n_iter 50 --seed $SEED --variant_name $name "$@"; then
    echo "FAILED $SUBJ $name" | tee -a "$FAILS"
  fi
}

run primary                          --label_col task_variant --retain_col retained --require_resolved
run labels_lookup                    --label_col label_lookup --retain_col retained
run trial_level_correct              --label_col task_variant --retain_col retain_trial_level --require_resolved
run equalized_counts                 --label_col task_variant --retain_col retained --require_resolved --equalize_counts
run perm_one_half                    --label_col task_variant --retain_col retained --require_resolved --permute_labels one_half
run perm_both                        --label_col task_variant --retain_col retained --require_resolved --permute_labels both
run seed_check_rerun                 --label_col task_variant --retain_col retained --require_resolved
if [[ $SUBJ == sub-01 ]]; then
  run inject_duplicates_DIAGNOSTIC   --label_col task_variant --retain_col retained --require_resolved --inject_duplicates
fi
run exclude_extreme_psc            --label_col task_variant --retain_col retained --require_resolved \
                                     --exclude_records_csv $BUNDLE/${SUBJ}_extreme_records.csv

# Same estimator on the old mis-indexed, duplicated pickles (diagnostic; not a manuscript result).
if ! python $SC/noise_ceiling_clean.py --subj $SUBJ --input_dir $LEGACY \
    --out_dir $RES/estimates/legacy_misindexed/$SUBJ --n_iter 50 --seed $SEED \
    --variant_name legacy_misindexed --label_col task_variant_lookup --retain_col retain_trial_level; then
  echo "FAILED $SUBJ legacy_misindexed" | tee -a "$FAILS"
fi
echo "DONE $SUBJ"

#!/bin/bash
#SBATCH --job-name=sync_betas_ss
#SBATCH --account=def-pbellec
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=/project/def-pbellec/xuan/fmri_dataset_project/logs/%x_%j.log
#SBATCH --error=/project/def-pbellec/xuan/fmri_dataset_project/logs/%x_%j.err

# Sync smoothed + PSC trial-level betas to the shared def-bashivan folder.
# rsync is resumable: if the job times out, resubmit and it picks up where it left off.

SRC=/project/def-pbellec/xuan/fmri_dataset_project/data/encoding_delay/trial_level_betas
DST=/project/def-bashivan/shared/fmri_betas_ss

set -u

echo "[$(date)] start sync $SRC -> $DST"

# Per-subject loop so the log shows coarse progress even when rsync output is sparse
for subj in sub-01 sub-02 sub-03 sub-05 sub-06; do
    echo "[$(date)] syncing ${subj} ..."
    rsync -a --partial --info=stats1 "${SRC}/${subj}" "${DST}/"
    echo "[$(date)] done ${subj} (rsync exit $?)"
done

# Top-level extras (grouped_betas, results, manifests)
echo "[$(date)] syncing top-level extras ..."
rsync -a --partial --info=stats1 --exclude 'sub-*' "${SRC}/" "${DST}/"
echo "[$(date)] all done"

#!/bin/bash
#SBATCH --account=def-pbellec
#SBATCH --job-name=tar_bids
#SBATCH --array=0-4
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=4:00:00
#SBATCH --output=/project/def-pbellec/xuan/fmri_dataset_project/logs/%x_%A_%a.log
#SBATCH --error=/project/def-pbellec/xuan/fmri_dataset_project/logs/%x_%A_%a.err

# Archive the BIDS release as one (uncompressed) tar per subject into the
# def-bashivan shared project space. NIfTI/CIFTI are already gzip'd, so no -z.
set -euo pipefail
shopt -s nullglob

SRC="${SCRATCH}/cneuromod_multfs_bids"
DEST="/project/def-bashivan/shared/cneuromod_multfs_bids"
mkdir -p "$DEST"

SUBJECTS=(sub-01 sub-02 sub-03 sub-05 sub-06)
SUB=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}

cd "$SRC"

echo "[$(date)] host=$(hostname) subject=${SUB} -> ${DEST}/${SUB}.tar"

PATHS=( "${SUB}" "derivatives/fmriprep/${SUB}" "derivatives/glm/${SUB}" )
PATHS+=( derivatives/fmriprep/${SUB}_ses-*.html )
[ -f "derivatives/fmriprep/${SUB}.html" ] && PATHS+=( "derivatives/fmriprep/${SUB}.html" )

tar -cf "${DEST}/${SUB}.tar" "${PATHS[@]}"
echo "[$(date)] ${SUB}.tar size: $(du -h "${DEST}/${SUB}.tar" | cut -f1)"

# Array task 0 also archives the shared top-level metadata.
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    echo "[$(date)] building metadata.tar"
    META=( dataset_description.json events.json README CHANGES .bidsignore )
    META+=( task-*_bold.json )
    META+=( derivatives/fmriprep/dataset_description.json )
    META+=( derivatives/glm/dataset_description.json )
    tar -cf "${DEST}/metadata.tar" "${META[@]}"
    echo "[$(date)] metadata.tar size: $(du -h "${DEST}/metadata.tar" | cut -f1)"
fi

echo "[$(date)] ${SUB} done."

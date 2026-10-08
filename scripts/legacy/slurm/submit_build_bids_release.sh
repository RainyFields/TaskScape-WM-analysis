#!/bin/bash
#SBATCH --account=def-pbellec
#SBATCH --job-name=build_bids
#SBATCH --array=0-4
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=8:00:00
#SBATCH --output=/project/def-pbellec/xuan/fmri_dataset_project/logs/%x_%A_%a.log
#SBATCH --error=/project/def-pbellec/xuan/fmri_dataset_project/logs/%x_%A_%a.err

# One array task per subject. Heavy I/O (rsync -L of raw + fMRIPrep + GLM).
set -euo pipefail

module load StdEnv/2023
module load python/3.11.5
module load scipy-stack/2025a
source /project/def-pbellec/xuan/nilearn/bin/activate

SUBJECTS=(sub-01 sub-02 sub-03 sub-05 sub-06)
SUB=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}

DATASET_ROOT="${SCRATCH}/cneuromod_multfs_bids"

echo "[$(date)] host=$(hostname) array_id=${SLURM_ARRAY_TASK_ID} subject=${SUB}"
echo "dataset_root=${DATASET_ROOT}"

python /project/def-pbellec/xuan/fmri_dataset_project/scripts/build_bids_release.py \
    --dataset_root "${DATASET_ROOT}" \
    --subjects "${SUB}" \
    --components raw events fmriprep glm

echo "[$(date)] subject ${SUB} done."

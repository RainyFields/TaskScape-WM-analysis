#!/bin/bash
#SBATCH --account=def-pbellec
#SBATCH --job-name=group_betas
#SBATCH --array=0-4
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output=/project/def-pbellec/xuan/fmri_dataset_project/logs/group_betas_%A_%a.log
#SBATCH --error=/project/def-pbellec/xuan/fmri_dataset_project/logs/group_betas_%A_%a.err

# Rebuild grouped condition PKLs from cleaned trial-level GLM outputs.
# Submit only after submit_glm_clean_regenerate.sh array has finished.

SUBJECTS=(sub-01 sub-02 sub-03 sub-05 sub-06)
SUBJ=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}

echo "Grouping betas for ${SUBJ}"
echo "Start: $(date)"

module load StdEnv/2023
module load python/3.11.5
source /project/def-pbellec/xuan/nilearn/bin/activate

GLM_ROOT=/project/def-pbellec/xuan/fmri_dataset_project/data/encoding_delay/trial_level_betas

python /project/def-pbellec/xuan/fmri_dataset_project/scripts/task_condition_identification.py \
    --subj "${SUBJ}" \
    --glm_root "${GLM_ROOT}" \
    --out_root "${GLM_ROOT}" \
    --tasks all \
    --correct_only \
    --target_selection delay_of_second

echo "End: $(date)"

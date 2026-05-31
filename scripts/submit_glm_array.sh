#!/bin/bash
#SBATCH --account=def-pbellec
#SBATCH --job-name=glm_glasser
#SBATCH --array=0-19
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/project/def-pbellec/xuan/fmri_dataset_project/logs/glm_%A_%a.log
#SBATCH --error=/project/def-pbellec/xuan/fmri_dataset_project/logs/glm_%A_%a.err

SUBJECTS=(sub-01 sub-02 sub-03 sub-05 sub-06)
TASKS=(1back ctxdm dms interdms)

# 20 combinations: 5 subjects x 4 tasks
SUBJ=${SUBJECTS[$((SLURM_ARRAY_TASK_ID / 4))]}
TASK=${TASKS[$((SLURM_ARRAY_TASK_ID % 4))]}

echo "Array task ${SLURM_ARRAY_TASK_ID}: ${SUBJ} ${TASK}"
echo "Start: $(date)"

module load StdEnv/2023
module load python/3.11.5
source /project/def-pbellec/xuan/nilearn/bin/activate

python /project/def-pbellec/xuan/fmri_dataset_project/scripts/glm_analysis.py \
    --subj ${SUBJ} \
    --tasks ${TASK} \
    --estimator ols \
    --fmri_root /project/def-pbellec/xuan/fmri_dataset_project/data/glasser_resampled \
    --out_root /project/def-pbellec/xuan/fmri_dataset_project/data/encoding_delay \
    --overwrite

echo "End: $(date)"

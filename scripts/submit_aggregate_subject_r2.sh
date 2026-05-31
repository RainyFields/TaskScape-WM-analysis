#!/bin/bash
#SBATCH --account=def-bashivan
#SBATCH --job-name=r2_agg
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

module purge
module load python/3.10
module load scipy-stack
source /project/def-pbellec/xuan/nilearn/bin/activate

cd /project/def-pbellec/xuan/fmri_dataset_project/scripts

SUBJ="${1:-sub-03}"
OUT_ROOT="/project/def-pbellec/xuan/fmri_dataset_project/data/r2_averaged_glasser64k"
mkdir -p "${OUT_ROOT}"
python aggregate_subject_r2_vertex_maps.py --subj "${SUBJ}" 2>&1 | tee "${OUT_ROOT}/${SUBJ}_aggregate.log"

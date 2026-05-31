#!/bin/bash
set -euo pipefail

mkdir -p /project/def-pbellec/xuan/fmri_dataset_project/logs
sbatch /project/def-pbellec/xuan/fmri_dataset_project/scripts/smooth_glasser_sm4_array.sh

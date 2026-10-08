#!/bin/bash
#SBATCH --account=def-pbellec
#SBATCH --job-name=smooth_glasser_sm4
#SBATCH --array=0-4
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=/project/def-pbellec/xuan/fmri_dataset_project/logs/smooth_sm4_%A_%a.log
#SBATCH --error=/project/def-pbellec/xuan/fmri_dataset_project/logs/smooth_sm4_%A_%a.err

set -euo pipefail

SUBJECTS=(sub-01 sub-02 sub-03 sub-05 sub-06)
SUBJ="${SUBJECTS[$SLURM_ARRAY_TASK_ID]}"

module load StdEnv/2023 connectomeworkbench/2.0.1

BASEDIR="/project/def-pbellec/xuan/fmri_dataset_project/data/glasser_resampled"
LEFT_SURF="/project/def-pbellec/xuan/fmri_dataset_project/data/fsLR_32k_midthickness/tpl-fsLR_hemi-L_den-32k_midthickness.surf.gii"
RIGHT_SURF="/project/def-pbellec/xuan/fmri_dataset_project/data/fsLR_32k_midthickness/tpl-fsLR_hemi-R_den-32k_midthickness.surf.gii"

SIGMA_SURF=1.6986
SIGMA_VOL=0
DIRECTION=COLUMN

output_path_for() {
    local input="$1"
    if [[ "$input" != *_space-Glasser64k_bold.dtseries.nii ]]; then
        echo "ERROR: Unexpected input filename pattern: $input" >&2
        exit 1
    fi
    echo "${input%_space-Glasser64k_bold.dtseries.nii}_space-Glasser64k_bold_sm4.dtseries.nii"
}

assert_paths_differ() {
    local input="$1"
    local output="$2"
    if [[ "$input" == "$output" ]]; then
        echo "ERROR: input path equals output path; refusing to overwrite original file." >&2
        echo "INPUT : $input" >&2
        echo "OUTPUT: $output" >&2
        exit 1
    fi
}

for n in $(seq 1 16); do
    SES="ses-$(printf '%03d' "$n")"
    DATADIR="${BASEDIR}/${SUBJ}/${SES}"
    [[ -d "$DATADIR" ]] || continue

    while IFS= read -r INPUT; do
        OUTPUT="$(output_path_for "$INPUT")"
        assert_paths_differ "$INPUT" "$OUTPUT"

        IN_NAME="$(basename "$INPUT")"
        OUT_NAME="$(basename "$OUTPUT")"

        if [[ -e "$OUTPUT" ]]; then
            echo "[${SUBJ}/${SES}] SKIP exists: ${IN_NAME} -> ${OUT_NAME}"
            continue
        fi

        wb_command -cifti-smoothing \
            "$INPUT" "$SIGMA_SURF" "$SIGMA_VOL" "$DIRECTION" "$OUTPUT" \
            -left-surface "$LEFT_SURF" \
            -right-surface "$RIGHT_SURF"

        echo "[${SUBJ}/${SES}] ${IN_NAME} -> ${OUT_NAME}"
    done < <(compgen -G "${DATADIR}/*_space-Glasser64k_bold.dtseries.nii" || true)
done

#!/usr/bin/env bash
set -euo pipefail

module load python/3.11.5 scipy-stack/2025a connectomeworkbench/2.0.1

BASEDIR="/project/def-pbellec/xuan/fmri_dataset_project/data/glasser_resampled"
FIGDIR="/project/def-pbellec/xuan/fmri_dataset_project/data/glasser_resampled/smoothing_verification"
LEFT_SURF="/project/def-pbellec/xuan/fmri_dataset_project/data/fsLR_32k_midthickness/tpl-fsLR_hemi-L_den-32k_midthickness.surf.gii"
RIGHT_SURF="/project/def-pbellec/xuan/fmri_dataset_project/data/fsLR_32k_midthickness/tpl-fsLR_hemi-R_den-32k_midthickness.surf.gii"
VERIFY_SCRIPT="/project/def-pbellec/xuan/fmri_dataset_project/scripts/verify_smoothing.py"

SUBJECTS=(sub-01 sub-02 sub-03 sub-05 sub-06)
SESSIONS=()
for n in $(seq 1 16); do
    SESSIONS+=("ses-$(printf '%03d' "$n")")
done

SIGMA_SURF=1.6986
SIGMA_VOL=0
DIRECTION=COLUMN

usage() {
    echo "Usage: $0 part1|part2"
}

output_path_for() {
    local input="$1"
    if [[ "$input" != *_space-Glasser64k_bold.dtseries.nii ]]; then
        echo "ERROR: Unexpected input filename pattern: $input" >&2
        exit 1
    fi
    local output="${input%_space-Glasser64k_bold.dtseries.nii}_space-Glasser64k_bold_sm4.dtseries.nii"
    echo "$output"
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

find_first_input() {
    local all_files=()
    local subj
    local ses
    for subj in "${SUBJECTS[@]}"; do
        for ses in "${SESSIONS[@]}"; do
            local datadir="${BASEDIR}/${subj}/${ses}"
            if [[ -d "$datadir" ]]; then
                while IFS= read -r f; do
                    all_files+=("$f")
                done < <(compgen -G "${datadir}/*_space-Glasser64k_bold.dtseries.nii" || true)
            fi
        done
    done

    if [[ ${#all_files[@]} -eq 0 ]]; then
        echo "ERROR: No input files found." >&2
        exit 1
    fi

    local sorted_files=()
    mapfile -t sorted_files < <(printf '%s\n' "${all_files[@]}" | sort)
    echo "${sorted_files[0]}"
}

run_smoothing() {
    local input="$1"
    local output="$2"
    wb_command -cifti-smoothing \
        "$input" "$SIGMA_SURF" "$SIGMA_VOL" "$DIRECTION" "$output" \
        -left-surface "$LEFT_SURF" \
        -right-surface "$RIGHT_SURF"
}

run_part1() {
    local first_input
    first_input="$(find_first_input)"
    local first_output
    first_output="$(output_path_for "$first_input")"

    assert_paths_differ "$first_input" "$first_output"

    echo "[Part 1] First input: $first_input"
    echo "[Part 1] Output    : $first_output"

    run_smoothing "$first_input" "$first_output"

    python "$VERIFY_SCRIPT" \
        --original "$first_input" \
        --smoothed "$first_output" \
        --figdir "$FIGDIR"

    echo
    echo "Part 1 completed and verification passed."
    echo "Pause here. Await user confirmation before running part2."
}

run_part2() {
    local subj
    local ses
    local input
    local output

    for subj in "${SUBJECTS[@]}"; do
        for ses in "${SESSIONS[@]}"; do
            local datadir="${BASEDIR}/${subj}/${ses}"
            [[ -d "$datadir" ]] || continue

            while IFS= read -r input; do
                output="$(output_path_for "$input")"
                assert_paths_differ "$input" "$output"

                local in_name
                local out_name
                in_name="$(basename "$input")"
                out_name="$(basename "$output")"

                if [[ -e "$output" ]]; then
                    echo "[$subj/$ses] SKIP exists: $in_name -> $out_name"
                    continue
                fi

                run_smoothing "$input" "$output"
                echo "[$subj/$ses] $in_name -> $out_name"
            done < <(compgen -G "${datadir}/*_space-Glasser64k_bold.dtseries.nii" || true)
        done
    done
}

if [[ $# -ne 1 ]]; then
    usage
    exit 1
fi

case "$1" in
    part1)
        run_part1
        ;;
    part2)
        run_part2
        ;;
    *)
        usage
        exit 1
        ;;
esac

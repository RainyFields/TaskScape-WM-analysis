#!/bin/bash
# Usage: bash smooth_bold.sh <input.dtseries.nii> <output.dtseries.nii> \
#                            <L.midthickness.surf.gii> <R.midthickness.surf.gii> \
#                            [fwhm_mm=4.0]
INPUT=$1
OUTPUT=$2
SURF_L=$3
SURF_R=$4
FWHM=${5:-4.0}
wb_command -cifti-smoothing "$INPUT" "$FWHM" "$FWHM" COLUMN "$OUTPUT" \
  -left-surface "$SURF_L" \
  -right-surface "$SURF_R"
echo "Done: $OUTPUT"

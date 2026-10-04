#!/bin/bash
# Surface-based smoothing of a CIFTI dense timeseries with Connectome Workbench.
#
# Usage: bash smooth_bold.sh <input.dtseries.nii> <output.dtseries.nii> \
#                            <L.midthickness.surf.gii> <R.midthickness.surf.gii> \
#                            [fwhm_mm=4.0]
#
# Note: wb_command -cifti-smoothing expects the kernel as a Gaussian *sigma* in mm,
# not FWHM. This script takes a true FWHM in mm and converts it to sigma via
# sigma = FWHM / (2 * sqrt(2 * ln 2)) = FWHM / 2.354820045. Volume smoothing is
# disabled (surface-only). Default 4.0 mm FWHM -> sigma 1.6986.
set -euo pipefail

INPUT=$1
OUTPUT=$2
SURF_L=$3
SURF_R=$4
FWHM=${5:-4.0}

SIGMA=$(awk -v f="$FWHM" 'BEGIN { printf "%.4f", f / 2.354820045 }')

wb_command -cifti-smoothing "$INPUT" "$SIGMA" 0 COLUMN "$OUTPUT" \
  -left-surface "$SURF_L" \
  -right-surface "$SURF_R"
echo "Done (FWHM ${FWHM} mm -> sigma ${SIGMA} mm): $OUTPUT"

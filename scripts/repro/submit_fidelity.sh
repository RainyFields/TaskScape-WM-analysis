#!/bin/bash
# Migration fidelity runs (Phase 5): re-run the GLM subset, the Fig. 8 clean chain and the clean noise ceiling on
# TaskScape-WM inputs from immutable code snapshots, then compare with the pre-migration outputs.
#
#   bash scripts/repro/submit_fidelity.sh [RUN_DIR]
#
# RUN_DIR default: /project/def-pbellec/xuan/TaskScape-WM-analysis/outputs/fidelity_<date>
# Code: `git archive` of branches taskscape-migration and taskscape-migration-nc -> RUN_DIR/code/<branch>
# References (read-only): `git archive` of tags pre-migration-2026-10 / -ncregen -> RUN_DIR/reference/<tag>;
#   GLM H5 and clean delay betas from the dataset (hardlinked pre-migration products).
# Every Python step installs config.paths.install_legacy_guard -> RUN_DIR/out/audit/*.opened.tsv.
set -euo pipefail
REPO=/project/def-pbellec/xuan/TaskScape-WM-analysis
RUN=${1:-$REPO/outputs/fidelity_$(date +%F)}
ACC=def-pbellec
mkdir -p "$RUN"/{code,reference,logs,out,compare}
cd "$REPO"
for br in taskscape-migration taskscape-migration-nc; do
  d=$RUN/code/$br; [ -e "$d" ] && { echo "refusing: $d exists (immutable snapshot)"; exit 1; }
  mkdir -p "$d"; git archive "$br" | tar -x -C "$d"; git rev-parse "$br" > "$d/COMMIT"
done
for t in pre-migration-2026-10 pre-migration-2026-10-ncregen; do
  d=$RUN/reference/$t; mkdir -p "$d"; git archive "$t" results figures | tar -x -C "$d"; git rev-parse "$t^{commit}" > "$d/COMMIT"
done
C=$RUN/code/taskscape-migration; CN=$RUN/code/taskscape-migration-nc; REF=$RUN/reference/pre-migration-2026-10
REFN=$RUN/reference/pre-migration-2026-10-ncregen; OUT=$RUN/out; CMP=$RUN/compare
PRE='module load StdEnv/2023 python/3.11.5 scipy-stack/2025a; source /project/def-pbellec/xuan/nilearn/bin/activate; export PYTHONDONTWRITEBYTECODE=1 TASKSCAPE_OUTPUT_ROOT='"$OUT"
sb() { # name time mem deps cmd...
  local name=$1 t=$2 m=$3 dep=$4; shift 4
  local d=""; [ -n "$dep" ] && d="--dependency=afterok:$dep"
  sbatch --parsable -A $ACC -J "fid_$name" -t "$t" -c 4 --mem="$m" $d -o "$RUN/logs/%x_%j.log" -e "$RUN/logs/%x_%j.err" \
    --wrap "$PRE; set -euo pipefail; $*"
}
SUBJ="sub-01 sub-02 sub-03 sub-05 sub-06"

# 1. GLM subset: every task family in 3 sessions (2 subjects + the sub-05/ses-012 exception session;
#    sub-01/ses-004 also has a __dup 1back acquisition). Flags = legacy submit_glm_array.sh (OLS).
J_GLM=""
for ss in "sub-01 ses-004" "sub-03 ses-002" "sub-05 ses-012"; do
  set -- $ss
  j=$(sb "glm_$1_$2" 03:00:00 32G "" python $C/scripts/glm_analysis.py --subj $1 --sessions $2 --tasks 1back ctxdm dms interdms --estimator ols --overwrite)
  J_GLM="$J_GLM:$j"
done
sb glm_compare 01:00:00 16G "${J_GLM#:}" python $C/scripts/repro/compare_fidelity.py glm --new $OUT/encoding_delay/trial_level_betas --out $CMP >/dev/null

# 2. Fig. 8 chain: run labels -> clean delay betas -> validation -> halves -> similarity -> figures
J_LAB=$(sb labels 01:00:00 16G "" python $C/scripts/verify_run_labels_from_blockfiles.py --out_dir $OUT/labels)
sb labels_compare 00:20:00 4G "$J_LAB" python $C/scripts/repro/compare_fidelity.py tables --new $OUT/labels --ref $REF/results/similarity_analysis/regenerated_clean --glob run_labels_verified.csv lookup_mismatch_report.csv --out $CMP --name labels >/dev/null
J_CLEAN=""
for s in $SUBJ; do
  j=$(sb "clean_$s" 04:00:00 48G "$J_LAB" python $C/scripts/build_clean_delay_betas.py --subj $s --labels $OUT/labels/run_labels_verified.csv --out_dir $OUT/clean_delay_betas)
  J_CLEAN="$J_CLEAN:$j"
done
J_CLEAN=${J_CLEAN#:}
sb clean_compare 01:00:00 32G "$J_CLEAN" python $C/scripts/repro/compare_fidelity.py clean --new $OUT/clean_delay_betas --out $CMP >/dev/null
J_VAL=$(sb clean_validate 01:00:00 32G "$J_CLEAN" python $C/scripts/validate_clean_delay_betas.py --data_dir $OUT/clean_delay_betas --out_dir $OUT/similarity_analysis/regenerated_clean)
J_HALF=""
for s in $SUBJ; do
  j=$(sb "halves_$s" 04:00:00 64G "$J_CLEAN" python $C/scripts/similarity_analysis_clean.py --stage halves --subj $s --clean_dir $OUT/clean_delay_betas --labels $OUT/labels/run_labels_verified.csv)
  J_HALF="$J_HALF:$j"
done
J_SIM=$(sb similarity 06:00:00 96G "${J_HALF#:}:$J_VAL" python $C/scripts/similarity_analysis_clean.py --stage similarity --clean_dir $OUT/clean_delay_betas --labels $OUT/labels/run_labels_verified.csv)
J_FIG=$(sb fig8 01:00:00 16G "$J_SIM" python $C/scripts/figure_making/plot_fig8_regenerated_clean.py)
sb sim_compare 01:00:00 16G "$J_FIG" "python $C/scripts/repro/compare_fidelity.py tables --new $OUT/similarity_analysis/regenerated_clean --ref $REF/results/similarity_analysis/regenerated_clean --out $CMP --name similarity_tables --exclude .md run_labels_verified.csv lookup_mismatch_report.csv --drop_variant legacy_pickle && python $C/scripts/repro/compare_fidelity.py tables --new $OUT/figures/fig8_regenerated_clean --ref $REF/figures/fig8_regenerated_clean --glob '*_source_data.csv' --out $CMP --name fig8_source_data --drop_variant legacy_pickle" >/dev/null

# 3. Clean noise ceiling, primary variant, from the dataset's derivatives/glm-clean-delay
J_NC=""
for s in $SUBJ; do
  j=$(sb "nc_$s" 02:00:00 32G "" bash $CN/scripts/noise_ceiling_clean/job_clean_primary_taskscape.sh $s $OUT)
  J_NC="$J_NC:$j"
done
sb nc_compare 00:30:00 8G "${J_NC#:}" python $C/scripts/repro/compare_fidelity.py tables --new $OUT/noise_ceiling/regenerated_clean/estimates/primary --ref $REFN/results/noise_ceiling/regenerated_clean/estimates/primary --out $CMP --name nc_primary >/dev/null
echo "submitted; run dir $RUN"

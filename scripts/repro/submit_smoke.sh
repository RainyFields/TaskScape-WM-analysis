#!/bin/bash
# Smoke-test a migrated script on a compute node, under strace (every open/openat logged) with the config.paths
# legacy guard active, outputs under RUN/out. One job per call; the fallback audit (scripts/repro/fallback_audit.py)
# can then be run on RUN to prove zero opens of the frozen legacy project.
#
#   [SMOKE_DEP=<jobid[:jobid...]>] bash scripts/repro/submit_smoke.sh RUN_DIR NAME TIME MEM -- <command...>
# e.g. bash scripts/repro/submit_smoke.sh outputs/smoke_2026-10-07 accuracy_level 00:30:00 8G -- python scripts/accuracy_level.py --subj sub-01
set -euo pipefail
REPO=/project/def-pbellec/xuan/TaskScape-WM-analysis
RUN=$1; NAME=$2; T=$3; M=$4; shift 4; [ "$1" == "--" ] && shift
case "$RUN" in /*) ;; *) RUN=$REPO/$RUN ;; esac
mkdir -p "$RUN"/{logs,out,jobs,strace}
printf '#!/bin/bash\nset -euo pipefail\ncd %s\n%s\n' "$REPO" "$*" > "$RUN/jobs/$NAME.sh"
PRE='module load StdEnv/2023 python/3.11.5 scipy-stack/2025a; source /project/def-pbellec/xuan/nilearn/bin/activate; export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg TASKSCAPE_OUTPUT_ROOT='"$RUN/out"
DEP=""; [ -n "${SMOKE_DEP:-}" ] && DEP="--dependency=afterok:$SMOKE_DEP"
id=$(sbatch --parsable -A def-pbellec -J "smoke_$NAME" -t "$T" -c 4 --mem="$M" $DEP -o "$RUN/logs/%x_%j.log" -e "$RUN/logs/%x_%j.err" \
  --wrap "$PRE; set -euo pipefail; strace -f -qq --seccomp-bpf -e trace=open,openat,openat2 -o $RUN/strace/$NAME.trace bash $RUN/jobs/$NAME.sh")
echo "$id" >> "$RUN/jobs/ids"; echo "$NAME $id"

#!/bin/bash
# Submit a batch job through run_on_compute.sh -b, retrying while the SLURM controller is unreachable.
# Usage: submit_retry.sh <run_on_compute options> -- <command>
# Prints the run_on_compute output; appends "<jobid> <name>" to $AGENT_JOB_LOGDIR/submitted_jobs.txt
set -uo pipefail
S=~/.cursor/skills/alliance-compute-jobs/scripts/run_on_compute.sh
LOGDIR=${AGENT_JOB_LOGDIR:-$PWD/logs}
mkdir -p "$LOGDIR"
for attempt in $(seq 1 30); do
  out=$($S -b "$@" 2>&1)
  if echo "$out" | grep -q "Unable to contact slurm controller\|Socket timed out\|slurm_load\|Connection refused"; then
    echo "[submit_retry] attempt $attempt: controller unreachable; retrying in 30 s" >&2
    sleep 30
    continue
  fi
  echo "$out"
  jid=$(echo "$out" | grep -oE "Submitted batch job [0-9]+" | grep -oE "[0-9]+")
  name=$(echo "$out" | grep -oE "job-name=[^ ]+" | head -1 | cut -d= -f2)
  if [[ -n "$jid" ]]; then
    echo "$jid $name $(date -Is) $*" >> "$LOGDIR/submitted_jobs.txt"
    exit 0
  fi
  echo "[submit_retry] submission failed (no job id):" >&2
  echo "$out" >&2
  exit 1
done
echo "[submit_retry] giving up after 30 attempts" >&2
exit 1

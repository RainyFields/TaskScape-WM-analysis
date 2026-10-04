#!/bin/bash
# Back up fmri_dataset_project/data/ to Dropbox via rclone.
#
# One-time setup (run on a login node):
#   rclone config
#     n) New remote
#     name: dropbox
#     storage: dropbox
#     client_id / client_secret: leave blank
#     follow the OAuth URL in your browser, then paste the token
#
# Dry run:
#   bash scripts/backup_data_to_dropbox.sh --dry-run
#
# Start backup (recommended in tmux/screen; ~87 GB, may take many hours):
#   nohup bash scripts/backup_data_to_dropbox.sh > logs/dropbox_backup.nohup.out 2>&1 &

set -euo pipefail

REMOTE="${RCLONE_REMOTE:-dropbox}"
REMOTE_PATH="${RCLONE_REMOTE_PATH:-fmri_dataset_project_backup/data}"
SOURCE="/project/def-pbellec/xuan/fmri_dataset_project/data"
LOG_DIR="/project/def-pbellec/xuan/fmri_dataset_project/logs"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/dropbox_backup_$(date +%Y%m%d_%H%M%S).log"

if ! rclone listremotes | grep -qx "${REMOTE}:"; then
    echo "Error: rclone remote '${REMOTE}:' is not configured." >&2
    echo "Run: rclone config   (create a remote named '${REMOTE}' with type dropbox)" >&2
    exit 1
fi

if [[ ! -d "$SOURCE" ]]; then
    echo "Error: source directory not found: $SOURCE" >&2
    exit 1
fi

RCLONE_ARGS=(
    copy
    "$SOURCE"
    "${REMOTE}:${REMOTE_PATH}"
    --progress
    --stats 30s
    --transfers 4
    --checkers 8
    --copy-links
    --log-file "$LOG_FILE"
    --log-level INFO
    --retries 10
    --low-level-retries 20
    --dropbox-batch-mode sync
)

if [[ "$DRY_RUN" -eq 1 ]]; then
    RCLONE_ARGS+=(--dry-run)
    echo "Dry run: listing what would be copied to ${REMOTE}:${REMOTE_PATH}"
fi

echo "Source:  $SOURCE"
echo "Dest:    ${REMOTE}:${REMOTE_PATH}"
echo "Log:     $LOG_FILE"
echo "Started: $(date)"

rclone "${RCLONE_ARGS[@]}"

echo "Finished: $(date)"
echo "Log saved to $LOG_FILE"

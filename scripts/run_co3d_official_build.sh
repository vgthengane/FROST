#!/bin/bash
# Build CO3D 1024 FPS + normal caches from official data/co3d/ pointcloud.ply files.
#
# Usage:
#   bash scripts/run_co3d_official_build.sh              # foreground
#   bash scripts/run_co3d_official_build.sh --screen     # background (screen co3d_build)
#
# Log: _experiments/logs/co3d_official_build.log
# Output: _datasets/CO3D/co3d_{train,test}_1024pts_fps_normal.dat

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${REPO}/_experiments/logs/co3d_official_build.log"
SCREEN_NAME="${SCREEN_NAME:-co3d_build}"

log() { echo "[$(date -Is)] $*" | tee -a "${LOG}"; }

run_build() {
  mkdir -p "$(dirname "${LOG}")" "${REPO}/_datasets/CO3D"
  log "CO3D official build start"
  log "Official root: ${REPO}/data/co3d"
  log "Output root:   ${REPO}/_datasets/CO3D"
  # --allow-partial: single-sequence subset lacks most train ply files.
  # Remove when full CO3D is downloaded.
  aprun --image climb3d_pp.sif python "${REPO}/scripts/build_co3d_official_1024_normals.py" \
    --official-root "${REPO}/data/co3d" \
    --output-root "${REPO}/_datasets/CO3D" \
    --export-pt \
    --allow-partial \
    --force \
    2>&1 | tee -a "${LOG}"
  log "CO3D official build finished"
}

if [[ "${1:-}" == "--screen" ]]; then
  if screen -list | grep -q "[.]${SCREEN_NAME}[[:space:]]"; then
    echo "Screen session '${SCREEN_NAME}' already running."
    echo "  screen -r ${SCREEN_NAME}"
    echo "  tail -f ${LOG}"
    exit 0
  fi
  screen -dmS "${SCREEN_NAME}" bash -lc \
    "cd \"${REPO}\" && \"${REPO}/scripts/run_co3d_official_build.sh\""
  echo "Started screen session: ${SCREEN_NAME}"
  echo "  tail -f ${LOG}"
  echo "  screen -r ${SCREEN_NAME}"
  exit 0
fi

run_build

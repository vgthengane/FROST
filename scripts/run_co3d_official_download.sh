#!/bin/bash
# Official CO3D single-sequence subset download (Open-Rig3R / co3d repo steps).
# Runs inside screen via: bash scripts/run_co3d_official_download.sh --screen
#
# Prereqs (done once):
#   git clone https://github.com/facebookresearch/co3d.git ../co3d
#   aprun python -m pip install --user -e ../co3d
#
# Data lands in: data/co3d/  (~8.9 GB single-sequence subset)
# Log: _experiments/logs/co3d_official_download.log

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CO3D_REPO="${CO3D_REPO:-$(dirname "${REPO}")/co3d}"
DOWNLOAD_FOLDER="${REPO}/data/co3d"
LOG="${REPO}/_experiments/logs/co3d_official_download.log"
SCREEN_NAME="${SCREEN_NAME:-co3d_dl}"

log() { echo "[$(date -Is)] $*" | tee -a "${LOG}"; }

run_download() {
  mkdir -p "${DOWNLOAD_FOLDER}" "$(dirname "${LOG}")"
  if [[ ! -d "${CO3D_REPO}/co3d" ]]; then
    echo "CO3D repo not found at ${CO3D_REPO}. Clone with:" >&2
    echo "  git clone https://github.com/facebookresearch/co3d.git ${CO3D_REPO}" >&2
    exit 1
  fi
  log "CO3D download start — subset=single_sequence (~8.9 GB)"
  log "Repo: ${CO3D_REPO}"
  log "Target: ${DOWNLOAD_FOLDER}"
  # climb3d_pp.sif has requests/tqdm/h5py; PYTHONPATH replaces pip -e (aprun sets PYTHONNOUSERSITE=1).
  aprun --image climb3d_pp.sif \
    --env "PYTHONPATH=${CO3D_REPO}/co3d" \
    python "${CO3D_REPO}/co3d/download_dataset.py" \
    --download_folder "${DOWNLOAD_FOLDER}" \
    --single_sequence_subset \
    --n_extract_workers 1 \
    2>&1 | tee -a "${LOG}"
  log "CO3D download finished"
}

if [[ "${1:-}" == "--screen" ]]; then
  if screen -list | grep -q "[.]${SCREEN_NAME}[[:space:]]"; then
    echo "Screen session '${SCREEN_NAME}' already running."
    echo "  screen -r ${SCREEN_NAME}"
    echo "  tail -f ${LOG}"
    exit 0
  fi
  screen -dmS "${SCREEN_NAME}" bash -lc \
    "cd \"${REPO}\" && \"${REPO}/scripts/run_co3d_official_download.sh\""
  echo "Started screen session: ${SCREEN_NAME}"
  echo "  tail -f ${LOG}"
  echo "  screen -r ${SCREEN_NAME}"
  exit 0
fi

run_download

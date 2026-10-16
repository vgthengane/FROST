#!/bin/bash
# Download CO3D full categories one-by-one, keep only FSCIL-index pointcloud.ply, delete zips.
#
# Usage:
#   bash datasets/preprocess/download_co3d_index_minimal.sh              # foreground
#   bash datasets/preprocess/download_co3d_index_minimal.sh --screen   # detached screen
#
# Log: _experiments/logs/co3d_index_minimal_download.log

set -euo pipefail
# Plain python by default; lab: export FROST_PYTHON="aprun python"
run_py() { # shellcheck disable=SC2086
  ${FROST_PYTHON:-python} "$@";
}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CO3D_REPO="${CO3D_REPO:-$REPO/third_party/co3d}"
OUT="${OUT:-$REPO/_datasets/co3d_minimal}"
INDEX="${INDEX:-$REPO/datasets/session_pclouds/co3d}"
LOG="${REPO}/_experiments/logs/co3d_index_minimal_download.log"
SCREEN_NAME="${SCREEN_NAME:-co3d_min_dl}"

log() { echo "[$(date -Is)] $*" | tee -a "${LOG}"; }

CATEGORIES=(
  apple backpack ball banana baseballbat baseballglove bench bicycle book bottle
  bowl broccoli cake car carrot cellphone chair couch cup donut frisbee hairdryer
  handbag hotdog hydrant keyboard kite laptop microwave motorcycle mouse orange
  parkingmeter pizza plant remote sandwich skateboard stopsign suitcase teddybear
  toaster toilet toybus toyplane toytruck tv umbrella vase wineglass
)

prune_category() {
  local CAT="$1"
  local CAT_DIR="$OUT/$CAT"
  local TMP="$OUT/_keep_tmp/$CAT"

  mkdir -p "$TMP"
  run_py - "$CAT" "$CAT_DIR" "$TMP" "$INDEX" <<'PY'
import shutil, sys
from pathlib import Path

cat, cat_dir, tmp, index_root = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
needed = set()
for split in ("train", "test"):
    for f in (index_root / split).glob("session_*.txt"):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            rel = line.split()[0].replace("v10-co3d/", "")
            parts = rel.split("/")
            if len(parts) == 3 and parts[0] == cat:
                needed.add(parts[2].removesuffix(".pt"))

kept = []
missing = []
for seq in sorted(needed):
    src = cat_dir / seq / "pointcloud.ply"
    if src.is_file():
        dst_dir = tmp / seq
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / "pointcloud.ply")
        kept.append(seq)
    else:
        missing.append(seq)

print(f"{cat}: needed={len(needed)} kept={len(kept)} missing={len(missing)}")
if missing[:5]:
    print(f"  missing examples: {missing[:5]}")
PY

  rm -rf "$CAT_DIR"
  mkdir -p "$CAT_DIR"
  if [ -d "$TMP" ] && [ "$(ls -A "$TMP" 2>/dev/null)" ]; then
    mv "$TMP"/* "$CAT_DIR"/
  fi
  rm -rf "$OUT/_keep_tmp"

  # delete zips for this category
  rm -f "$OUT/${CAT}"_*.zip
}
download_category() {
  local CAT="$1"
  log "Download start: $CAT"
  run_py "$CO3D_REPO/co3d/download_dataset.py" \
    --download_folder "$OUT" \
    --download_categories "$CAT" \
    --n_download_workers 2 \
    --n_extract_workers 1 \
    --clear_archives_after_unpacking \
    2>&1 | tee -a "${LOG}"

  log "Prune start: $CAT"
  prune_category "$CAT"
  log "Done: $CAT"
}

run_all() {
  mkdir -p "$OUT" "$(dirname "$LOG")"
  log "CO3D index-minimal download start -> $OUT"
  for CAT in "${CATEGORIES[@]}"; do
    download_category "$CAT"
  done
  log "All categories finished"
}

if [[ "${1:-}" == "--screen" ]]; then
  if screen -list | grep -q "[.]${SCREEN_NAME}[[:space:]]"; then
    echo "Screen '${SCREEN_NAME}' already running."
    echo "  screen -r ${SCREEN_NAME}"
    echo "  tail -f ${LOG}"
    exit 0
  fi
  screen -dmS "${SCREEN_NAME}" bash -lc \
    "cd \"$REPO\" && bash \"$REPO/datasets/preprocess/download_co3d_index_minimal.sh\""
  echo "Started screen: ${SCREEN_NAME}"
  echo "  tail -f ${LOG}"
  echo "  screen -r ${SCREEN_NAME}"
  exit 0
fi

run_all

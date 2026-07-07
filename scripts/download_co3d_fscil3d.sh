#!/bin/bash
# Download FSCIL-3D / microshape preprocessed CO3D point clouds and build
# 1024-point FPS + normal caches for this repo.
#
# Source: microshape ECCV'22 / FSCIL-3D Mega.nz mirror (1024 FPS .pt per object).
#   https://mega.nz/folder/fBliUDBL#e-whIrJOCSjfXff7_S5iAQ
#
# Usage:
#   bash scripts/download_co3d_fscil3d.sh
#   bash scripts/download_co3d_fscil3d.sh --skip-download   # build only
#   bash scripts/download_co3d_fscil3d.sh --from-zip /path/to/dataset.zip
#   bash scripts/download_co3d_fscil3d.sh --data-root /custom/CO3D
#
# Output layout:
#   _datasets/CO3D/<category>/{train,test}/*.pt
#   _datasets/CO3D/co3d_train_1024pts_fps_normal.dat
#   _datasets/CO3D/co3d_test_1024pts_fps_normal.dat
#   _datasets/CO3D/co3d_class_names.txt

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATA_ROOT="_datasets/CO3D"
DOWNLOAD_DIR="_datasets/_downloads"
MEGA_FOLDER="https://mega.nz/folder/fBliUDBL#e-whIrJOCSjfXff7_S5iAQ"
MEGA_DL_URL="https://mega.nz/#F!fBliUDBL!e-whIrJOCSjfXff7_S5iAQ"
SKIP_DOWNLOAD=0
FROM_ZIP=""
FORCE_BUILD=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --skip-download)
      SKIP_DOWNLOAD=1
      shift
      ;;
    --from-zip)
      FROM_ZIP="$2"
      shift 2
      ;;
    --force)
      FORCE_BUILD=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--data-root PATH] [--skip-download] [--from-zip ZIP] [--force]" >&2
      exit 1
      ;;
  esac
done

log() { echo "[$(date -Is)] $*"; }

has_co3d_tree() {
  [[ -d "${DATA_ROOT}/apple/train" ]] || [[ -d "${DATA_ROOT}/co3d/apple/train" ]]
}

extract_co3d_from_tree() {
  local src="$1"
  if [[ -d "${src}/apple/train" ]]; then
    log "CO3D tree already at ${src}"
    if [[ "$(realpath "${src}")" != "$(realpath "${DATA_ROOT}")" ]]; then
      mkdir -p "${DATA_ROOT}"
      log "Syncing ${src} -> ${DATA_ROOT}"
      rsync -a "${src}/" "${DATA_ROOT}/"
    fi
    return 0
  fi
  if [[ -d "${src}/co3d/apple/train" ]]; then
    mkdir -p "${DATA_ROOT}"
    log "Copying ${src}/co3d -> ${DATA_ROOT}"
    rsync -a "${src}/co3d/" "${DATA_ROOT}/"
    return 0
  fi
  if [[ -d "${src}/dataset/co3d/apple/train" ]]; then
    mkdir -p "${DATA_ROOT}"
    log "Copying ${src}/dataset/co3d -> ${DATA_ROOT}"
    rsync -a "${src}/dataset/co3d/" "${DATA_ROOT}/"
    return 0
  fi
  return 1
}

extract_zip() {
  local zip_path="$1"
  if [[ ! -f "${zip_path}" ]]; then
    echo "Zip not found: ${zip_path}" >&2
    exit 1
  fi
  local tmp="${DOWNLOAD_DIR}/co3d_extract_$$"
  mkdir -p "${tmp}"
  log "Extracting ${zip_path} -> ${tmp}"
  unzip -q "${zip_path}" -d "${tmp}"
  if ! extract_co3d_from_tree "${tmp}"; then
    # Search one level down (dataset.zip layouts vary).
    local found
    found="$(find "${tmp}" -type d -path '*/apple/train' 2>/dev/null | head -1 || true)"
    if [[ -n "${found}" ]]; then
      local parent
      parent="$(dirname "$(dirname "${found}")")"
      extract_co3d_from_tree "${parent}"
    else
      echo "Could not find CO3D apple/train inside ${zip_path}" >&2
      echo "Extract manually and pass --skip-download once ${DATA_ROOT}/apple/train exists." >&2
      exit 1
    fi
  fi
  rm -rf "${tmp}"
}

download_mega() {
  mkdir -p "${DOWNLOAD_DIR}"
  if command -v megadl >/dev/null 2>&1; then
    log "Downloading FSCIL-3D dataset archive via megadl (may take a while)..."
    megadl --no-progress --path "${DOWNLOAD_DIR}" "${MEGA_DL_URL}" || true
    local zip
    zip="$(find "${DOWNLOAD_DIR}" -maxdepth 2 -iname 'dataset.zip' -o -iname '*co3d*.zip' 2>/dev/null | head -1 || true)"
    if [[ -n "${zip}" ]]; then
      extract_zip "${zip}"
      return 0
    fi
    if extract_co3d_from_tree "${DOWNLOAD_DIR}"; then
      return 0
    fi
  fi

  if command -v rclone >/dev/null 2>&1; then
    log "Trying rclone copyto from Mega folder..."
    if rclone copy "${MEGA_FOLDER}" "${DOWNLOAD_DIR}" --mega-show-all; then
      local zip
      zip="$(find "${DOWNLOAD_DIR}" -maxdepth 2 -iname 'dataset.zip' 2>/dev/null | head -1 || true)"
      if [[ -n "${zip}" ]]; then
        extract_zip "${zip}"
        return 0
      fi
    fi
  fi

  cat >&2 <<EOF
Could not auto-download CO3D (megadl/rclone not available or Mega fetch failed).

Manual steps:
  1. Open: ${MEGA_FOLDER}
  2. Download dataset.zip (FSCIL-3D / microshape mirror)
  3. Run:
       bash scripts/download_co3d_fscil3d.sh --from-zip /path/to/dataset.zip

Or install megatools and retry:
  sudo apt-get install -y megatools   # Debian/Ubuntu
  bash scripts/download_co3d_fscil3d.sh
EOF
  return 1
}

mkdir -p "${DATA_ROOT}" "${DOWNLOAD_DIR}"

if [[ -n "${FROM_ZIP}" ]]; then
  extract_zip "${FROM_ZIP}"
elif [[ "${SKIP_DOWNLOAD}" -eq 0 ]] && ! has_co3d_tree; then
  download_mega
elif has_co3d_tree; then
  log "CO3D raw tree present under ${DATA_ROOT}"
else
  echo "No CO3D data at ${DATA_ROOT}. Download manually or omit --skip-download." >&2
  exit 1
fi

if ! has_co3d_tree; then
  echo "CO3D layout not found after download. Expected ${DATA_ROOT}/apple/train/*.pt" >&2
  exit 1
fi

BUILD_ARGS=(--data-root "${DATA_ROOT}")
if [[ "${FORCE_BUILD}" -eq 1 ]]; then
  BUILD_ARGS+=(--force)
fi

log "Building 1024 FPS + normal caches..."
aprun python scripts/build_co3d_1024_normals.py "${BUILD_ARGS[@]}"

log "CO3D setup complete — ${DATA_ROOT}"

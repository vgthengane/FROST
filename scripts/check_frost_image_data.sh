#!/usr/bin/env bash
# Preflight for FROST image FSCIL: offline ViT weights + LGSP datasets.
#
# Usage:
#   bash scripts/check_frost_image_data.sh
#   bash scripts/check_frost_image_data.sh --data-path _datasets/images --protocol cub200
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_PATH="${REPO}/_datasets/images"
PROTOCOL="all"
VIT_DIR="${REPO}/_experiments/pretrained_weights/ViT"
VIT_CKPT="${VIT_DIR}/vit_base_patch16_224.augreg_in21k.safetensors"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-path) DATA_PATH="${2:?}"; shift 2 ;;
    --protocol) PROTOCOL="${2:?}"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

# Resolve relative data path against repo root.
if [[ "${DATA_PATH}" != /* ]]; then
  DATA_PATH="${REPO}/${DATA_PATH}"
fi

ok=1
need() {
  local path="$1"
  local hint="$2"
  if [[ -e "${path}" ]]; then
    echo "OK   ${path}"
  else
    echo "MISS ${path}"
    echo "     -> ${hint}"
    ok=0
  fi
}

echo "=== ViT offline weights ==="
need "${VIT_CKPT}" "bash datasets/preprocess/download_vit_in21k.sh && sync folder to Leonardo"

check_cub() {
  need "${DATA_PATH}/CUB200/images" "bash datasets/preprocess/download_cub200.sh"
}
check_aircraft() {
  need "${DATA_PATH}/FGVC-Aircraft/data/images" "bash datasets/preprocess/download_fgvc_aircraft.sh"
}
check_inf200() {
  need "${DATA_PATH}/iNF200/train_mini" "bash datasets/preprocess/download_inf200.sh  (manual LGSP/HF staging)"
  need "${DATA_PATH}/iNF200/val" "bash datasets/preprocess/download_inf200.sh  (manual LGSP/HF staging)"
}

echo "=== Datasets under ${DATA_PATH} (protocol=${PROTOCOL}) ==="
case "${PROTOCOL}" in
  all)
    check_cub; check_aircraft; check_inf200
    ;;
  cub200) check_cub ;;
  aircraft) check_aircraft ;;
  inf200) check_inf200 ;;
  *)
    echo "Unknown protocol: ${PROTOCOL} (expected cub200|aircraft|inf200|all)" >&2
    exit 1
    ;;
esac

if [[ "${ok}" -ne 1 ]]; then
  echo
  echo "Preflight FAILED. Fix missing paths before sbatch."
  exit 1
fi
echo
echo "Preflight OK."

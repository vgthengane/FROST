#!/usr/bin/env bash
# Download microshape FSCIL-3D bicycle class and patch local ShapeNet55.
#
# Microshape hosts the preprocessed ShapeNet bicycle point clouds used by the
# shapenet_scanobjectnn protocol (class id 4). The Point-BERT ShapeNet55 zip
# in RL-CL3D / SelfEvo does NOT include bicycle (synset 02834778).
#
# Usage:
#   bash scripts/download_microshape_bicycle.sh
#   bash scripts/download_microshape_bicycle.sh --rebuild-cache
#
# Manual fallback (if automated mega download fails):
#   1. Open https://mega.nz/folder/fBliUDBL#e-whIrJOCSjfXff7_S5iAQ
#   2. Download dataset.zip and unzip under:
#        third_party/microshape/data/dataset/
#   3. Run:
#        aprun python scripts/patch_shapenet55_bicycle.py \
#          --microshape-root third_party/microshape/data/dataset/v9-shapenet-scanobjectnn/4 \
#          --rebuild-cache
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MICRO_DATASET="${REPO}/third_party/microshape/data/dataset"
BICYCLE_DIR="${MICRO_DATASET}/v9-shapenet-scanobjectnn/4"
MEGA_FOLDER="https://mega.nz/folder/fBliUDBL#e-whIrJOCSjfXff7_S5iAQ"
REBUILD=0

for arg in "$@"; do
  case "$arg" in
    --rebuild-cache) REBUILD=1 ;;
    -h|--help)
      sed -n '1,20p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $arg" >&2
      exit 1
      ;;
  esac
done

if [[ -d "${BICYCLE_DIR}/train" ]] && compgen -G "${BICYCLE_DIR}/train/model_*.pt" >/dev/null; then
  echo "[ok] microshape bicycle folder already present: ${BICYCLE_DIR}"
else
  echo "[info] microshape bicycle not found locally."
  echo "[info] Download microshape dataset from: ${MEGA_FOLDER}"
  echo "[info] Unzip so this path exists: ${BICYCLE_DIR}/train/model_*.pt"

  if command -v megadl >/dev/null 2>&1; then
    echo "[try] megadl folder download ..."
    mkdir -p "${MICRO_DATASET}"
    megadl --path "${MICRO_DATASET}" "${MEGA_FOLDER}" || true
  fi

  if [[ ! -d "${BICYCLE_DIR}/train" ]]; then
    echo "[error] Bicycle .pt files still missing after download attempt." >&2
    echo "        Unzip microshape dataset.zip manually, then re-run this script." >&2
    exit 1
  fi
fi

PATCH_ARGS=(--microshape-root "${BICYCLE_DIR}")
if [[ "${REBUILD}" -eq 1 ]]; then
  PATCH_ARGS+=(--rebuild-cache)
fi

echo "[run] aprun python scripts/patch_shapenet55_bicycle.py ${PATCH_ARGS[*]}"
cd "${REPO}"
aprun python scripts/patch_shapenet55_bicycle.py "${PATCH_ARGS[@]}"

echo "[run] aprun python scripts/smoke_test_cross_scanobjectnn.py"
aprun python scripts/smoke_test_cross_scanobjectnn.py

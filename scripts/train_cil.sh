#!/usr/bin/env bash
# Run 3D continual learning on ModelNet40 via Uni3D + LAMDA-PILOT.
#
# Usage:
#   bash scripts/train_cil.sh                          # full run, mos_modelnet40
#   bash scripts/train_cil.sh simplecil_modelnet40     # full run
#   bash scripts/train_cil.sh mos_modelnet40 --smoke   # short dry-run
#
# Requires ModelNet40 at _datasets/modelnet40_normal_resampled (see README).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="mos_modelnet40"
SMOKE=false

for arg in "$@"; do
  case "${arg}" in
    --smoke) SMOKE=true ;;
    *) CONFIG="${arg}" ;;
  esac
done

CONFIG_PATH="${REPO_ROOT}/configs/${CONFIG}.json"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

SMOKE_ARGS=()
if [[ "${SMOKE}" == true ]]; then
  # 2 tasks × 4 classes/task (full run: 10 tasks × 4 = 40 classes)
  export CIL_SMOKE_PREFIX=smoke_mn40
  export CIL_SMOKE_MAX_CLASSES=8
  export CIL_SMOKE_MAX_TRAIN_PER_CLASS=10
  export CIL_SMOKE_MAX_TEST_PER_CLASS=5
  export CIL_SMOKE_TUNED_EPOCH=2
  export CIL_SMOKE_CRCT_EPOCHS=2
  export CIL_SMOKE_BATCH_SIZE=4
  export CIL_SMOKE_NUM_WORKERS=2
  SMOKE_ARGS=(--smoke)
fi

cd "${REPO_ROOT}"
aprun python train_cil.py --config "${CONFIG_PATH}" "${SMOKE_ARGS[@]}"

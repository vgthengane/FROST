#!/usr/bin/env bash
# Local CUB AdaptFormer LR × epoch probe (task-0 only) + convergence logs.
#
# Goal: see whether higher LR reaches the same base test acc in fewer epochs.
# Does NOT need Leonardo. Prefer GPU; falls back to CPU if CUDA is unavailable.
#
# Usage:
#   bash scripts/run_pride_image_lr_epoch_probe.sh
#   bash scripts/run_pride_image_lr_epoch_probe.sh --fast          # capped samples
#   bash scripts/run_pride_image_lr_epoch_probe.sh --epochs 12
#   bash scripts/run_pride_image_lr_epoch_probe.sh --lrs 0.01,0.03,0.05
#   bash scripts/run_pride_image_lr_epoch_probe.sh --device -1     # force CPU
# Plot:
#   aprun python scripts/plot_pride_image_lr_epoch_probe.py
# =============================================================================

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RUNNER=(aprun python)
LOGROOT="_experiments/00_PRIDE_IMAGE_LR_EPOCH_PROBE"
DATA_PATH="_datasets/images"
VIT_CKPT="_experiments/pretrained_weights/ViT/vit_base_patch16_224.augreg_in21k.safetensors"
export PRIDE_VIT_CKPT="$(pwd)/${VIT_CKPT}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

EPOCHS=20
LRS=(0.01 0.03 0.05)
FAST=0
DEVICE_ARGS=()
SKIP_DONE=0
SEED=2008

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fast) FAST=1; shift ;;
    --epochs) EPOCHS="${2:?}"; shift 2 ;;
    --lrs)
      IFS=',' read -r -a LRS <<< "${2:?}"
      shift 2
      ;;
    --device)
      DEVICE_ARGS=(--device "${2:?}")
      shift 2
      ;;
    --skip-done) SKIP_DONE=1; shift ;;
    --seed) SEED="${2:?}"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "${LOGROOT}"

# Auto-pick CPU if CUDA is broken on this host.
if [[ ${#DEVICE_ARGS[@]} -eq 0 ]]; then
  if ! "${RUNNER[@]}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
    echo "[probe] CUDA unavailable — using CPU (--device -1)"
    DEVICE_ARGS=(--device -1)
  else
    echo "[probe] CUDA available — using GPU 0"
    DEVICE_ARGS=(--device 0)
  fi
fi

log_has_curve() {
  local f="$1"
  [[ -f "${f}" ]] || return 1
  grep -q "warmup_curve task=0" "${f}"
}

run_one() {
  local lr="$1"
  local lr_tag
  lr_tag="$(echo "${lr}" | tr '.' 'p')"
  local method="probe_cub_lr${lr_tag}_ep${EPOCHS}"
  if [[ "${FAST}" -eq 1 ]]; then
    method="fast_${method}"
  fi
  local logfile="${LOGROOT}/${method}.log"

  if [[ "${SKIP_DONE}" -eq 1 ]] && log_has_curve "${logfile}"; then
    local n
    n="$(grep -c "warmup_curve task=0" "${logfile}" || true)"
    if [[ "${n}" -ge "${EPOCHS}" ]]; then
      echo "[probe] SKIP ${method} (have ${n}/${EPOCHS} epochs)"
      return 0
    fi
  fi

  local -a extra=(
    --set "tuned_epoch=${EPOCHS}"
    --set "task0_init_lr=${lr}"
    --set "init_lr=${lr}"
    --set "save_checkpoint=false"
    --set "task0_save_warmup_ckpts=false"
    --set "resume=false"
  )
  if [[ "${FAST}" -eq 1 ]]; then
    # Still full 100 base classes, but fewer images — for quick convergence shape.
    extra+=(
      --set max_train_per_class=20
      --set max_test_per_class=10
      --set batch_size=16
      --set num_workers=2
    )
  fi

  echo "[probe] === ${method} lr=${lr} epochs=${EPOCHS} fast=${FAST} ==="
  "${RUNNER[@]}" train_fscil_image.py \
    --config configs/pride_image_cub200_buffree.json \
    --data-path "${DATA_PATH}" \
    --logdir "${LOGROOT}" \
    --method-name "${method}" \
    --log-layout flat \
    --seeds "${SEED}" \
    --max-task 0 \
    "${DEVICE_ARGS[@]}" \
    "${extra[@]}" \
    2>&1 | tee "${logfile}"
  echo "[probe] === done ${method} ==="
}

for lr in "${LRS[@]}"; do
  run_one "${lr}"
done

echo "[probe] finished → ${LOGROOT}"
echo "[probe] plot: aprun python scripts/plot_pride_image_lr_epoch_probe.py"

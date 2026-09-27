#!/bin/bash
#SBATCH --job-name=pride_img_a05_buf_ep10
#SBATCH --account=EUHPC_D36_199
#SBATCH --partition=boost_usr_prod
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=_experiments/slurm_logs/%A_%a_pride_img_a05_buf_ep10.out

# =============================================================================
# PRidE IMAGE twin — α=0.5 buffer, LGSP FSCIL protocols (EP10)
# =============================================================================
# Same as scripts/simpl_runs_pride_image_alpha05_ep10.sh except:
#   fscil_preset=canonical + exemplar_buffer_shots=1 (1-shot old-class buffer)
#
# Array map:
#   0  cub200     configs/pride_image_cub200_buf.json
#   1  aircraft   configs/pride_image_aircraft_buf.json
#   2  inf200     configs/pride_image_inf200_buf.json
#
# Logs:
#   _experiments/00_PRIDE_IMAGE_ALPHA05_BUF_EP10/final10_<proto>_mean_close_st_buf_a05.log
#
# On Leonardo (after sync + image data + offline ViT weights):
#   bash scripts/download_vit_in21k.sh && bash scripts/check_pride_image_data.sh
#   bash sync_eurohpc.sh --folder=_experiments/pretrained_weights/ViT
#   sbatch scripts/simpl_runs_pride_image_alpha05_buf_ep10.sh
# =============================================================================

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

RUNNER=(aprun python)
LOGROOT="_experiments/00_PRIDE_IMAGE_ALPHA05_BUF_EP10"
SEED=2008
SKIP_DONE=0
ONLY_IDX=""
ONLY_PROTOCOL=""
SMOKE=0
DATA_PATH="_datasets/images"
VIT_CKPT="_experiments/pretrained_weights/ViT/vit_base_patch16_224.augreg_in21k.safetensors"
export PRIDE_VIT_CKPT="$(pwd)/${VIT_CKPT}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local) RUNNER=(aprun python); shift ;;
    --skip-done) SKIP_DONE=1; shift ;;
    --smoke) SMOKE=1; shift ;;
    --data-path)
      DATA_PATH="${2:?}"
      shift 2
      ;;
    --array)
      ONLY_IDX="${2:?}"
      shift 2
      ;;
    --only-protocol)
      ONLY_PROTOCOL="${2:?}"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "${LOGROOT}" _experiments/slurm_logs

JOBS=(
  "cub200|configs/pride_image_cub200_buf.json|cub200|11|CUB-200"
  "aircraft|configs/pride_image_aircraft_buf.json|aircraft|11|FGVC-Aircraft"
  "inf200|configs/pride_image_inf200_buf.json|inf200|11|iNF200"
)

BASE_FLAGS=(
  "resume=false"
  "save_checkpoint=false"
  "task0_save_warmup_ckpts=false"
  "fscil_adapter_base_only=true"
  "task0_adapter_warmup=true"
  "increment_adapter_warmup=false"
  "fscil_classifier_warmup_after_protos=false"
  "fscil_task1_adapter_warmup=false"
  "tuned_epoch=10"
  "conflict_replay=true"
  "conflict_replay_on_test=false"
  "conflict_replay_use_test_cm=false"
  "model_name=selfevo"
  "fscil_preset=canonical"
  "exemplar_buffer_shots=1"
  "exemplar_buffer_ridge_refresh_shots=1"
  "prototype_pooling=max"
  "pooling_temperature=0.07"
  "fscil_shot_prototypes=false"
  "fscil_cluster_new_classes=false"
  "fscil_mean_best2_prototypes=true"
  "fscil_mean_close_far_prototypes=false"
  "fscil_mean_nearmean_prototypes=false"
  "fscil_mean_far2_prototypes=false"
  "insert_strategy=mean_best2"
  "max_prototypes_per_class=3"
  "fscil_increment_max_prototypes=3"
  "evolution_policy=static"
  "classifier_mode=ridge_cache"
  "ridge_cache_alpha=0.5"
  "fscil_increment_ridge_alpha=0.5"
  "fscil_task1_ridge_alpha=0.5"
  "fscil_ridge_alpha_ramp=false"
  "backbone_type=pretrained_vit_b16_224_in21k_adapter"
  "ffn_num=64"
)

log() { echo "[$(date -Is)] $*"; }

log_is_complete() {
  local logfile="$1"
  local n_sess="$2"
  [[ -f "${logfile}" ]] || return 1
  local last
  last="$(grep "CNN top1 curve:" "${logfile}" 2>/dev/null | tail -n 1 || true)"
  [[ -n "${last}" ]] || return 1
  local n
  n="$(echo "${last}" | tr -cd ',' | wc -c)"
  n=$((n + 1))
  [[ "${n}" -ge "${n_sess}" ]]
}

run_idx() {
  local idx="$1"
  IFS='|' read -r run_id config prefix n_sess label <<< "${JOBS[$idx]}"

  if [[ -n "${ONLY_PROTOCOL}" && "${run_id}" != "${ONLY_PROTOCOL}" ]]; then
    return 0
  fi

  bash scripts/check_pride_image_data.sh --data-path "${DATA_PATH}" --protocol "${run_id}"

  local method="final10_${prefix}_mean_close_st_buf_a05"
  local logfile="${LOGROOT}/${method}.log"

  if [[ "${SKIP_DONE}" -eq 1 ]] && log_is_complete "${logfile}" "${n_sess}"; then
    log "SKIP ${method}"
    return 0
  fi

  local -a extra_args=()
  for flag in "${BASE_FLAGS[@]}"; do
    extra_args+=(--set "${flag}")
  done

  if [[ "${SMOKE}" -eq 1 ]]; then
    # Mini schedule must match max_classes or task-1 loaders are empty.
    extra_args+=(
      --max-task 1
      --set init_cls=10
      --set increment=10
      --set tuned_epoch=1
      --set task0_save_warmup_ckpts=false
      --set max_classes=20
      --set max_train_per_class=5
      --set max_test_per_class=3
      --device -1
    )
    method="smoke_${method}"
    logfile="${LOGROOT}/${method}.log"
  fi

  log "=== [${idx}] ${method} | ${label} | mean_close_st | buf | max | ep10 | α=0.5 | ViT-IN21k-adapt ==="
  "${RUNNER[@]}" train_fscil_image.py \
    --config "${config}" \
    --data-path "${DATA_PATH}" \
    --logdir "${LOGROOT}" \
    --method-name "${method}" \
    --log-layout flat \
    --seeds "${SEED}" \
    "${extra_args[@]}" \
    2>&1 | tee "${logfile}"
  log "=== ${method} done ==="
}

if [[ -n "${ONLY_IDX}" ]]; then
  run_idx "${ONLY_IDX}"
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  run_idx "${SLURM_ARRAY_TASK_ID}"
else
  for idx in $(seq 0 2); do
    run_idx "${idx}"
  done
fi

log "PRidE image α=0.5 ep10 buffer twin finished → ${LOGROOT}"

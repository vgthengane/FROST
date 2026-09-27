#!/bin/bash
#SBATCH --job-name=pride_img_gauss
#SBATCH --account=EUHPC_D36_199
#SBATCH --partition=boost_usr_prod
#SBATCH --array=0-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=_experiments/slurm_logs/%A_%a_pride_img_gauss.out

# =============================================================================
# PRidE IMAGE — novel-focused cache scoring sweep (buffer-free, mean-only)
# =============================================================================
# Motivation: Novel@Last is the bottleneck on iNF/Aircraft; allshot5 hurt.
# This sweep compares cache scoring:
#   max         — current default (cosine max over keys)
#   logsumexp   — soft mixture over keys (already implemented)
#   diag_gauss  — NEW: shrunk diagonal Gaussian from all 5-shot embeddings
#
# Fixed recipe (best overnight mean-only style):
#   insert=mean | buffree | static | base-only AdaptFormer ep80 | rp=2000
#   α ∈ {0.35, 0.5} | seed=2008
#
# Array = protocol(3) × pooling(3) × alpha(2)  →  18 jobs if expanded;
# default array 0-11 covers aircraft+inf200 × 3 pool × 2 α (skip CUB).
#   flat = proto*6 + pool*2 + alpha_i
#   proto: 0=aircraft, 1=inf200  (add cub with --include-cub → remap)
#
# Logs:
#   _experiments/00_PRIDE_IMAGE_NOVEL_GAUSS/
#     gauss_<proto>_mean1_<pool>_a<alpha>_ep80.log
#
# On Leonardo:
#   bash sync_eurohpc.sh --code --force
#   sbatch -A EUHPC_D36_199 -p boost_usr_prod \
#     scripts/simpl_runs_pride_image_novel_gauss.sh
# Local one job:
#   bash scripts/simpl_runs_pride_image_novel_gauss.sh --local --array 0
# =============================================================================

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

RUNNER=(aprun python)
LOGROOT="_experiments/00_PRIDE_IMAGE_NOVEL_GAUSS"
SEED=2008
SKIP_DONE=0
ONLY_IDX=""
ONLY_PROTOCOL=""
SMOKE=0
INCLUDE_CUB=0
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
    --include-cub) INCLUDE_CUB=1; shift ;;
    --data-path) DATA_PATH="${2:?}"; shift 2 ;;
    --array) ONLY_IDX="${2:?}"; shift 2 ;;
    --only-protocol) ONLY_PROTOCOL="${2:?}"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "${LOGROOT}" _experiments/slurm_logs

PROTOCOLS=(
  "aircraft|configs/pride_image_aircraft_buffree.json|aircraft|11|FGVC-Aircraft"
  "inf200|configs/pride_image_inf200_buffree.json|inf200|11|iNF200"
)
if [[ "${INCLUDE_CUB}" -eq 1 ]]; then
  PROTOCOLS=(
    "cub200|configs/pride_image_cub200_buffree.json|cub200|11|CUB-200"
    "${PROTOCOLS[@]}"
  )
fi

POOLS=(max logsumexp diag_gauss)
ALPHAS=(0.35 0.5)
N_POOL=${#POOLS[@]}
N_ALPHA=${#ALPHAS[@]}
N_PROTO=${#PROTOCOLS[@]}
N_PER_PROTO=$((N_POOL * N_ALPHA))

BASE_FLAGS=(
  "resume=false"
  "save_checkpoint=false"
  "task0_save_warmup_ckpts=false"
  "fscil_adapter_base_only=true"
  "task0_adapter_warmup=true"
  "increment_adapter_warmup=false"
  "fscil_classifier_warmup_after_protos=false"
  "fscil_task1_adapter_warmup=false"
  "tuned_epoch=80"
  "conflict_replay=true"
  "conflict_replay_on_test=false"
  "conflict_replay_use_test_cm=false"
  "model_name=selfevo"
  "exemplar_buffer_shots=0"
  "prototype_pooling=max"
  "pooling_temperature=0.07"
  "fscil_shot_prototypes=false"
  "fscil_cluster_new_classes=false"
  "fscil_mean_best2_prototypes=false"
  "fscil_mean_close_far_prototypes=false"
  "fscil_mean_nearmean_prototypes=false"
  "fscil_mean_far2_prototypes=false"
  "insert_strategy=mean"
  "max_prototypes_per_class=1"
  "fscil_increment_max_prototypes=1"
  "evolution_policy=static"
  "classifier_mode=ridge_cache"
  "fscil_ridge_alpha_ramp=false"
  "ridge_rp_dim=2000"
  "backbone_type=pretrained_vit_b16_224_in21k_adapter"
  "ffn_num=64"
  "cache_gauss_var_floor=1e-4"
  "cache_gauss_var_shrink=0.5"
  "cache_gauss_use_logdet=false"
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

run_flat() {
  local flat="$1"
  local max=$((N_PROTO * N_PER_PROTO))
  if [[ "${flat}" -lt 0 || "${flat}" -ge "${max}" ]]; then
    echo "Array index ${flat} out of range [0, $((max - 1))]" >&2
    exit 1
  fi
  local proto_idx=$((flat / N_PER_PROTO))
  local rem=$((flat % N_PER_PROTO))
  local pool_idx=$((rem / N_ALPHA))
  local alpha_idx=$((rem % N_ALPHA))

  IFS='|' read -r run_id config prefix n_sess label <<< "${PROTOCOLS[$proto_idx]}"
  local pool="${POOLS[$pool_idx]}"
  local alpha="${ALPHAS[$alpha_idx]}"
  local a_tag
  a_tag="$(echo "${alpha}" | tr '.' 'p')"

  if [[ -n "${ONLY_PROTOCOL}" && "${run_id}" != "${ONLY_PROTOCOL}" ]]; then
    return 0
  fi

  bash scripts/check_pride_image_data.sh --data-path "${DATA_PATH}" --protocol "${run_id}"

  local method="gauss_${prefix}_mean1_${pool}_a${a_tag}_ep80"
  local logfile="${LOGROOT}/${method}.log"

  if [[ "${SKIP_DONE}" -eq 1 ]] && log_is_complete "${logfile}" "${n_sess}"; then
    log "SKIP ${method}"
    return 0
  fi

  local -a extra_args=()
  for flag in "${BASE_FLAGS[@]}"; do
    extra_args+=(--set "${flag}")
  done
  extra_args+=(
    --set "prototype_pooling=${pool}"
    --set "ridge_cache_alpha=${alpha}"
    --set "fscil_increment_ridge_alpha=${alpha}"
    --set "fscil_task1_ridge_alpha=${alpha}"
  )

  if [[ "${SMOKE}" -eq 1 ]]; then
    extra_args+=(
      --max-task 1
      --set init_cls=10
      --set increment=10
      --set tuned_epoch=1
      --set max_classes=20
      --set max_train_per_class=5
      --set max_test_per_class=3
    )
    method="smoke_${method}"
    logfile="${LOGROOT}/${method}.log"
  fi

  log "=== [${flat}] ${method} | ${label} | pool=${pool} α=${alpha} mean1 ep80 rp2000 ==="
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
  grep -E "CNN top1 curve:|FSCIL novel accuracy curve:|FSCIL base retention curve:|Average Accuracy" \
    "${logfile}" | tail -n 8 || true
}

if [[ -n "${ONLY_IDX}" ]]; then
  run_flat "${ONLY_IDX}"
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  run_flat "${SLURM_ARRAY_TASK_ID}"
else
  for flat in $(seq 0 $((N_PROTO * N_PER_PROTO - 1))); do
    run_flat "${flat}"
  done
fi

log "novel gauss sweep finished → ${LOGROOT}"

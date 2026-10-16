#!/bin/bash
#SBATCH --job-name=frost_image
#SBATCH --account=EUHPC_D36_199
#SBATCH --partition=boost_usr_prod
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/tmp/frost_image_%A_%a.out

# =============================================================================
# FROST reproduce — image FSCIL (CUB-200, iNF200, FGVC-Aircraft)
# =============================================================================
# Locked paper row = buffer-free mean-only, α=0.5, rp=2000.
# Image paper table has no exemplar-buffer twin (unlike 3D EF/EB).
#
# Default: first paper seed only (1947). Pass --fullseeds for all 5.
# Array (1 seed): 3 protocols; --fullseeds: 15 = 5 × 3
#   seeds: 1947, 1991, 2001, 2008, 2019
#   proto: 0=cub200(ep100) 1=inf200(ep100) 2=aircraft(ep120)
#
# Logs:
#   _experiments/00_PRIDE_IMAGE_MEANONLY_A05_RP2000_5SEEDS/
#
# Examples:
#   sbatch scripts/reproduce_image.sh
#   sbatch --array=0-14 scripts/reproduce_image.sh --fullseeds
#   bash scripts/reproduce_image.sh --array 0 --only-protocol cub200
#
# Summarize:
#   bash scripts/summarize_frost_image_multiseed.sh \
#     --log-root _experiments/00_PRIDE_IMAGE_MEANONLY_A05_RP2000_5SEEDS \
#     --insert mean1
#
# See scripts/README.md for data/ and _archive/ layout.
# =============================================================================

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"
# shellcheck source=frost_common.sh
source "${REPO_ROOT}/scripts/frost_common.sh"

RUNNER=(aprun python)
FULLSEEDS=0
SEEDS_OVERRIDE=0
SEEDS=()
DATA_PATH="_datasets/images"
VIT_CKPT="_experiments/pretrained_weights/ViT/vit_base_patch16_224.augreg_in21k.safetensors"
export PRIDE_VIT_CKPT="$(pwd)/${VIT_CKPT}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

SKIP_DONE=0
ONLY_IDX="${SLURM_ARRAY_TASK_ID:-}"
ONLY_PROTOCOL=""
ONLY_SEED=""
SMOKE=0
ALPHA=0.5
RP=2000
BUDGET=600

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local) RUNNER=(aprun python); shift ;;
    --fullseeds) FULLSEEDS=1; shift ;;
    --skip-done) SKIP_DONE=1; shift ;;
    --smoke) SMOKE=1; shift ;;
    --array) ONLY_IDX="${2:?}"; shift 2 ;;
    --only-protocol) ONLY_PROTOCOL="${2:?}"; shift 2 ;;
    --only-seed) ONLY_SEED="${2:?}"; shift 2 ;;
    --data-path) DATA_PATH="${2:?}"; shift 2 ;;
    --seeds)
      IFS=',' read -r -a SEEDS <<< "${2:?}"
      SEEDS_OVERRIDE=1
      shift 2
      ;;
    -h|--help)
      sed -n '15,45p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if [[ "${SEEDS_OVERRIDE}" -eq 0 ]]; then
  frost_select_seeds SEEDS FROST_IMAGE_SEEDS "${FULLSEEDS}"
fi

# proto|config|prefix|ep|n_sess  (paper = buffer-free only)
PROTOCOLS=(
  "cub200|configs/frost_img_cub200.json|cub200|100|11"
  "inf200|configs/frost_img_inf200.json|inf200|100|11"
  "aircraft|configs/frost_img_aircraft.json|aircraft|120|11"
)

N_PROTO=${#PROTOCOLS[@]}
N_SEED=${#SEEDS[@]}
N_JOBS=$((N_PROTO * N_SEED))

run_image_job() {
  local proto_idx="$1"
  local seed="$2"

  IFS='|' read -r proto config prefix ep n_sess <<< "${PROTOCOLS[$proto_idx]}"

  if [[ -n "${ONLY_PROTOCOL}" && "${proto}" != "${ONLY_PROTOCOL}" ]]; then
    return 0
  fi
  if [[ -n "${ONLY_SEED}" && "${seed}" != "${ONLY_SEED}" ]]; then
    return 0
  fi

  local logroot="${FROST_IMAGE_LOGROOT_EF}"
  mkdir -p "${logroot}"

  local ep_run="${ep}"
  local method="final_img_${prefix}_ep${ep_run}_mean1_a05_rp2000"
  local -a smoke_args=()
  if [[ "${SMOKE}" -eq 1 ]]; then
    ep_run=1
    method="smoke_${method}"
    smoke_args=(--max-task 1)
  fi
  local logfile="${logroot}/${method}_seed=${seed}.log"

  if [[ "${SKIP_DONE}" -eq 1 ]] && frost_log_is_complete "${logfile}" "${n_sess}"; then
    frost_log "SKIP ${method} seed=${seed}"
    return 0
  fi

  bash scripts/check_frost_image_data.sh --data-path "${DATA_PATH}" --protocol "${proto}"

  # Recipe lives in configs/base/images.json; only run-varying knobs here.
  local -a extra=(
    --set resume=false
    --set save_checkpoint=false
    --set task0_save_warmup_ckpts=true
    --set task0_use_best_warmup=true
    --set num_workers=4
    --set "tuned_epoch=${ep_run}"
    --set "cache_budget=${BUDGET}"
    --set "ridge_cache_alpha=${ALPHA}"
    --set "fscil_increment_ridge_alpha=${ALPHA}"
    --set "fscil_task1_ridge_alpha=${ALPHA}"
    --set "ridge_rp_dim=${RP}"
    --set "fscil_old_shot_replay_seed=${seed}"
    --set "fscil_base_ridge_refresh_seed=${seed}"
  )

  frost_log "=== ${method} | ${proto} ep=${ep_run} α=${ALPHA} rp=${RP} seed=${seed} | buffree ==="
  "${RUNNER[@]}" train_fscil_image.py \
    --config "${config}" \
    --data-path "${DATA_PATH}" \
    --logdir "${logroot}" \
    --method-name "${method}" \
    --log-layout flat \
    --seeds "${seed}" \
    "${smoke_args[@]}" \
    "${extra[@]}" \
    2>&1 | tee "${logfile}"
  frost_log "=== ${method} seed=${seed} done ==="
}

run_flat() {
  local flat="$1"
  if [[ "${flat}" -lt 0 || "${flat}" -ge "${N_JOBS}" ]]; then
    echo "array index ${flat} out of range [0, $((N_JOBS - 1))]" >&2
    exit 1
  fi
  local seed_idx=$((flat / N_PROTO))
  local proto_idx=$((flat % N_PROTO))
  run_image_job "${proto_idx}" "${SEEDS[$seed_idx]}"
}

if [[ -z "${ONLY_IDX}" ]]; then
  echo "Defined ${N_JOBS} image jobs (buffree-only protocols=${N_PROTO} seeds=${N_SEED}). Pass --array N or sbatch."
  exit 0
fi

run_flat "${ONLY_IDX}"

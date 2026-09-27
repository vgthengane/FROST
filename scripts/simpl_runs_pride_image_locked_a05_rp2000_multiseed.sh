#!/bin/bash
#SBATCH --job-name=pride_img_ms
#SBATCH --account=EUHPC_D36_199
#SBATCH --partition=boost_usr_prod
#SBATCH --array=0-14
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=_experiments/slurm_logs/%A_%a_pride_img_ms.out
# SBATCH footprint matched to scripts/simpl_runs_pride_final_sub_mn40_logsumexp_rp1000.sh

# =============================================================================
# PRidE IMAGE — locked table recipe, 5 paper seeds
# =============================================================================
# Recipe (locked for paper table):
#   mean_best2 | α=0.5 | rp=2000 | budget=600 | buffer-free | static
#   base-only AdaptFormer | conflict train-CM | best-warmup restore
#   CUB ep100 | iNF ep100 | Aircraft ep120
#
# Array (15 jobs = 3 protocols × 5 seeds):
#   task_id = seed_idx * 3 + protocol_idx
#   protocol_idx: 0=cub200(ep100)  1=inf200(ep100)  2=aircraft(ep120)
#   seed_idx:     0..4 → 1947, 1991, 2001, 2008, 2019
#
# Logs:
#   _experiments/00_PRIDE_IMAGE_LOCKED_A05_RP2000_5SEEDS/
#     final_img_<proto>_ep<N>_mean_best2_a05_rp2000_seed=<seed>.log
#
# Submit (Leonardo):
#   sbatch scripts/simpl_runs_pride_image_locked_a05_rp2000_multiseed.sh
# From Surrey:
#   bash scripts/leonardo_remote.sh sync-file scripts/simpl_runs_pride_image_locked_a05_rp2000_multiseed.sh
#   bash scripts/leonardo_remote.sh sbatch scripts/simpl_runs_pride_image_locked_a05_rp2000_multiseed.sh
#
# Summarize:
#   bash scripts/summarize_pride_image_locked_multiseed.sh
# One job smoke:
#   sbatch --array=0 scripts/simpl_runs_pride_image_locked_a05_rp2000_multiseed.sh
# Skip finished:
#   bash scripts/simpl_runs_pride_image_locked_a05_rp2000_multiseed.sh --local --skip-done --array 0
# =============================================================================

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

RUNNER=(aprun python)
LOGROOT="_experiments/00_PRIDE_IMAGE_LOCKED_A05_RP2000_5SEEDS"
DATA_PATH="_datasets/images"
VIT_CKPT="_experiments/pretrained_weights/ViT/vit_base_patch16_224.augreg_in21k.safetensors"
export PRIDE_VIT_CKPT="$(pwd)/${VIT_CKPT}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

DEFAULT_SEEDS=(1947 1991 2001 2008 2019)
SEEDS=("${DEFAULT_SEEDS[@]}")
ONLY_IDX="${SLURM_ARRAY_TASK_ID:-}"
SKIP_DONE=0
ONLY_PROTOCOL=""
ONLY_SEED=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local) RUNNER=(aprun python); shift ;;
    --skip-done) SKIP_DONE=1; shift ;;
    --array) ONLY_IDX="${2:?}"; shift 2 ;;
    --only-protocol) ONLY_PROTOCOL="${2:?}"; shift 2 ;;
    --only-seed) ONLY_SEED="${2:?}"; shift 2 ;;
    --seeds)
      IFS=',' read -r -a SEEDS <<< "${2:?}"
      shift 2
      ;;
    --data-path) DATA_PATH="${2:?}"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "${LOGROOT}" _experiments/slurm_logs

# protocol_idx → proto|config|prefix|epochs|n_sess
PROTOCOLS=(
  "cub200|configs/pride_image_cub200_buffree.json|cub200|100|11"
  "inf200|configs/pride_image_inf200_buffree.json|inf200|100|11"
  "aircraft|configs/pride_image_aircraft_buffree.json|aircraft|120|11"
)
N_PROTO=${#PROTOCOLS[@]}
N_SEED=${#SEEDS[@]}
N_JOBS=$((N_PROTO * N_SEED))

if [[ -z "${ONLY_IDX}" ]]; then
  echo "Defined ${N_JOBS} jobs (protocols=${N_PROTO} seeds=${N_SEED}). Pass --array N or sbatch."
  exit 0
fi

if [[ "${ONLY_IDX}" -lt 0 || "${ONLY_IDX}" -ge "${N_JOBS}" ]]; then
  echo "array index ${ONLY_IDX} out of range [0, $((N_JOBS - 1))]" >&2
  exit 1
fi

SEED_IDX=$((ONLY_IDX / N_PROTO))
PROTO_IDX=$((ONLY_IDX % N_PROTO))
SEED="${SEEDS[$SEED_IDX]}"
IFS='|' read -r PROTO CONFIG PREFIX EP N_SESS <<< "${PROTOCOLS[$PROTO_IDX]}"

if [[ -n "${ONLY_PROTOCOL}" && "${PROTO}" != "${ONLY_PROTOCOL}" ]]; then
  echo "skip (only-protocol=${ONLY_PROTOCOL})"
  exit 0
fi
if [[ -n "${ONLY_SEED}" && "${SEED}" != "${ONLY_SEED}" ]]; then
  echo "skip (only-seed=${ONLY_SEED})"
  exit 0
fi

ALPHA=0.5
RP=2000
BUDGET=600
METHOD="final_img_${PREFIX}_ep${EP}_mean_best2_a05_rp2000"
LOGFILE="${LOGROOT}/${METHOD}_seed=${SEED}.log"

log() { echo "[$(date -Is)] $*"; }

log_is_complete() {
  local logfile="$1"
  local n_sess="$2"
  [[ -f "${logfile}" ]] || return 1
  local last
  last="$(grep "CNN top1 curve:" "${logfile}" 2>/dev/null | tail -n 1 || true)"
  [[ -n "${last}" ]] || return 1
  # Prefer a real [..] curve line (avoid tqdm-interleaved false positives).
  local n
  n="$(echo "${last}" | grep -oE '\[[0-9.,[:space:]]+\]' | tail -n 1 | tr -cd ',' | wc -c || true)"
  [[ -n "${n}" ]] || return 1
  n=$((n + 1))
  [[ "${n}" -ge "${n_sess}" ]]
}

if [[ "${SKIP_DONE}" -eq 1 ]] && log_is_complete "${LOGFILE}" "${N_SESS}"; then
  log "SKIP ${METHOD} seed=${SEED}"
  exit 0
fi

bash scripts/check_pride_image_data.sh --data-path "${DATA_PATH}" --protocol "${PROTO}"

log "=== [${ONLY_IDX}/${N_JOBS}] ${METHOD} | ${PROTO} ep=${EP} α=${ALPHA} rp=${RP} seed=${SEED} ==="
"${RUNNER[@]}" train_fscil_image.py \
  --config "${CONFIG}" \
  --data-path "${DATA_PATH}" \
  --logdir "${LOGROOT}" \
  --method-name "${METHOD}" \
  --log-layout flat \
  --seeds "${SEED}" \
  --set resume=false \
  --set save_checkpoint=false \
  --set task0_save_warmup_ckpts=true \
  --set task0_use_best_warmup=true \
  --set fscil_adapter_base_only=true \
  --set task0_adapter_warmup=true \
  --set increment_adapter_warmup=false \
  --set fscil_classifier_warmup_after_protos=false \
  --set fscil_task1_adapter_warmup=false \
  --set conflict_replay=true \
  --set conflict_replay_on_test=false \
  --set conflict_replay_use_test_cm=false \
  --set model_name=selfevo \
  --set exemplar_buffer_shots=0 \
  --set prototype_pooling=max \
  --set pooling_temperature=0.07 \
  --set fscil_shot_protocol=false \
  --set fscil_cluster_new_classes=false \
  --set fscil_mean_close_far_prototypes=false \
  --set fscil_mean_nearmean_prototypes=false \
  --set fscil_mean_far2_prototypes=false \
  --set fscil_mean_best2_prototypes=true \
  --set fscil_shot_prototypes=false \
  --set insert_strategy=mean_best2 \
  --set max_prototypes_per_class=3 \
  --set fscil_increment_max_prototypes=3 \
  --set evolution_policy=static \
  --set classifier_mode=ridge_cache \
  --set fscil_ridge_alpha_ramp=false \
  --set backbone_type=pretrained_vit_b16_224_in21k_adapter \
  --set ffn_num=64 \
  --set num_workers=4 \
  --set tuned_epoch="${EP}" \
  --set cache_budget="${BUDGET}" \
  --set ridge_cache_alpha="${ALPHA}" \
  --set fscil_increment_ridge_alpha="${ALPHA}" \
  --set fscil_task1_ridge_alpha="${ALPHA}" \
  --set ridge_rp_dim="${RP}" \
  --set "fscil_old_shot_replay_seed=${SEED}" \
  --set "fscil_base_ridge_refresh_seed=${SEED}" \
  2>&1 | tee "${LOGFILE}"

log "=== ${METHOD} seed=${SEED} done ==="

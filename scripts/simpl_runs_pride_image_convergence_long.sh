#!/bin/bash
#SBATCH --job-name=pride_img_conv
#SBATCH --account=EUHPC_D36_199
#SBATCH --partition=boost_usr_prod
#SBATCH --array=0-8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=_experiments/slurm_logs/%A_%a_pride_img_conv.out
# Resource footprint matched to scripts/simpl_runs_pride_final_sub_ep10.sh
# (8 CPU / 32G) with a longer 5h wall so CUB/iNF ep120 full cannot TIMEOUT.

# =============================================================================
# PRidE IMAGE — long AdaptFormer convergence check (ep100 / ep120)
# =============================================================================
# Goal: see whether task-0 adapter warmup keeps improving past overnight ep80.
#
# Grid (9 jobs):
#   protocol:     aircraft | cub200 | inf200
#   tuned_epoch:  100 | 120
#   mode:         full FSCIL (all sessions) for ep100 on all 3
#                 + task0-only ep120 on all 3 (cheap deep warmup curves)
#                 + full FSCIL ep120 on aircraft only (fastest protocol)
#
# Locked recipe (matches overnight / 3D PRidE insert):
#   mean_best2 | α=0.5 flat | rp=2000 | budget=600 | static | buffer-free
#   base-only AdaptFormer | conflict train-CM | seed 2008
#
# Logs: _experiments/00_PRIDE_IMAGE_CONVERGENCE_LONG/
#   conv_<proto>_ep<N>_mean_best2_a05_rp2000.log
#   conv_<proto>_ep<N>_mean_best2_a05_rp2000_task0.log
#
# Per-epoch AdaptFormer test acc is already logged as:
#   "Task 0 warmup Epoch k/N => ... Test_accy ..."
#   "warmup_curve task=0 epoch=k/N ..."
#
# Morning summary:
#   bash scripts/summarize_pride_image_convergence.sh
#
# Submit (Leonardo):
#   sbatch scripts/simpl_runs_pride_image_convergence_long.sh
# From Surrey:
#   bash scripts/leonardo_remote.sh sync-file scripts/simpl_runs_pride_image_convergence_long.sh
#   bash scripts/leonardo_remote.sh sbatch scripts/simpl_runs_pride_image_convergence_long.sh
#
# Smoke one job locally / single array task:
#   bash scripts/simpl_runs_pride_image_convergence_long.sh --local --array 0
#   sbatch --array=0 scripts/simpl_runs_pride_image_convergence_long.sh
# =============================================================================

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

RUNNER=(aprun python)
LOGROOT="_experiments/00_PRIDE_IMAGE_CONVERGENCE_LONG"
SEED=2008
DATA_PATH="_datasets/images"
VIT_CKPT="_experiments/pretrained_weights/ViT/vit_base_patch16_224.augreg_in21k.safetensors"
export PRIDE_VIT_CKPT="$(pwd)/${VIT_CKPT}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

ONLY_IDX="${SLURM_ARRAY_TASK_ID:-}"
SKIP_DONE=0
LOCAL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local) LOCAL=1; RUNNER=(aprun python); shift ;;
    --skip-done) SKIP_DONE=1; shift ;;
    --array) ONLY_IDX="${2:?}"; shift 2 ;;
    --data-path) DATA_PATH="${2:?}"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "${LOGROOT}" _experiments/slurm_logs

# Index map (must match #SBATCH --array=0-8):
#  0 aircraft ep100 full
#  1 cub200   ep100 full
#  2 inf200   ep100 full
#  3 aircraft ep120 task0
#  4 cub200   ep120 task0
#  5 inf200   ep120 task0
#  6 aircraft ep120 full   (cheap full beyond 100)
#  7 cub200   ep120 full
#  8 inf200   ep120 full
mapfile -t JOBS < <(
  echo "aircraft|100|full"
  echo "cub200|100|full"
  echo "inf200|100|full"
  echo "aircraft|120|task0"
  echo "cub200|120|task0"
  echo "inf200|120|task0"
  echo "aircraft|120|full"
  echo "cub200|120|full"
  echo "inf200|120|full"
)

N_JOBS="${#JOBS[@]}"
if [[ -z "${ONLY_IDX}" ]]; then
  echo "Defined ${N_JOBS} convergence jobs. Pass --array N or submit via sbatch."
  exit 0
fi

if [[ "${ONLY_IDX}" -lt 0 || "${ONLY_IDX}" -ge "${N_JOBS}" ]]; then
  echo "array index ${ONLY_IDX} out of range [0, $((N_JOBS - 1))]" >&2
  exit 1
fi

IFS='|' read -r PROTO EP MODE <<< "${JOBS[$ONLY_IDX]}"

case "${PROTO}" in
  cub200)
    CONFIG="configs/pride_image_cub200_buffree.json"
    PREFIX="cub200"
    N_SESS=11
    ;;
  aircraft)
    CONFIG="configs/pride_image_aircraft_buffree.json"
    PREFIX="aircraft"
    N_SESS=11
    ;;
  inf200)
    CONFIG="configs/pride_image_inf200_buffree.json"
    PREFIX="inf200"
    N_SESS=11
    ;;
  *)
    echo "bad protocol ${PROTO}" >&2
    exit 1
    ;;
esac

ALPHA=0.5
RP=2000
BUDGET=600
A_TAG="05"
METHOD="conv_${PREFIX}_ep${EP}_mean_best2_a${A_TAG}_rp${RP}"
if [[ "${MODE}" == "task0" ]]; then
  METHOD="${METHOD}_task0"
fi
LOGFILE="${LOGROOT}/${METHOD}.log"

log() { echo "[$(date -Is)] $*"; }

log_is_complete() {
  local logfile="$1"
  local mode="$2"
  local n_sess="$3"
  local ep="$4"
  [[ -f "${logfile}" ]] || return 1
  if [[ "${mode}" == "task0" ]]; then
    # Need a full warmup curve (ep lines) or final warmup epoch line.
    local n
    n="$(grep -c "warmup_curve task=0" "${logfile}" 2>/dev/null || true)"
    if [[ "${n}" -ge "${ep}" ]]; then
      return 0
    fi
    grep -q "Task 0 warmup Epoch ${ep}/${ep}" "${logfile}" 2>/dev/null
    return $?
  fi
  local last
  last="$(grep "CNN top1 curve:" "${logfile}" 2>/dev/null | tail -n 1 || true)"
  [[ -n "${last}" ]] || return 1
  local n
  n="$(echo "${last}" | tr -cd ',' | wc -c)"
  n=$((n + 1))
  [[ "${n}" -ge "${n_sess}" ]]
}

if [[ "${SKIP_DONE}" -eq 1 ]] && log_is_complete "${LOGFILE}" "${MODE}" "${N_SESS}" "${EP}"; then
  log "SKIP ${METHOD}"
  exit 0
fi

bash scripts/check_pride_image_data.sh --data-path "${DATA_PATH}" --protocol "${PROTO}"

MAX_TASK_ARGS=()
if [[ "${MODE}" == "task0" ]]; then
  MAX_TASK_ARGS=(--max-task 0)
fi

log "=== [${ONLY_IDX}/${N_JOBS}] ${METHOD} | ${PROTO} ep=${EP} mode=${MODE} α=${ALPHA} rp=${RP} ==="
"${RUNNER[@]}" train_fscil_image.py \
  --config "${CONFIG}" \
  --data-path "${DATA_PATH}" \
  --logdir "${LOGROOT}" \
  --method-name "${METHOD}" \
  --log-layout flat \
  --seeds "${SEED}" \
  "${MAX_TASK_ARGS[@]}" \
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
  2>&1 | tee "${LOGFILE}"

log "=== ${METHOD} done ==="

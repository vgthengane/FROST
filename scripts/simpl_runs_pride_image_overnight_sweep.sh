#!/bin/bash
#SBATCH --job-name=pride_img_ovn
#SBATCH --account=EUHPC_D36_199
#SBATCH --partition=boost_usr_prod
#SBATCH --array=0-323
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=_experiments/slurm_logs/%A_%a_pride_img_ovn.out

# =============================================================================
# PRidE IMAGE — overnight hyperparameter sweep (buffer-free)
# =============================================================================
# Axes (cartesian, 324 jobs):
#   protocol:     cub200 | aircraft | inf200
#   tuned_epoch:  20 | 40 | 80          (deeper AdaptFormer base warmup)
#   insert:       mean_best2 | allshot5 | mean1
#                   allshot5 = every novel 5-shot sample as a prototype
#                   mean1    = single mean proto from all available shots
#                   mean_best2 = locked mean + two closest (max 3)
#   cache_budget: 600 | 2000            (higher sizes)
#   alpha:        0.35 | 0.5 | 0.7
#   ridge_rp_dim: 1000 | 2000
#
# Logs: _experiments/00_PRIDE_IMAGE_OVERNIGHT_SWEEP/<method>.log
# Morning summary:
#   bash scripts/summarize_pride_image_overnight.sh
#
# Submit (Leonardo):
#   sbatch scripts/simpl_runs_pride_image_overnight_sweep.sh
# From Surrey (with live Cineca cert):
#   bash scripts/leonardo_remote.sh sync-file models/selfevo.py
#   bash scripts/leonardo_remote.sh sync-file scripts/simpl_runs_pride_image_overnight_sweep.sh
#   bash scripts/leonardo_remote.sh sbatch scripts/simpl_runs_pride_image_overnight_sweep.sh
# =============================================================================

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

RUNNER=(aprun python)
LOGROOT="_experiments/00_PRIDE_IMAGE_OVERNIGHT_SWEEP"
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

# Build cartesian grid once (order defines array indices).
mapfile -t JOBS < <(
  for proto in cub200 aircraft inf200; do
    for ep in 20 40 80; do
      for insert in mean_best2 allshot5 mean1; do
        for budget in 600 2000; do
          for alpha in 0.35 0.5 0.7; do
            for rp in 1000 2000; do
              echo "${proto}|${ep}|${insert}|${budget}|${alpha}|${rp}"
            done
          done
        done
      done
    done
  done
)

N_JOBS="${#JOBS[@]}"
if [[ -z "${ONLY_IDX}" ]]; then
  echo "Defined ${N_JOBS} sweep jobs. Pass --array N or submit via sbatch."
  exit 0
fi

if [[ "${ONLY_IDX}" -lt 0 || "${ONLY_IDX}" -ge "${N_JOBS}" ]]; then
  echo "array index ${ONLY_IDX} out of range [0, $((N_JOBS - 1))]" >&2
  exit 1
fi

IFS='|' read -r PROTO EP INSERT BUDGET ALPHA RP <<< "${JOBS[$ONLY_IDX]}"

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

A_TAG="${ALPHA/./}"
METHOD="ovn_${PREFIX}_ep${EP}_${INSERT}_b${BUDGET}_a${A_TAG}_rp${RP}"
LOGFILE="${LOGROOT}/${METHOD}.log"

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

if [[ "${SKIP_DONE}" -eq 1 ]] && log_is_complete "${LOGFILE}" "${N_SESS}"; then
  log "SKIP ${METHOD}"
  exit 0
fi

bash scripts/check_pride_image_data.sh --data-path "${DATA_PATH}" --protocol "${PROTO}"

# Shared recipe knobs (buffer-free overnight).
EXTRA=(
  --set resume=false
  --set save_checkpoint=false
  --set task0_save_warmup_ckpts=false
  --set fscil_adapter_base_only=true
  --set task0_adapter_warmup=true
  --set increment_adapter_warmup=false
  --set fscil_classifier_warmup_after_protos=false
  --set fscil_task1_adapter_warmup=false
  --set conflict_replay=true
  --set conflict_replay_on_test=false
  --set conflict_replay_use_test_cm=false
  --set model_name=selfevo
  --set exemplar_buffer_shots=0
  --set prototype_pooling=max
  --set pooling_temperature=0.07
  --set fscil_shot_protocol=false
  --set fscil_cluster_new_classes=false
  --set fscil_mean_close_far_prototypes=false
  --set fscil_mean_nearmean_prototypes=false
  --set fscil_mean_far2_prototypes=false
  --set evolution_policy=static
  --set classifier_mode=ridge_cache
  --set fscil_ridge_alpha_ramp=false
  --set backbone_type=pretrained_vit_b16_224_in21k_adapter
  --set ffn_num=64
  --set tuned_epoch="${EP}"
  --set cache_budget="${BUDGET}"
  --set ridge_cache_alpha="${ALPHA}"
  --set fscil_increment_ridge_alpha="${ALPHA}"
  --set fscil_task1_ridge_alpha="${ALPHA}"
  --set ridge_rp_dim="${RP}"
)

case "${INSERT}" in
  mean_best2)
    EXTRA+=(
      --set insert_strategy=mean_best2
      --set fscil_mean_best2_prototypes=true
      --set fscil_shot_prototypes=false
      --set max_prototypes_per_class=3
      --set fscil_increment_max_prototypes=3
    )
    ;;
  allshot5)
    # Base: mean_best2; novel: every 5-shot sample as a prototype.
    EXTRA+=(
      --set insert_strategy=mean_best2
      --set fscil_mean_best2_prototypes=false
      --set fscil_shot_prototypes=true
      --set max_prototypes_per_class=3
      --set fscil_increment_max_prototypes=5
    )
    ;;
  mean1)
    EXTRA+=(
      --set insert_strategy=mean
      --set fscil_mean_best2_prototypes=false
      --set fscil_shot_prototypes=false
      --set max_prototypes_per_class=1
      --set fscil_increment_max_prototypes=1
    )
    ;;
  *)
    echo "bad insert ${INSERT}" >&2
    exit 1
    ;;
esac

log "=== [${ONLY_IDX}/${N_JOBS}] ${METHOD} | ${PROTO} ep=${EP} insert=${INSERT} budget=${BUDGET} α=${ALPHA} rp=${RP} ==="
"${RUNNER[@]}" train_fscil_image.py \
  --config "${CONFIG}" \
  --data-path "${DATA_PATH}" \
  --logdir "${LOGROOT}" \
  --method-name "${METHOD}" \
  --log-layout flat \
  --seeds "${SEED}" \
  "${EXTRA[@]}" \
  2>&1 | tee "${LOGFILE}"
log "=== ${METHOD} done ==="

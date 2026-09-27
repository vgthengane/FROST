#!/bin/bash
#SBATCH --job-name=pride_img_ep20ai
#SBATCH --account=EUHPC_D36_199
#SBATCH --partition=boost_usr_prod
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=_experiments/slurm_logs/%A_%a_pride_img_ep20ai.out

# =============================================================================
# PRidE IMAGE — Aircraft + iNF200 with deeper AdaptFormer (ep20 / ep40)
# =============================================================================
# Uses CUB-winning recipe knobs:
#   mean_best2 | static | ridge_cache | buffer-free | conflict train-CM
# Sweep (16 jobs):
#   protocol ∈ {aircraft, inf200}
#   tuned_epoch ∈ {20, 40}
#   alpha ∈ {0.35, 0.5}
#   ridge_rp_dim ∈ {1000, 2000}
#   cache_budget = 600
#
# Submit:
#   sbatch scripts/simpl_runs_pride_image_aircraft_inf200_ep20.sh
# =============================================================================

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

RUNNER=(aprun python)
LOGROOT="_experiments/00_PRIDE_IMAGE_AIR_INF_EP20"
SEED=2008
DATA_PATH="_datasets/images"
export PRIDE_VIT_CKPT="$(pwd)/_experiments/pretrained_weights/ViT/vit_base_patch16_224.augreg_in21k.safetensors"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

IDX="${SLURM_ARRAY_TASK_ID:-${1:-}}"
[[ -n "${IDX}" ]] || { echo "Need SLURM_ARRAY_TASK_ID or arg index"; exit 1; }

mkdir -p "${LOGROOT}" _experiments/slurm_logs

mapfile -t JOBS < <(
  for proto in aircraft inf200; do
    for ep in 20 40; do
      for alpha in 0.35 0.5; do
        for rp in 1000 2000; do
          echo "${proto}|${ep}|${alpha}|${rp}"
        done
      done
    done
  done
)

IFS='|' read -r PROTO EP ALPHA RP <<< "${JOBS[$IDX]}"

case "${PROTO}" in
  aircraft)
    CONFIG="configs/pride_image_aircraft_buffree.json"
    PREFIX="aircraft"
    ;;
  inf200)
    CONFIG="configs/pride_image_inf200_buffree.json"
    PREFIX="inf200"
    ;;
  *)
    echo "bad proto ${PROTO}" >&2
    exit 1
    ;;
esac

A_TAG="${ALPHA/./}"
METHOD="airinf_${PREFIX}_ep${EP}_mean_best2_b600_a${A_TAG}_rp${RP}"
LOGFILE="${LOGROOT}/${METHOD}.log"

bash scripts/check_pride_image_data.sh --data-path "${DATA_PATH}" --protocol "${PROTO}"

echo "[$(date -Is)] === [${IDX}/16] ${METHOD} ==="
"${RUNNER[@]}" train_fscil_image.py \
  --config "${CONFIG}" \
  --data-path "${DATA_PATH}" \
  --logdir "${LOGROOT}" \
  --method-name "${METHOD}" \
  --log-layout flat \
  --seeds "${SEED}" \
  --set resume=false \
  --set save_checkpoint=false \
  --set task0_save_warmup_ckpts=false \
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
  --set tuned_epoch="${EP}" \
  --set cache_budget=600 \
  --set ridge_cache_alpha="${ALPHA}" \
  --set fscil_increment_ridge_alpha="${ALPHA}" \
  --set fscil_task1_ridge_alpha="${ALPHA}" \
  --set ridge_rp_dim="${RP}" \
  2>&1 | tee "${LOGFILE}"

echo "[$(date -Is)] === ${METHOD} done ==="

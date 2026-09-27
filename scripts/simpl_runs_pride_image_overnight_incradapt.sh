#!/bin/bash
#SBATCH --job-name=pride_img_ovn_ia
#SBATCH --account=EUHPC_D36_199
#SBATCH --partition=boost_usr_prod
#SBATCH --array=0-23
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=_experiments/slurm_logs/%A_%a_pride_img_ovn_ia.out

# Companion overnight: deeper base + increment adapter warmup on allshot5.
# 24 jobs: proto×{cub,air,inf} × ep×{40,80} × alpha×{0.5,0.7} × rp×{1000,2000}
# budget fixed 2000, insert=allshot5, increment_adapter_warmup=true

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

RUNNER=(aprun python)
LOGROOT="_experiments/00_PRIDE_IMAGE_OVERNIGHT_SWEEP"
SEED=2008
DATA_PATH="_datasets/images"
export PRIDE_VIT_CKPT="$(pwd)/_experiments/pretrained_weights/ViT/vit_base_patch16_224.augreg_in21k.safetensors"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

IDX="${SLURM_ARRAY_TASK_ID:?}"
mkdir -p "${LOGROOT}" _experiments/slurm_logs

mapfile -t JOBS < <(
  for proto in cub200 aircraft inf200; do
    for ep in 40 80; do
      for alpha in 0.5 0.7; do
        for rp in 1000 2000; do
          echo "${proto}|${ep}|${alpha}|${rp}"
        done
      done
    done
  done
)

IFS='|' read -r PROTO EP ALPHA RP <<< "${JOBS[$IDX]}"
case "${PROTO}" in
  cub200) CONFIG="configs/pride_image_cub200_buffree.json"; PREFIX=cub200 ;;
  aircraft) CONFIG="configs/pride_image_aircraft_buffree.json"; PREFIX=aircraft ;;
  inf200) CONFIG="configs/pride_image_inf200_buffree.json"; PREFIX=inf200 ;;
esac

A_TAG="${ALPHA/./}"
METHOD="ovn_${PREFIX}_ep${EP}_allshot5_b2000_a${A_TAG}_rp${RP}_incradapt"
LOGFILE="${LOGROOT}/${METHOD}.log"

bash scripts/check_pride_image_data.sh --data-path "${DATA_PATH}" --protocol "${PROTO}"

echo "[$(date -Is)] === [${IDX}] ${METHOD} ==="
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
  --set increment_adapter_warmup=true \
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
  --set fscil_mean_best2_prototypes=false \
  --set fscil_shot_prototypes=true \
  --set insert_strategy=mean_best2 \
  --set max_prototypes_per_class=3 \
  --set fscil_increment_max_prototypes=5 \
  --set evolution_policy=static \
  --set classifier_mode=ridge_cache \
  --set fscil_ridge_alpha_ramp=false \
  --set backbone_type=pretrained_vit_b16_224_in21k_adapter \
  --set ffn_num=64 \
  --set tuned_epoch="${EP}" \
  --set cache_budget=2000 \
  --set ridge_cache_alpha="${ALPHA}" \
  --set fscil_increment_ridge_alpha="${ALPHA}" \
  --set fscil_task1_ridge_alpha="${ALPHA}" \
  --set ridge_rp_dim="${RP}" \
  2>&1 | tee "${LOGFILE}"
echo "[$(date -Is)] === ${METHOD} done ==="

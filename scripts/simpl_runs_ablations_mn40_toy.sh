#!/bin/bash
#SBATCH --job-name=selfevo-toy-mn40-evo
#SBATCH --account=EUHPC_D36_199
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=_experiments/slurm_logs/%j_simpl_abl_mn40_toy.out

# Toy overnight (fast) sanity-check for evolution ablations on FSCIL MN40.
#
# What it runs:
#   1) full (ridge_cache + evolution_policy=rules)  [baseline]
#   2) static (ridge_cache + evolution_policy=static)
#   3) ridge-only (classifier_mode=ridge; cache/evolution bypass)
#
# Output logs:
#   _experiments/00_SIMPL_TOY_ABLATIONS_MN40/<method>_seed=2008.log
#
# Submit on Leonardo:
#   sbatch scripts/simpl_runs_ablations_mn40_toy.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

CONFIG="configs/selfevo_fscil_mn40_5t_flex_compare.json"
LOGROOT="_experiments/00_SIMPL_TOY_ABLATIONS_MN40"
mkdir -p "${LOGROOT}" _experiments/slurm_logs

COMMON_OVERRIDES=(
  # Make warmups fast in toy mode.
  --set tuned_epoch=1
  --set increment_tuned_epoch=1
  --set fscil_task1_warmup_epochs=1
  --set fscil_last_task_warmup_epochs=1
  --set task0_save_warmup_ckpts=false
)

SEEDS=(--seeds 2008)

run_one() {
  local method="$1"
  shift
  local -a extra=("$@")

  echo "=== ${method} toy start ==="
  aprun python train_fscil.py \
    --config "${CONFIG}" \
    --logdir "${LOGROOT}" \
    --method-name "${method}" \
    --log-layout flat \
    --toy \
    "${SEEDS[@]}" \
    "${COMMON_OVERRIDES[@]}" \
    --max-task 1 \
    "${extra[@]}" \
    2>&1 | tee "${LOGROOT}/${method}_seed=2008.log"
  echo "=== ${method} toy done ==="

  echo "--- ${method} last curve ---"
  grep "CNN top1 curve" "${LOGROOT}/${method}_seed=2008.log" | tail -1 || true
  echo "--- ${method} average ---"
  grep "Average Accuracy (CNN):" "${LOGROOT}/${method}_seed=2008.log" | tail -1 || true
}

# 1) Full (rules)
run_one "toy_mn40_abl_full" \
  # no extra overrides; config already uses ridge_cache + rules

# 2) Static evolution
run_one "toy_mn40_abl_static" \
  --set evolution_policy=static

# 3) Ridge-only bypass (cache not used at inference)
run_one "toy_mn40_abl_ridge_only" \
  --set classifier_mode=ridge \
  --set evolution_policy=static

echo "Toy ablation logs are under: ${LOGROOT}"


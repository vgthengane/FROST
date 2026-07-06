#!/bin/bash
#SBATCH --job-name=rlrazor
#SBATCH --account=EUHPC_D36_199
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=_experiments/slurm_logs/%j_rlrazor.out

# RL's Razor ablation — isolated from core.sh / ours.sh / full_runs_5seeds.
#
# Compares RanPAC task-0 adapter warmup variants on ModelNet40 (10 tasks):
#   ranpac_onpolicy  — 1-0 REINFORCE (on-policy)
#   ranpac_kl_reg    — SFT + forward-KL penalty to warmup-start policy
#
# Baseline ranpac (SFT warmup) stays in _experiments/full_runs_5seeds/ranpac/.
#
# Usage:
#   bash scripts/run_rlrazor_ablation.sh
#   SEEDS=1993 bash scripts/run_rlrazor_ablation.sh
#   sbatch scripts/run_rlrazor_ablation.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

LOG_ROOT="_experiments/rlrazor_ablation"
SEEDS="${SEEDS:-1947,1991,2001,2008,2019}"
METHODS=(ranpac_onpolicy ranpac_kl_reg)

mkdir -p "${LOG_ROOT}" _experiments/slurm_logs

for METHOD in "${METHODS[@]}"; do
  CONFIG="configs/${METHOD}_modelnet40.json"
  if [[ ! -f "${CONFIG}" ]]; then
    ARCHIVE_CONFIG="_archive/configs/cil/${METHOD}_modelnet40.json"
    if [[ -f "${ARCHIVE_CONFIG}" ]]; then
      CONFIG="${ARCHIVE_CONFIG}"
    else
      echo "Config not found: ${CONFIG} (and missing archive ${ARCHIVE_CONFIG})" >&2
      exit 1
    fi
  fi
  echo "========== RL Razor ablation: ${METHOD} (seeds=${SEEDS}) =========="
  aprun python train_cil.py \
    --config "${CONFIG}" \
    --logdir="${LOG_ROOT}/${METHOD}" \
    --method-name="${METHOD}" \
    --log-layout flat \
    --seeds "${SEEDS}"
done

echo "========== Summary =========="
aprun python scripts/summarize_camera_results.py --log-root "${LOG_ROOT}"

echo "Compare against baseline: aprun python scripts/summarize_camera_results.py --log-root _experiments/full_runs_5seeds"

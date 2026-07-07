#!/bin/bash
#SBATCH --job-name=simpl_aug_sgd
#SBATCH --account=EUHPC_D36_199
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=_experiments/slurm_logs/%j_simpl_runs_aug_sgd.out

# FSCIL ablation: task-0 point aug (z-rot + scale + shift) + SGD warmup.
#   task0_init_lr=0.001, task0_weight_decay=0.0001
# Same session schedule as simpl_runs; separate log root for comparison.
#
# Submit on Leonardo:
#   sbatch scripts/simpl_runs_aug_sgd.sh
#   sbatch scripts/simpl_runs_aug_sgd.sh --mn40-only
#   sbatch scripts/simpl_runs_aug_sgd.sh --sn55-only
#
# Local (GPU node):
#   bash scripts/simpl_runs_aug_sgd.sh --local
#   bash scripts/simpl_runs_aug_sgd.sh --local --mn40-only
#
# Logs: _experiments/00_SIMPL_RUNs_aug_sgd/<method>_seed=<seed>.log
# Table: _experiments/00_SIMPL_RUNs_aug_sgd/results_table.md

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

LOGROOT="_experiments/00_SIMPL_RUNs_aug_sgd"
METHOD_MN40="seel_mn40_aug_sgd"
METHOD_SN55="seel_sn55_aug_sgd"
mkdir -p "${LOGROOT}" _experiments/slurm_logs

RUNNER=(aprun python)
RUN_MN40=1
RUN_SN55=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --local)
      RUNNER=(aprun python)
      shift
      ;;
    --mn40-only)
      RUN_MN40=1
      RUN_SN55=0
      shift
      ;;
    --sn55-only)
      RUN_MN40=0
      RUN_SN55=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ "${RUN_MN40}" -eq 0 && "${RUN_SN55}" -eq 0 ]]; then
  echo "Nothing to run: use --mn40-only and/or --sn55-only (default: both)" >&2
  exit 1
fi

log() { echo "[$(date -Is)] $*"; }

# Task-0 warmup: microshape-style geom aug + SGD (increment clf-warmup stays config default).
TASK0_OVERRIDES=(
  --set "task0_point_augment=true"
  --set "task0_aug_rotate_z=true"
  --set "task0_aug_scale_low=0.8"
  --set "task0_aug_scale_high=1.25"
  --set "task0_aug_shift_range=0.1"
  --set "task0_optimizer=sgd"
  --set "task0_init_lr=0.001"
  --set "task0_weight_decay=0.0001"
)

run_one() {
  local config="$1"
  local method="$2"
  log "=== ${method} start (config=${config}, aug+sgd lr=0.001 wd=1e-4) ==="
  "${RUNNER[@]}" train_fscil.py \
    --config "${config}" \
    --logdir "${LOGROOT}" \
    --method-name "${method}" \
    --log-layout flat \
    "${TASK0_OVERRIDES[@]}" \
    2>&1 | tee "${LOGROOT}/${method}.log"
  log "=== ${method} done ==="
  grep "CNN top1 curve" "${LOGROOT}/${method}_seed="*.log || true
}

log "simpl_runs_aug_sgd — FSCIL-3D task-0 aug + SGD (seed 2008)"

if [[ "${RUN_MN40}" -eq 1 ]]; then
  run_one "configs/selfevo_fscil_mn40_5t_flex_compare.json" "${METHOD_MN40}"
fi
if [[ "${RUN_SN55}" -eq 1 ]]; then
  run_one "configs/selfevo_fscil_sn55_7t_flex_compare.json" "${METHOD_SN55}"
fi

log "=== Per-dataset summaries ==="
"${RUNNER[@]}" scripts/summarize_fscil_results.py \
  --log-root "${LOGROOT}" \
  --methods aug_sgd \
  --out "${LOGROOT}/results.md"

log "=== Combined 3D-FLEG comparison table ==="
"${RUNNER[@]}" scripts/summarize_fscil_paper_table.py \
  --campaign simpl_runs_aug_sgd \
  --out "${LOGROOT}/results_table.md"

log "simpl_runs_aug_sgd finished — see ${LOGROOT}/results_table.md"

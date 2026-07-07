#!/bin/bash
#SBATCH --job-name=t0_lr_fscil
#SBATCH --account=EUHPC_D36_199
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=_experiments/slurm_logs/%j_t0_lr_sweep_fscil.out

# Task-0 adapter LR sweep on canonical FSCIL-3D MN40 (session 0 only).
# Verifies FSCIL class order + train_n=7438 before trusting warmup metrics.
#
# Submit on Leonardo:
#   sbatch scripts/t0_lr_sweep_fscil.sh
#
# Local:
#   bash scripts/t0_lr_sweep_fscil.sh --local
#   bash scripts/t0_lr_sweep_fscil.sh --local --lr 0.005
#
# Logs: _experiments/flex_compare/t0_lr_sweep_fscil/<method>_seed=2008.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

CONFIG="configs/selfevo_fscil_mn40_5t_flex_compare.json"
LOGROOT="_experiments/flex_compare/t0_lr_sweep_fscil"
EXPECTED_TRAIN_N=7438
FSCIL_ORDER_PREFIX="FSCIL-3D class_order: [0, 2, 4, 5, 7"

RUNNER=(aprun python)
LRS=(0.003 0.005 0.01)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local)
      RUNNER=(aprun python)
      shift
      ;;
    --lr)
      LRS=("$2")
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--local] [--lr LR]" >&2
      exit 1
      ;;
  esac
done

mkdir -p "${LOGROOT}" _experiments/slurm_logs

log() { echo "[$(date -Is)] $*"; }

verify_fscil_protocol() {
  local logfile="$1"
  if ! grep -qF "${FSCIL_ORDER_PREFIX}" "${logfile}"; then
    log "FAIL: missing FSCIL-3D class order in ${logfile}" >&2
    return 1
  fi
  if ! grep -q "train_n=${EXPECTED_TRAIN_N}" "${logfile}"; then
    log "FAIL: expected train_n=${EXPECTED_TRAIN_N} in ${logfile}" >&2
    grep "Base session:" "${logfile}" >&2 || true
    return 1
  fi
  return 0
}

run_one() {
  local lr="$1"
  local method="seel_mn40_t0lr${lr}"
  local logfile="${LOGROOT}/${method}_seed=2008.log"

  log "=== ${method} task0_init_lr=${lr} ==="
  "${RUNNER[@]}" train_fscil.py \
    --config "${CONFIG}" \
    --max-task 0 \
    --set "task0_init_lr=${lr}" \
    --logdir "${LOGROOT}" \
    --method-name "${method}" \
    --log-layout flat \
    2>&1 | tee "${logfile}"

  verify_fscil_protocol "${logfile}"
  local best
  best=$(grep -oP 'Test_accy \K[0-9.]+' "${logfile}" | sort -n | tail -1 || true)
  log "=== ${method} done — peak warmup Test_accy=${best:-unknown} ==="
}

log "t0_lr_sweep_fscil — MN40 FSCIL-3D task 0 only (LRs: ${LRS[*]})"

for lr in "${LRS[@]}"; do
  run_one "${lr}"
done

log "=== Summary (peak warmup Test_accy) ==="
for lr in "${LRS[@]}"; do
  method="seel_mn40_t0lr${lr}"
  logfile="${LOGROOT}/${method}_seed=2008.log"
  if [[ -f "${logfile}" ]]; then
    best=$(grep -oP 'Test_accy \K[0-9.]+' "${logfile}" | sort -n | tail -1 || echo "?")
    echo "  lr=${lr}: ${best}%  (${logfile})"
  fi
done

log "t0_lr_sweep_fscil finished — see ${LOGROOT}/"

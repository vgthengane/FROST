#!/bin/bash
#SBATCH --job-name=fscil_mn40_oracle
#SBATCH --account=EUHPC_D36_199
#SBATCH --partition=boost_usr_prod
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=_experiments/slurm_logs/%A_%a_fscil_mn40_oracle.out

# MN40-only Joint/FT wrapper. Prefer the full 6-protocol launcher:
#   sbatch scripts/simpl_runs_fscil_oracle_ft_all.sh
#
# This script still runs MN40 joint (array 0) and finetune (array 1).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

RUNNER=(aprun python)
LOGROOT="_experiments/00_FSCIL_ORACLE_FT"
SEED=2008
SKIP_DONE=0
ONLY=""
SMOKE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local) RUNNER=(aprun python); shift ;;
    --skip-done) SKIP_DONE=1; shift ;;
    --smoke) SMOKE=1; shift ;;
    --only)
      ONLY="${2:?}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "${LOGROOT}" _experiments/slurm_logs

NAMES=(joint finetune)
CONFIGS=(
  configs/fscil_joint_mn40_5t.json
  configs/fscil_finetune_mn40_5t.json
)
METHODS=(
  oracle_mn40_joint_ep10_5
  oracle_mn40_finetune_ep10_5
)

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && -z "${ONLY}" ]]; then
  IDX="${SLURM_ARRAY_TASK_ID}"
else
  IDX=""
fi

log() { echo "[$(date -Is)] $*"; }

run_one() {
  local idx="$1"
  local name="${NAMES[$idx]}"
  local config="${CONFIGS[$idx]}"
  local method="${METHODS[$idx]}"
  local logfile="${LOGROOT}/${method}.log"

  if [[ "${SKIP_DONE}" -eq 1 && -f "${logfile}" ]] && grep -q "CNN top1 curve" "${logfile}"; then
    log "SKIP ${method} (done)"
    return 0
  fi

  log "=== ${method} start (config=${config}) ==="
  local extra=()
  if [[ "${SMOKE}" -eq 1 ]]; then
    extra+=(--set base_epoch=1 --set increment_epoch=1)
  fi
  "${RUNNER[@]}" train_fscil.py \
    --config "${config}" \
    --logdir "${LOGROOT}" \
    --method-name "${method}" \
    --log-layout flat \
    --seeds "${SEED}" \
    "${extra[@]}" \
    2>&1 | tee "${logfile}"
  log "=== ${method} done ==="
  grep "CNN top1 curve" "${LOGROOT}/${method}"*.log 2>/dev/null || true
}

if [[ -n "${ONLY}" ]]; then
  case "${ONLY}" in
    joint|0) run_one 0 ;;
    finetune|ft|1) run_one 1 ;;
    *)
      echo "--only must be joint|finetune" >&2
      exit 1
      ;;
  esac
elif [[ -n "${IDX}" ]]; then
  run_one "${IDX}"
else
  run_one 0
  run_one 1
fi

log "oracle MN40 joint/FT finished → ${LOGROOT}"

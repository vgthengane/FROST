#!/bin/bash
#SBATCH --job-name=scanobj_xd_bf
#SBATCH --account=EUHPC_D36_199
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=_experiments/slurm_logs/%j_scanobj_xd_bf.out

# Cross-dataset FSCIL (ModelNet40/ShapeNet55 -> ScanObjectNN) with buffer-free SelfEvo.
# Usage:
#   sbatch scripts/run_bufferfree_scanobjectnn.sh
#   sbatch scripts/run_bufferfree_scanobjectnn.sh --mn40-only
#   sbatch scripts/run_bufferfree_scanobjectnn.sh --sn55-only

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

LOGROOT="_experiments/00_SCANOBJ_XD_RUNs"
METHOD_MN40="seel_mn40_scanobj_4t_bufferfree"
METHOD_SN55="seel_sn55_scanobj_4t_bufferfree"
mkdir -p "${LOGROOT}" _experiments/slurm_logs

RUNNER=(aprun python)
RUN_MN40=1
RUN_SN55=1
while [[ $# -gt 0 ]]; do
  case "$1" in
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

run_one() {
  local config="$1"
  local method="$2"
  log "=== ${method} start (buffer-free, config=${config}) ==="
  "${RUNNER[@]}" train_fscil.py \
    --config "${config}" \
    --logdir "${LOGROOT}" \
    --method-name "${method}" \
    --log-layout flat \
    --set model_name=selfevo_bufferfree \
    2>&1 | tee "${LOGROOT}/${method}.log"
  log "=== ${method} done ==="
  rg "CNN top1 curve" "${LOGROOT}/${method}_seed="*.log || true
}

log "scanobj_xd buffer-free — cross-dataset FSCIL (seed 2008)"

if [[ "${RUN_MN40}" -eq 1 ]]; then
  run_one "configs/selfevo_fscil_mn40_scanobjectnn_4t.json" "${METHOD_MN40}"
fi
if [[ "${RUN_SN55}" -eq 1 ]]; then
  run_one "configs/selfevo_fscil_sn55_scanobjectnn_4t.json" "${METHOD_SN55}"
fi

log "scanobj_xd buffer-free finished — logs under ${LOGROOT}/"

#!/bin/bash
#SBATCH --job-name=co3d-fscil
#SBATCH --account=EUHPC_D36_199
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=_experiments/slurm_logs/%j_co3d_fscil.out

# CO3D FSCIL runs (microshape protocol).
#
# In-dataset:  CO3D 25+5x5 over 6 sessions  (configs/selfevo_fscil_co3d_6t.json)
# Cross-domain: ShapeNet55 -> CO3D 39+5x10 over 11 sessions
#               (configs/selfevo_fscil_sn55_co3d_11t.json)
#
# Submit on Leonardo:
#   sbatch scripts/simpl_runs_co3d.sh
#   sbatch scripts/simpl_runs_co3d.sh --co3d-only
#   sbatch scripts/simpl_runs_co3d.sh --cross-only
#
# Local (GPU node):
#   bash scripts/simpl_runs_co3d.sh --local
#   bash scripts/simpl_runs_co3d.sh --local --co3d-only
#
# Logs: _experiments/00_CO3D_RUNs/<method>_seed=<seed>.log
# Smoke (before submit):
#   aprun python scripts/smoke_test_co3d_runs.py

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

LOGROOT="_experiments/00_CO3D_RUNs"
METHOD_CO3D="seel_co3d_6t"
METHOD_CROSS="seel_sn55_co3d_11t"
mkdir -p "${LOGROOT}" _experiments/slurm_logs

RUNNER=(aprun python)
RUN_CO3D=1
RUN_CROSS=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local)
      RUNNER=(aprun python)
      shift
      ;;
    --co3d-only)
      RUN_CO3D=1
      RUN_CROSS=0
      shift
      ;;
    --cross-only)
      RUN_CO3D=0
      RUN_CROSS=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ "${RUN_CO3D}" -eq 0 && "${RUN_CROSS}" -eq 0 ]]; then
  echo "Nothing to run: use --co3d-only and/or --cross-only (default: both)" >&2
  exit 1
fi

log() { echo "[$(date -Is)] $*"; }

run_one() {
  local config="$1"
  local method="$2"
  log "=== ${method} start (config=${config}) ==="
  "${RUNNER[@]}" train_fscil.py \
    --config "${config}" \
    --logdir "${LOGROOT}" \
    --method-name "${method}" \
    --log-layout flat \
    2>&1 | tee "${LOGROOT}/${method}.log"
  log "=== ${method} done ==="
  grep "CNN top1 curve" "${LOGROOT}/${method}_seed="*.log || true
}

log "co3d_fscil — in-dataset + cross-dataset (seed 2008)"

if [[ "${RUN_CO3D}" -eq 1 ]]; then
  run_one "configs/selfevo_fscil_co3d_6t.json" "${METHOD_CO3D}"
fi
if [[ "${RUN_CROSS}" -eq 1 ]]; then
  run_one "configs/selfevo_fscil_sn55_co3d_11t.json" "${METHOD_CROSS}"
fi

log "co3d_fscil finished — logs under ${LOGROOT}/"

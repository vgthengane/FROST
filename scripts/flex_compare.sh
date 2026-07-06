#!/bin/bash
#SBATCH --job-name=flex_compare
#SBATCH --account=EUHPC_D36_199
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=_experiments/slurm_logs/%j_flex_compare.out

# Canonical FSCIL track — SelfEvo ridge_cache vs 3D-FLEG (FSCIL-3D class order).
# Task definition: _notes/fscil_canonical.md
#
# Submit on Leonardo:
#   sbatch scripts/flex_compare.sh
#   sbatch scripts/flex_compare.sh --mn40-only
#   sbatch scripts/flex_compare.sh --sn55-only
#
# Local (GPU node):
#   bash scripts/flex_compare.sh --local
#   bash scripts/flex_compare.sh --local --mn40-only
#   bash scripts/flex_compare.sh --local --sn55-only
#
# Logs: _experiments/flex_compare/<method>_seed=<seed>.log
# Table: _experiments/flex_compare/results_table.md

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

LOGROOT="_experiments/flex_compare"
METHOD_MN40="seel_mn40_flex_compare"
METHOD_SN55="seel_sn55_flex_compare"
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
  echo "Nothing to run: enable --mn40-only and/or --sn55-only" >&2
  exit 1
fi

log() { echo "[$(date -Is)] $*"; }

run_one() {
  local config="$1"
  local method="$2"
  log "=== ${method} start (config=${config}) ==="
  "${RUNNER[@]}" train_cil.py \
    --config "${config}" \
    --logdir "${LOGROOT}" \
    --method-name "${method}" \
    --log-layout flat \
    2>&1 | tee "${LOGROOT}/${method}.log"
  log "=== ${method} done ==="
  grep "CNN top1 curve" "${LOGROOT}/${method}_seed="*.log || true
}

log "flex_compare — SEEL FSCIL-3D (1-shot old refresh, seed 2008)"

if [[ "${RUN_MN40}" -eq 1 ]]; then
  run_one "configs/selfevo_fscil_mn40_5t_flex_compare.json" "${METHOD_MN40}"
fi
if [[ "${RUN_SN55}" -eq 1 ]]; then
  run_one "configs/selfevo_fscil_sn55_7t_flex_compare.json" "${METHOD_SN55}"
fi

log "=== Per-dataset summaries ==="
"${RUNNER[@]}" scripts/summarize_fscil_results.py \
  --log-root "${LOGROOT}" \
  --methods flex_compare \
  --out "${LOGROOT}/results.md"

log "=== Combined 3D-FLEG comparison table ==="
"${RUNNER[@]}" scripts/summarize_fscil_paper_table.py \
  --campaign flex_compare \
  --out "${LOGROOT}/results_table.md"

log "flex_compare finished — see ${LOGROOT}/results_table.md"

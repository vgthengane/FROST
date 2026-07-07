#!/bin/bash
#SBATCH --job-name=simpl_ep10_ep5
#SBATCH --account=EUHPC_D36_199
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=_experiments/slurm_logs/%j_simpl_runs_ep10_ep5.out

# FSCIL epoch ablation: 10 epochs task 0, 5 epochs on increment clf-warmup (tasks 1+).
# Task-0: SGD lr=0.001, wd=1e-4 (canonical config). MN40 + SN55.
#
# Submit:
#   sbatch scripts/simpl_runs_ep10_ep5.sh
#   sbatch scripts/simpl_runs_ep10_ep5.sh --mn40-only
#   bash scripts/simpl_runs_ep10_ep5.sh --local --sn55-only
#
# Logs: _experiments/00_SIMPL_RUNs_ep10_ep5/

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

LOGROOT="_experiments/00_SIMPL_RUNs_ep10_ep5"
METHOD_MN40="seel_mn40_ep10_ep5"
METHOD_SN55="seel_sn55_ep10_ep5"
EPOCH_ARGS=(
  --set "tuned_epoch=10"
  --set "increment_tuned_epoch=5"
  --set "fscil_task1_warmup_epochs=5"
  --set "fscil_last_task_warmup_epochs=5"
)

mkdir -p "${LOGROOT}" _experiments/slurm_logs

RUNNER=(aprun python)
RUN_MN40=1
RUN_SN55=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --local) RUNNER=(aprun python); shift ;;
    --mn40-only) RUN_MN40=1; RUN_SN55=0; shift ;;
    --sn55-only) RUN_MN40=0; RUN_SN55=1; shift ;;
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
  log "=== ${method} start (t0=10 ep, incr=5 ep, config=${config}) ==="
  "${RUNNER[@]}" train_fscil.py \
    --config "${config}" \
    "${EPOCH_ARGS[@]}" \
    --logdir "${LOGROOT}" \
    --method-name "${method}" \
    --log-layout flat \
    2>&1 | tee "${LOGROOT}/${method}.log"
  log "=== ${method} done ==="
  grep "CNN top1 curve" "${LOGROOT}/${method}.log" || true
}

log "simpl_runs_ep10_ep5 — task0=10 ep, increments=5 ep (seed 2008)"

if [[ "${RUN_MN40}" -eq 1 ]]; then
  run_one "configs/selfevo_fscil_mn40_5t_flex_compare.json" "${METHOD_MN40}"
fi
if [[ "${RUN_SN55}" -eq 1 ]]; then
  run_one "configs/selfevo_fscil_sn55_7t_flex_compare.json" "${METHOD_SN55}"
fi

log "=== Summaries ==="
"${RUNNER[@]}" scripts/summarize_fscil_results.py \
  --log-root "${LOGROOT}" \
  --methods ep10_ep5 \
  --out "${LOGROOT}/results.md"
"${RUNNER[@]}" scripts/summarize_fscil_paper_table.py \
  --campaign simpl_runs_ep10_ep5 \
  --out "${LOGROOT}/results_table.md"

log "simpl_runs_ep10_ep5 finished — see ${LOGROOT}/results_table.md"

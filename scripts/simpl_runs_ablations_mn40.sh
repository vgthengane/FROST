#!/bin/bash
#SBATCH --job-name=simpl_abl_mn40
#SBATCH --account=EUHPC_D36_199
#SBATCH --partition=boost_usr_prod
#SBATCH --array=0-13
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=_experiments/slurm_logs/%A_%a_simpl_abl_mn40.out

# FSCIL ModelNet40 ablation campaign (seed 2008).
#
# Table A — module ablations (classifier, evolution, replay, gated growth)
# Table B — hyperparameter ablations (α, cache budget B, prototypes per task P)
#
# Submit on Leonardo (14 parallel array jobs, one ablation each):
#   bash sync_eurohpc.sh --code --force          # from local workstation
#   sbatch scripts/simpl_runs_ablations_mn40.sh  # on Leonardo login node
#
# After jobs finish (on Leonardo):
#   aprun python scripts/summarize_fscil_ablations_mn40.py
#
# Smoke on Leonardo debug QoS (one ablation, 2 sessions):
#   sbatch --qos=boost_qos_dbg --time=01:00:00 --array=0 \
#     scripts/simpl_runs_ablations_mn40.sh --smoke
#
# Re-run / single ablation on Leonardo:
#   sbatch --array=0 scripts/simpl_runs_ablations_mn40.sh   # full baseline only
#   sbatch --array=1 scripts/simpl_runs_ablations_mn40.sh   # ridge only
#
# Logs: _experiments/00_SIMPL_ABLATIONS_MN40/<method>_seed=2008.log
# Tables: aprun python scripts/summarize_fscil_ablations_mn40.py

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

CONFIG="configs/selfevo_fscil_mn40_5t_flex_compare.json"
LOGROOT="_experiments/00_SIMPL_ABLATIONS_MN40"
RUNNER=(aprun python)

# method_id | tier | human label (for logs)
ABLATIONS=(
  "full|modules|Full SIMPL (baseline)"
  "ridge|modules|Ridge only (no cache at inference)"
  "cache|modules|Cache only (no ridge)"
  "static|modules|Ridge+cache, static evolution"
  "noreplay|modules|w/o conflict replay"
  "nogated|modules|w/o gated growth"
  "a05|hparams|Fusion α=0.5"
  "a07|hparams|Fusion α=0.7"
  "a10|hparams|Fusion α=1.0 (ridge only)"
  "B100|hparams|Cache budget B=100"
  "B300|hparams|Cache budget B=300"
  "p33|hparams|Fixed P=3 (base + incr)"
  "p35|hparams|Variable P: base=3, incr=5"
  "pshot|hparams|Per-shot protos, P≤5"
)

SMOKE=0
SKIP_DONE=0
TIER=""
ONLY=""
LOCAL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local) LOCAL=1; shift ;;
    --smoke) SMOKE=1; shift ;;
    --skip-done) SKIP_DONE=1; shift ;;
    --tier)
      TIER="${2:?--tier requires modules or hparams}"
      shift 2
      ;;
    --only)
      ONLY="${2:?--only requires an ablation id (e.g. ridge, a05, p33)}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--local] [--smoke] [--skip-done] [--tier modules|hparams] [--only ID]" >&2
      exit 1
      ;;
  esac
done

mkdir -p "${LOGROOT}" _experiments/slurm_logs

log() { echo "[$(date -Is)] $*"; }

smoke_args() {
  echo tuned_epoch=1
  echo increment_tuned_epoch=1
  echo fscil_task1_warmup_epochs=1
  echo fscil_last_task_warmup_epochs=1
  echo task0_save_warmup_ckpts=false
}

sets_for_id() {
  local id="$1"
    case "${id}" in
    full) ;;
    ridge)
      echo fscil_preset=ridge_only
      ;;
    cache)
      echo fscil_preset=cache_only
      ;;
    static)
      echo fscil_preset=static_fusion
      ;;
    noreplay)
      echo fscil_preset=no_conflict_replay
      ;;
    nogated)
      echo enable_gated_growth=false
      ;;
    a05) echo ridge_cache_alpha=0.5 ;;
    a07) echo ridge_cache_alpha=0.7 ;;
    a10) echo ridge_cache_alpha=1.0 ;;
    B100) echo cache_budget=100 ;;
    B300) echo cache_budget=300 ;;
    p33)
      echo max_prototypes_per_class=3
      echo fscil_increment_max_prototypes=3
      ;;
    p35)
      echo max_prototypes_per_class=3
      echo fscil_increment_max_prototypes=5
      ;;
    pshot)
      echo fscil_shot_prototypes=true
      echo fscil_increment_max_prototypes=5
      ;;
    *)
      echo "Unknown ablation id: ${id}" >&2
      return 1
      ;;
  esac
}

method_for_id() {
  echo "seel_mn40_abl_${1}"
}

log_is_complete() {
  local logfile="$1"
  [[ -f "${logfile}" ]] || return 1
  grep -q "CNN top1 curve: \[[^]]*,[^]]*\]" "${logfile}" 2>/dev/null || return 1
  local n
  n="$(grep -o "CNN top1 curve:" "${logfile}" | wc -l)"
  [[ "${n}" -ge 5 ]]
}

run_one() {
  local id="$1"
  local tier="$2"
  local label="$3"
  local method
  method="$(method_for_id "${id}")"
  local logfile="${LOGROOT}/${method}_seed=2008.log"

  if [[ -n "${TIER}" && "${tier}" != "${TIER}" ]]; then
    return 0
  fi
  if [[ -n "${ONLY}" && "${id}" != "${ONLY}" ]]; then
    return 0
  fi

  if [[ "${SKIP_DONE}" -eq 1 ]] && log_is_complete "${logfile}"; then
    log "SKIP ${method} (${label}) — complete log exists"
    return 0
  fi

  local -a extra_args=()
  while IFS= read -r line; do
    [[ -n "${line}" ]] && extra_args+=(--set "${line}")
  done < <(sets_for_id "${id}")

  if [[ "${SMOKE}" -eq 1 ]]; then
    extra_args+=(--max-task 1)
    while IFS= read -r line; do
      [[ -n "${line}" ]] && extra_args+=(--set "${line}")
    done < <(smoke_args)
    logfile="${LOGROOT}/${method}_smoke_seed=2008.log"
  fi

  log "=== ${method} [${tier}] ${label} ==="
  "${RUNNER[@]}" train_fscil.py \
    --config "${CONFIG}" \
    --logdir "${LOGROOT}" \
    --method-name "${method}" \
    --log-layout flat \
    "${extra_args[@]}" \
    2>&1 | tee "${LOGROOT}/${method}$([[ ${SMOKE} -eq 1 ]] && echo _smoke).log"
  log "=== ${method} done ==="
  grep "CNN top1 curve" "${LOGROOT}/${method}"*.log | tail -1 || true
}

select_indices() {
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "${SLURM_ARRAY_TASK_ID}"
    return
  fi
  seq 0 $((${#ABLATIONS[@]} - 1))
}

log "simpl_runs_ablations_mn40 — MN40 FSCIL ablations (seed 2008, smoke=${SMOKE})"

for idx in $(select_indices); do
  IFS='|' read -r id tier label <<< "${ABLATIONS[$idx]}"
  run_one "${id}" "${tier}" "${label}"
done

if [[ "${SMOKE}" -eq 0 && -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  log "=== Summaries ==="
  "${RUNNER[@]}" scripts/summarize_fscil_ablations_mn40.py \
    --log-root "${LOGROOT}" \
    --out "${LOGROOT}/ablation_tables.md"
fi

log "simpl_runs_ablations_mn40 finished — see ${LOGROOT}/"

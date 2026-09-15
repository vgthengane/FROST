#!/bin/bash
#SBATCH --job-name=cross_incr_wu
#SBATCH --account=EUHPC_D36_199
#SBATCH --partition=boost_usr_prod
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=_experiments/slurm_logs/%A_%a_cross_incr_wu.out

# Cross-dataset increment warmup ablation (seed 2008).
#
# Base-only adapter + task-1 novel warmup for domain shift.
# Increment sessions: 2 epochs, restore task-0 best anchor, use last epoch weights.
#
# Array layout (dataset × warmup mode):
#   0 = MN40->ScanObj, pre-proto adapter warmup (task 1 only)
#   1 = MN40->ScanObj, post-proto clf warmup (task 1 only)
#   2 = MN40->ScanObj, pre + post (task 1 only)
#   3 = SN55->ScanObj, pre-proto
#   4 = SN55->ScanObj, post-proto
#   5 = SN55->ScanObj, pre + post
#
# Submit:
#   sbatch scripts/simpl_runs_cross_increment_warmup_ablation.sh
#
# Local:
#   bash scripts/simpl_runs_cross_increment_warmup_ablation.sh --local

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

LOGROOT="_experiments/00_SCANOBJ_ABL_ADAPTER"
RUNNER=(aprun python)
SKIP_DONE=0
ONLY_IDX=""
SMOKE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local) RUNNER=(aprun python); shift ;;
    --skip-done) SKIP_DONE=1; shift ;;
    --smoke) SMOKE=1; shift ;;
    --array)
      ONLY_IDX="${2:?--array needs index 0-5}"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "${LOGROOT}" _experiments/slurm_logs

# run_id|config|method_prefix|n_sess|label
RUNS=(
  "mn_pre|configs/selfevo_fscil_mn40_scanobjectnn_4t.json|abl_xd_mn_incr_pre|4|MN40->ScanObj pre-proto"
  "mn_post|configs/selfevo_fscil_mn40_scanobjectnn_4t.json|abl_xd_mn_incr_post|4|MN40->ScanObj post-proto"
  "mn_both|configs/selfevo_fscil_mn40_scanobjectnn_4t.json|abl_xd_mn_incr_both|4|MN40->ScanObj pre+post"
  "sn_pre|configs/selfevo_fscil_sn55_scanobjectnn_4t.json|abl_xd_sn_incr_pre|4|SN55->ScanObj pre-proto"
  "sn_post|configs/selfevo_fscil_sn55_scanobjectnn_4t.json|abl_xd_sn_incr_post|4|SN55->ScanObj post-proto"
  "sn_both|configs/selfevo_fscil_sn55_scanobjectnn_4t.json|abl_xd_sn_incr_both|4|SN55->ScanObj pre+post"
)

SHARED_FLAGS=(
  "fscil_adapter_base_only=true"
  "task0_adapter_warmup=true"
  "fscil_task1_adapter_warmup=true"
  "increment_adapter_warmup=true"
  "increment_tuned_epoch=2"
  "fscil_task1_warmup_epochs=2"
  "fscil_last_task_warmup_epochs=2"
  "fscil_restore_task0_adapter_before_increment_warmup=true"
)

mode_flags() {
  case "$1" in
    *_pre)
      echo fscil_classifier_warmup_after_protos=false
      echo fscil_increment_warmup_pre_and_post=false
      ;;
    *_post)
      echo fscil_classifier_warmup_after_protos=true
      echo fscil_increment_warmup_pre_and_post=false
      ;;
    *_both)
      echo fscil_classifier_warmup_after_protos=true
      echo fscil_increment_warmup_pre_and_post=true
      ;;
    *)
      echo "Unknown mode in run_id: $1" >&2
      exit 1
      ;;
  esac
}

log() { echo "[$(date -Is)] $*"; }

log_is_complete() {
  local logfile="$1"
  local n_sess="$2"
  [[ -f "${logfile}" ]] || return 1
  local last
  last="$(rg "CNN top1 curve:" "${logfile}" | tail -n 1 || true)"
  [[ -n "${last}" ]] || return 1
  local n
  n="$(echo "${last}" | tr -cd ',' | wc -c)"
  n=$((n + 1))
  [[ "${n}" -ge "${n_sess}" ]]
}

run_entry() {
  local run_id="$1"
  local config="$2"
  local method="$3"
  local n_sess="$4"
  local label="$5"

  local logfile="${LOGROOT}/${method}_seed=2008.log"
  if [[ "${SKIP_DONE}" -eq 1 ]] && log_is_complete "${logfile}" "${n_sess}"; then
    log "SKIP ${method} — complete"
    return 0
  fi

  local -a extra_args=()
  for flag in "${SHARED_FLAGS[@]}"; do
    extra_args+=(--set "${flag}")
  done
  while IFS= read -r flag; do
    extra_args+=(--set "${flag}")
  done < <(mode_flags "${run_id}")

  if [[ "${SMOKE}" -eq 1 ]]; then
    extra_args+=(--max-task 1 --set tuned_epoch=1 --set increment_tuned_epoch=1)
    method="smoke_${method}"
  fi

  log "=== ${method} | ${label} | 2-ep increment, task0-best anchor, last weights ==="
  "${RUNNER[@]}" train_fscil.py \
    --config "${config}" \
    --logdir "${LOGROOT}" \
    --method-name "${method}" \
    --log-layout flat \
    --seeds 2008 \
    "${extra_args[@]}" \
    2>&1 | tee "${LOGROOT}/${method}.log"
  log "=== ${method} done ==="
}

dispatch_idx() {
  local idx="$1"
  IFS='|' read -r run_id config method n_sess label <<< "${RUNS[$idx]}"
  run_entry "${run_id}" "${config}" "${method}" "${n_sess}" "${label}"
}

if [[ -n "${ONLY_IDX}" ]]; then
  dispatch_idx "${ONLY_IDX}"
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  dispatch_idx "${SLURM_ARRAY_TASK_ID}"
else
  for idx in "${!RUNS[@]}"; do
    dispatch_idx "${idx}"
  done
fi

log "cross increment warmup ablation finished"

#!/bin/bash
#SBATCH --job-name=baseonly_buf
#SBATCH --account=EUHPC_D36_199
#SBATCH --partition=boost_usr_prod
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=_experiments/slurm_logs/%A_%a_baseonly_buf.out

# Base-only adapter + buffer (1 old-shot replay per class per increment task).
#
# Adapter SGD only on task 0. Increment sessions: frozen backbone, ridge/cache update,
# cache replay + fscil_old_shot_replay=1 (one sample per old class each task).
#
# Runs 4 jobs via array index:
#   0 = MN40 in-domain        (already done as abl_adapter_mn40_base_only_adapter)
#   1 = SN55 in-domain        (already done as abl_adapter_sn55_base_only_adapter)
#   2 = MN40 -> ScanObjectNN
#   3 = SN55 -> ScanObjectNN
#
# Submit on Leonardo (4 parallel jobs):
#   sbatch scripts/simpl_runs_baseonly_adapter_buffer_campaign.sh
#
# Skip completed logs (recommended — skips in-domain if already run):
#   sbatch scripts/simpl_runs_baseonly_adapter_buffer_campaign.sh
#   # then locally on Leonardo if re-submitting:
#   bash scripts/simpl_runs_baseonly_adapter_buffer_campaign.sh --local --skip-done
#
# Cross-dataset only:
#   sbatch --array=2-3 scripts/simpl_runs_baseonly_adapter_buffer_campaign.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

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
      ONLY_IDX="${2:?--array needs index 0-3}"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      echo "Usage: $0 [--local] [--skip-done] [--smoke] [--array N]" >&2
      exit 1
      ;;
  esac
done

mkdir -p _experiments/slurm_logs

# run_id|logroot|config|method|n_sess|label
RUNS=(
  "mn40_id|_experiments/00_SIMPL_ABL_ADAPTER|configs/selfevo_fscil_mn40_5t_flex_compare.json|abl_adapter_mn40_base_only_adapter|5|MN40 in-domain"
  "sn55_id|_experiments/00_SIMPL_ABL_ADAPTER|configs/selfevo_fscil_sn55_7t_flex_compare.json|abl_adapter_sn55_base_only_adapter|7|SN55 in-domain"
  "mn40_xd|_experiments/00_SCANOBJ_ABL_ADAPTER|configs/selfevo_fscil_mn40_scanobjectnn_4t.json|abl_adapter_mn_scanobj_base_only|4|MN40->ScanObj"
  "sn55_xd|_experiments/00_SCANOBJ_ABL_ADAPTER|configs/selfevo_fscil_sn55_scanobjectnn_4t.json|abl_adapter_sn_scanobj_base_only|4|SN55->ScanObj"
)

# Base-only adapter: warmup at task 0 only, no increment adapter/classifier SGD.
BASE_ONLY_FLAGS=(
  "fscil_adapter_base_only=true"
  "task0_adapter_warmup=true"
  "increment_adapter_warmup=false"
  "fscil_classifier_warmup_after_protos=false"
)

CROSS_TASK1_WARMUP_FLAGS=(
  "fscil_task1_adapter_warmup=true"
  "increment_tuned_epoch=2"
  "fscil_task1_warmup_epochs=2"
  "fscil_restore_task0_adapter_before_increment_warmup=true"
  "fscil_classifier_warmup_after_protos=false"
  "fscil_increment_warmup_pre_and_post=false"
)

# Exemplar buffer: cache proxies + 1 raw old-class shot per class per increment task.
BUFFER_FLAGS=(
  "model_name=selfevo"
  "old_data_source=cache"
  "fscil_old_shot_replay=1"
)

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
  local logroot="$2"
  local config="$3"
  local method="$4"
  local n_sess="$5"
  local label="$6"

  mkdir -p "${logroot}"
  local logfile="${logroot}/${method}_seed=2008.log"

  if [[ "${SKIP_DONE}" -eq 1 ]] && log_is_complete "${logfile}" "${n_sess}"; then
    log "SKIP ${method} — complete (${n_sess} sessions)"
    return 0
  fi

  local -a extra_args=()
  for flag in "${BASE_ONLY_FLAGS[@]}" "${BUFFER_FLAGS[@]}"; do
    extra_args+=(--set "${flag}")
  done
  if [[ "${run_id}" == *"_xd" ]]; then
    for flag in "${CROSS_TASK1_WARMUP_FLAGS[@]}"; do
      extra_args+=(--set "${flag}")
    done
  fi

  if [[ "${SMOKE}" -eq 1 ]]; then
    extra_args+=(
      --max-task 1
      --set tuned_epoch=1
      --set increment_tuned_epoch=1
      --set task0_save_warmup_ckpts=false
    )
    method="smoke_${method}"
  fi

  log "=== [${run_id}] ${method} | ${label} | base-only adapter + buffer (1-shot) ==="
  "${RUNNER[@]}" train_fscil.py \
    --config "${config}" \
    --logdir "${logroot}" \
    --method-name "${method}" \
    --log-layout flat \
    --seeds 2008 \
    "${extra_args[@]}" \
    2>&1 | tee "${logroot}/${method}.log"
  log "=== ${method} done ==="
  rg "CNN top1 curve|FSCIL novel accuracy curve|FSCIL harmonic curve" "${logroot}/${method}"*.log || true
}

dispatch_idx() {
  local idx="$1"
  IFS='|' read -r run_id logroot config method n_sess label <<< "${RUNS[$idx]}"
  run_entry "${run_id}" "${logroot}" "${config}" "${method}" "${n_sess}" "${label}"
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

log "baseonly+buffer campaign finished"

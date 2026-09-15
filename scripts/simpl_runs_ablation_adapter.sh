#!/bin/bash
#SBATCH --job-name=simpl_abl_adapter
#SBATCH --account=EUHPC_D36_199
#SBATCH --partition=boost_usr_prod
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=_experiments/slurm_logs/%A_%a_simpl_abl_adapter.out

# Adapter ablation for FSCIL SIMPL (seed 2008).
#
# Runs 6 jobs via array index:
#   0 = MN40 with adapter warmup
#   1 = MN40 without adapter warmup
#   2 = MN40 base-only adapter (task-0 only)
#   3 = SN55 with adapter warmup
#   4 = SN55 without adapter warmup
#   5 = SN55 base-only adapter (task-0 only)
#
# Submit on Leonardo:
#   sbatch scripts/simpl_runs_ablation_adapter.sh
#
# Optional single run:
#   sbatch --array=1 scripts/simpl_runs_ablation_adapter.sh
#
# Local / sequential:
#   bash scripts/simpl_runs_ablation_adapter.sh --local
#
# Logs:
#   _experiments/00_SIMPL_ABL_ADAPTER/<method>_seed=2008.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

LOGROOT="_experiments/00_SIMPL_ABL_ADAPTER"
RUNNER=(aprun python)
SKIP_DONE=0
ONLY_DATASET=""
ONLY_VARIANT=""
SMOKE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local) RUNNER=(aprun python); shift ;;
    --skip-done) SKIP_DONE=1; shift ;;
    --smoke) SMOKE=1; shift ;;
    --dataset)
      ONLY_DATASET="${2:?--dataset needs mn40 or sn55}"
      shift 2
      ;;
    --variant)
      ONLY_VARIANT="${2:?--variant needs with_adapter|without_adapter|base_only_adapter}"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      echo "Usage: $0 [--local] [--skip-done] [--smoke] [--dataset mn40|sn55] [--variant with_adapter|without_adapter|base_only_adapter]" >&2
      exit 1
      ;;
  esac
done

mkdir -p "${LOGROOT}" _experiments/slurm_logs

# dataset_id|config|method_prefix|expected_sessions
DATASETS=(
  "mn40|configs/selfevo_fscil_mn40_5t_flex_compare.json|abl_adapter_mn40|5"
  "sn55|configs/selfevo_fscil_sn55_7t_flex_compare.json|abl_adapter_sn55|7"
)

# variant_id|label
VARIANTS=(
  "with_adapter|Adapter warmup ON"
  "without_adapter|Adapter warmup OFF"
  "base_only_adapter|Adapter ON only at base task"
)

log() { echo "[$(date -Is)] $*"; }

sets_for_variant() {
  case "$1" in
    with_adapter)
      echo task0_adapter_warmup=true
      echo increment_adapter_warmup=true
      echo fscil_classifier_warmup_after_protos=true
      ;;
    without_adapter)
      echo task0_adapter_warmup=false
      echo increment_adapter_warmup=false
      echo fscil_classifier_warmup_after_protos=false
      echo backbone_type=uni3d_base_patch14_448
      ;;
    base_only_adapter)
      echo fscil_adapter_base_only=true
      echo task0_adapter_warmup=true
      echo increment_adapter_warmup=false
      echo fscil_classifier_warmup_after_protos=false
      ;;
    *)
      echo "Unknown variant: $1" >&2
      return 1
      ;;
  esac
}

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

run_one() {
  local ds_id="$1"
  local config="$2"
  local prefix="$3"
  local n_sess="$4"
  local var_id="$5"
  local var_label="$6"

  if [[ -n "${ONLY_DATASET}" && "${ds_id}" != "${ONLY_DATASET}" ]]; then
    return 0
  fi
  if [[ -n "${ONLY_VARIANT}" && "${var_id}" != "${ONLY_VARIANT}" ]]; then
    return 0
  fi

  local method="${prefix}_${var_id}"
  local logfile="${LOGROOT}/${method}_seed=2008.log"

  if [[ "${SKIP_DONE}" -eq 1 ]] && log_is_complete "${logfile}" "${n_sess}"; then
    log "SKIP ${method} — complete (${n_sess} sessions)"
    return 0
  fi

  local -a extra_args=()
  while IFS= read -r line; do
    [[ -n "${line}" ]] && extra_args+=(--set "${line}")
  done < <(sets_for_variant "${var_id}")

  if [[ "${SMOKE}" -eq 1 ]]; then
    extra_args+=(
      --max-task 1
      --set tuned_epoch=1
      --set increment_tuned_epoch=1
      --set fscil_task1_warmup_epochs=1
      --set fscil_last_task_warmup_epochs=1
      --set task0_save_warmup_ckpts=false
    )
    method="smoke_${method}"
  fi

  log "=== ${method} | ${ds_id} | ${var_label} ==="
  "${RUNNER[@]}" train_fscil.py \
    --config "${config}" \
    --logdir "${LOGROOT}" \
    --method-name "${method}" \
    --log-layout flat \
    --seeds 2008 \
    "${extra_args[@]}" \
    2>&1 | tee "${LOGROOT}/${method}.log"
  log "=== ${method} done ==="
  rg "CNN top1 curve|FSCIL novel accuracy curve|FSCIL harmonic curve" "${LOGROOT}/${method}"*.log || true
}

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  idx="${SLURM_ARRAY_TASK_ID}"
  d_idx=$((idx / 3))
  v_idx=$((idx % 3))
  IFS='|' read -r ds_id config prefix n_sess <<< "${DATASETS[$d_idx]}"
  IFS='|' read -r var_id var_label <<< "${VARIANTS[$v_idx]}"
  run_one "${ds_id}" "${config}" "${prefix}" "${n_sess}" "${var_id}" "${var_label}"
else
  for d_entry in "${DATASETS[@]}"; do
    IFS='|' read -r ds_id config prefix n_sess <<< "${d_entry}"
    for v_entry in "${VARIANTS[@]}"; do
      IFS='|' read -r var_id var_label <<< "${v_entry}"
      run_one "${ds_id}" "${config}" "${prefix}" "${n_sess}" "${var_id}" "${var_label}"
    done
  done
fi

log "simpl_runs_ablation_adapter finished — see ${LOGROOT}/"

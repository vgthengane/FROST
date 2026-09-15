#!/bin/bash
#SBATCH --job-name=abl_adapt_id
#SBATCH --account=EUHPC_D36_199
#SBATCH --partition=boost_usr_prod
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=_experiments/slurm_logs/%A_%a_abl_adapt_id.out

# In-domain adapter campaign (MN40 + SN55, seed 2008).
#
# Compares three adapter regimes at explicit epoch budgets:
#   full_2ep   — task-0: 10 ep, all increments: 2 ep (current flex_compare default)
#   full_10ep  — task-0: 10 ep, all increments: 10 ep (legacy ablation setting)
#   base_only  — task-0: 10 ep, increments: frozen adapter (no SGD)
#
# Array layout (dataset × variant):
#   0 = MN40  full_2ep
#   1 = MN40  full_10ep
#   2 = MN40  base_only
#   3 = SN55  full_2ep
#   4 = SN55  full_10ep
#   5 = SN55  base_only
#
# Submit on Leonardo:
#   sbatch scripts/simpl_runs_adapter_ablation_indomain.sh
#
# Single cell:
#   sbatch --array=0 scripts/simpl_runs_adapter_ablation_indomain.sh
#
# Skip completed:
#   sbatch scripts/simpl_runs_adapter_ablation_indomain.sh  # add --skip-done via env:
#   SKIP_DONE=1 sbatch --export=ALL,SKIP_DONE=1 scripts/simpl_runs_adapter_ablation_indomain.sh
#
# Local:
#   bash scripts/simpl_runs_adapter_ablation_indomain.sh --local --skip-done
#
# Logs: _experiments/00_ADAPTER_CAMPAIGN_ID/<method>_seed=2008.log
# Table: aprun python scripts/summarize_fscil_adapter_campaign.py --scope indomain

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

LOGROOT="_experiments/00_ADAPTER_CAMPAIGN_ID"
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
      ONLY_VARIANT="${2:?--variant needs full_2ep|full_10ep|base_only}"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      echo "Usage: $0 [--local] [--skip-done] [--smoke] [--dataset mn40|sn55] [--variant full_2ep|full_10ep|base_only]" >&2
      exit 1
      ;;
  esac
done

mkdir -p "${LOGROOT}" _experiments/slurm_logs

DATASETS=(
  "mn40|configs/selfevo_fscil_mn40_5t_flex_compare.json|abl_id_mn40|5"
  "sn55|configs/selfevo_fscil_sn55_7t_flex_compare.json|abl_id_sn55|7"
)

VARIANTS=(
  "full_2ep|Full adapter 10/2 (base/novel)"
  "full_10ep|Full adapter 10/10"
  "base_only|Base-only adapter (task-0 only)"
)

log() { echo "[$(date -Is)] $*"; }

flags_for_variant() {
  case "$1" in
    full_2ep)
      echo fscil_adapter_base_only=false
      echo task0_adapter_warmup=true
      echo increment_adapter_warmup=true
      echo fscil_classifier_warmup_after_protos=true
      echo tuned_epoch=10
      echo increment_tuned_epoch=2
      echo fscil_task1_warmup_epochs=2
      echo fscil_last_task_warmup_epochs=2
      ;;
    full_10ep)
      echo fscil_adapter_base_only=false
      echo task0_adapter_warmup=true
      echo increment_adapter_warmup=true
      echo fscil_classifier_warmup_after_protos=true
      echo tuned_epoch=10
      echo increment_tuned_epoch=10
      echo fscil_task1_warmup_epochs=10
      echo fscil_last_task_warmup_epochs=10
      ;;
    base_only)
      echo fscil_adapter_base_only=true
      echo task0_adapter_warmup=true
      echo increment_adapter_warmup=false
      echo fscil_classifier_warmup_after_protos=false
      echo fscil_task1_adapter_warmup=false
      echo tuned_epoch=10
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
  done < <(flags_for_variant "${var_id}")

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

log "adapter ablation (in-domain) finished — see ${LOGROOT}/"

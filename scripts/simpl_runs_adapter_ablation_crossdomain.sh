#!/bin/bash
#SBATCH --job-name=abl_adapt_xd
#SBATCH --account=EUHPC_D36_199
#SBATCH --partition=boost_usr_prod
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=_experiments/slurm_logs/%A_%a_abl_adapt_xd.out

# Cross-dataset adapter campaign (MN40/SN55 -> ScanObjectNN, seed 2008).
#
# Variants:
#   full_2ep    — full adapter, task-0: 10 ep, all increments: 2 ep
#   full_10ep   — full adapter, task-0: 10 ep, all increments: 10 ep
#   base_t1_2ep — base-only + 2-ep adapter warmup on task 1 only (domain shift)
#   base_only   — base-only adapter, no increment warmup
#
# Array layout (dataset × variant):
#   0 = MN40->ScanObj  full_2ep
#   1 = MN40->ScanObj  full_10ep
#   2 = MN40->ScanObj  base_t1_2ep
#   3 = MN40->ScanObj  base_only
#   4 = SN55->ScanObj  full_2ep
#   5 = SN55->ScanObj  full_10ep
#   6 = SN55->ScanObj  base_t1_2ep
#   7 = SN55->ScanObj  base_only
#
# Submit:
#   sbatch scripts/simpl_runs_adapter_ablation_crossdomain.sh
#
# Cross-dataset only (MN40):
#   sbatch --array=0-3 scripts/simpl_runs_adapter_ablation_crossdomain.sh
#
# Local:
#   bash scripts/simpl_runs_adapter_ablation_crossdomain.sh --local --skip-done
#
# Logs: _experiments/00_ADAPTER_CAMPAIGN_XD/<method>_seed=2008.log
# Table: aprun python scripts/summarize_fscil_adapter_campaign.py --scope crossdomain

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

LOGROOT="_experiments/00_ADAPTER_CAMPAIGN_XD"
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
      ONLY_DATASET="${2:?--dataset needs mn_scanobj or sn_scanobj}"
      shift 2
      ;;
    --variant)
      ONLY_VARIANT="${2:?--variant needs full_2ep|full_10ep|base_t1_2ep|base_only}"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      echo "Usage: $0 [--local] [--skip-done] [--smoke] [--dataset mn_scanobj|sn_scanobj] [--variant ...]" >&2
      exit 1
      ;;
  esac
done

mkdir -p "${LOGROOT}" _experiments/slurm_logs

DATASETS=(
  "mn_scanobj|configs/selfevo_fscil_mn40_scanobjectnn_4t.json|abl_xd_mn|4"
  "sn_scanobj|configs/selfevo_fscil_sn55_scanobjectnn_4t.json|abl_xd_sn|4"
)

VARIANTS=(
  "full_2ep|Full adapter 10/2 (all increments)"
  "full_10ep|Full adapter 10/10 (all increments)"
  "base_t1_2ep|Base-only + task-1 adapter 2 ep"
  "base_only|Base-only (task-0 only)"
)

log() { echo "[$(date -Is)] $*"; }

flags_for_variant() {
  case "$1" in
    full_2ep)
      echo fscil_adapter_base_only=false
      echo task0_adapter_warmup=true
      echo increment_adapter_warmup=true
      echo fscil_classifier_warmup_after_protos=true
      echo fscil_task1_adapter_warmup=false
      echo tuned_epoch=10
      echo increment_tuned_epoch=2
      echo fscil_task1_warmup_epochs=2
      echo fscil_last_task_warmup_epochs=2
      echo fscil_restore_task0_adapter_before_increment_warmup=true
      ;;
    full_10ep)
      echo fscil_adapter_base_only=false
      echo task0_adapter_warmup=true
      echo increment_adapter_warmup=true
      echo fscil_classifier_warmup_after_protos=true
      echo fscil_task1_adapter_warmup=false
      echo tuned_epoch=10
      echo increment_tuned_epoch=10
      echo fscil_task1_warmup_epochs=10
      echo fscil_last_task_warmup_epochs=10
      echo fscil_restore_task0_adapter_before_increment_warmup=true
      ;;
    base_t1_2ep)
      echo fscil_adapter_base_only=true
      echo task0_adapter_warmup=true
      echo fscil_task1_adapter_warmup=true
      echo increment_adapter_warmup=true
      echo fscil_classifier_warmup_after_protos=false
      echo fscil_increment_warmup_pre_and_post=false
      echo tuned_epoch=10
      echo increment_tuned_epoch=2
      echo fscil_task1_warmup_epochs=2
      echo fscil_last_task_warmup_epochs=2
      echo fscil_restore_task0_adapter_before_increment_warmup=true
      ;;
    base_only)
      echo fscil_adapter_base_only=true
      echo task0_adapter_warmup=true
      echo fscil_task1_adapter_warmup=false
      echo increment_adapter_warmup=false
      echo fscil_classifier_warmup_after_protos=false
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
  d_idx=$((idx / 4))
  v_idx=$((idx % 4))
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

log "adapter ablation (cross-domain) finished — see ${LOGROOT}/"

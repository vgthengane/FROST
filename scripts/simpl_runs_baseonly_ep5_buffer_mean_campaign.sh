#!/bin/bash
#SBATCH --job-name=ep5_mean_buf
#SBATCH --account=EUHPC_D36_199
#SBATCH --partition=boost_usr_prod
#SBATCH --array=0-11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=_experiments/slurm_logs/%A_%a_ep5_mean_buf.out

# Base-only adapter warmup (5 epochs) + exemplar buffer vs buffer-free.
# KeyCache: one MEAN prototype per class (NOT per-shot few-shot features).
#
# Distinct from simpl_runs_baseonly_ep5_buffer_campaign.sh (per-shot KeyCache).
#
# Protocols (6):
#   MN40, SN55, CO3D in-domain
#   MN40->ScanObjectNN, SN55->ScanObjectNN, SN55->CO3D cross-domain
#
# Variants (2 per protocol, 12 jobs total):
#   buffer      — fscil_preset=canonical (1 old-shot exemplar per old class)
#   buffer-free — model_name=selfevo_bufferfree (no old-class disk re-read)
#
# Array layout: idx = protocol_idx * 2 + variant_idx
#   variant 0 = buffer, 1 = buffer-free
#
# Submit on Leonardo (12 parallel jobs):
#   sbatch scripts/simpl_runs_baseonly_ep5_buffer_mean_campaign.sh
#
# Buffer only (6 jobs):
#   sbatch --array=0,2,4,6,8,10 scripts/simpl_runs_baseonly_ep5_buffer_mean_campaign.sh
#
# Buffer-free only (6 jobs):
#   sbatch --array=1,3,5,7,9,11 scripts/simpl_runs_baseonly_ep5_buffer_mean_campaign.sh
#
# Local:
#   bash scripts/simpl_runs_baseonly_ep5_buffer_mean_campaign.sh --local
#   bash scripts/simpl_runs_baseonly_ep5_buffer_mean_campaign.sh --local --skip-done
#   bash scripts/simpl_runs_baseonly_ep5_buffer_mean_campaign.sh --local --only-protocol mn40
#   bash scripts/simpl_runs_baseonly_ep5_buffer_mean_campaign.sh --local --only-variant bufferfree
#
# Logs: _experiments/00_BUF_CAMPAIGN_EP5_MEAN/<method>_seed=2008.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

RUNNER=(aprun python)
LOGROOT="_experiments/00_BUF_CAMPAIGN_EP5_MEAN"
SEED=2008
SKIP_DONE=0
ONLY_IDX=""
ONLY_PROTOCOL=""
ONLY_VARIANT=""
SMOKE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local) RUNNER=(aprun python); shift ;;
    --skip-done) SKIP_DONE=1; shift ;;
    --smoke) SMOKE=1; shift ;;
    --array)
      ONLY_IDX="${2:?--array needs index 0-11}"
      shift 2
      ;;
    --only-protocol)
      ONLY_PROTOCOL="${2:?--only-protocol needs mn40|sn55|co3d|mn_scanobj|sn_scanobj|sn_co3d}"
      shift 2
      ;;
    --only-variant)
      ONLY_VARIANT="${2:?--only-variant needs buffer|bufferfree}"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      echo "Usage: $0 [--local] [--skip-done] [--smoke] [--array N]" >&2
      echo "       [--only-protocol mn40|sn55|co3d|mn_scanobj|sn_scanobj|sn_co3d]" >&2
      echo "       [--only-variant buffer|bufferfree]" >&2
      exit 1
      ;;
  esac
done

mkdir -p "${LOGROOT}" _experiments/slurm_logs

# run_id|config|method_buf|method_bf|n_sess|label|cross
PROTOCOLS=(
  "mn40|configs/selfevo_fscil_mn40_5t_flex_compare.json|abl_mn40_ep5_mean_buf|abl_mn40_ep5_mean_buffree|5|MN40 in-domain|0"
  "sn55|configs/selfevo_fscil_sn55_7t_flex_compare.json|abl_sn55_ep5_mean_buf|abl_sn55_ep5_mean_buffree|7|SN55 in-domain|0"
  "co3d|configs/selfevo_fscil_co3d_6t.json|abl_co3d_ep5_mean_buf|abl_co3d_ep5_mean_buffree|6|CO3D in-domain|0"
  "mn_scanobj|configs/selfevo_fscil_mn40_scanobjectnn_4t.json|abl_mn_scanobj_ep5_mean_buf|abl_mn_scanobj_ep5_mean_buffree|4|MN40->ScanObj|1"
  "sn_scanobj|configs/selfevo_fscil_sn55_scanobjectnn_4t.json|abl_sn_scanobj_ep5_mean_buf|abl_sn_scanobj_ep5_mean_buffree|4|SN55->ScanObj|1"
  "sn_co3d|configs/selfevo_fscil_sn55_co3d_11t.json|abl_sn_co3d_ep5_mean_buf|abl_sn_co3d_ep5_mean_buffree|11|SN55->CO3D|1"
)

BASE_ONLY_FLAGS=(
  "fscil_adapter_base_only=true"
  "task0_adapter_warmup=true"
  "increment_adapter_warmup=false"
  "fscil_classifier_warmup_after_protos=false"
  "tuned_epoch=5"
)

MEAN_PROTO_FLAGS=(
  "fscil_shot_prototypes=false"
  "fscil_cluster_new_classes=false"
  "insert_strategy=mean"
)

CROSS_TASK1_WARMUP_FLAGS=(
  "fscil_task1_adapter_warmup=true"
  "increment_tuned_epoch=2"
  "fscil_task1_warmup_epochs=2"
  "fscil_restore_task0_adapter_before_increment_warmup=true"
  "fscil_classifier_warmup_after_protos=false"
  "fscil_increment_warmup_pre_and_post=false"
)

BUFFER_FLAGS=(
  "model_name=selfevo"
  "fscil_preset=canonical"
)

BUFFERFREE_FLAGS=(
  "model_name=selfevo_bufferfree"
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

run_one() {
  local run_id="$1"
  local config="$2"
  local method="$3"
  local n_sess="$4"
  local label="$5"
  local cross="$6"
  local variant="$7"
  shift 7
  local -a variant_flags=("$@")

  mkdir -p "${LOGROOT}"
  local logfile="${LOGROOT}/${method}_seed=${SEED}.log"

  if [[ "${SKIP_DONE}" -eq 1 ]] && log_is_complete "${logfile}" "${n_sess}"; then
    log "SKIP ${method} — complete (${n_sess} sessions)"
    return 0
  fi

  local -a extra_args=()
  for flag in "${BASE_ONLY_FLAGS[@]}" "${MEAN_PROTO_FLAGS[@]}" "${variant_flags[@]}"; do
    extra_args+=(--set "${flag}")
  done
  if [[ "${cross}" -eq 1 ]]; then
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

  log "=== [${run_id}] ${method} | ${label} | ${variant} | base-only ep5 mean-proto ==="
  "${RUNNER[@]}" train_fscil.py \
    --config "${config}" \
    --logdir "${LOGROOT}" \
    --method-name "${method}" \
    --log-layout flat \
    --seeds "${SEED}" \
    "${extra_args[@]}" \
    2>&1 | tee "${LOGROOT}/${method}.log"
  log "=== ${method} done ==="
  rg "CNN top1 curve|FSCIL novel accuracy curve|FSCIL harmonic curve" \
    "${LOGROOT}/${method}"*.log || true
}

dispatch_idx() {
  local idx="$1"
  local proto_idx=$((idx / 2))
  local variant_idx=$((idx % 2))

  IFS='|' read -r run_id config method_buf method_bf n_sess label cross <<< "${PROTOCOLS[$proto_idx]}"

  if [[ -n "${ONLY_PROTOCOL}" && "${run_id}" != "${ONLY_PROTOCOL}" ]]; then
    return 0
  fi

  local variant method
  local -a variant_flags=()
  if [[ "${variant_idx}" -eq 0 ]]; then
    variant="buffer"
    method="${method_buf}"
    variant_flags=("${BUFFER_FLAGS[@]}")
  else
    variant="buffer-free"
    method="${method_bf}"
    variant_flags=("${BUFFERFREE_FLAGS[@]}")
  fi

  if [[ -n "${ONLY_VARIANT}" ]]; then
    case "${ONLY_VARIANT}" in
      buffer) [[ "${variant_idx}" -eq 0 ]] || return 0 ;;
      bufferfree) [[ "${variant_idx}" -eq 1 ]] || return 0 ;;
      *) echo "Unknown --only-variant ${ONLY_VARIANT}" >&2; exit 1 ;;
    esac
  fi

  run_one "${run_id}" "${config}" "${method}" "${n_sess}" "${label}" "${cross}" "${variant}" \
    "${variant_flags[@]}"
}

if [[ -n "${ONLY_IDX}" ]]; then
  dispatch_idx "${ONLY_IDX}"
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  dispatch_idx "${SLURM_ARRAY_TASK_ID}"
else
  for idx in $(seq 0 11); do
    dispatch_idx "${idx}"
  done
fi

log "baseonly ep5 buffer mean-proto campaign finished"

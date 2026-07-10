#!/bin/bash
#SBATCH --job-name=simpl_abl_adapt_xd
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
#SBATCH --output=_experiments/slurm_logs/%A_%a_simpl_abl_adapt_xd.out

# DEPRECATED — use the unified buffer-free campaign instead:
#   sbatch scripts/simpl_runs_baseonly_adapter_campaign.sh
#   (array indices 2-3 = cross-dataset MN40/SN55)
#
# Cross-dataset adapter ablation: base-only adapter warmup (seed 2008).
#
# Runs 4 jobs via array index:
#   0 = MN40 -> ScanObjectNN, buffer (cache replay)
#   1 = SN55 -> ScanObjectNN, buffer (cache replay)
#   2 = MN40 -> ScanObjectNN, buffer-free
#   3 = SN55 -> ScanObjectNN, buffer-free
#
# Submit on Leonardo:
#   sbatch scripts/simpl_runs_ablation_adapter_scanobjectnn.sh
#
# Optional single run:
#   sbatch --array=1 scripts/simpl_runs_ablation_adapter_scanobjectnn.sh
#
# Local / sequential:
#   bash scripts/simpl_runs_ablation_adapter_scanobjectnn.sh --local
#
# Smoke (task 0 only):
#   bash scripts/simpl_runs_ablation_adapter_scanobjectnn.sh --local --smoke
#
# Logs:
#   _experiments/00_SCANOBJ_ABL_ADAPTER/<method>.log
#   _experiments/00_SCANOBJ_ABL_ADAPTER/<method>_seed=2008.log  (flat layout)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

LOGROOT="_experiments/00_SCANOBJ_ABL_ADAPTER"
RUNNER=(aprun python)
SKIP_DONE=0
ONLY_DATASET=""
ONLY_MODE=""
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
    --mode)
      ONLY_MODE="${2:?--mode needs buffer or bufferfree}"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      echo "Usage: $0 [--local] [--skip-done] [--smoke] [--dataset mn_scanobj|sn_scanobj] [--mode buffer|bufferfree]" >&2
      exit 1
      ;;
  esac
done

mkdir -p "${LOGROOT}" _experiments/slurm_logs

# dataset_id|config|method_prefix|expected_sessions
DATASETS=(
  "mn_scanobj|configs/selfevo_fscil_mn40_scanobjectnn_4t.json|abl_adapter_mn_scanobj|4"
  "sn_scanobj|configs/selfevo_fscil_sn55_scanobjectnn_4t.json|abl_adapter_sn_scanobj|4"
)

# mode_id|label|model_name
MODES=(
  "buffer|Buffer (cache replay)|selfevo"
  "bufferfree|Buffer-free|selfevo_bufferfree"
)

# Base-only adapter flags (shared).
BASE_ONLY_FLAGS=(
  "fscil_adapter_base_only=true"
  "task0_adapter_warmup=true"
  "increment_adapter_warmup=false"
  "fscil_classifier_warmup_after_protos=false"
)

log() { echo "[$(date -Is)] $*"; }

sets_for_mode() {
  case "$1" in
    buffer)
      ;;
    bufferfree)
      echo model_name=selfevo_bufferfree
      ;;
    *)
      echo "Unknown mode: $1" >&2
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
  local mode_id="$5"
  local mode_label="$6"
  local model_name="$7"

  if [[ -n "${ONLY_DATASET}" && "${ds_id}" != "${ONLY_DATASET}" ]]; then
    return 0
  fi
  if [[ -n "${ONLY_MODE}" && "${mode_id}" != "${ONLY_MODE}" ]]; then
    return 0
  fi

  local method="${prefix}_base_only"
  if [[ "${mode_id}" == "bufferfree" ]]; then
    method="${method}_bf"
  fi
  local logfile="${LOGROOT}/${method}_seed=2008.log"

  if [[ "${SKIP_DONE}" -eq 1 ]] && log_is_complete "${logfile}" "${n_sess}"; then
    log "SKIP ${method} — complete (${n_sess} sessions)"
    return 0
  fi

  local -a extra_args=(
    --set "model_name=${model_name}"
  )
  for flag in "${BASE_ONLY_FLAGS[@]}"; do
    extra_args+=(--set "${flag}")
  done
  while IFS= read -r line; do
    [[ -n "${line}" ]] && extra_args+=(--set "${line}")
  done < <(sets_for_mode "${mode_id}")

  if [[ "${SMOKE}" -eq 1 ]]; then
    extra_args+=(
      --max-task 1
      --set tuned_epoch=1
      --set increment_tuned_epoch=1
      --set task0_save_warmup_ckpts=false
    )
    method="smoke_${method}"
  fi

  log "=== ${method} | ${ds_id} | ${mode_label} | base-only adapter ==="
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
  d_idx=$((idx / 2))
  m_idx=$((idx % 2))
  IFS='|' read -r ds_id config prefix n_sess <<< "${DATASETS[$d_idx]}"
  IFS='|' read -r mode_id mode_label model_name <<< "${MODES[$m_idx]}"
  run_one "${ds_id}" "${config}" "${prefix}" "${n_sess}" "${mode_id}" "${mode_label}" "${model_name}"
else
  for d_entry in "${DATASETS[@]}"; do
    IFS='|' read -r ds_id config prefix n_sess <<< "${d_entry}"
    for m_entry in "${MODES[@]}"; do
      IFS='|' read -r mode_id mode_label model_name <<< "${m_entry}"
      run_one "${ds_id}" "${config}" "${prefix}" "${n_sess}" "${mode_id}" "${mode_label}" "${model_name}"
    done
  done
fi

log "simpl_runs_ablation_adapter_scanobjectnn finished — see ${LOGROOT}/"

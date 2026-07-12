#!/bin/bash
#SBATCH --job-name=h3_buffree_max
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
#SBATCH --output=_experiments/slurm_logs/%A_%a_h3_buffree_max.out

# Locked PRidE main recipe (buffer-free): hybrid3_st + max pooling on all 6 protocols.
# See FINAL_METHOD_BUFFERFREE.md / _notes/pride_final.md.
# 6 jobs only (no ablations).
#
# Logs: _experiments/00_FINAL_EP5_MAX_POOL/final_<proto>_hybrid3_st_buffree_max.log
#
# Submit:
#   sbatch scripts/simpl_runs_final_hybrid3_buffree_max.sh
#
# Local:
#   bash scripts/simpl_runs_final_hybrid3_buffree_max.sh --local
#   bash scripts/simpl_runs_final_hybrid3_buffree_max.sh --local --skip-done
#   bash scripts/simpl_runs_final_hybrid3_buffree_max.sh --local --only-protocol mn40

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

RUNNER=(aprun python)
LOGROOT="_experiments/00_FINAL_EP5_MAX_POOL"
SEED=2008
SKIP_DONE=0
ONLY_IDX=""
ONLY_PROTOCOL=""
SMOKE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local) RUNNER=(aprun python); shift ;;
    --skip-done) SKIP_DONE=1; shift ;;
    --smoke) SMOKE=1; shift ;;
    --array)
      ONLY_IDX="${2:?}"
      shift 2
      ;;
    --only-protocol)
      ONLY_PROTOCOL="${2:?}"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "${LOGROOT}" _experiments/slurm_logs

# run_id|config|prefix|n_sess|label
JOBS=(
  "mn40|configs/selfevo_fscil_mn40_5t_flex_compare.json|mn40|5|MN40"
  "sn55|configs/selfevo_fscil_sn55_7t_flex_compare.json|sn55|7|SN55"
  "co3d|configs/selfevo_fscil_co3d_6t.json|co3d|6|CO3D"
  "mn_scanobj|configs/selfevo_fscil_mn40_scanobjectnn_4t.json|mn_scanobj|4|MN→ScanObj"
  "sn_scanobj|configs/selfevo_fscil_sn55_scanobjectnn_4t.json|sn_scanobj|4|SN→ScanObj"
  "sn_co3d|configs/selfevo_fscil_sn55_co3d_11t.json|sn_co3d|11|SN→CO3D"
)

BASE_FLAGS=(
  "fscil_adapter_base_only=true"
  "task0_adapter_warmup=true"
  "increment_adapter_warmup=false"
  "fscil_classifier_warmup_after_protos=false"
  "fscil_task1_adapter_warmup=false"
  "tuned_epoch=5"
  "conflict_replay=true"
  "conflict_replay_on_test=false"
  "conflict_replay_use_test_cm=false"
  "model_name=selfevo_bufferfree"
  "prototype_pooling=max"
  "pooling_temperature=0.07"
  "fscil_shot_prototypes=false"
  "fscil_cluster_new_classes=false"
  "fscil_mean_best2_prototypes=false"
  "fscil_mean_close_far_prototypes=true"
  "fscil_mean_nearmean_prototypes=false"
  "insert_strategy=mean"
  "max_prototypes_per_class=3"
  "fscil_increment_max_prototypes=3"
  "evolution_policy=static"
)

log() { echo "[$(date -Is)] $*"; }

log_is_complete() {
  local logfile="$1"
  local n_sess="$2"
  [[ -f "${logfile}" ]] || return 1
  local last
  last="$(grep "CNN top1 curve:" "${logfile}" 2>/dev/null | tail -n 1 || true)"
  [[ -n "${last}" ]] || return 1
  local n
  n="$(echo "${last}" | tr -cd ',' | wc -c)"
  n=$((n + 1))
  [[ "${n}" -ge "${n_sess}" ]]
}

run_idx() {
  local idx="$1"
  IFS='|' read -r run_id config prefix n_sess label <<< "${JOBS[$idx]}"

  if [[ -n "${ONLY_PROTOCOL}" && "${run_id}" != "${ONLY_PROTOCOL}" ]]; then
    return 0
  fi

  local method="final_${prefix}_hybrid3_st_buffree_max"
  local logfile="${LOGROOT}/${method}.log"

  if [[ "${SKIP_DONE}" -eq 1 ]] && log_is_complete "${logfile}" "${n_sess}"; then
    log "SKIP ${method}"
    return 0
  fi

  local -a extra_args=()
  for flag in "${BASE_FLAGS[@]}"; do
    extra_args+=(--set "${flag}")
  done

  if [[ "${SMOKE}" -eq 1 ]]; then
    extra_args+=(
      --max-task 1
      --set tuned_epoch=1
      --set task0_save_warmup_ckpts=false
    )
    method="smoke_${method}"
    logfile="${LOGROOT}/${method}.log"
  fi

  log "=== [${idx}] ${method} | ${label} | hybrid3_st | buffree | max ==="
  "${RUNNER[@]}" train_fscil.py \
    --config "${config}" \
    --logdir "${LOGROOT}" \
    --method-name "${method}" \
    --log-layout flat \
    --seeds "${SEED}" \
    "${extra_args[@]}" \
    2>&1 | tee "${logfile}"
  log "=== ${method} done ==="
}

if [[ -n "${ONLY_IDX}" ]]; then
  run_idx "${ONLY_IDX}"
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  run_idx "${SLURM_ARRAY_TASK_ID}"
else
  for idx in $(seq 0 5); do
    run_idx "${idx}"
  done
fi

log "hybrid3_st buffree+max campaign finished → ${LOGROOT}"

#!/bin/bash
#SBATCH --job-name=pride_a05_ep10
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
#SBATCH --output=_experiments/slurm_logs/%A_%a_pride_a05_ep10.out

# =============================================================================
# PRidE FINAL — α=0.5 buffer-free, all 6 FSCIL protocols (EP10)
# =============================================================================
# Locked main paper recipe (`_notes/pride_final.md`):
#   ridge_cache_alpha = fscil_increment_ridge_alpha = fscil_task1_ridge_alpha = 0.5
#   fscil_ridge_alpha_ramp = false
#   → flat α=0.5 on session 0 and every increment
#
# Array map:
#   0  mn40       configs/selfevo_fscil_mn40_5t_flex_compare.json
#   1  sn55       configs/selfevo_fscil_sn55_7t_flex_compare.json
#   2  co3d       configs/selfevo_fscil_co3d_6t.json
#   3  mn_scanobj configs/selfevo_fscil_mn40_scanobjectnn_4t.json
#   4  sn_scanobj configs/selfevo_fscil_sn55_scanobjectnn_4t.json
#   5  sn_co3d    configs/selfevo_fscil_sn55_co3d_11t.json
#
# Shared recipe:
#   buffer-free | mean_close_st (mean + two close) | max pool | static | base-only
#   tuned_epoch=10 | conflict train-CM only | seed=2008 | Uni3D-B frozen
#
# Logs:
#   _experiments/00_ABLATION_ALPHA05_EP10/final10_<proto>_mean_close_st_buffree_a05.log
#
# Buffer twin (same α=0.5, exemplar buffer on):
#   scripts/simpl_runs_pride_alpha05_buf_ep10.sh
#   → _experiments/00_ABLATION_ALPHA05_BUF_EP10/final10_<proto>_mean_close_st_buf_a05.log
#
# On Leonardo (after code sync):
#   bash sync_eurohpc.sh --code
#   sbatch scripts/simpl_runs_pride_alpha05_ep10.sh
# One protocol:
#   bash scripts/simpl_runs_pride_alpha05_ep10.sh --local --only-protocol mn_scanobj
# Skip finished:
#   bash scripts/simpl_runs_pride_alpha05_ep10.sh --local --skip-done
# =============================================================================

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

RUNNER=(aprun python)
LOGROOT="_experiments/00_ABLATION_ALPHA05_EP10"
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
  "tuned_epoch=10"
  "conflict_replay=true"
  "conflict_replay_on_test=false"
  "conflict_replay_use_test_cm=false"
  "model_name=selfevo_bufferfree"
  "prototype_pooling=max"
  "pooling_temperature=0.07"
  "fscil_shot_prototypes=false"
  "fscil_cluster_new_classes=false"
  "fscil_mean_best2_prototypes=true"
  "fscil_mean_close_far_prototypes=false"
  "fscil_mean_nearmean_prototypes=false"
  "fscil_mean_far2_prototypes=false"
  "insert_strategy=mean"
  "max_prototypes_per_class=3"
  "fscil_increment_max_prototypes=3"
  "evolution_policy=static"
  "classifier_mode=ridge_cache"
  "ridge_cache_alpha=0.5"
  "fscil_increment_ridge_alpha=0.5"
  "fscil_task1_ridge_alpha=0.5"
  "fscil_ridge_alpha_ramp=false"
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

  local method="final10_${prefix}_mean_close_st_buffree_a05"
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

  log "=== [${idx}] ${method} | ${label} | mean_close_st | buffree | max | ep10 | α=0.5 ==="
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

log "α=0.5 ep10 campaign finished → ${LOGROOT}"

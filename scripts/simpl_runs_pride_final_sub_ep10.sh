#!/bin/bash
#SBATCH --job-name=pride_fsub_bf
#SBATCH --account=EUHPC_D36_199
#SBATCH --partition=boost_usr_prod
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=_experiments/slurm_logs/%A_%a_pride_fsub_bf.out

# =============================================================================
# PRidE final_sub — mean-only proto bank, α=0.5, rp=2000, EP10 (buffer-free)
# =============================================================================
# Seed 2008 first pass over all 6 FSCIL protocols (3 in-domain + 3 cross).
#
# Recipe vs locked mean_close_st main:
#   insert = mean only (1 key/class)
#   ridge_rp_dim = 2000   (G dim)
#   ridge_cache α = 0.5 flat
#   tuned_epoch = 10, static, base-only, max pool, conflict train-CM
#   buffer-free
#
# Array map:
#   0  mn40       1  sn55       2  co3d
#   3  mn_scanobj 4  sn_scanobj 5  sn_co3d
#
# Logs:
#   _experiments/00_FINAL_SUB_MEAN_A05_RP2000_EP10/
#     final_sub_<proto>_mean_st_buffree_a05_rp2000.log
#
# Buffer twin:
#   sbatch scripts/simpl_runs_pride_final_sub_buf_ep10.sh
# Multi-seed (2008 first, then 1947,1991,2001,2019):
#   sbatch scripts/simpl_runs_pride_final_sub_ep10_multiseed.sh
#
# On Leonardo:
#   bash sync_eurohpc.sh --code
#   sbatch scripts/simpl_runs_pride_final_sub_ep10.sh
# Local one protocol:
#   bash scripts/simpl_runs_pride_final_sub_ep10.sh --local --only-protocol mn40
# Skip finished:
#   bash scripts/simpl_runs_pride_final_sub_ep10.sh --local --skip-done
# =============================================================================

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

RUNNER=(aprun python)
LOGROOT="_experiments/00_FINAL_SUB_MEAN_A05_RP2000_EP10"
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
  "fscil_mean_best2_prototypes=false"
  "fscil_mean_close_far_prototypes=false"
  "fscil_mean_nearmean_prototypes=false"
  "fscil_mean_far2_prototypes=false"
  "insert_strategy=mean"
  "max_prototypes_per_class=1"
  "fscil_increment_max_prototypes=1"
  "evolution_policy=static"
  "classifier_mode=ridge_cache"
  "ridge_cache_alpha=0.5"
  "fscil_increment_ridge_alpha=0.5"
  "fscil_task1_ridge_alpha=0.5"
  "fscil_ridge_alpha_ramp=false"
  "ridge_rp_dim=2000"
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

  local method="final_sub_${prefix}_mean_st_buffree_a05_rp2000"
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

  log "=== [${idx}] ${method} | ${label} | mean_st | buffree | ep10 | α=0.5 | rp=2000 ==="
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

log "final_sub buffree ep10 (seed ${SEED}) finished → ${LOGROOT}"

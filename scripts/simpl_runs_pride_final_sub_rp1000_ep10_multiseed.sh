#!/bin/bash
#SBATCH --job-name=pride_fsub1k_ms
#SBATCH --account=EUHPC_D36_199
#SBATCH --partition=boost_usr_prod
#SBATCH --array=0-29
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=_experiments/slurm_logs/%A_%a_pride_fsub1k_ms.out

# =============================================================================
# PRidE final_sub multi-seed — mean-only, α=0.5, rp=1000, EP10 (buffer-free)
# =============================================================================
# Same recipe as rp=2000 final_sub, but ridge_rp_dim=1000.
# All 5 paper seeds × 6 protocols in one array (submit together).
#
# Array layout (30 jobs = 6 protocols × 5 seeds):
#   task_id = seed_idx * 6 + protocol_idx
#   seed order: 1947, 1991, 2001, 2008, 2019
#   protocol_idx: 0..5  (mn40, sn55, co3d, mn_scanobj, sn_scanobj, sn_co3d)
#
# Logs:
#   _experiments/00_FINAL_SUB_MEAN_A05_RP1000_EP10_5SEEDS/
#     final_sub_<proto>_mean_st_buffree_a05_rp1000_seed=<seed>.log
#
# Buffer twin (submit both at once):
#   sbatch scripts/simpl_runs_pride_final_sub_rp1000_ep10_multiseed.sh
#   sbatch scripts/simpl_runs_pride_final_sub_rp1000_buf_ep10_multiseed.sh
#
# Walltime note: 08:00:00 default (02:00:00 was too short for SN55 / SN→CO3D).
# Override: sbatch --time=12:00:00 ...
#
# Summarize:
#   aprun python scripts/summarize_pride_multiseed.py --variant final_sub_rp1000_both
# =============================================================================

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

RUNNER=(aprun python)
LOGROOT="_experiments/00_FINAL_SUB_MEAN_A05_RP1000_EP10_5SEEDS"
DEFAULT_SEEDS=(1947 1991 2001 2008 2019)
SEEDS=("${DEFAULT_SEEDS[@]}")
SKIP_DONE=0
ONLY_IDX=""
ONLY_PROTOCOL=""
ONLY_SEED=""
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
    --only-seed)
      ONLY_SEED="${2:?}"
      shift 2
      ;;
    --seeds)
      IFS=',' read -r -a SEEDS <<< "${2:?}"
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
N_PROTO=${#JOBS[@]}
N_SEED=${#SEEDS[@]}

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
  "ridge_rp_dim=1000"
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

run_pair() {
  local proto_idx="$1"
  local seed="$2"

  IFS='|' read -r run_id config prefix n_sess label <<< "${JOBS[$proto_idx]}"

  if [[ -n "${ONLY_PROTOCOL}" && "${run_id}" != "${ONLY_PROTOCOL}" ]]; then
    return 0
  fi
  if [[ -n "${ONLY_SEED}" && "${seed}" != "${ONLY_SEED}" ]]; then
    return 0
  fi

  local method="final_sub_${prefix}_mean_st_buffree_a05_rp1000"
  local logfile="${LOGROOT}/${method}_seed=${seed}.log"

  if [[ "${SKIP_DONE}" -eq 1 ]] && log_is_complete "${logfile}" "${n_sess}"; then
    log "SKIP ${method} seed=${seed}"
    return 0
  fi

  local -a extra_args=()
  for flag in "${BASE_FLAGS[@]}"; do
    extra_args+=(--set "${flag}")
  done
  extra_args+=(
    --set "fscil_old_shot_replay_seed=${seed}"
    --set "fscil_base_ridge_refresh_seed=${seed}"
  )

  if [[ "${SMOKE}" -eq 1 ]]; then
    extra_args+=(
      --max-task 1
      --set tuned_epoch=1
      --set task0_save_warmup_ckpts=false
    )
    method="smoke_${method}"
    logfile="${LOGROOT}/${method}_seed=${seed}.log"
  fi

  log "=== [${proto_idx}] ${method} | ${label} | seed=${seed} | buffree | ep10 | α=0.5 | rp=1000 ==="
  "${RUNNER[@]}" train_fscil.py \
    --config "${config}" \
    --logdir "${LOGROOT}" \
    --method-name "${method}" \
    --log-layout flat \
    --seeds "${seed}" \
    "${extra_args[@]}" \
    2>&1 | tee "${logfile}"
  log "=== ${method} seed=${seed} done ==="
}

run_flat_idx() {
  local flat="$1"
  local max=$((N_PROTO * N_SEED))
  if [[ "${flat}" -lt 0 || "${flat}" -ge "${max}" ]]; then
    echo "Array index ${flat} out of range [0, $((max - 1))] (N_PROTO=${N_PROTO} N_SEED=${N_SEED})" >&2
    exit 1
  fi
  local seed_idx=$((flat / N_PROTO))
  local proto_idx=$((flat % N_PROTO))
  run_pair "${proto_idx}" "${SEEDS[$seed_idx]}"
}

if [[ -n "${ONLY_IDX}" ]]; then
  run_flat_idx "${ONLY_IDX}"
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  run_flat_idx "${SLURM_ARRAY_TASK_ID}"
else
  for seed in "${SEEDS[@]}"; do
    for proto_idx in $(seq 0 $((N_PROTO - 1))); do
      run_pair "${proto_idx}" "${seed}"
    done
  done
fi

log "final_sub rp1000 multi-seed buffree finished → ${LOGROOT}"

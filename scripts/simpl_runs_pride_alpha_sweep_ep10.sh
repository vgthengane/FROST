#!/bin/bash
#SBATCH --job-name=pride_a_sweep
#SBATCH --account=EUHPC_D36_199
#SBATCH --partition=boost_usr_prod
#SBATCH --array=0-29
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=_experiments/slurm_logs/%A_%a_pride_a_sweep_ep10.out

# =============================================================================
# PRidE flat fusion-α sweep — EP10 paper final recipe (buffer-free)
# =============================================================================
# Alphas: 0.8, 0.7, 0.6, 0.4, 0.3  (α=0.5 has its own launcher:
#   scripts/simpl_runs_pride_alpha05_ep10.sh)
#
# For each α, fusion is FLAT on all sessions:
#   ridge_cache_alpha = fscil_increment_ridge_alpha = fscil_task1_ridge_alpha = α
#   fscil_ridge_alpha_ramp = false
#
# Array layout (30 jobs = 5 alphas × 6 protocols):
#   idx = alpha_i * 6 + proto_i
#   alpha_i: 0→0.8, 1→0.7, 2→0.6, 3→0.4, 4→0.3
#   proto_i: 0 mn40, 1 sn55, 2 co3d, 3 mn_scanobj, 4 sn_scanobj, 5 sn_co3d
#
# Shared recipe (same as locked final10):
#   buffer-free | mean_close_st | max | static | base-only | tuned_epoch=10
#   conflict train-CM only | seed=2008 | Uni3D-B frozen
#
# Logs:
#   _experiments/00_ABLATION_ALPHA_SWEEP_EP10/final10_<proto>_mean_close_st_buffree_a<tag>.log
#   tag: a08 a07 a06 a04 a03
#
# On Leonardo:
#   bash sync_eurohpc.sh --code
#   sbatch scripts/simpl_runs_pride_alpha_sweep_ep10.sh
#
# Local / one α / one protocol:
#   bash scripts/simpl_runs_pride_alpha_sweep_ep10.sh --local --only-alpha 0.7
#   bash scripts/simpl_runs_pride_alpha_sweep_ep10.sh --local --only-alpha 0.4 --only-protocol mn40
#   bash scripts/simpl_runs_pride_alpha_sweep_ep10.sh --local --skip-done
# =============================================================================

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

RUNNER=(aprun python)
LOGROOT="_experiments/00_ABLATION_ALPHA_SWEEP_EP10"
SEED=2008
SKIP_DONE=0
ONLY_IDX=""
ONLY_PROTOCOL=""
ONLY_ALPHA=""
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
    --only-alpha)
      ONLY_ALPHA="${2:?}"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "${LOGROOT}" _experiments/slurm_logs

# alpha value | log tag (a08, …)
ALPHAS=(
  "0.8|a08"
  "0.7|a07"
  "0.6|a06"
  "0.4|a04"
  "0.3|a03"
)

JOBS=(
  "mn40|configs/selfevo_fscil_mn40_5t_flex_compare.json|mn40|5|MN40"
  "sn55|configs/selfevo_fscil_sn55_7t_flex_compare.json|sn55|7|SN55"
  "co3d|configs/selfevo_fscil_co3d_6t.json|co3d|6|CO3D"
  "mn_scanobj|configs/selfevo_fscil_mn40_scanobjectnn_4t.json|mn_scanobj|4|MN→ScanObj"
  "sn_scanobj|configs/selfevo_fscil_sn55_scanobjectnn_4t.json|sn_scanobj|4|SN→ScanObj"
  "sn_co3d|configs/selfevo_fscil_sn55_co3d_11t.json|sn_co3d|11|SN→CO3D"
)

N_PROTO=${#JOBS[@]}
N_ALPHA=${#ALPHAS[@]}

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
  "fscil_ridge_alpha_ramp=false"
)

log() { echo "[$(date -Is)] $*"; }

alpha_eq() {
  # Numeric equality for --only-alpha (avoids host python dependency).
  awk -v a="${1}" -v b="${2}" 'BEGIN { exit !(a + 0 == b + 0) }'
}

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

run_flat_idx() {
  local flat="$1"
  local a_i=$((flat / N_PROTO))
  local p_i=$((flat % N_PROTO))

  if [[ "${a_i}" -ge "${N_ALPHA}" || "${p_i}" -ge "${N_PROTO}" ]]; then
    log "SKIP flat=${flat} out of range"
    return 0
  fi

  IFS='|' read -r alpha atag <<< "${ALPHAS[$a_i]}"
  IFS='|' read -r run_id config prefix n_sess label <<< "${JOBS[$p_i]}"

  if [[ -n "${ONLY_PROTOCOL}" && "${run_id}" != "${ONLY_PROTOCOL}" ]]; then
    return 0
  fi
  if [[ -n "${ONLY_ALPHA}" ]] && ! alpha_eq "${ONLY_ALPHA}" "${alpha}"; then
    return 0
  fi

  local method="final10_${prefix}_mean_close_st_buffree_${atag}"
  local logfile="${LOGROOT}/${method}.log"

  if [[ "${SKIP_DONE}" -eq 1 ]] && log_is_complete "${logfile}" "${n_sess}"; then
    log "SKIP ${method}"
    return 0
  fi

  local -a extra_args=()
  for flag in "${BASE_FLAGS[@]}"; do
    extra_args+=(--set "${flag}")
  done
  extra_args+=(
    --set "ridge_cache_alpha=${alpha}"
    --set "fscil_increment_ridge_alpha=${alpha}"
    --set "fscil_task1_ridge_alpha=${alpha}"
  )

  if [[ "${SMOKE}" -eq 1 ]]; then
    extra_args+=(
      --max-task 1
      --set tuned_epoch=1
      --set task0_save_warmup_ckpts=false
    )
    method="smoke_${method}"
    logfile="${LOGROOT}/${method}.log"
  fi

  log "=== [flat=${flat} a=${a_i} p=${p_i}] ${method} | ${label} | α=${alpha} | ep10 buffree mean_close max ==="
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
  run_flat_idx "${ONLY_IDX}"
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  run_flat_idx "${SLURM_ARRAY_TASK_ID}"
else
  local_n=$((N_ALPHA * N_PROTO))
  for flat in $(seq 0 $((local_n - 1))); do
    run_flat_idx "${flat}"
  done
fi

log "α sweep ep10 finished → ${LOGROOT} (alphas 0.8/0.7/0.6/0.4/0.3 × 6 protocols)"

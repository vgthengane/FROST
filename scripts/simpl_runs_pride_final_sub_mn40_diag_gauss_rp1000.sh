#!/bin/bash
#SBATCH --job-name=pride_fsub_gauss
#SBATCH --account=EUHPC_D36_199
#SBATCH --partition=boost_usr_prod
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=_experiments/slurm_logs/%A_%a_pride_fsub_gauss.out

# =============================================================================
# PRidE final_sub — MN40: diag_gauss vs mean-only max (seed 2008 check)
# =============================================================================
# Same recipe as final_sub rp=1000 mean-only, except:
#   prototype_pooling = diag_gauss
# Seed 2008 only (no multi-seed until this looks promising).
#
# Array: 0=buffree, 1=buf
#
# Logs:
#   _experiments/00_FINAL_SUB_MEAN_DIAG_GAUSS_A05_RP1000_MN40/
#     final_sub_mn40_mean_diag_gauss_<buffree|buf>_a05_rp1000_seed=2008.log
#
# Launch:
#   bash sync_eurohpc.sh --code
#   sbatch scripts/simpl_runs_pride_final_sub_mn40_diag_gauss_rp1000.sh
#   sbatch scripts/simpl_runs_pride_final_sub_mn40_logsumexp_rp1000.sh   # twin
#
# Compare:
#   aprun python scripts/summarize_pride_distribution_mn40.py
# =============================================================================

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

RUNNER=(aprun python)
LOGROOT="_experiments/00_FINAL_SUB_MEAN_DIAG_GAUSS_A05_RP1000_MN40"
CONFIG="configs/selfevo_fscil_mn40_5t_flex_compare.json"
SEED=2008
N_SESS=5
SKIP_DONE=0
ONLY_IDX=""
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
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "${LOGROOT}" _experiments/slurm_logs

VARIANTS=(
  "buffree|selfevo_bufferfree|"
  "buf|selfevo|canonical"
)

COMMON_FLAGS=(
  "fscil_adapter_base_only=true"
  "task0_adapter_warmup=true"
  "increment_adapter_warmup=false"
  "fscil_classifier_warmup_after_protos=false"
  "fscil_task1_adapter_warmup=false"
  "tuned_epoch=10"
  "conflict_replay=true"
  "conflict_replay_on_test=false"
  "conflict_replay_use_test_cm=false"
  "prototype_pooling=diag_gauss"
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

run_idx() {
  local idx="$1"
  IFS='|' read -r var_name model_name preset <<< "${VARIANTS[$idx]}"

  local method="final_sub_mn40_mean_diag_gauss_${var_name}_a05_rp1000"
  local logfile="${LOGROOT}/${method}_seed=${SEED}.log"

  if [[ "${SKIP_DONE}" -eq 1 ]] && log_is_complete "${logfile}" "${N_SESS}"; then
    log "SKIP ${method} seed=${SEED}"
    return 0
  fi

  local -a extra_args=()
  for flag in "${COMMON_FLAGS[@]}"; do
    extra_args+=(--set "${flag}")
  done
  extra_args+=(--set "model_name=${model_name}")
  if [[ -n "${preset}" ]]; then
    extra_args+=(--set "fscil_preset=${preset}")
  fi
  extra_args+=(
    --set "fscil_old_shot_replay_seed=${SEED}"
    --set "fscil_base_ridge_refresh_seed=${SEED}"
  )

  if [[ "${SMOKE}" -eq 1 ]]; then
    extra_args+=(
      --max-task 1
      --set tuned_epoch=1
      --set task0_save_warmup_ckpts=false
    )
    method="smoke_${method}"
    logfile="${LOGROOT}/${method}_seed=${SEED}.log"
  fi

  log "=== [${idx}] ${method} | MN40 | seed=${SEED} | ${var_name} | diag_gauss | ep10 | α=0.5 | rp=1000 ==="
  "${RUNNER[@]}" train_fscil.py \
    --config "${CONFIG}" \
    --logdir "${LOGROOT}" \
    --method-name "${method}" \
    --log-layout flat \
    --seeds "${SEED}" \
    "${extra_args[@]}" \
    2>&1 | tee "${logfile}"
  log "=== ${method} seed=${SEED} done ==="
}

if [[ -n "${ONLY_IDX}" ]]; then
  run_idx "${ONLY_IDX}"
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  run_idx "${SLURM_ARRAY_TASK_ID}"
else
  for idx in 0 1; do
    run_idx "${idx}"
  done
fi

log "final_sub MN40 diag_gauss seed ${SEED} finished → ${LOGROOT}"

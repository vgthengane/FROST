#!/bin/bash
#SBATCH --job-name=final_ep10
#SBATCH --account=EUHPC_D36_199
#SBATCH --partition=boost_usr_prod
#SBATCH --array=0-71
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=_experiments/slurm_logs/%A_%a_final_ep10.out

# Same matrix as simpl_runs_final_ep5_baseonly.sh but task-0 adapter warmup:
#   tuned_epoch=10, task0_use_best_warmup=true (best test-epoch checkpoint)
#
# Protocols (6) × buffer/buffer-free (2) × proto regimes (5) = 60 jobs
#
# Submit on Leonardo:
#   sbatch scripts/simpl_runs_final_ep10_baseonly.sh
#
# Local:
#   bash scripts/simpl_runs_final_ep10_baseonly.sh --local --skip-done
#
# Logs: _experiments/00_FINAL_EP10_BASEONLY/final10_<protocol>_<regime>_<buf|buffree>.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

RUNNER=(aprun python)
LOGROOT="_experiments/00_FINAL_EP10_BASEONLY"
METHOD_PREFIX="final10"
SEED=2008
SKIP_DONE=0
ONLY_IDX=""
ONLY_PROTOCOL=""
ONLY_BUFFER=""
ONLY_REGIME=""
SMOKE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local) RUNNER=(aprun python); shift ;;
    --skip-done) SKIP_DONE=1; shift ;;
    --smoke) SMOKE=1; shift ;;
    --array)
      ONLY_IDX="${2:?--array needs index 0-71}"
      shift 2
      ;;
    --only-protocol)
      ONLY_PROTOCOL="${2:?--only-protocol needs mn40|sn55|co3d|mn_scanobj|sn_scanobj|sn_co3d}"
      shift 2
      ;;
    --only-buffer)
      ONLY_BUFFER="${2:?--only-buffer needs buffer|bufferfree}"
      shift 2
      ;;
    --only-regime)
      ONLY_REGIME="${2:?--only-regime needs mean_st|pshot_evo|pshot_evo5|pshot_st|hybrid3_st|hybrid4_evo}"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "${LOGROOT}" _experiments/slurm_logs

PROTOCOLS=(
  "mn40|configs/selfevo_fscil_mn40_5t_flex_compare.json|mn40|5|MN40 in-domain"
  "sn55|configs/selfevo_fscil_sn55_7t_flex_compare.json|sn55|7|SN55 in-domain"
  "co3d|configs/selfevo_fscil_co3d_6t.json|co3d|6|CO3D in-domain"
  "mn_scanobj|configs/selfevo_fscil_mn40_scanobjectnn_4t.json|mn_scanobj|4|MN40->ScanObj"
  "sn_scanobj|configs/selfevo_fscil_sn55_scanobjectnn_4t.json|sn_scanobj|4|SN55->ScanObj"
  "sn_co3d|configs/selfevo_fscil_sn55_co3d_11t.json|sn_co3d|11|SN55->CO3D"
)

REGIMES=(
  "mean_st"
  "pshot_evo"
  "pshot_evo5"
  "pshot_st"
  "hybrid3_st"
  "hybrid4_evo"
)

BASE_FLAGS=(
  "fscil_adapter_base_only=true"
  "task0_adapter_warmup=true"
  "increment_adapter_warmup=false"
  "fscil_classifier_warmup_after_protos=false"
  "fscil_task1_adapter_warmup=false"
  "tuned_epoch=10"
  "task0_use_best_warmup=true"
  "task0_save_warmup_ckpts=true"
  "conflict_replay=true"
  "conflict_replay_on_test=false"
  "conflict_replay_use_test_cm=false"
)

BUFFER_FLAGS=(
  "model_name=selfevo"
  "fscil_preset=canonical"
)

BUFFERFREE_FLAGS=(
  "model_name=selfevo_bufferfree"
)

regime_flags() {
  local regime="$1"
  case "${regime}" in
    mean_st)
      printf '%s\n' \
        "fscil_shot_prototypes=false" \
        "fscil_cluster_new_classes=false" \
        "fscil_mean_best2_prototypes=false" \
        "fscil_mean_close_far_prototypes=false" \
        "fscil_mean_nearmean_prototypes=false" \
        "insert_strategy=mean" \
        "max_prototypes_per_class=1" \
        "fscil_increment_max_prototypes=1" \
        "evolution_policy=static"
      ;;
    pshot_evo)
      printf '%s\n' \
        "fscil_shot_prototypes=true" \
        "fscil_cluster_new_classes=false" \
        "fscil_mean_best2_prototypes=false" \
        "fscil_mean_close_far_prototypes=false" \
        "fscil_mean_nearmean_prototypes=false" \
        "insert_strategy=mean" \
        "max_prototypes_per_class=3" \
        "fscil_increment_max_prototypes=8" \
        "evolution_policy=rules"
      ;;
    pshot_evo5)
      printf '%s\n' \
        "fscil_shot_prototypes=true" \
        "fscil_cluster_new_classes=false" \
        "fscil_mean_best2_prototypes=false" \
        "fscil_mean_close_far_prototypes=false" \
        "fscil_mean_nearmean_prototypes=false" \
        "insert_strategy=mean" \
        "max_prototypes_per_class=5" \
        "fscil_increment_max_prototypes=5" \
        "evolution_policy=rules"
      ;;
    pshot_st)
      printf '%s\n' \
        "fscil_shot_prototypes=true" \
        "fscil_cluster_new_classes=false" \
        "fscil_mean_best2_prototypes=false" \
        "fscil_mean_close_far_prototypes=false" \
        "fscil_mean_nearmean_prototypes=false" \
        "insert_strategy=mean" \
        "max_prototypes_per_class=5" \
        "fscil_increment_max_prototypes=5" \
        "evolution_policy=static"
      ;;
    hybrid3_st)
      printf '%s\n' \
        "fscil_shot_prototypes=false" \
        "fscil_cluster_new_classes=false" \
        "fscil_mean_best2_prototypes=false" \
        "fscil_mean_close_far_prototypes=true" \
        "insert_strategy=mean" \
        "max_prototypes_per_class=3" \
        "fscil_increment_max_prototypes=3" \
        "evolution_policy=static"
      ;;
    hybrid4_evo)
      printf '%s\n' \
        "fscil_shot_prototypes=false" \
        "fscil_cluster_new_classes=false" \
        "fscil_mean_best2_prototypes=false" \
        "fscil_mean_close_far_prototypes=true" \
        "insert_strategy=mean" \
        "max_prototypes_per_class=3" \
        "fscil_increment_max_prototypes=4" \
        "evolution_policy=rules"
      ;;
    *)
      echo "Unknown regime: ${regime}" >&2
      return 1
      ;;
  esac
}

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
  local buffer_tag="$6"
  local regime="$7"
  shift 7
  local -a buffer_flags=("$@")

  mkdir -p "${LOGROOT}"
  local logfile="${LOGROOT}/${method}.log"

  if [[ "${SKIP_DONE}" -eq 1 ]] && log_is_complete "${logfile}" "${n_sess}"; then
    log "SKIP ${method} — complete (${n_sess} sessions)"
    return 0
  fi

  local -a extra_args=()
  for flag in "${BASE_FLAGS[@]}" "${buffer_flags[@]}"; do
    extra_args+=(--set "${flag}")
  done
  while IFS= read -r flag; do
    [[ -n "${flag}" ]] || continue
    extra_args+=(--set "${flag}")
  done < <(regime_flags "${regime}")

  if [[ "${SMOKE}" -eq 1 ]]; then
    extra_args+=(
      --max-task 1
      --set tuned_epoch=1
      --set task0_save_warmup_ckpts=false
    )
    method="smoke_${method}"
  fi

  log "=== [${run_id}] ${method} | ${label} | ${regime} | ${buffer_tag} | ep10 best ==="
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

dispatch_idx() {
  local idx="$1"
  local protocol_idx=$((idx / 12))
  local rem=$((idx % 12))
  local buffer_idx=$((rem / 6))
  local regime_idx=$((rem % 6))

  IFS='|' read -r run_id config prefix n_sess label <<< "${PROTOCOLS[$protocol_idx]}"
  local regime="${REGIMES[$regime_idx]}"

  if [[ -n "${ONLY_PROTOCOL}" && "${run_id}" != "${ONLY_PROTOCOL}" ]]; then
    return 0
  fi
  if [[ -n "${ONLY_REGIME}" && "${regime}" != "${ONLY_REGIME}" ]]; then
    return 0
  fi

  local buffer_tag method
  local -a buffer_flags=()
  if [[ "${buffer_idx}" -eq 0 ]]; then
    buffer_tag="buf"
    buffer_flags=("${BUFFER_FLAGS[@]}")
  else
    buffer_tag="buffree"
    buffer_flags=("${BUFFERFREE_FLAGS[@]}")
  fi

  if [[ -n "${ONLY_BUFFER}" ]]; then
    case "${ONLY_BUFFER}" in
      buffer) [[ "${buffer_idx}" -eq 0 ]] || return 0 ;;
      bufferfree) [[ "${buffer_idx}" -eq 1 ]] || return 0 ;;
      *) echo "Unknown --only-buffer ${ONLY_BUFFER}" >&2; exit 1 ;;
    esac
  fi

  method="${METHOD_PREFIX}_${prefix}_${regime}_${buffer_tag}"
  run_one "${run_id}" "${config}" "${method}" "${n_sess}" "${label}" "${buffer_tag}" "${regime}" \
    "${buffer_flags[@]}"
}

if [[ -n "${ONLY_IDX}" ]]; then
  dispatch_idx "${ONLY_IDX}"
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  dispatch_idx "${SLURM_ARRAY_TASK_ID}"
else
  for idx in $(seq 0 71); do
    dispatch_idx "${idx}"
  done
fi

log "final ep10 baseonly (best warmup) campaign finished"

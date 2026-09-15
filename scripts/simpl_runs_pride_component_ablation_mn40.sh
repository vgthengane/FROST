#!/bin/bash
#SBATCH --job-name=pride_comp_mn40
#SBATCH --account=EUHPC_D36_199
#SBATCH --partition=boost_usr_prod
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=_experiments/slurm_logs/%A_%a_pride_comp_mn40.out

# Controlled PRidE component ablation on FSCIL-3D ModelNet40 (seed 2008).
#
#   0 linear          frozen Uni3D; linear CE head trained every session
#   1 adapter_linear  base-only adapter + linear CE; linear trains every session
#   2 ridge           base-only adapter; analytical ridge inference
#   3 cache           base-only adapter; prototype-cache inference
#   4 ridge_cache     exact locked final PRidE MN40 recipe (α=0.5, EP10)
#
# All variants are buffer-free. The three analytical/prototype variants inherit
# the final representation and prototype recipe. Row 4 intentionally matches
# scripts/simpl_runs_pride_alpha05_ep10.sh --only mn40.
#
# Submit:
#   sbatch scripts/simpl_runs_pride_component_ablation_mn40.sh
# Smoke one row:
#   sbatch --qos=boost_qos_dbg --time=01:00:00 --array=1 \
#     scripts/simpl_runs_pride_component_ablation_mn40.sh --smoke
# Summarize:
#   aprun python scripts/summarize_pride_component_ablation_mn40.py

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

CONFIG="configs/selfevo_fscil_mn40_5t_flex_compare.json"
LOGROOT="_experiments/00_PRIDE_COMPONENT_ABLATION_MN40"
SEED=2008
RUNNER=(aprun python)

VARIANTS=(
  "linear|Linear (frozen Uni3D + CE head)"
  "adapter_linear|Adapter + linear (adapter base-only)"
  "ridge|Ridge only (adapter base-only)"
  "cache|Cache only (adapter base-only)"
  "ridge_cache|Ridge + cache, α=0.5 (locked final PRidE)"
)

SMOKE=0
SKIP_DONE=0
ONLY=""
LOCAL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local) LOCAL=1; shift ;;
    --smoke) SMOKE=1; shift ;;
    --skip-done) SKIP_DONE=1; shift ;;
    --only) ONLY="${2:?--only requires a variant id}"; shift 2 ;;
    *)
      echo "Usage: $0 [--local] [--smoke] [--skip-done] [--only linear|adapter_linear|ridge|cache|ridge_cache]" >&2
      exit 1
      ;;
  esac
done

if [[ "${LOCAL}" -eq 1 ]]; then
  RUNNER=(aprun python)
fi

mkdir -p "${LOGROOT}" _experiments/slurm_logs

log() { echo "[$(date -Is)] $*"; }

log_is_complete() {
  local logfile="$1"
  [[ -f "${logfile}" ]] || return 1
  local last
  last="$(rg "CNN top1 curve:" "${logfile}" 2>/dev/null | tail -n 1 || true)"
  [[ -n "${last}" ]] || return 1
  local count
  count="$(echo "${last}" | tr -cd ',' | wc -c)"
  [[ $((count + 1)) -ge 5 ]]
}

common_buffer_free_flags() {
  echo "exemplar_buffer_shots=0"
  echo "exemplar_buffer_ridge_refresh_shots=0"
  echo "fscil_old_shot_replay=0"
  echo "fscil_base_ridge_refresh_per_class=0"
}

linear_flags() {
  local use_adapter="$1"
  echo "model_name=fscil_finetune"
  echo "linprobe_mode=finetune"
  echo "linprobe_use_adapter=${use_adapter}"
  echo "backbone_type=uni3d_base_patch14_448"
  echo "base_epoch=10"
  echo "increment_epoch=10"
  echo "task0_init_lr=0.001"
  echo "task0_optimizer=sgd"
  echo "task0_weight_decay=0.0001"
  echo "init_lr=0.005"
  echo "optimizer=sgd"
  echo "weight_decay=0.0005"
  common_buffer_free_flags
}

final_representation_flags() {
  echo "model_name=selfevo_bufferfree"
  echo "fscil_adapter_base_only=true"
  echo "task0_adapter_warmup=true"
  echo "increment_adapter_warmup=false"
  echo "fscil_classifier_warmup_after_protos=false"
  echo "fscil_task1_adapter_warmup=false"
  echo "tuned_epoch=10"
  echo "conflict_replay=true"
  echo "conflict_replay_on_test=false"
  echo "conflict_replay_use_test_cm=false"
  echo "prototype_pooling=max"
  echo "pooling_temperature=0.07"
  echo "fscil_shot_prototypes=false"
  echo "fscil_cluster_new_classes=false"
  echo "fscil_mean_best2_prototypes=true"
  echo "fscil_mean_close_far_prototypes=false"
  echo "fscil_mean_nearmean_prototypes=false"
  echo "insert_strategy=mean"
  echo "max_prototypes_per_class=3"
  echo "fscil_increment_max_prototypes=3"
  echo "evolution_policy=static"
  echo "ridge_cache_alpha=0.5"
  echo "fscil_increment_ridge_alpha=0.5"
  echo "fscil_task1_ridge_alpha=0.5"
  echo "fscil_ridge_alpha_ramp=false"
  common_buffer_free_flags
}

flags_for_variant() {
  case "$1" in
    linear)
      linear_flags false
      ;;
    adapter_linear)
      linear_flags true
      ;;
    ridge)
      final_representation_flags
      echo "classifier_mode=ridge"
      ;;
    cache)
      final_representation_flags
      echo "classifier_mode=cache"
      ;;
    ridge_cache)
      final_representation_flags
      echo "classifier_mode=ridge_cache"
      ;;
    *)
      echo "Unknown variant: $1" >&2
      return 1
      ;;
  esac
}

run_variant() {
  local id="$1"
  local label="$2"
  [[ -z "${ONLY}" || "${ONLY}" == "${id}" ]] || return 0

  local method="pride_mn40_comp_${id}"
  local logfile="${LOGROOT}/${method}_seed=${SEED}.log"
  if [[ "${SMOKE}" -eq 1 ]]; then
    method="${method}_smoke"
    logfile="${LOGROOT}/${method}_seed=${SEED}.log"
  elif [[ "${SKIP_DONE}" -eq 1 ]] && log_is_complete "${logfile}"; then
    log "SKIP ${method}: complete log exists"
    return 0
  fi

  local -a extra_args=()
  while IFS= read -r flag; do
    [[ -n "${flag}" ]] && extra_args+=(--set "${flag}")
  done < <(flags_for_variant "${id}")

  if [[ "${SMOKE}" -eq 1 ]]; then
    extra_args+=(
      --max-task 1
      --set base_epoch=1
      --set increment_epoch=1
      --set tuned_epoch=1
      --set task0_save_warmup_ckpts=false
    )
  fi

  log "=== ${method}: ${label} ==="
  "${RUNNER[@]}" train_fscil.py \
    --config "${CONFIG}" \
    --logdir "${LOGROOT}" \
    --method-name "${method}" \
    --log-layout flat \
    --seeds "${SEED}" \
    "${extra_args[@]}" \
    2>&1 | tee "${logfile}"
  log "=== ${method} done ==="
}

select_indices() {
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "${SLURM_ARRAY_TASK_ID}"
  else
    seq 0 $((${#VARIANTS[@]} - 1))
  fi
}

for idx in $(select_indices); do
  IFS='|' read -r id label <<< "${VARIANTS[$idx]}"
  run_variant "${id}" "${label}"
done

if [[ "${SMOKE}" -eq 0 && -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  "${RUNNER[@]}" scripts/summarize_pride_component_ablation_mn40.py \
    --log-root "${LOGROOT}"
fi

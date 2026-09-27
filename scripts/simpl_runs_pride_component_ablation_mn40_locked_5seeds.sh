#!/bin/bash
#SBATCH --job-name=pride_comp_mn40
#SBATCH --account=EUHPC_D36_199
#SBATCH --partition=boost_usr_prod
#SBATCH --array=0-19
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=_experiments/slurm_logs/%A_%a_pride_comp_mn40.out

# =============================================================================
# PRidE COMPONENT ablation only — MN40, locked setting, 5 seeds, buffer-free
# =============================================================================
# Locked base: mean-only, rp=2000, α=0.5, static, base-only EP10.
#
# Rows (paper ablation.tex):
#   0 linear           frozen Uni3D + CE
#   1 adapter_linear   base-only adapter + CE
#   2 ridge            GramBank (ridge only)     ← must run @ mean + rp2000
#   3 cache            ProtoBank (cache only)    ← must run @ mean + rp2000
#   Gram+ProtoBank     REUSED from final_sub (not in this array)
#
# Array = variant_idx * 5 + seed_idx   (4 × 5 = 20 jobs)
#   seeds: 2008, 1947, 1991, 2001, 2019
#
# Submit:
#   sbatch scripts/simpl_runs_pride_component_ablation_mn40_locked_5seeds.sh
# Summarize (includes reused default row):
#   aprun python scripts/summarize_pride_locked_ablation_mn40.py --latex
# =============================================================================

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

CONFIG="configs/selfevo_fscil_mn40_5t_flex_compare.json"
LOGROOT="_experiments/00_PRIDE_LOCKED_ABLATION_MN40_5SEEDS"
EXPECTED_SESSIONS=5
RUNNER=(aprun python)
SEEDS=(2008 1947 1991 2001 2019)

VARIANTS=(
  "linear|Linear (frozen Uni3D + CE)"
  "adapter_linear|Adapter + Linear"
  "ridge|GramBank (ridge only)"
  "cache|ProtoBank (cache only)"
)

SMOKE=0
SKIP_DONE=0
ONLY=""
ONLY_SEED=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local) shift ;;
    --smoke) SMOKE=1; shift ;;
    --skip-done) SKIP_DONE=1; shift ;;
    --only) ONLY="${2:?}"; shift 2 ;;
    --only-seed) ONLY_SEED="${2:?}"; shift 2 ;;
    *)
      echo "Usage: $0 [--local] [--smoke] [--skip-done] [--only ID] [--only-seed SEED]" >&2
      exit 1
      ;;
  esac
done

mkdir -p "${LOGROOT}" _experiments/slurm_logs
log() { echo "[$(date -Is)] $*"; }

log_is_complete() {
  local logfile="$1"
  [[ -f "${logfile}" ]] || return 1
  local last
  last="$(grep "CNN top1 curve:" "${logfile}" 2>/dev/null | tail -n 1 || true)"
  [[ -n "${last}" ]] || return 1
  local count
  count="$(echo "${last}" | tr -cd ',' | wc -c)"
  [[ $((count + 1)) -ge "${EXPECTED_SESSIONS}" ]]
}

locked_base_flags() {
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
  echo "fscil_mean_best2_prototypes=false"
  echo "fscil_mean_close_far_prototypes=false"
  echo "fscil_mean_nearmean_prototypes=false"
  echo "fscil_mean_far2_prototypes=false"
  echo "insert_strategy=mean"
  echo "max_prototypes_per_class=1"
  echo "fscil_increment_max_prototypes=1"
  echo "evolution_policy=static"
  echo "classifier_mode=ridge_cache"
  echo "ridge_cache_alpha=0.5"
  echo "fscil_increment_ridge_alpha=0.5"
  echo "fscil_task1_ridge_alpha=0.5"
  echo "fscil_ridge_alpha_ramp=false"
  echo "ridge_rp_dim=2000"
  echo "exemplar_buffer_shots=0"
  echo "exemplar_buffer_ridge_refresh_shots=0"
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
  echo "exemplar_buffer_shots=0"
  echo "exemplar_buffer_ridge_refresh_shots=0"
  echo "fscil_old_shot_replay=0"
  echo "fscil_base_ridge_refresh_per_class=0"
}

flags_for_variant() {
  case "$1" in
    linear) linear_flags false ;;
    adapter_linear) linear_flags true ;;
    ridge) locked_base_flags; echo "classifier_mode=ridge" ;;
    cache) locked_base_flags; echo "classifier_mode=cache" ;;
    *) echo "Unknown variant: $1" >&2; return 1 ;;
  esac
}

run_cell() {
  local id="$1" label="$2" seed="$3"
  [[ -z "${ONLY}" || "${ONLY}" == "${id}" ]] || return 0
  [[ -z "${ONLY_SEED}" || "${ONLY_SEED}" == "${seed}" ]] || return 0

  local method="pride_mn40_abl_${id}"
  local logfile="${LOGROOT}/${method}_seed=${seed}.log"
  if [[ "${SMOKE}" -eq 1 ]]; then
    method="${method}_smoke"
    logfile="${LOGROOT}/${method}_seed=${seed}.log"
  elif [[ "${SKIP_DONE}" -eq 1 ]] && log_is_complete "${logfile}"; then
    log "SKIP ${method} seed=${seed}"
    return 0
  fi

  local -a extra_args=()
  while IFS= read -r flag; do
    [[ -n "${flag}" ]] && extra_args+=(--set "${flag}")
  done < <(flags_for_variant "${id}")

  if [[ "${SMOKE}" -eq 1 ]]; then
    extra_args+=(
      --max-task 1
      --set base_epoch=1 --set increment_epoch=1 --set tuned_epoch=1
      --set task0_save_warmup_ckpts=false
    )
  fi

  log "=== ${method} seed=${seed}: ${label} ==="
  "${RUNNER[@]}" train_fscil.py \
    --config "${CONFIG}" \
    --logdir "${LOGROOT}" \
    --method-name "${method}" \
    --log-layout flat \
    --seeds "${seed}" \
    "${extra_args[@]}" \
    2>&1 | tee "${logfile}"
  log "=== ${method} seed=${seed} done ==="
}

N_VAR=${#VARIANTS[@]}
N_SEED=${#SEEDS[@]}

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  task="${SLURM_ARRAY_TASK_ID}"
  vidx=$((task / N_SEED))
  sidx=$((task % N_SEED))
  [[ "${vidx}" -lt "${N_VAR}" ]] || exit 0
  IFS='|' read -r id label <<< "${VARIANTS[$vidx]}"
  run_cell "${id}" "${label}" "${SEEDS[$sidx]}"
else
  for vidx in $(seq 0 $((N_VAR - 1))); do
    IFS='|' read -r id label <<< "${VARIANTS[$vidx]}"
    for seed in "${SEEDS[@]}"; do
      run_cell "${id}" "${label}" "${seed}"
    done
  done
fi

#!/bin/bash
#SBATCH --job-name=pride_hp_mn40
#SBATCH --account=EUHPC_D36_199
#SBATCH --partition=boost_usr_prod
#SBATCH --array=0-12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=_experiments/slurm_logs/%A_%a_pride_hp_mn40.out

# One-factor hyperparameter ablations around locked buffer-free PRidE on MN40.
#
# Existing exact-match runs are reused by the summarizer:
#   alpha={0.3,0.7}: 00_ABLATION_ALPHA_SWEEP_EP10
#   shared default:   00_ABLATION_ALPHA05_EP10
#
# The shared default is alpha=0.5, mean+two-close insertion, adapter width 64,
# and ridge random-projection dimension 1000. Therefore, only the thirteen
# non-default/non-existing settings below need new jobs.
#
# Prototype labels:
#   mean         class mean only
#   close_far    class mean + one close + one far (ablation)
#   mean_far2    class mean + two shots farthest from the mean
#   all          all five novel-class shots
#   (default)    mean + two close  — reused from locked final10 log
#
# Submit:
#   sbatch scripts/simpl_runs_pride_hparam_ablation_mn40.sh
# Summarize after syncing results:
#   aprun python scripts/summarize_pride_hparam_ablation_mn40.py

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

CONFIG="${CONFIG:-configs/selfevo_fscil_mn40_5t_flex_compare.json}"
LOGROOT="${LOGROOT:-_experiments/00_PRIDE_HPARAM_ABLATION_MN40}"
SEED="${SEED:-2008}"
METHOD_PREFIX="${METHOD_PREFIX:-pride_mn40_hp}"
EXPECTED_SESSIONS="${EXPECTED_SESSIONS:-5}"
SUMMARY_SCRIPT="${SUMMARY_SCRIPT:-scripts/summarize_pride_hparam_ablation_mn40.py}"
RUN_ALL_ALPHAS="${RUN_ALL_ALPHAS:-0}"
RUN_COMPONENTS="${RUN_COMPONENTS:-0}"
RUNNER=(aprun python)

VARIANTS=(
  "alpha01|Fusion alpha=0.1"
  "alpha09|Fusion alpha=0.9"
  "proto_mean|Prototype insertion: mean"
  "proto_close_far|Prototype insertion: mean + close + far"
  "proto_mean_far2|Prototype insertion: mean + two far"
  "proto_all|Prototype insertion: all five shots"
  "adapter32|Adapter width=32"
  "adapter128|Adapter width=128"
  "adapter256|Adapter width=256"
  "ridge250|Ridge RP dimension=250"
  "ridge500|Ridge RP dimension=500"
  "ridge2000|Ridge RP dimension=2000"
  "ridge4000|Ridge RP dimension=4000"
)
if [[ "${RUN_ALL_ALPHAS}" -eq 1 ]]; then
  VARIANTS+=(
    "alpha03|Fusion alpha=0.3"
    "alpha05|Fusion alpha=0.5"
    "alpha07|Fusion alpha=0.7"
  )
fi
if [[ "${RUN_COMPONENTS}" -eq 1 ]]; then
  VARIANTS+=(
    "comp_linear|Component: linear CE head"
    "comp_adapter_linear|Component: base adapter + linear CE head"
    "comp_ridge|Component: ridge only"
    "comp_cache|Component: cache only"
  )
fi

SMOKE=0
SKIP_DONE=0
ONLY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local) shift ;;
    --smoke) SMOKE=1; shift ;;
    --skip-done) SKIP_DONE=1; shift ;;
    --only) ONLY="${2:?--only requires a variant id}"; shift 2 ;;
    *)
      echo "Usage: $0 [--local] [--smoke] [--skip-done] [--only VARIANT]" >&2
      exit 1
      ;;
  esac
done

mkdir -p "${LOGROOT}" _experiments/slurm_logs

log() { echo "[$(date -Is)] $*"; }

log_is_complete() {
  local logfile="$1"
  [[ -f "${logfile}" ]] || return 1
  local curve
  curve="$(rg "CNN top1 curve:" "${logfile}" 2>/dev/null | tail -n 1 || true)"
  [[ -n "${curve}" ]] || return 1
  local count
  count="$(echo "${curve}" | tr -cd ',' | wc -c)"
  [[ $((count + 1)) -ge "${EXPECTED_SESSIONS}" ]]
}

base_flags() {
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
  echo "fscil_mean_far2_prototypes=false"
  echo "fscil_mean_nearmean_prototypes=false"
  echo "insert_strategy=mean"
  echo "max_prototypes_per_class=3"
  echo "fscil_increment_max_prototypes=3"
  echo "evolution_policy=static"
  echo "classifier_mode=ridge_cache"
  echo "ridge_cache_alpha=0.5"
  echo "fscil_increment_ridge_alpha=0.5"
  echo "fscil_task1_ridge_alpha=0.5"
  echo "fscil_ridge_alpha_ramp=false"
  echo "ffn_num=64"
  echo "ridge_rp_dim=1000"
  echo "exemplar_buffer_shots=0"
  echo "exemplar_buffer_ridge_refresh_shots=0"
  # These are part of the locked FSCIL protocol and are distinct from the
  # exemplar buffer controlled by the two settings above.
  echo "fscil_old_shot_replay=1"
  echo "fscil_base_ridge_refresh_per_class=1"
}

flags_for_variant() {
  case "$1" in
    alpha01)
      echo "ridge_cache_alpha=0.1"
      echo "fscil_increment_ridge_alpha=0.1"
      echo "fscil_task1_ridge_alpha=0.1"
      ;;
    alpha03)
      echo "ridge_cache_alpha=0.3"
      echo "fscil_increment_ridge_alpha=0.3"
      echo "fscil_task1_ridge_alpha=0.3"
      ;;
    alpha05)
      ;;
    alpha07)
      echo "ridge_cache_alpha=0.7"
      echo "fscil_increment_ridge_alpha=0.7"
      echo "fscil_task1_ridge_alpha=0.7"
      ;;
    alpha09)
      echo "ridge_cache_alpha=0.9"
      echo "fscil_increment_ridge_alpha=0.9"
      echo "fscil_task1_ridge_alpha=0.9"
      ;;
    proto_mean)
      echo "fscil_mean_best2_prototypes=false"
      ;;
    proto_close_far)
      echo "fscil_mean_best2_prototypes=false"
      echo "fscil_mean_close_far_prototypes=true"
      ;;
    proto_mean_far2)
      echo "fscil_mean_best2_prototypes=false"
      echo "fscil_mean_far2_prototypes=true"
      ;;
    proto_all)
      echo "fscil_mean_best2_prototypes=false"
      echo "fscil_shot_prototypes=true"
      echo "fscil_increment_max_prototypes=5"
      ;;
    adapter32) echo "ffn_num=32" ;;
    adapter128) echo "ffn_num=128" ;;
    adapter256) echo "ffn_num=256" ;;
    ridge250) echo "ridge_rp_dim=250" ;;
    ridge500) echo "ridge_rp_dim=500" ;;
    ridge2000) echo "ridge_rp_dim=2000" ;;
    ridge4000) echo "ridge_rp_dim=4000" ;;
    comp_linear)
      echo "model_name=fscil_finetune"
      echo "linprobe_mode=finetune"
      echo "linprobe_use_adapter=false"
      echo "backbone_type=uni3d_base_patch14_448"
      echo "base_epoch=10"
      echo "increment_epoch=10"
      echo "task0_init_lr=0.001"
      echo "task0_optimizer=sgd"
      echo "task0_weight_decay=0.0001"
      echo "init_lr=0.005"
      echo "optimizer=sgd"
      echo "weight_decay=0.0005"
      ;;
    comp_adapter_linear)
      echo "model_name=fscil_finetune"
      echo "linprobe_mode=finetune"
      echo "linprobe_use_adapter=true"
      echo "backbone_type=uni3d_base_patch14_448"
      echo "base_epoch=10"
      echo "increment_epoch=10"
      echo "task0_init_lr=0.001"
      echo "task0_optimizer=sgd"
      echo "task0_weight_decay=0.0001"
      echo "init_lr=0.005"
      echo "optimizer=sgd"
      echo "weight_decay=0.0005"
      ;;
    comp_ridge) echo "classifier_mode=ridge" ;;
    comp_cache) echo "classifier_mode=cache" ;;
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

  local method="${METHOD_PREFIX}_${id}"
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
  done < <(base_flags)
  while IFS= read -r flag; do
    [[ -n "${flag}" ]] && extra_args+=(--set "${flag}")
  done < <(flags_for_variant "${id}")

  if [[ "${SMOKE}" -eq 1 ]]; then
    extra_args+=(
      --max-task 1
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
  "${RUNNER[@]}" "${SUMMARY_SCRIPT}" \
    --log-root "${LOGROOT}"
fi

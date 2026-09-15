#!/bin/bash
#SBATCH --job-name=max_fill
#SBATCH --account=EUHPC_D36_199
#SBATCH --partition=boost_usr_prod
#SBATCH --array=0-17
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=_experiments/slurm_logs/%A_%a_max_fill.out

# Fill MN40/SN55 under prototype_pooling=max (flex_compare JSONs default to logsumexp).
# Includes buffer AND buffer-free for paper-aligned pooling.
#
# 18 jobs (array 0-17):
#   MN40 (0-8):  mean_st buf/bf, nearmean bf, hybrid3 buf/bf, hybrid4 buf/bf, pshot_st buf/bf
#   SN55 (9-17): same
#
# For hybrid3_st buf+max on CO3D + cross too, also submit:
#   sbatch scripts/simpl_runs_final_hybrid3_buf_max.sh
#
# For full insert-ladder buf (config pooling, all 6 protocols):
#   sbatch scripts/simpl_runs_final_hybrid_closefar_buf.sh
#
# Submit:
#   sbatch scripts/simpl_runs_max_pool_mn40_sn55_fill.sh
#   sbatch scripts/simpl_runs_max_pool_mn40_sn55_fill.sh --skip-done   # after partial runs
#
# Local:
#   bash scripts/simpl_runs_max_pool_mn40_sn55_fill.sh --local
#   bash scripts/simpl_runs_max_pool_mn40_sn55_fill.sh --local --skip-done
#   bash scripts/simpl_runs_max_pool_mn40_sn55_fill.sh --local --only-protocol mn40
#
# Logs: _experiments/00_FINAL_EP5_MAX_POOL/final_<proto>_<regime>_<buf|buffree>_max.log

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
      ONLY_PROTOCOL="${2:?--only-protocol needs mn40|sn55}"
      shift 2
      ;;
    *)
      echo "Unknown: $1" >&2
      echo "Usage: $0 [--local] [--skip-done] [--smoke] [--array N] [--only-protocol mn40|sn55]" >&2
      exit 1
      ;;
  esac
done

mkdir -p "${LOGROOT}" _experiments/slurm_logs

# protocol|config|prefix|n_sess|regime|buffer|model
JOBS=(
  # MN40 (0-8)
  "mn40|configs/selfevo_fscil_mn40_5t_flex_compare.json|mn40|5|mean_st|buf|selfevo"
  "mn40|configs/selfevo_fscil_mn40_5t_flex_compare.json|mn40|5|mean_st|buffree|selfevo_bufferfree"
  "mn40|configs/selfevo_fscil_mn40_5t_flex_compare.json|mn40|5|hybrid_nearmean_st|buffree|selfevo_bufferfree"
  "mn40|configs/selfevo_fscil_mn40_5t_flex_compare.json|mn40|5|hybrid3_st|buf|selfevo"
  "mn40|configs/selfevo_fscil_mn40_5t_flex_compare.json|mn40|5|hybrid3_st|buffree|selfevo_bufferfree"
  "mn40|configs/selfevo_fscil_mn40_5t_flex_compare.json|mn40|5|hybrid4_evo|buf|selfevo"
  "mn40|configs/selfevo_fscil_mn40_5t_flex_compare.json|mn40|5|hybrid4_evo|buffree|selfevo_bufferfree"
  "mn40|configs/selfevo_fscil_mn40_5t_flex_compare.json|mn40|5|pshot_st|buf|selfevo"
  "mn40|configs/selfevo_fscil_mn40_5t_flex_compare.json|mn40|5|pshot_st|buffree|selfevo_bufferfree"
  # SN55 (9-17)
  "sn55|configs/selfevo_fscil_sn55_7t_flex_compare.json|sn55|7|mean_st|buf|selfevo"
  "sn55|configs/selfevo_fscil_sn55_7t_flex_compare.json|sn55|7|mean_st|buffree|selfevo_bufferfree"
  "sn55|configs/selfevo_fscil_sn55_7t_flex_compare.json|sn55|7|hybrid_nearmean_st|buffree|selfevo_bufferfree"
  "sn55|configs/selfevo_fscil_sn55_7t_flex_compare.json|sn55|7|hybrid3_st|buf|selfevo"
  "sn55|configs/selfevo_fscil_sn55_7t_flex_compare.json|sn55|7|hybrid3_st|buffree|selfevo_bufferfree"
  "sn55|configs/selfevo_fscil_sn55_7t_flex_compare.json|sn55|7|hybrid4_evo|buf|selfevo"
  "sn55|configs/selfevo_fscil_sn55_7t_flex_compare.json|sn55|7|hybrid4_evo|buffree|selfevo_bufferfree"
  "sn55|configs/selfevo_fscil_sn55_7t_flex_compare.json|sn55|7|pshot_st|buf|selfevo"
  "sn55|configs/selfevo_fscil_sn55_7t_flex_compare.json|sn55|7|pshot_st|buffree|selfevo_bufferfree"
)

BASE=(
  "fscil_adapter_base_only=true"
  "task0_adapter_warmup=true"
  "increment_adapter_warmup=false"
  "fscil_classifier_warmup_after_protos=false"
  "fscil_task1_adapter_warmup=false"
  "tuned_epoch=5"
  "conflict_replay=true"
  "conflict_replay_on_test=false"
  "conflict_replay_use_test_cm=false"
  "prototype_pooling=max"
  "pooling_temperature=0.1"
)

regime_flags() {
  case "$1" in
    mean_st)
      printf '%s\n' "fscil_shot_prototypes=false" "fscil_cluster_new_classes=false" \
        "fscil_mean_best2_prototypes=false" "fscil_mean_close_far_prototypes=false" \
        "fscil_mean_nearmean_prototypes=false" "insert_strategy=mean" \
        "max_prototypes_per_class=1" "fscil_increment_max_prototypes=1" "evolution_policy=static"
      ;;
    hybrid_nearmean_st)
      printf '%s\n' "fscil_shot_prototypes=false" "fscil_cluster_new_classes=false" \
        "fscil_mean_best2_prototypes=false" "fscil_mean_close_far_prototypes=false" \
        "fscil_mean_nearmean_prototypes=true" "insert_strategy=mean" \
        "max_prototypes_per_class=2" "fscil_increment_max_prototypes=2" "evolution_policy=static"
      ;;
    hybrid3_st)
      printf '%s\n' "fscil_shot_prototypes=false" "fscil_cluster_new_classes=false" \
        "fscil_mean_best2_prototypes=false" "fscil_mean_close_far_prototypes=true" \
        "fscil_mean_nearmean_prototypes=false" "insert_strategy=mean" \
        "max_prototypes_per_class=3" "fscil_increment_max_prototypes=3" "evolution_policy=static"
      ;;
    hybrid4_evo)
      printf '%s\n' "fscil_shot_prototypes=false" "fscil_cluster_new_classes=false" \
        "fscil_mean_best2_prototypes=false" "fscil_mean_close_far_prototypes=true" \
        "fscil_mean_nearmean_prototypes=false" "insert_strategy=mean" \
        "max_prototypes_per_class=3" "fscil_increment_max_prototypes=4" "evolution_policy=rules"
      ;;
    pshot_st)
      printf '%s\n' "fscil_shot_prototypes=true" "fscil_cluster_new_classes=false" \
        "fscil_mean_best2_prototypes=false" "fscil_mean_close_far_prototypes=false" \
        "fscil_mean_nearmean_prototypes=false" "insert_strategy=mean" \
        "max_prototypes_per_class=5" "fscil_increment_max_prototypes=5" "evolution_policy=static"
      ;;
    *)
      echo "Unknown regime: $1" >&2
      return 1
      ;;
  esac
}

log() { echo "[$(date -Is)] $*"; }

log_ok() {
  local f="$1" n="$2"
  [[ -f "$f" ]] || return 1
  local last
  last=$(grep "CNN top1 curve:" "$f" 2>/dev/null | tail -n1 || true)
  [[ -n "$last" ]] || return 1
  local c
  c=$(echo "$last" | tr -cd ',' | wc -c)
  c=$((c + 1))
  [[ "$c" -ge "$n" ]]
}

run_idx() {
  local idx="$1"
  [[ "$idx" -lt "${#JOBS[@]}" ]] || { log "idx $idx out of range (0-$((${#JOBS[@]}-1)))"; return 0; }
  IFS='|' read -r run_id config prefix n_sess regime buffer model <<< "${JOBS[$idx]}"

  if [[ -n "${ONLY_PROTOCOL}" && "${run_id}" != "${ONLY_PROTOCOL}" ]]; then
    return 0
  fi

  local method="final_${prefix}_${regime}_${buffer}_max"
  local logfile="${LOGROOT}/${method}.log"
  if [[ "${SKIP_DONE}" -eq 1 ]] && log_ok "${logfile}" "${n_sess}"; then
    log "SKIP ${method}"
    return 0
  fi

  log "=== [$idx] ${method} ==="
  local -a extra=(--set "model_name=${model}")
  if [[ "${model}" == "selfevo" ]]; then
    extra+=(--set "fscil_preset=canonical")
  fi
  for f in "${BASE[@]}"; do
    extra+=(--set "$f")
  done
  while IFS= read -r f; do
    [[ -n "$f" ]] && extra+=(--set "$f")
  done < <(regime_flags "${regime}")

  if [[ "${SMOKE}" -eq 1 ]]; then
    extra+=(--max-task 1 --set tuned_epoch=1 --set task0_save_warmup_ckpts=false)
    method="smoke_${method}"
    logfile="${LOGROOT}/${method}.log"
  fi

  "${RUNNER[@]}" train_fscil.py \
    --config "${config}" \
    --logdir "${LOGROOT}" \
    --method-name "${method}" \
    --log-layout flat \
    --seeds "${SEED}" \
    "${extra[@]}" \
    2>&1 | tee "${logfile}"
  log "=== ${method} done ==="
}

if [[ -n "${ONLY_IDX}" ]]; then
  run_idx "${ONLY_IDX}"
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  run_idx "${SLURM_ARRAY_TASK_ID}"
else
  for i in $(seq 0 $((${#JOBS[@]} - 1))); do
    run_idx "$i"
  done
fi

log "max-pool MN40/SN55 fill finished → ${LOGROOT}"

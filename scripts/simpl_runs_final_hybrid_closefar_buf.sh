#!/bin/bash
#SBATCH --job-name=hybrid_cf_buf
#SBATCH --account=EUHPC_D36_199
#SBATCH --partition=boost_usr_prod
#SBATCH --array=0-23
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=_experiments/slurm_logs/%A_%a_hybrid_cf_buf.out

# EP5 hybrid close–far insert ladder — WITH exemplar buffer (canonical 1-shot).
# Twin of scripts/simpl_runs_final_hybrid_closefar.sh (which is buffree-only).
#
# Why this exists:
#   00_FINAL_EP5_HYBRID_CLOSEFAR/ only has *_buffree.log. Paper buffer ablation
#   needs matching *_buf.log for hybrid3 / hybrid4 / nearmean / fps5.
#
# Regimes (4) × protocols (6) = 24 jobs:
#   hybrid3_st         — mean + 1 close + 1 far, static (3 keys)
#   hybrid4_evo        — mean + close + far + 1 evo slot (max 4 keys)
#   hybrid_nearmean_st — mean + 1 close to mean, static (2 keys)
#   fps5_st            — farthest-point on 5 shots, static (5 keys)
#
# Pooling: same as buffree twin (config default — MN40/SN55 logsumexp,
# CO3D+cross max). For forced max on MN40/SN55 see:
#   scripts/simpl_runs_max_pool_mn40_sn55_fill.sh
# For hybrid3_st buf + max on all 6 protocols see:
#   scripts/simpl_runs_final_hybrid3_buf_max.sh
#
# Submit on Leonardo:
#   sbatch scripts/simpl_runs_final_hybrid_closefar_buf.sh
#   sbatch --array=0-5 scripts/simpl_runs_final_hybrid_closefar_buf.sh   # hybrid3 only (all protocols)
#
# Local:
#   bash scripts/simpl_runs_final_hybrid_closefar_buf.sh --local
#   bash scripts/simpl_runs_final_hybrid_closefar_buf.sh --local --skip-done
#   bash scripts/simpl_runs_final_hybrid_closefar_buf.sh --local --only-regime hybrid3_st
#
# Logs: _experiments/00_FINAL_EP5_HYBRID_CLOSEFAR/final_<protocol>_<regime>_buf.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

RUNNER=(aprun python)
LOGROOT="_experiments/00_FINAL_EP5_HYBRID_CLOSEFAR"
METHOD_PREFIX="final"
SEED=2008
SKIP_DONE=0
ONLY_IDX=""
ONLY_PROTOCOL=""
ONLY_REGIME=""
SMOKE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local) RUNNER=(aprun python); shift ;;
    --skip-done) SKIP_DONE=1; shift ;;
    --smoke) SMOKE=1; shift ;;
    --array)
      ONLY_IDX="${2:?--array needs index 0-23}"
      shift 2
      ;;
    --only-protocol)
      ONLY_PROTOCOL="${2:?--only-protocol needs mn40|sn55|co3d|mn_scanobj|sn_scanobj|sn_co3d}"
      shift 2
      ;;
    --only-regime)
      ONLY_REGIME="${2:?--only-regime needs hybrid3_st|hybrid4_evo|hybrid_nearmean_st|fps5_st}"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      echo "Usage: $0 [--local] [--skip-done] [--smoke] [--array N]" >&2
      echo "       [--only-protocol PROTO] [--only-regime REGIME]" >&2
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
  "hybrid3_st"
  "hybrid4_evo"
  "hybrid_nearmean_st"
  "fps5_st"
)

BASE_FLAGS=(
  "fscil_adapter_base_only=true"
  "task0_adapter_warmup=true"
  "increment_adapter_warmup=false"
  "fscil_classifier_warmup_after_protos=false"
  "fscil_task1_adapter_warmup=false"
  "tuned_epoch=5"
  "conflict_replay=true"
  "conflict_replay_on_test=false"
  "conflict_replay_use_test_cm=false"
  "model_name=selfevo"
  "fscil_preset=canonical"
)

regime_flags() {
  local regime="$1"
  case "${regime}" in
    hybrid3_st)
      printf '%s\n' \
        "fscil_shot_prototypes=false" \
        "fscil_cluster_new_classes=false" \
        "fscil_mean_best2_prototypes=false" \
        "fscil_mean_close_far_prototypes=true" \
        "fscil_mean_nearmean_prototypes=false" \
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
        "fscil_mean_nearmean_prototypes=false" \
        "insert_strategy=mean" \
        "max_prototypes_per_class=3" \
        "fscil_increment_max_prototypes=4" \
        "evolution_policy=rules"
      ;;
    hybrid_nearmean_st)
      printf '%s\n' \
        "fscil_shot_prototypes=false" \
        "fscil_cluster_new_classes=false" \
        "fscil_mean_best2_prototypes=false" \
        "fscil_mean_close_far_prototypes=false" \
        "fscil_mean_nearmean_prototypes=true" \
        "insert_strategy=mean" \
        "max_prototypes_per_class=2" \
        "fscil_increment_max_prototypes=2" \
        "evolution_policy=static"
      ;;
    fps5_st)
      printf '%s\n' \
        "fscil_shot_prototypes=false" \
        "fscil_cluster_new_classes=true" \
        "fscil_mean_best2_prototypes=false" \
        "fscil_mean_close_far_prototypes=false" \
        "fscil_mean_nearmean_prototypes=false" \
        "insert_strategy=mean" \
        "max_prototypes_per_class=5" \
        "fscil_increment_max_prototypes=5" \
        "evolution_policy=static"
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
  last="$(grep "CNN top1 curve:" "${logfile}" 2>/dev/null | tail -n 1 || true)"
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
  local regime="$6"

  mkdir -p "${LOGROOT}"
  local logfile="${LOGROOT}/${method}.log"

  if [[ "${SKIP_DONE}" -eq 1 ]] && log_is_complete "${logfile}" "${n_sess}"; then
    log "SKIP ${method} — complete (${n_sess} sessions)"
    return 0
  fi

  local -a extra_args=()
  for flag in "${BASE_FLAGS[@]}"; do
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
    logfile="${LOGROOT}/${method}.log"
  fi

  log "=== [${run_id}] ${method} | ${label} | ${regime} | buf | ep5 ==="
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
  local protocol_idx=$((idx / 4))
  local regime_idx=$((idx % 4))

  IFS='|' read -r run_id config prefix n_sess label <<< "${PROTOCOLS[$protocol_idx]}"
  local regime="${REGIMES[$regime_idx]}"

  if [[ -n "${ONLY_PROTOCOL}" && "${run_id}" != "${ONLY_PROTOCOL}" ]]; then
    return 0
  fi
  if [[ -n "${ONLY_REGIME}" && "${regime}" != "${ONLY_REGIME}" ]]; then
    return 0
  fi

  local method="${METHOD_PREFIX}_${prefix}_${regime}_buf"
  run_one "${run_id}" "${config}" "${method}" "${n_sess}" "${label}" "${regime}"
}

if [[ -n "${ONLY_IDX}" ]]; then
  dispatch_idx "${ONLY_IDX}"
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  dispatch_idx "${SLURM_ARRAY_TASK_ID}"
else
  for idx in $(seq 0 23); do
    dispatch_idx "${idx}"
  done
fi

log "final ep5 hybrid closefar BUFFER campaign finished → ${LOGROOT}"

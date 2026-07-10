#!/bin/bash
#SBATCH --job-name=simpl_abl_reg
#SBATCH --account=EUHPC_D36_199
#SBATCH --partition=boost_usr_prod
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=_experiments/slurm_logs/%A_%a_simpl_abl_regimes.out

# Regime ablations for the evolution / fusion story (seed 2008).
#
# Protocols (rows):
#   0  ModelNet40 within-dataset FSCIL (5 sessions)
#   1  ShapeNet55 within-dataset FSCIL (7 sessions)
#   2  ModelNet40 -> ScanObjectNN cross-dataset (4 sessions)
#   3  ShapeNet55 -> ScanObjectNN cross-dataset (4 sessions)
#
# Variants (columns):
#   0  ridge only          classifier_mode=ridge
#   1  cache only          classifier_mode=cache          (+ rules so cache can adapt)
#   2  ridge + cache       classifier_mode=ridge_cache    + evolution_policy=static
#   3  ridge + cache + evo classifier_mode=ridge_cache    + evolution_policy=rules
#
# Array layout: index = protocol*4 + variant  →  0..15
#
# Submit on Leonardo (full training):
#   bash sync_eurohpc.sh --code --force
#   sbatch scripts/simpl_runs_ablations_regimes.sh
#
# Toy overnight (first; see scripts/simpl_runs_ablations_regimes_toy.sh):
#   sbatch scripts/simpl_runs_ablations_regimes_toy.sh
#
# Summary:
#   aprun python scripts/summarize_fscil_regime_ablations.py

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

LOGROOT="_experiments/00_SIMPL_REGIME_ABLATIONS"
RUNNER=(aprun python)
SKIP_DONE=0
TOY=0
ONLY_PROTO=""
ONLY_VAR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local) RUNNER=(aprun python); shift ;;
    --skip-done) SKIP_DONE=1; shift ;;
    --toy) TOY=1; shift ;;
    --protocol)
      ONLY_PROTO="${2:?--protocol needs mn40|sn55|mn_scanobj|sn_scanobj}"
      shift 2
      ;;
    --variant)
      ONLY_VAR="${2:?--variant needs ridge|cache|static|rules}"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "${LOGROOT}" _experiments/slurm_logs

# protocol_id|config|method_prefix|expected_sessions
PROTOCOLS=(
  "mn40|configs/selfevo_fscil_mn40_5t_flex_compare.json|abl_mn40|5"
  "sn55|configs/selfevo_fscil_sn55_7t_flex_compare.json|abl_sn55|7"
  "mn_scanobj|configs/selfevo_fscil_mn40_scanobjectnn_4t.json|abl_mn_scanobj|4"
  "sn_scanobj|configs/selfevo_fscil_sn55_scanobjectnn_4t.json|abl_sn_scanobj|4"
)

# variant_id|label|extra --set pairs as KEY=VALUE lines
VARIANTS=(
  "ridge|Ridge only"
  "cache|Cache only"
  "static|Ridge+cache (static)"
  "rules|Ridge+cache+evolution"
)

log() { echo "[$(date -Is)] $*"; }

sets_for_variant() {
  case "$1" in
    ridge) echo fscil_preset=ridge_only ;;
    cache) echo fscil_preset=cache_only ;;
    static) echo fscil_preset=static_fusion ;;
    rules) echo fscil_preset=canonical ;;
    *)
      echo "Unknown variant: $1" >&2
      return 1
      ;;
  esac
}

log_is_complete() {
  local logfile="$1"
  local n_sess="$2"
  [[ -f "${logfile}" ]] || return 1
  local last
  last="$(grep "CNN top1 curve:" "${logfile}" | tail -1 || true)"
  [[ -n "${last}" ]] || return 1
  # Count commas + 1 ≈ number of sessions reported in the last curve.
  local n
  n="$(echo "${last}" | tr -cd ',' | wc -c)"
  n=$((n + 1))
  [[ "${n}" -ge "${n_sess}" ]]
}

run_one() {
  local proto_id="$1"
  local config="$2"
  local prefix="$3"
  local n_sess="$4"
  local var_id="$5"
  local var_label="$6"

  if [[ -n "${ONLY_PROTO}" && "${proto_id}" != "${ONLY_PROTO}" ]]; then
    return 0
  fi
  if [[ -n "${ONLY_VAR}" && "${var_id}" != "${ONLY_VAR}" ]]; then
    return 0
  fi

  local method="${prefix}_${var_id}"
  local logfile="${LOGROOT}/${method}_seed=2008.log"

  if [[ "${SKIP_DONE}" -eq 1 ]] && log_is_complete "${logfile}" "${n_sess}"; then
    log "SKIP ${method} — complete (${n_sess} sessions)"
    return 0
  fi

  local -a extra_args=()
  while IFS= read -r line; do
    [[ -n "${line}" ]] && extra_args+=(--set "${line}")
  done < <(sets_for_variant "${var_id}")

  # NOTE: do NOT use train_fscil.py --toy here — it forces max_classes=20 and
  # breaks SN55 (25 base) / ScanObjectNN protocols. Fast mode keeps the real
  # class schedule and only shortens warmups + stops after session 1.
  local -a toy_args=()
  if [[ "${TOY}" -eq 1 ]]; then
    toy_args=(
      --max-task 1
      --set tuned_epoch=1
      --set increment_tuned_epoch=1
      --set fscil_task1_warmup_epochs=1
      --set fscil_last_task_warmup_epochs=1
      --set task0_save_warmup_ckpts=false
    )
    method="toy_${prefix}_${var_id}"
    logfile="${LOGROOT}/${method}_seed=2008.log"
  fi

  log "=== ${method} | ${proto_id} | ${var_label} (config=${config}) ==="
  "${RUNNER[@]}" train_fscil.py \
    --config "${config}" \
    --logdir "${LOGROOT}" \
    --method-name "${method}" \
    --log-layout flat \
    --seeds 2008 \
    "${toy_args[@]}" \
    "${extra_args[@]}" \
    2>&1 | tee "${LOGROOT}/${method}.log"
  log "=== ${method} done ==="
  grep "CNN top1 curve" "${LOGROOT}/${method}"*.log | tail -1 || true
  grep "FSCIL novel accuracy curve" "${LOGROOT}/${method}"*.log | tail -1 || true
  grep "FSCIL harmonic curve" "${LOGROOT}/${method}"*.log | tail -1 || true
}

# --- select which (protocol, variant) to run ---
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  idx="${SLURM_ARRAY_TASK_ID}"
  p_idx=$((idx / 4))
  v_idx=$((idx % 4))
  IFS='|' read -r proto_id config prefix n_sess <<< "${PROTOCOLS[$p_idx]}"
  IFS='|' read -r var_id var_label <<< "${VARIANTS[$v_idx]}"
  run_one "${proto_id}" "${config}" "${prefix}" "${n_sess}" "${var_id}" "${var_label}"
else
  # Sequential (local / no array): all combinations.
  for p_entry in "${PROTOCOLS[@]}"; do
    IFS='|' read -r proto_id config prefix n_sess <<< "${p_entry}"
    for v_entry in "${VARIANTS[@]}"; do
      IFS='|' read -r var_id var_label <<< "${v_entry}"
      run_one "${proto_id}" "${config}" "${prefix}" "${n_sess}" "${var_id}" "${var_label}"
    done
  done
  log "=== Summaries ==="
  "${RUNNER[@]}" scripts/summarize_fscil_regime_ablations.py \
    --log-root "${LOGROOT}" \
    --out "${LOGROOT}/regime_ablation_tables.md" || true
fi

log "simpl_runs_ablations_regimes finished — see ${LOGROOT}/"

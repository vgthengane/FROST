#!/bin/bash
#SBATCH --job-name=pride_hp_mn40
#SBATCH --account=EUHPC_D36_199
#SBATCH --partition=boost_usr_prod
#SBATCH --array=0-44
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=_experiments/slurm_logs/%A_%a_pride_hp_mn40.out

# =============================================================================
# PRidE HPARAM ablation — MN40, locked mean-only, 5 seeds, buffer-free
# =============================================================================
# Locked base (all rows inherit unless overridden):
#   mean-only ProtoBank, ridge_rp_dim=2000, α=0.5, ffn_num=64,
#   max pool, static, base-only EP10, conflict train-CM
#
# Grid (paper-style):
#   α:            0.1, 0.3, 0.5, 0.7, 0.9
#   adapter width: 32, 64, 128, 256   (128 replaces typo "120")
#   ridge RP:     500, 1000, 2000, 4000
#
# Defaults are NEVER scheduled here (reuse final_sub / summarizer):
#   α=0.5 + ffn=64 + rp=2000  ← 00_FINAL_SUB_MEAN_A05_RP2000_*
#   rp=1000                   ← 00_FINAL_SUB_MEAN_A05_RP1000_*
#
# This array only launches the 9 non-default cells × 5 seeds = 45 jobs:
#   0–4   alpha01     5–9   alpha03     10–14 alpha07     15–19 alpha09
#   20–24 adapter32   25–29 adapter128  30–34 adapter256
#   35–39 rp500       40–44 rp4000
#
# Logs (same root as component ablation):
#   _experiments/00_PRIDE_LOCKED_ABLATION_MN40_5SEEDS/
#     pride_mn40_abl_<id>_seed=<seed>.log
#
# Skip-done is ON by default: complete mean-only logs are not re-run.
#
# Submit:
#   sbatch scripts/simpl_runs_pride_hparam_ablation_mn40_locked_5seeds.sh
# Force re-run a cell:
#   sbatch scripts/simpl_runs_pride_hparam_ablation_mn40_locked_5seeds.sh --force
# Summarize full grid (incl. reused defaults):
#   aprun python scripts/summarize_pride_locked_ablation_mn40.py
# =============================================================================

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

CONFIG="configs/selfevo_fscil_mn40_5t_flex_compare.json"
LOGROOT="_experiments/00_PRIDE_LOCKED_ABLATION_MN40_5SEEDS"
EXPECTED_SESSIONS=5
RUNNER=(aprun python)
SEEDS=(2008 1947 1991 2001 2019)

# Only non-default variants (defaults reused by summarizer).
VARIANTS=(
  "alpha01|Fusion alpha=0.1"
  "alpha03|Fusion alpha=0.3"
  "alpha07|Fusion alpha=0.7"
  "alpha09|Fusion alpha=0.9"
  "adapter32|Adapter width ffn_num=32"
  "adapter128|Adapter width ffn_num=128"
  "adapter256|Adapter width ffn_num=256"
  "rp500|Ridge RP dim=500"
  "rp4000|Ridge RP dim=4000"
)

SMOKE=0
SKIP_DONE=1
ONLY=""
ONLY_SEED=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local) shift ;;
    --smoke) SMOKE=1; shift ;;
    --skip-done) SKIP_DONE=1; shift ;;
    --force|--no-skip-done) SKIP_DONE=0; shift ;;
    --only) ONLY="${2:?}"; shift 2 ;;
    --only-seed) ONLY_SEED="${2:?}"; shift 2 ;;
    *)
      echo "Usage: $0 [--local] [--smoke] [--skip-done|--force] [--only ID] [--only-seed SEED]" >&2
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
  echo "ffn_num=64"
  echo "exemplar_buffer_shots=0"
  echo "exemplar_buffer_ridge_refresh_shots=0"
}

flags_for_variant() {
  case "$1" in
    alpha01)
      locked_base_flags
      echo "ridge_cache_alpha=0.1"
      echo "fscil_increment_ridge_alpha=0.1"
      echo "fscil_task1_ridge_alpha=0.1"
      ;;
    alpha03)
      locked_base_flags
      echo "ridge_cache_alpha=0.3"
      echo "fscil_increment_ridge_alpha=0.3"
      echo "fscil_task1_ridge_alpha=0.3"
      ;;
    alpha07)
      locked_base_flags
      echo "ridge_cache_alpha=0.7"
      echo "fscil_increment_ridge_alpha=0.7"
      echo "fscil_task1_ridge_alpha=0.7"
      ;;
    alpha09)
      locked_base_flags
      echo "ridge_cache_alpha=0.9"
      echo "fscil_increment_ridge_alpha=0.9"
      echo "fscil_task1_ridge_alpha=0.9"
      ;;
    adapter32)
      locked_base_flags
      echo "ffn_num=32"
      ;;
    adapter128)
      locked_base_flags
      echo "ffn_num=128"
      ;;
    adapter256)
      locked_base_flags
      echo "ffn_num=256"
      ;;
    rp500)
      locked_base_flags
      echo "ridge_rp_dim=500"
      ;;
    rp4000)
      locked_base_flags
      echo "ridge_rp_dim=4000"
      ;;
    *)
      echo "Unknown variant: $1" >&2
      return 1
      ;;
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
      --set tuned_epoch=1
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
  if [[ "${SMOKE}" -eq 0 ]]; then
    "${RUNNER[@]}" scripts/summarize_pride_locked_ablation_mn40.py \
      --log-root "${LOGROOT}"
  fi
fi

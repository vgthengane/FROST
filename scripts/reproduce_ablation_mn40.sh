#!/bin/bash
#SBATCH --job-name=frost_abl_mn40
#SBATCH --account=EUHPC_D36_199
#SBATCH --partition=boost_usr_prod
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/tmp/frost_abl_mn40_%A_%a.out

# =============================================================================
# FROST paper ablations on ModelNet40 (exemplar-free), single seed (1947).
# Matches Sec. ablation tables: component / α / adapter width / CorrMem V.
#
# Array 0-15 (16 jobs):
#   0-2   component: corrmem | protomem | fusion
#   3-7   alpha: 0.1 0.3 0.5 0.7 0.9
#   8-11  ffn_num: 32 64 128 256
#   12-15 ridge_rp_dim: 500 1000 2000 4000
#
#   sbatch scripts/reproduce_ablation_mn40.sh
#   bash scripts/reproduce_ablation_mn40.sh --array 0   # local/smoke one job
# =============================================================================

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"
# shellcheck source=frost_common.sh
source "${REPO_ROOT}/scripts/frost_common.sh"

frost_init_runner
SEED=1947
CONFIG="configs/frost_pc_in_mn40.json"
LOGROOT="_experiments/00_ABLATION_MN40_SEED${SEED}"
ONLY_IDX="${SLURM_ARRAY_TASK_ID:-}"
SMOKE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --array) ONLY_IDX="${2:?}"; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    --local) frost_init_runner; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "${LOGROOT}"

# tag|extra --set flags (space-separated key=val)
VARIANTS=(
  "comp_corrmem|classifier_mode=ridge"
  "comp_protomem|classifier_mode=cache"
  "comp_fusion|classifier_mode=ridge_cache"
  "alpha_0.1|ridge_cache_alpha=0.1 fscil_increment_ridge_alpha=0.1 fscil_task1_ridge_alpha=0.1"
  "alpha_0.3|ridge_cache_alpha=0.3 fscil_increment_ridge_alpha=0.3 fscil_task1_ridge_alpha=0.3"
  "alpha_0.5|ridge_cache_alpha=0.5 fscil_increment_ridge_alpha=0.5 fscil_task1_ridge_alpha=0.5"
  "alpha_0.7|ridge_cache_alpha=0.7 fscil_increment_ridge_alpha=0.7 fscil_task1_ridge_alpha=0.7"
  "alpha_0.9|ridge_cache_alpha=0.9 fscil_increment_ridge_alpha=0.9 fscil_task1_ridge_alpha=0.9"
  "ffn_32|ffn_num=32"
  "ffn_64|ffn_num=64"
  "ffn_128|ffn_num=128"
  "ffn_256|ffn_num=256"
  "rp_500|ridge_rp_dim=500"
  "rp_1000|ridge_rp_dim=1000"
  "rp_2000|ridge_rp_dim=2000"
  "rp_4000|ridge_rp_dim=4000"
)

N=${#VARIANTS[@]}
IDX="${ONLY_IDX:-0}"
if [[ -z "${ONLY_IDX}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  frost_log "No array index; running all ${N} variants sequentially"
  for ((IDX=0; IDX<N; IDX++)); do
    ONLY_IDX="${IDX}" "$0" --array "${IDX}" ${SMOKE:+--smoke}
  done
  exit 0
fi

if [[ "${IDX}" -lt 0 || "${IDX}" -ge "${N}" ]]; then
  frost_log "Skip idx=${IDX} (out of range 0..$((N-1)))"
  exit 0
fi

entry="${VARIANTS[$IDX]}"
tag="${entry%%|*}"
extra="${entry#*|}"

METHOD="abl_mn40_${tag}_seed=${SEED}"
LOGFILE="${LOGROOT}/${METHOD}.log"

if frost_log_is_complete "${LOGFILE}" 5; then
  frost_log "Skip done: ${LOGFILE}"
  exit 0
fi

EXTRA_ARGS=()
while IFS= read -r flag; do
  [[ -n "${flag}" ]] || continue
  EXTRA_ARGS+=(--set "${flag}")
done < <(frost_pc_recipe_flags; frost_pc_buffer_flags "buffree"; printf '%s\n' ${extra})
EXTRA_ARGS+=(
  --set "fscil_old_shot_replay_seed=${SEED}"
  --set "fscil_base_ridge_refresh_seed=${SEED}"
  --set task0_save_warmup_ckpts=false
  --set save_checkpoint=false
)
if [[ "${SMOKE}" -eq 1 ]]; then
  EXTRA_ARGS+=(--max-task 1 --set tuned_epoch=1)
fi

frost_log "Ablation idx=${IDX} tag=${tag} -> ${LOGFILE}"
mkdir -p "$(dirname "${LOGFILE}")"
"${RUNNER[@]}" train_fscil.py \
  --config "${CONFIG}" \
  --logdir "${LOGROOT}" \
  --method-name "${METHOD}" \
  --log-layout flat \
  --seeds "${SEED}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${LOGFILE}"

frost_log "Done ${METHOD}"

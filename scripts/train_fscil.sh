#!/bin/bash
#SBATCH --job-name=selfevo-cl-train-full
#SBATCH --account=EUHPC_D36_199
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=_experiments/slurm_logs/%j_%x.out

# Run from repo root:
#   bash scripts/train_cil.sh simplecil mos
#   bash scripts/train_cil.sh mos --seeds 1947,1991,2001,2008,2019
#
# Logs (flat layout):
#   _experiments/full_runs_5seeds/<method>/<method>_seed=<seed>.log
#   bash scripts/train_cil.sh mos --save-checkpoint          # keep per-task checkpoints
#   sbatch scripts/train_cil.sh l2p dualprompt coda_prompt

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

ARCHIVE_CONFIG_DIR="_archive/configs/cil"

RESUME=false
SAVE_CHECKPOINT=false
SEEDS=""
SKIP_SEEDS=""
METHODS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume) RESUME=true; shift ;;
    --save-checkpoint) SAVE_CHECKPOINT=true; shift ;;
    --seeds=*) SEEDS="${1#*=}"; shift ;;
    --seeds)
      SEEDS="${2:?--seeds requires a comma-separated list}"
      shift 2
      ;;
    --skip-seeds=*) SKIP_SEEDS="${1#*=}"; shift ;;
    --skip-seeds)
      SKIP_SEEDS="${2:?--skip-seeds requires a comma-separated list}"
      shift 2
      ;;
    -*) echo "Unknown option: $1" >&2; exit 1 ;;
    *) METHODS+=("$1"); shift ;;
  esac
done

if [[ ${#METHODS[@]} -eq 0 ]]; then
  echo "Usage: bash scripts/train_cil.sh <method> [method ...] [--resume] [--save-checkpoint] [--seeds S1,S2] [--skip-seeds S]" >&2
  exit 1
fi

LOG_ROOT="_experiments/full_runs_5seeds"
RESUME_ARGS=()
CHECKPOINT_ARGS=()
SEED_ARGS=()
if [[ "${RESUME}" == true ]]; then
  RESUME_ARGS=(--resume)
fi
if [[ "${SAVE_CHECKPOINT}" == true ]]; then
  CHECKPOINT_ARGS=(--save-checkpoint)
fi
if [[ -n "${SEEDS}" ]]; then
  SEED_ARGS+=(--seeds "${SEEDS}")
fi
if [[ -n "${SKIP_SEEDS}" ]]; then
  SEED_ARGS+=(--skip-seeds "${SKIP_SEEDS}")
fi

for METHOD in "${METHODS[@]}"; do
  case "${METHOD}" in
    selfevo) config_path="configs/selfevo_modelnet40.json" ;;
    selfevo_static) config_path="configs/selfevo_static_modelnet40.json" ;;
    selfevo_bandit) config_path="configs/selfevo_bandit_modelnet40.json" ;;
    selfevo_rl) config_path="configs/selfevo_rl_modelnet40.json" ;;
    selfevo_ridge) config_path="configs/selfevo_ridge_modelnet40.json" ;;
    selfevo_ridge_replay) config_path="configs/selfevo_ridge_replay_modelnet40.json" ;;
    selfevo_ridge_static) config_path="configs/selfevo_ridge_static_modelnet40.json" ;;
    selfevo_ridge_dataset) config_path="configs/selfevo_ridge_dataset_modelnet40.json" ;;
    selfevo_ridge_pool) config_path="configs/selfevo_ridge_pool_modelnet40.json" ;;
    selfevo_ridge_hardneg) config_path="configs/selfevo_ridge_hardneg_modelnet40.json" ;;
    selfevo_ridge_gated) config_path="configs/selfevo_ridge_gated_modelnet40.json" ;;
    selfevo_ridge_cache) config_path="configs/selfevo_ridge_cache_modelnet40.json" ;;
    selfevo_ridge_cache_gated) config_path="configs/selfevo_ridge_cache_gated_modelnet40.json" ;;
    selfevo_cache) config_path="configs/selfevo_cache_modelnet40.json" ;;
    selfevo_conflict) config_path="configs/selfevo_conflict_modelnet40.json" ;;
    *) config_path="configs/${METHOD}_modelnet40.json" ;;
  esac
  log_dir="${LOG_ROOT}/${METHOD}"

  if [[ ! -f "${config_path}" ]]; then
    archive_candidate="${ARCHIVE_CONFIG_DIR}/$(basename "${config_path}")"
    if [[ -f "${archive_candidate}" ]]; then
      config_path="${archive_candidate}"
    else
      echo "Config not found: ${config_path}" >&2
      exit 1
    fi
  fi

  echo "========== Starting RUN: ${METHOD} =========="
  aprun python train_cil.py \
    --config "${config_path}" \
    --logdir="${log_dir}" \
    --method-name="${METHOD}" \
    --log-layout flat \
    "${SEED_ARGS[@]}" \
    "${RESUME_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}"
  echo "========== Finished RUN: ${METHOD} ==========" >> "${LOG_ROOT}/${METHOD}.log"
done

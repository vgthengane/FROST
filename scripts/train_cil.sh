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
#   bash scripts/train_cil.sh l2p --resume
#   sbatch scripts/train_cil.sh l2p dualprompt coda_prompt

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

RESUME=false
METHODS=()
for arg in "$@"; do
  case "${arg}" in
    --resume) RESUME=true ;;
    -*) echo "Unknown option: ${arg}" >&2; exit 1 ;;
    *) METHODS+=("${arg}") ;;
  esac
done

if [[ ${#METHODS[@]} -eq 0 ]]; then
  echo "Usage: bash scripts/train_cil.sh <method> [method ...] [--resume]" >&2
  exit 1
fi

LOG_ROOT="_experiments/modelnet40/methods"
RESUME_ARGS=()
if [[ "${RESUME}" == true ]]; then
  RESUME_ARGS=(--resume)
fi

for METHOD in "${METHODS[@]}"; do
  config_path="configs/${METHOD}_modelnet40.json"
  log_dir="${LOG_ROOT}/${METHOD}"

  if [[ ! -f "${config_path}" ]]; then
    echo "Config not found: ${config_path}" >&2
    exit 1
  fi

  echo "========== Starting RUN: ${METHOD} =========="
  aprun python train_cil.py \
    --config "${config_path}" \
    --logdir="${log_dir}" \
    "${RESUME_ARGS[@]}"
  echo "========== Finished RUN: ${METHOD} ==========" >> "${LOG_ROOT}/${METHOD}.log"
done

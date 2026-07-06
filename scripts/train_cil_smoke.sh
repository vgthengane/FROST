#!/bin/bash
#SBATCH --job-name=selfevo-cl-smoke-test
#SBATCH --account=EUHPC_D36_199
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=20:00
#SBATCH --output=_experiments/slurm_logs/%j_%x.out

# Run from repo root:
#   bash scripts/train_cil_smoke.sh
#   bash scripts/train_cil_smoke.sh simplecil mos
#   sbatch scripts/train_cil_smoke.sh l2p dualprompt

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

ARCHIVE_CONFIG_DIR="_archive/configs/cil"

if [[ $# -eq 0 ]]; then
  set -- simplecil mos l2p dualprompt coda_prompt slca aper_adapter ease ranpac tuna
fi

LOG_ROOT="_experiments/modelnet40/methods_smoke"

for METHOD in "$@"; do
  config_path="configs/${METHOD}_modelnet40.json"
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

  echo "========== Starting SMOKE RUN: ${METHOD} =========="
  aprun python train_cil.py \
    --config "${config_path}" \
    --logdir="${log_dir}" \
    --smoke
  echo "========== Finished SMOKE RUN: ${METHOD} ==========" >> "${LOG_ROOT}/${METHOD}.log"
done

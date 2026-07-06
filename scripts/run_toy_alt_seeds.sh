#!/bin/bash
#SBATCH --job-name=toy-5seeds
#SBATCH --account=EUHPC_D36_199
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=_experiments/slurm_logs/%j_toy_5seeds.out

# Quick toy sweep (20 classes, 5 tasks) across alternative seeds.
#
#   sbatch scripts/run_toy_alt_seeds.sh
#   bash scripts/run_toy_alt_seeds.sh ranpac mos

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

SEEDS="1947,1991,2001,2008,2019"
METHODS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds=*) SEEDS="${1#*=}"; shift ;;
    --seeds) SEEDS="${2:?--seeds requires list}"; shift 2 ;;
    *) METHODS+=("$1"); shift ;;
  esac
done

if [[ ${#METHODS[@]} -eq 0 ]]; then
  METHODS=(simplecil ranpac mos)
fi

LOG_ROOT="_experiments/toy_runs_5seeds"
mkdir -p "${LOG_ROOT}" _experiments/slurm_logs

echo "Toy alt-seed sweep"
echo "  Seeds:   ${SEEDS}"
echo "  Methods: ${METHODS[*]}"
echo "  Log dir: ${LOG_ROOT}"
echo ""

resolve_config() {
  case "$1" in
    selfevo) echo "configs/selfevo_modelnet40.json" ;;
    selfevo_ridge) echo "configs/selfevo_ridge_modelnet40.json" ;;
    selfevo_ridge_cache) echo "configs/selfevo_ridge_cache_modelnet40.json" ;;
    *) echo "configs/${1}_modelnet40.json" ;;
  esac
}

for METHOD in "${METHODS[@]}"; do
  config_path="$(resolve_config "${METHOD}")"
  if [[ ! -f "${config_path}" ]]; then
    archive_candidate="_archive/configs/cil/$(basename "${config_path}")"
    if [[ ! -f "${archive_candidate}" ]]; then
      echo "SKIP ${METHOD} (missing ${config_path} and ${archive_candidate})" >&2
      continue
    fi
    config_path="${archive_candidate}"
  fi
  echo "========== TOY: ${METHOD} =========="
  aprun python train_cil.py \
    --config "${config_path}" \
    --logdir "${LOG_ROOT}/${METHOD}" \
    --method-name "${METHOD}" \
    --log-layout flat \
    --toy \
    --seeds "${SEEDS}"
done

echo ""
aprun python scripts/summarize_toy_multiseed.py --log-root "${LOG_ROOT}"

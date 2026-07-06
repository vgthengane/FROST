#!/bin/bash
#SBATCH --job-name=selfevo-toy
#SBATCH --account=EUHPC_D36_199
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=_experiments/slurm_logs/%j_%x.out

# Overnight toy comparison: SelfEvo variants vs SimpleCIL / MOS / EASE / TUNA.
#
# Toy settings (--toy): 20 classes, 5 tasks, 25 train / 10 test per class, seed 1993.
#
# Run from repo root:
#   bash scripts/train_cil_toy.sh
#   sbatch scripts/train_cil_toy.sh
#   bash scripts/train_cil_toy.sh selfevo mos
#
# Results summary (after run):
#   aprun python scripts/summarize_toy_results.py

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

if [[ $# -eq 0 ]]; then
  set -- \
    simplecil \
    selfevo \
    selfevo_static \
    selfevo_bandit \
    selfevo_rl \
    mos \
    ease \
    tuna
fi

LOG_ROOT="_experiments/modelnet40/toy"
RESULTS_FILE="${LOG_ROOT}/toy_results.txt"
mkdir -p "${LOG_ROOT}" _experiments/slurm_logs 2>/dev/null || true
: > "${RESULTS_FILE}"

echo "========================================" | tee -a "${RESULTS_FILE}"
echo " SelfEvo toy experiment — $(date -Iseconds)" | tee -a "${RESULTS_FILE}"
echo "========================================" | tee -a "${RESULTS_FILE}"

resolve_config() {
  local method="$1"
  case "${method}" in
    selfevo) echo "configs/selfevo_modelnet40.json" ;;
    selfevo_static) echo "configs/selfevo_static_modelnet40.json" ;;
    selfevo_bandit) echo "configs/selfevo_bandit_modelnet40.json" ;;
    selfevo_rl) echo "configs/selfevo_rl_modelnet40.json" ;;
    selfevo_ridge) echo "configs/selfevo_ridge_modelnet40.json" ;;
    selfevo_conflict) echo "configs/selfevo_conflict_modelnet40.json" ;;
    *) echo "configs/${method}_modelnet40.json" ;;
  esac
}

for METHOD in "$@"; do
  config_path="$(resolve_config "${METHOD}")"
  log_dir="${LOG_ROOT}/${METHOD}"

  if [[ ! -f "${config_path}" ]]; then
    archive_candidate="_archive/configs/cil/$(basename "${config_path}")"
    if [[ -f "${archive_candidate}" ]]; then
      config_path="${archive_candidate}"
    else
      echo "SKIP  ${METHOD}  (config missing: ${config_path})" | tee -a "${RESULTS_FILE}"
      continue
    fi
  fi

  echo "" | tee -a "${RESULTS_FILE}"
  echo ">>> ${METHOD}  config=${config_path}" | tee -a "${RESULTS_FILE}"
  run_log="/tmp/toy_${METHOD}_$$.log"

  if aprun python train_cil.py \
    --config "${config_path}" \
    --logdir="${log_dir}" \
    --toy >"${run_log}" 2>&1; then
    avg=$(grep -E "Average Accuracy \(CNN\)" "${run_log}" | tail -1 || true)
    last=$(grep -E "CNN top1 curve" "${run_log}" | tail -1 || true)
    echo "PASS  ${METHOD}  ${avg}" | tee -a "${RESULTS_FILE}"
    [[ -n "${last}" ]] && echo "      ${last}" | tee -a "${RESULTS_FILE}"
  else
    err=$(tail -5 "${run_log}" | tr '\n' ' ')
    echo "FAIL  ${METHOD}  ${err}" | tee -a "${RESULTS_FILE}"
  fi
done

echo "" | tee -a "${RESULTS_FILE}"
echo "========================================" | tee -a "${RESULTS_FILE}"
aprun python scripts/summarize_toy_results.py --log-root "${LOG_ROOT}" | tee -a "${RESULTS_FILE}"
echo "Full summary appended to ${RESULTS_FILE}"

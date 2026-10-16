#!/bin/bash
#SBATCH --job-name=frost_indomain
#SBATCH --account=EUHPC_D36_199
#SBATCH --partition=boost_usr_prod
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/tmp/frost_indomain_%A_%a.out

# =============================================================================
# FROST reproduce — in-domain 3D FSCIL (MN40, SN55, CO3D)
# =============================================================================
# Locked recipe: mean-only, α=0.5, rp=2000, EP10, static, test-CM off.
# Default runs BOTH buffer-free (EF) and buffer (EB).
#
# Default: first paper seed only (1947). Pass --fullseeds for all 5.
# Array (default both modes, 1 seed): 6 = 2 modes × 1 seed × 3 protocols
#   --fullseeds both modes: 30 = 2 × 5 × 3
#   flat = mode_idx * (n_seed*n_proto) + seed_idx * n_proto + proto_idx
#   mode 0 = buffree → 00_FINAL_SUB_MEAN_A05_RP2000_EP10_5SEEDS
#   mode 1 = buf     → 00_FINAL_SUB_MEAN_A05_RP2000_BUF_EP10_5SEEDS
#   seeds: 1947, 1991, 2001, 2008, 2019
#   proto: 0=mn40 1=sn55 2=co3d
#
# Examples:
#   sbatch scripts/reproduce_indomain.sh
#   sbatch --array=0-29 scripts/reproduce_indomain.sh --fullseeds
#   sbatch --array=0-14 scripts/reproduce_indomain.sh --buffer-free --fullseeds
#   bash scripts/reproduce_indomain.sh --only-protocol mn40 --only-seed 2008 --buffer-free
#
# Summarize:
#   python scripts/summarize_pride_multiseed.py --variant final_sub_both
# =============================================================================

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"
# shellcheck source=frost_common.sh
source "${REPO_ROOT}/scripts/frost_common.sh"

frost_init_runner
WANT_EF=0
WANT_EB=0
FULLSEEDS=0
SEEDS_OVERRIDE=0
SKIP_DONE=0
ONLY_IDX=""
ONLY_PROTOCOL=""
ONLY_SEED=""
SMOKE=0
SEEDS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local) frost_init_runner; shift ;;
    --buffer-free|--buffree|--ef) WANT_EF=1; shift ;;
    --buffer|--buf|--eb) WANT_EB=1; shift ;;
    --fullseeds) FULLSEEDS=1; shift ;;
    --skip-done) SKIP_DONE=1; shift ;;
    --smoke) SMOKE=1; shift ;;
    --array) ONLY_IDX="${2:?}"; shift 2 ;;
    --only-protocol) ONLY_PROTOCOL="${2:?}"; shift 2 ;;
    --only-seed) ONLY_SEED="${2:?}"; shift 2 ;;
    --seeds)
      IFS=',' read -r -a SEEDS <<< "${2:?}"
      SEEDS_OVERRIDE=1
      shift 2
      ;;
    -h|--help)
      sed -n '15,45p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if [[ "${SEEDS_OVERRIDE}" -eq 0 ]]; then
  frost_select_seeds SEEDS FROST_PC_SEEDS "${FULLSEEDS}"
fi

frost_parse_modes "${WANT_EF}" "${WANT_EB}"

JOBS=(
  "mn40|configs/frost_pc_in_mn40.json|mn40|5|MN40"
  "sn55|configs/frost_pc_in_sn55.json|sn55|7|SN55"
  "co3d|configs/frost_pc_in_co3d.json|co3d|6|CO3D"
)

frost_dispatch_pc JOBS SEEDS MODES RUNNER \
  "${ONLY_IDX}" "${SKIP_DONE}" "${SMOKE}" "${ONLY_PROTOCOL}" "${ONLY_SEED}"

frost_log "reproduce_indomain finished (modes=${MODES[*]})"

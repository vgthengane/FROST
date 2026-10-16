#!/bin/bash
#SBATCH --job-name=frost_xdomain
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
#SBATCH --output=/tmp/frost_xdomain_%A_%a.out

# =============================================================================
# FROST reproduce — cross-domain 3D FSCIL (MN→Scan, SN→Scan, SN→CO3D)
# =============================================================================
# Same locked recipe as reproduce_indomain.sh. Default: BOTH EF and EB.
#
# Default: first paper seed only. Pass --fullseeds for all 5.
# Array (default both modes, 1 seed): 6 = 2 × 1 × 3; --fullseeds: 30 = 2 × 5 × 3
#   proto: 0=mn_scanobj 1=sn_scanobj 2=sn_co3d
#
# Examples:
#   sbatch scripts/reproduce_crossdomain.sh
#   sbatch --array=0-29 scripts/reproduce_crossdomain.sh --fullseeds
#   sbatch --array=0-14 scripts/reproduce_crossdomain.sh --buffer-free --fullseeds
#   bash scripts/reproduce_crossdomain.sh --only-protocol sn_co3d --only-seed 2008 --buffer
#
# Summarize:
#   python scripts/summarize_frost_multiseed.py --variant final_sub_both
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
      sed -n '15,40p' "$0"
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
  "mn_scanobj|configs/frost_pc_cross_mn40_scanobj.json|mn_scanobj|4|MN→ScanObj"
  "sn_scanobj|configs/frost_pc_cross_sn55_scanobj.json|sn_scanobj|4|SN→ScanObj"
  "sn_co3d|configs/frost_pc_cross_sn55_co3d.json|sn_co3d|11|SN→CO3D"
)

frost_dispatch_pc JOBS SEEDS MODES RUNNER \
  "${ONLY_IDX}" "${SKIP_DONE}" "${SMOKE}" "${ONLY_PROTOCOL}" "${ONLY_SEED}"

frost_log "reproduce_crossdomain finished (modes=${MODES[*]})"

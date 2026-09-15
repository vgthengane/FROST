#!/bin/bash
#SBATCH --job-name=abl_adapt_all
#SBATCH --account=EUHPC_D36_199
#SBATCH --partition=boost_usr_prod
#SBATCH --array=0-13
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=_experiments/slurm_logs/%A_%a_abl_adapt_all.out

# Unified adapter campaign — MN40 + SN55, in-domain + cross-domain (seed 2008).
#
# Array 0-5  → in-domain  (see simpl_runs_adapter_ablation_indomain.sh)
# Array 6-13 → cross-domain (see simpl_runs_adapter_ablation_crossdomain.sh)
#
# Submit both datasets in one job:
#   sbatch scripts/simpl_runs_adapter_ablation_all.sh
#
# Or use the launcher:
#   bash scripts/run_adapter_ablation_campaign.sh submit-all

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

IDX="${SLURM_ARRAY_TASK_ID:-}"

if [[ -z "${IDX}" ]]; then
  echo "Set SLURM_ARRAY_TASK_ID or submit via sbatch (array 0-13)." >&2
  exit 1
fi

if [[ "${IDX}" -le 5 ]]; then
  export SLURM_ARRAY_TASK_ID="${IDX}"
  exec bash scripts/simpl_runs_adapter_ablation_indomain.sh "$@"
else
  export SLURM_ARRAY_TASK_ID="$((IDX - 6))"
  exec bash scripts/simpl_runs_adapter_ablation_crossdomain.sh "$@"
fi

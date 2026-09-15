#!/bin/bash
#SBATCH --job-name=simpl_abl_reg_toy
#SBATCH --account=EUHPC_D36_199
#SBATCH --partition=boost_usr_prod
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=_experiments/slurm_logs/%A_%a_simpl_abl_regimes_toy.out

# Fast overnight / morning checks for the 4-variant × 4-protocol regime matrix.
# Protocol-preserving "toy": real FSCIL class order, but 1 warmup epoch and
# --max-task 1 (only sessions 0–1). Does NOT use train_fscil.py --toy (that
# would clamp max_classes=20 and break SN55 / ScanObjectNN).
#
# Same story matrix as scripts/simpl_runs_ablations_regimes.sh:
#   ridge | cache | ridge_cache+static | ridge_cache+rules
# × MN40 | SN55 | MN→ScanObj | SN→ScanObj
#
# On Leonardo (preferred — 16 parallel jobs):
#   bash sync_eurohpc.sh --code --force
#   sbatch scripts/simpl_runs_ablations_regimes_toy.sh
#
# After finished / sync experiments back:
#   aprun python scripts/summarize_fscil_regime_ablations.py --toy
#
# Then launch FULL training in the morning:
#   sbatch scripts/simpl_runs_ablations_regimes.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

exec bash scripts/simpl_runs_ablations_regimes.sh --toy "$@"

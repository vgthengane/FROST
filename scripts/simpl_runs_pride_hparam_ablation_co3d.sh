#!/bin/bash
#SBATCH --job-name=pride_hp_co3d
#SBATCH --account=EUHPC_D36_199
#SBATCH --partition=boost_usr_prod
#SBATCH --array=0-19
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=_experiments/slurm_logs/%A_%a_pride_hp_co3d.out

# CO3D counterpart of the controlled PRidE hyperparameter ablation.
# The underlying launcher keeps every non-ablated value at the locked final
# default. All five alpha values run fresh under the same canonical config.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

export CONFIG="configs/selfevo_fscil_co3d_6t.json"
export LOGROOT="_experiments/00_PRIDE_HPARAM_ABLATION_CO3D"
export METHOD_PREFIX="pride_co3d_hp"
export EXPECTED_SESSIONS=6
export SUMMARY_SCRIPT="scripts/summarize_pride_hparam_ablation_co3d.py"
export RUN_ALL_ALPHAS=1
export RUN_COMPONENTS=1

exec bash scripts/simpl_runs_pride_hparam_ablation_mn40.sh "$@"

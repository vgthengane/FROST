#!/bin/bash
#SBATCH --job-name=ranpac_fscil_mn40
#SBATCH --account=EUHPC_D36_199
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=_experiments/slurm_logs/%j_ranpac_fscil_mn40.out

# Fair RanPAC FSCIL-3D MN40 vs PRidE component ablation (seed 2008, base adapter ep10).
# Submit: sbatch scripts/run_ranpac_fscil_mn40_quick.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p _experiments/slurm_logs _experiments/00_RANPAC_FSCIL_MN40

aprun python train_fscil.py \
  --config configs/ranpac_fscil_mn40_5t_fscil3d.json \
  --logdir _experiments/00_RANPAC_FSCIL_MN40 \
  --method-name ranpac_fscil_mn40 \
  --log-layout flat \
  --seeds 2008

#!/bin/bash
#SBATCH --job-name=mn40_bf_cache1
#SBATCH --account=EUHPC_D36_199
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=_experiments/slurm_logs/%j_mn40_bf_cache1.out

# ModelNet40 FSCIL-3D buffer-free run with a single cache prototype per class.
# Usage:
#   sbatch scripts/run_bufferfree_mn40_singlecache.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

LOGROOT="_experiments/00_SIMPL_RUNs"
METHOD="seel_mn40_bufferfree_cache1"
mkdir -p "${LOGROOT}" _experiments/slurm_logs

echo "[$(date -Is)] === ${METHOD} start (buffer-free + single-cache, config=selfevo_fscil_mn40_5t_flex_compare.json) ==="

aprun python train_fscil.py \
  --config configs/selfevo_fscil_mn40_5t_flex_compare.json \
  --logdir "${LOGROOT}" \
  --method-name "${METHOD}" \
  --log-layout flat \
  --set model_name=selfevo_bufferfree \
  --set max_prototypes_per_class=1 \
  --set fscil_increment_max_prototypes=1

echo "[$(date -Is)] === ${METHOD} done ==="

#!/bin/bash
#SBATCH --job-name=exp08-smoke
#SBATCH --partition=defq
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/exp08_smoke_%j.out

source /home/leej70/miniconda3/etc/profile.d/conda.sh
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="/tmp/exp08_smoke_mpl_${SLURM_JOB_ID}"

python src/analyze_exp08_background_mixed_effects.py \
  --output outputs/exp08_background_mixed_smoke \
  --scanners gt450 \
  --bands high \
  --cv-folds 3 \
  --workers 1

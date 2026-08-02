#!/bin/bash
#SBATCH --job-name=exp08-bg-mix
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/exp08_background_mixed_%j.out

source /home/leej70/miniconda3/etc/profile.d/conda.sh
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="/tmp/exp08_background_mixed_mpl_${SLURM_JOB_ID}"

python src/analyze_exp08_background_mixed_effects.py \
  --output outputs/exp08_background_mixed_effects \
  --cv-folds 5 \
  --workers "$SLURM_CPUS_PER_TASK"

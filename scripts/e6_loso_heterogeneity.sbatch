#!/bin/bash
#SBATCH --job-name=e6-hetero
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/e6_loso_heterogeneity_%j.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONNOUSERSITE=1

python src/analyze_e6_loso_heterogeneity.py --workers "$SLURM_CPUS_PER_TASK" "$@"

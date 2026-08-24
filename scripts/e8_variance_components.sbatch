#!/bin/bash
#SBATCH --job-name=e8-varcomp
#SBATCH --partition=defq
#SBATCH --array=0-3
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/e8_variance_components_%A_%a.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$JOB_WORKDIR/src"

python src/analyze_e8_variance_components.py --encoder-index "$SLURM_ARRAY_TASK_ID" "$@"

#!/bin/bash
#SBATCH --job-name=e8-manifold
#SBATCH --partition=defq
#SBATCH --array=0-1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/e8_manifold_distance_%A_%a.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$JOB_WORKDIR/src"

python src/analyze_e8_manifold_distance.py --encoder-index "$SLURM_ARRAY_TASK_ID" "$@"

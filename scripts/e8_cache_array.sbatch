#!/bin/bash
#SBATCH --job-name=e8-cache
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --array=0-108%24
#SBATCH --output=logs/e8_cache_%A_%a.out

source ~/.bashrc
conda activate "${JOB_CONDA_PREFIX:-cpath}"
set -euo pipefail
cd "${JOB_WORKDIR:-/mnt/isilon/oldridge_lab/leej/prenorm}"

export MPLCONFIGDIR="$PWD/.mplconfig"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/src"

python src/build_e8_cache.py --task-index "$SLURM_ARRAY_TASK_ID"

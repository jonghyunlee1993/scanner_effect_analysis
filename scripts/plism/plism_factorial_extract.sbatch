#!/bin/bash
#SBATCH --job-name=plism-fact-img
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=06:00:00
#SBATCH --array=0-90%30

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR="$JOB_OUTPUT_ROOT/.matplotlib"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export VIPS_CONCURRENCY="$SLURM_CPUS_PER_TASK"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python src/plism_factorial_external.py \
  --plism-root "$JOB_PLISM_ROOT" \
  --output-root "$JOB_OUTPUT_ROOT" \
  extract-task --task-index "$SLURM_ARRAY_TASK_ID" \
  --workers "$SLURM_CPUS_PER_TASK" --batch-size 16

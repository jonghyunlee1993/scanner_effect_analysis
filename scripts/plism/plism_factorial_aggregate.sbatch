#!/bin/bash
#SBATCH --job-name=plism-fact-agg
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

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

python src/plism_factorial_external.py \
  --plism-root "$JOB_PLISM_ROOT" \
  --output-root "$JOB_OUTPUT_ROOT" \
  aggregate --pannormal-root "$JOB_PANNORMAL_ROOT"

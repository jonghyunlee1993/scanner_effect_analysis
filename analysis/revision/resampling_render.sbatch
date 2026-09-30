#!/bin/bash
#SBATCH --job-name=rv10-render
#SBATCH --partition=defq
#SBATCH --array=0-102
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=analysis/revision/results/resampling_robustness/logs/render_%A_%a.log

# RV10 Stage B: one array task per cohort slide (needs Stage A geometry shards).
export PATH="${JOB_CONDA_PREFIX}/bin:${PATH}"
export PYTHONDONTWRITEBYTECODE=1
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
cd "${JOB_WORKDIR}"
set -euo pipefail

python analysis/revision/resampling_render.py

#!/bin/bash
#SBATCH --job-name=rv10-geom
#SBATCH --partition=defq
#SBATCH --array=0-102
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=analysis/revision/results/resampling_robustness/logs/geometry_%A_%a.log

# RV10 Stage A: one array task per cohort slide; aggregate with
#   python analysis/revision/resampling_geometry.py aggregate
# (RV10_COMMAND=aggregate and --array unset for the aggregation job).
export PATH="${JOB_CONDA_PREFIX}/bin:${PATH}"
export PYTHONDONTWRITEBYTECODE=1
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
cd "${JOB_WORKDIR}"
set -euo pipefail

python analysis/revision/resampling_geometry.py "${RV10_COMMAND:-slide}"

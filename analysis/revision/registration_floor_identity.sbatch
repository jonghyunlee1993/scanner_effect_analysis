#!/bin/bash
#SBATCH --job-name=rv08-identity
#SBATCH --partition=defq
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=analysis/revision/results/registration_floor/logs/identity_%j.log

# RV08 stage 1 (gate): re-read every set20 AT2 patch from the WSI and require
# bit-identity with the paired cache. Exits non-zero on any mismatch, so the
# embed/aggregate jobs submitted with --dependency=afterok never start.

export PATH="${JOB_CONDA_PREFIX}/bin:${PATH}"
export PYTHONPATH="${JOB_WORKDIR}:${JOB_WORKDIR}/src:${JOB_WORKDIR}/scripts"
export PYTHONDONTWRITEBYTECODE=1
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1
cd "${JOB_WORKDIR}"
set -euo pipefail

python analysis/revision/registration_floor.py identity --workers "${SLURM_CPUS_PER_TASK}"

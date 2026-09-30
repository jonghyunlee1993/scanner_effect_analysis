#!/bin/bash
#SBATCH --job-name=rv-p0b-agg
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=analysis/revision/results/set20_reaggregation/logs/aggregate_%j.log

export PATH="${JOB_CONDA_PREFIX}/bin:${PATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
cd "${JOB_WORKDIR}"
set -euo pipefail
export PYTHONPATH="${JOB_WORKDIR}/src"
export PYTHONDONTWRITEBYTECODE=1

python analysis/revision/set20_reaggregation.py aggregate

#!/bin/bash
#SBATCH --job-name=rv02-band-agg
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=analysis/revision/results/band_manipulation_set20/logs/aggregate_%j.log

export PATH="${JOB_CONDA_PREFIX}/bin:${PATH}"
export PYTHONPATH="${JOB_WORKDIR}/src:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${JOB_WORKDIR}/analysis/revision/results/band_manipulation_set20/.mplconfig"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
mkdir -p "${MPLCONFIGDIR}"
cd "${JOB_WORKDIR}"
set -euo pipefail

python analysis/revision/band_manipulation_set20.py

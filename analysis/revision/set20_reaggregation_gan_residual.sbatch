#!/bin/bash
#SBATCH --job-name=rv-p0b-gan
#SBATCH --partition=defq
#SBATCH --array=0-102
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:30:00
#SBATCH --output=analysis/revision/results/set20_reaggregation/logs/gan_residual_%A_%a.log

export PATH="${JOB_CONDA_PREFIX}/bin:${PATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
cd "${JOB_WORKDIR}"
set -euo pipefail
export PYTHONPATH="${JOB_WORKDIR}/src"
export PYTHONDONTWRITEBYTECODE=1

python analysis/revision/set20_reaggregation.py gan-residual \
    --task-index "${SLURM_ARRAY_TASK_ID}" --workers "${SLURM_CPUS_PER_TASK}"

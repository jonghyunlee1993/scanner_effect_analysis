#!/bin/bash
#SBATCH --job-name=aug-strong-sum
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=00:30:00
#SBATCH --output=analysis/paper/results/augmentation_strong_blur/logs/aggregate_%j.log

export PATH="${JOB_CONDA_PREFIX}/bin:${PATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
cd "${JOB_WORKDIR}"
set -euo pipefail

python analysis/paper/aggregate_augmentation_strong_blur.py

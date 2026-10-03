#!/bin/bash
#SBATCH --job-name=aug-strong-fig
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=analysis/paper/results/augmentation_strong_blur/logs/plot_%j.log

export PATH="${JOB_CONDA_PREFIX}/bin:${PATH}"
export MPLCONFIGDIR="${JOB_WORKDIR}/analysis/paper/results/augmentation_strong_blur/.mplconfig"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
mkdir -p "${MPLCONFIGDIR}"
cd "${JOB_WORKDIR}"
set -euo pipefail

python analysis/paper/plot_augmentation_strong_blur.py

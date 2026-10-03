#!/bin/bash
#SBATCH --job-name=aug-metric-aggregate
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=analysis/paper/results/augmentation_paired_metrics/logs/aggregate_%j.log

export PATH="${JOB_CONDA_PREFIX}/bin:${PATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
cd "${JOB_WORKDIR}"
set -euo pipefail

python analysis/paper/aggregate_augmentation_paired_metrics.py

#!/bin/bash
# RV04b: correction-independent and nonlinear target detectability.
# With --array=0-87: one dataset x PFM x method per task (all three classifiers).
# Without --array: aggregate (bootstrap CIs and summary.md).
#SBATCH --job-name=rv04b-detect
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=analysis/revision/results/detectability_within_fold/logs/rv04b_%A_%a.log

export PATH="${JOB_CONDA_PREFIX}/bin:${PATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONDONTWRITEBYTECODE=1
cd "${JOB_WORKDIR}"
set -euo pipefail

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  python analysis/revision/detectability_within_fold.py task --task-index "${SLURM_ARRAY_TASK_ID}"
else
  python analysis/revision/detectability_within_fold.py aggregate
fi

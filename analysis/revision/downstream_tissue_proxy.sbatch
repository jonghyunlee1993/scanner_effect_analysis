#!/bin/bash
# RV05 downstream tissue proxy. With --array: one task per dataset x PFM (nearest centroid) or
# dataset x PFM x scanner (logistic regression); list with
#   python analysis/revision/downstream_tissue_proxy.py list-tasks [--source smoke]
# Without --array: aggregate (recovery, Spearman, summary.md). JOB_SOURCE=smoke runs the
# smoke adapter into results/downstream_tissue_proxy/smoke.
#SBATCH --job-name=rv05-tissue
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=analysis/revision/results/downstream_tissue_proxy/logs/rv05_%A_%a.log

export PATH="${JOB_CONDA_PREFIX}/bin:${PATH}"
export PYTHONDONTWRITEBYTECODE=1
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
cd "${JOB_WORKDIR}"
set -euo pipefail

SOURCE="${JOB_SOURCE:-rv03}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  python analysis/revision/downstream_tissue_proxy.py task --task-index "${SLURM_ARRAY_TASK_ID}" --source "${SOURCE}"
else
  python analysis/revision/downstream_tissue_proxy.py aggregate --source "${SOURCE}"
fi

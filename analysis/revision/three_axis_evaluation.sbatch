#!/bin/bash
# RV04 three-axis evaluation. With --array: one task per dataset x PFM (metrics) or
# dataset x PFM x method (detectability); list with
#   python analysis/revision/three_axis_evaluation.py list-tasks [--source smoke]
# Without --array: aggregate (bootstrap, parity, summary.md). JOB_SOURCE=smoke runs the
# smoke adapter into results/three_axis_evaluation/smoke.
#SBATCH --job-name=rv04-3axis
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=analysis/revision/results/three_axis_evaluation/logs/rv04_%A_%a.log

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
  python analysis/revision/three_axis_evaluation.py task --task-index "${SLURM_ARRAY_TASK_ID}" --source "${SOURCE}"
else
  python analysis/revision/three_axis_evaluation.py aggregate --source "${SOURCE}"
fi

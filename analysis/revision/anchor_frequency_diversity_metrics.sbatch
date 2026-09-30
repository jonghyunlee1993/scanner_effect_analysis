#!/bin/bash
# RV13 stage 2 (CPU). With --array=0-38: one model x {detect, probe6, ri} per task; list with
#   python analysis/revision/anchor_frequency_diversity_metrics.py list-tasks
# Without --array: aggregate (bootstrap CIs, QC, H1-H4 read-outs, summary.md).
#SBATCH --job-name=rv13-metrics
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=analysis/revision/results/anchor_frequency_diversity/logs/metrics_%A_%a.log

export PATH="${JOB_CONDA_PREFIX}/bin:${PATH}"
export PYTHONPATH="${JOB_WORKDIR}:${JOB_WORKDIR}/src:${JOB_WORKDIR}/scripts:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export HDF5_USE_FILE_LOCKING=FALSE
export MPLCONFIGDIR="${JOB_WORKDIR}/analysis/revision/results/anchor_frequency_diversity/.mplconfig"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
mkdir -p "${MPLCONFIGDIR}"
cd "${JOB_WORKDIR}"
set -euo pipefail

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  python analysis/revision/anchor_frequency_diversity_metrics.py task --task-index "${SLURM_ARRAY_TASK_ID}"
else
  python analysis/revision/anchor_frequency_diversity_metrics.py aggregate
fi

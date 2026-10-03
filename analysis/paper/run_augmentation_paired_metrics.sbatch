#!/bin/bash
#SBATCH --job-name=aug-paired-metrics
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=analysis/paper/results/augmentation_paired_metrics/logs/%A_%a.log

export PATH="${JOB_CONDA_PREFIX}/bin:${PATH}"
export PYTHONPATH="${JOB_WORKDIR}/src:${PYTHONPATH:-}"
export TORCH_HOME="${JOB_TORCH_HOME}"
export MPLCONFIGDIR="${JOB_WORKDIR}/analysis/paper/results/augmentation_paired_metrics/.mplconfig"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
mkdir -p "${MPLCONFIGDIR}"
cd "${JOB_WORKDIR}"
set -euo pipefail

python analysis/paper/augmentation_paired_metrics.py \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --task-count "${JOB_TASK_COUNT}" \
  --max-slides "${JOB_MAX_SLIDES:-0}" \
  --batch-size 16

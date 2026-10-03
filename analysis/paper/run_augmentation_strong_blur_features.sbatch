#!/bin/bash
#SBATCH --job-name=aug-strong-blur
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=analysis/paper/results/augmentation_strong_blur/logs/%A_%a.log

export PATH="${JOB_CONDA_PREFIX}/bin:${PATH}"
export PYTHONPATH="${JOB_WORKDIR}/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="${JOB_TORCH_HOME}"
export MPLCONFIGDIR="${JOB_WORKDIR}/analysis/paper/results/augmentation_strong_blur/.mplconfig"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
mkdir -p "${MPLCONFIGDIR}"
cd "${JOB_WORKDIR}"
set -euo pipefail

python analysis/paper/augmentation_strong_blur_features.py \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --task-count "${JOB_TASK_COUNT}" \
  --max-slides "${JOB_MAX_SLIDES:-0}"

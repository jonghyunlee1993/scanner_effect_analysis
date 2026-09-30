#!/bin/bash
#SBATCH --job-name=rv01-blur-feat
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-7
#SBATCH --output=analysis/revision/results/blur_sharpen_set20/logs/features_%A_%a.log

export PATH="${JOB_CONDA_PREFIX}/bin:${PATH}"
export PYTHONPATH="${JOB_WORKDIR}/src:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR="${JOB_WORKDIR}/analysis/revision/results/blur_sharpen_set20/.mplconfig"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
mkdir -p "${MPLCONFIGDIR}"
cd "${JOB_WORKDIR}"
set -euo pipefail

python analysis/revision/blur_sharpen_set20_features.py \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --task-count "${JOB_TASK_COUNT:-${SLURM_ARRAY_TASK_COUNT}}" \
  --max-slides "${JOB_MAX_SLIDES:-0}"

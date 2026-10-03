#!/usr/bin/env bash
#SBATCH --job-name=aug-ood-uni
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00

source ~/.bashrc
conda activate "${JOB_CONDA_PREFIX}"
set -euo pipefail

cd "${JOB_WORKDIR}"
mkdir -p "${JOB_OUTPUT_ROOT}/logs"
export MPLCONFIGDIR="${JOB_OUTPUT_ROOT}/.matplotlib"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

python src/augmentation_ood_uni.py \
  --source-root "${JOB_SOURCE_ROOT}" \
  --output-root "${JOB_OUTPUT_ROOT}" \
  extract-task \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --task-count "${JOB_TASK_COUNT}" \
  --max-slides "${JOB_MAX_SLIDES:-0}" \
  --batch-size 32

#!/usr/bin/env bash
#SBATCH --job-name=scan-batch-uni
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:a100-40g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=05:00:00

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

python src/scanner_batch_extensions.py \
  --source-root "${JOB_SOURCE_ROOT}" \
  --augmentation-root "${JOB_AUGMENTATION_ROOT}" \
  --output-root "${JOB_OUTPUT_ROOT}" \
  uni-extract-task --task-index "${SLURM_ARRAY_TASK_ID}" --task-count "${JOB_TASK_COUNT}" --batch-size 32

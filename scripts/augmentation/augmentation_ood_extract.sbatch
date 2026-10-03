#!/usr/bin/env bash
#SBATCH --job-name=aug-ood
#SBATCH --partition=defq
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00

source ~/.bashrc
conda activate "${JOB_CONDA_PREFIX}"
set -euo pipefail

cd "${JOB_WORKDIR}"
mkdir -p "${JOB_OUTPUT_ROOT}/logs"
export MPLCONFIGDIR="${JOB_OUTPUT_ROOT}/.matplotlib"
export NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

python src/augmentation_ood_study.py \
  --source-root "${JOB_SOURCE_ROOT}" \
  --output-root "${JOB_OUTPUT_ROOT}" \
  extract-slide \
  --slide-index "${SLURM_ARRAY_TASK_ID}" \
  --workers "${SLURM_CPUS_PER_TASK}"

#!/usr/bin/env bash
#SBATCH --job-name=final-img
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=04:00:00

source ~/.bashrc
conda activate "${JOB_CONDA_PREFIX}"
set -euo pipefail

cd "${JOB_WORKDIR}"
mkdir -p "${JOB_OUTPUT_ROOT}/logs"
export MPLCONFIGDIR="${JOB_OUTPUT_ROOT}/.matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

python src/final_image_study.py \
  --output-root "${JOB_OUTPUT_ROOT}" \
  extract-slide \
  --slide-index "${SLURM_ARRAY_TASK_ID}" \
  --workers "${SLURM_CPUS_PER_TASK}"

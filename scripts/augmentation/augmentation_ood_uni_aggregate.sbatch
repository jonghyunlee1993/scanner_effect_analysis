#!/usr/bin/env bash
#SBATCH --job-name=aug-ood-uni-agg
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00

source ~/.bashrc
conda activate "${JOB_CONDA_PREFIX}"
set -euo pipefail

cd "${JOB_WORKDIR}"
mkdir -p "${JOB_OUTPUT_ROOT}/logs"
export MPLCONFIGDIR="${JOB_OUTPUT_ROOT}/.matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

python src/augmentation_ood_uni.py \
  --source-root "${JOB_SOURCE_ROOT}" \
  --output-root "${JOB_OUTPUT_ROOT}" \
  aggregate

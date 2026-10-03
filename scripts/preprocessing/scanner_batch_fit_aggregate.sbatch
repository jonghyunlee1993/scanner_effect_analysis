#!/usr/bin/env bash
#SBATCH --job-name=scan-batch-param
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

source ~/.bashrc
conda activate "${JOB_CONDA_PREFIX}"
set -euo pipefail
cd "${JOB_WORKDIR}"
mkdir -p "${JOB_OUTPUT_ROOT}/logs"
export MPLCONFIGDIR="${JOB_OUTPUT_ROOT}/.matplotlib"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

python src/scanner_batch_extensions.py \
  --source-root "${JOB_SOURCE_ROOT}" \
  --augmentation-root "${JOB_AUGMENTATION_ROOT}" \
  --output-root "${JOB_OUTPUT_ROOT}" \
  fit-aggregate

#!/usr/bin/env bash
#SBATCH --job-name=paper-uni-feature
#SBATCH --partition=defq
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=analysis/paper/results/feature_uni_v1_internal_%j.out

source ~/.bashrc
conda activate "${JOB_CONDA_PREFIX}"
set -euo pipefail
cd "${JOB_WORKDIR}"
export PYTHONPATH="${JOB_WORKDIR}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
python -m manuscript_completion.feature_uni_v1 \
  --direction at2_to_scanner --train-panel 40 \
  --output "${JOB_OUTPUT_ROOT:-analysis/paper/results/feature_uni_v1/feature40_forward}"

#!/bin/bash
#SBATCH --job-name=rv10-uni
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=analysis/revision/results/resampling_robustness/logs/uni_%A_%a.log

# RV10 Stage C: UNI v1 embeddings of the Stage B renders; slides are split over
# the array as cohort[task::task_count]. CPU fallback (GPU cap reached):
#   sbatch -p defq --gres=none --cpus-per-task=16 --mem=48G --array=0-25 ...
export PATH="${JOB_CONDA_PREFIX}/bin:${PATH}"
export PYTHONDONTWRITEBYTECODE=1
export HDF5_USE_FILE_LOCKING=FALSE
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
cd "${JOB_WORKDIR}"
set -euo pipefail

python analysis/revision/resampling_uni.py ${RV10_UNI_ARGS:-}

#!/bin/bash
#SBATCH --job-name=band-crosspfm
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-14%3
#SBATCH --output=outputs/bandwise_crosspfm_2026-09-25/logs/%x_%A_%a.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"
export PYTHONPATH="$JOB_WORKDIR/src:$JOB_WORKDIR/scripts"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export HDF5_USE_FILE_LOCKING=FALSE

python -m scripts.frequency.review_bandwise_crosspfm \
  --task-index "$SLURM_ARRAY_TASK_ID" --batch-size 8

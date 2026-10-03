#!/bin/bash
#SBATCH --job-name=feature-fit
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/feature_crossencoder_review_2026-09-25/logs/%x_%A_%a.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"
export PYTHONPATH="$JOB_WORKDIR/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export HDF5_USE_FILE_LOCKING=FALSE

models=(uni2 virchow2 hoptimus1)
scanners=(versa akoya gt450 s360 s60)
task="${JOB_TASK_INDEX:-${SLURM_ARRAY_TASK_ID:-}}"
model_index=$((task / 5))
scanner_index=$((task % 5))
python -m scripts.features.review_feature_crossencoder_fit \
  --model "${models[$model_index]}" --scanner "${scanners[$scanner_index]}"

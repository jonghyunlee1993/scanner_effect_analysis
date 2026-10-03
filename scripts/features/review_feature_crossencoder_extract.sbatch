#!/bin/bash
#SBATCH --job-name=feature-raw
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/feature_crossencoder_review_2026-09-25/logs/%x_%j.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"
export PYTHONPATH="$JOB_WORKDIR/src:$JOB_WORKDIR/scripts:${PYTHONPATH:-}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export HDF5_USE_FILE_LOCKING=FALSE

task_index="${JOB_TASK_INDEX:-${SLURM_ARRAY_TASK_ID:-}}"
extra=()
if [[ -n "${JOB_MAX_SLIDES:-}" ]]; then
  extra+=(--max-slides "$JOB_MAX_SLIDES")
fi
python -m scripts.features.review_feature_crossencoder_extract \
  --model "$JOB_MODEL" --task-index "$task_index" \
  --batch-size "${JOB_BATCH_SIZE:-8}" "${extra[@]}"

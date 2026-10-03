#!/bin/bash
#SBATCH --job-name=gan-encoder
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/gan_encoder_review_2026-09-25/logs/%x_%j.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"
export PYTHONPATH="$JOB_WORKDIR/src:$JOB_WORKDIR/scripts:${PYTHONPATH:-}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export HDF5_USE_FILE_LOCKING=FALSE

fold="${JOB_FOLD:-${SLURM_ARRAY_TASK_ID:-}}"
extra=()
if [[ -n "${JOB_MAX_SLIDES:-}" ]]; then
  extra+=(--max-slides "$JOB_MAX_SLIDES")
fi
python -m scripts.features.review_gan_crossencoder \
  --model "$JOB_MODEL" --fold "$fold" --batch-size "${JOB_BATCH_SIZE:-8}" \
  "${extra[@]}"

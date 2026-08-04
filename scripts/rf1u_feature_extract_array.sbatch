#!/bin/bash
#SBATCH --job-name=rf1u-feat
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/rf1u_feature_%A_%a.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONNOUSERSITE=1

TARGETS=(at2 gt450 s60)
TARGET=${TARGETS[$((SLURM_ARRAY_TASK_ID / 4))]}
ENCODER=$((SLURM_ARRAY_TASK_ID % 4))

python src/extract_rf1u_features.py \
  --encoder-index "$ENCODER" \
  --target "$TARGET"

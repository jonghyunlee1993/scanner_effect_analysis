#!/bin/bash
#SBATCH --job-name=e4-conch-tail
#SBATCH --partition=gpuq
#SBATCH --array=40-108%4
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/e4_control_feature_conch_tail_%A_%a.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONNOUSERSITE=1

python src/extract_e4_control_features.py \
  --encoder-index 2 \
  --slide-index "$SLURM_ARRAY_TASK_ID" \
  "$@"

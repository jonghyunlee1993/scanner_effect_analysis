#!/bin/bash
#SBATCH --job-name=rf1-paired
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-14%6
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/rf1_paired_robust_%A_%a.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

python src/analyze_rf1_paired_robust.py \
  --task-index "$SLURM_ARRAY_TASK_ID" \
  --output outputs/rf1_improvement_pilot/paired_robust \
  --batch-size 4

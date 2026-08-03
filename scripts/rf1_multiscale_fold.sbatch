#!/bin/bash
#SBATCH --job-name=rf1-ms-fold
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-4%2
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=logs/rf1_multiscale_%A_%a.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export PYTHONPATH="$JOB_WORKDIR/src${PYTHONPATH:+:$PYTHONPATH}"
python src/analyze_rf1_multiscale.py \
  --mode fold \
  --fold "$SLURM_ARRAY_TASK_ID" \
  --batch-size 6

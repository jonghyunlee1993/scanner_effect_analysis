#!/bin/bash
#SBATCH --job-name=rf1-gamut-reinhard
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-14%4
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/rf1_gamut_reinhard_%A_%a.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONNOUSERSITE=1

python src/analyze_rf1_gamut_reinhard.py --task-index "$SLURM_ARRAY_TASK_ID"

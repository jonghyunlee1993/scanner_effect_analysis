#!/bin/bash
#SBATCH --job-name=e0-pfm-smoke
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:a100-40g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/e0_pfm_encoder_smoke_%j.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONNOUSERSITE=1

python src/smoke_e0_pfm_encoders.py "$@"

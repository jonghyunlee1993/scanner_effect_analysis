#!/bin/bash
#SBATCH --job-name=e4-hf-reference
#SBATCH --partition=gpuq
#SBATCH --array=0-2
#SBATCH --gres=gpu:a100-40g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/e4_global_hf_reference_%A_%a.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONNOUSERSITE=1

python src/build_e4_global_hf_reference.py --fov-index "$SLURM_ARRAY_TASK_ID" "$@"

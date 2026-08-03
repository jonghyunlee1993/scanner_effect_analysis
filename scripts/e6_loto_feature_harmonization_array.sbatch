#!/bin/bash
#SBATCH --job-name=e6-loto-feat
#SBATCH --partition=gpuq
#SBATCH --array=0-147%20
#SBATCH --gres=gpu:a100-40g:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/e6_loto_feature_harmonization_%A_%a.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONNOUSERSITE=1

python src/extract_e6_loto_feature_harmonization.py --task-index "$SLURM_ARRAY_TASK_ID" "$@"

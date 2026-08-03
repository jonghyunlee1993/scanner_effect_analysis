#!/bin/bash
#SBATCH --job-name=e5-rf1-feat
#SBATCH --partition=gpuq
#SBATCH --array=0-3
#SBATCH --gres=gpu:a100-40g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/e5_rf1_feature_%A_%a.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONNOUSERSITE=1

python src/extract_e5_rf1_features.py --encoder-index "$SLURM_ARRAY_TASK_ID" "$@"


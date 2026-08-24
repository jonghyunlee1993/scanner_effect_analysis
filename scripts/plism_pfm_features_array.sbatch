#!/bin/bash
#SBATCH --job-name=plism-pfm
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/plism_pfm_%A_%a.out

source ~/.bashrc
conda activate "${JOB_CONDA_PREFIX:-cpath}"
set -euo pipefail
cd "${JOB_WORKDIR:-/mnt/isilon/oldridge_lab/leej/prenorm}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="$PWD/.mplconfig"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/src"

python src/extract_plism_pfm_features.py \
  --condition "$PLISM_CONDITION" \
  --encoder-index "$SLURM_ARRAY_TASK_ID"

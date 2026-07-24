#!/bin/bash
# Exp-02 publication figure rendering from extracted pairwise UNI metrics.
#SBATCH --job-name=exp02_render
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/exp02_render_%j.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"
export PYTHONPATH=$PWD/src
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/tmp/matplotlib-$SLURM_JOB_ID
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

python src/render_exp02_pairwise_uni.py \
  --input "${EXP02_INPUT:-outputs/exp02_visualization}" \
  --output "${EXP02_OUTPUT:-outputs/exp02_visualization/figures}"

#!/bin/bash
# Exp-02 high-frequency perturbation trajectory metrics and figures.
#SBATCH --job-name=exp02_hf_render
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/exp02_hf_render_%j.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"
export PYTHONPATH=$PWD/src
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/tmp/matplotlib-$SLURM_JOB_ID
export NUMBA_CACHE_DIR=/tmp/numba-$SLURM_JOB_ID
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TF_CPP_MIN_LOG_LEVEL=2

python src/render_exp02_hf_trajectory.py \
  --input "${EXP02_HF_INPUT:-outputs/exp02_hf_trajectory}" \
  --output "${EXP02_HF_OUTPUT:-outputs/exp02_hf_trajectory/figures}"

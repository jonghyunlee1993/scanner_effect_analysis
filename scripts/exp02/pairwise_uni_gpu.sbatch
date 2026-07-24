#!/bin/bash
# Exp-02 pairwise scanner rendering + UNI/blur-frontier extraction.
#SBATCH --job-name=exp02_pair_uni
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/exp02_pairwise_uni_%j.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"
export PYTHONPATH=$PWD/src
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  python src/run_exp02_pairwise_uni.py \
    --config "${PRENORM_CONFIG:-configs/experiments/exp01a_lf16_10slide.yaml}" \
    --output "${EXP02_OUTPUT:-outputs/exp02_visualization}" \
    --fit-max-locations "${EXP02_FIT_MAX:-600}" \
    --eval-max-locations "${EXP02_EVAL_MAX:-160}"

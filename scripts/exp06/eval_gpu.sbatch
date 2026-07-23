#!/bin/bash
# Band-specific Exp-06 evaluation.
#SBATCH --job-name=exp06_eval
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/exp06_eval_%j.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"
export PYTHONPATH=$PWD/src
: "${EVAL_CHECKPOINT:?Set EVAL_CHECKPOINT to an Exp-06 checkpoint}"
srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  python src/eval_exp06.py \
    --config "${PRENORM_CONFIG:-configs/experiments/exp06a_lf16_10slide.yaml}" \
    --checkpoint "$EVAL_CHECKPOINT"

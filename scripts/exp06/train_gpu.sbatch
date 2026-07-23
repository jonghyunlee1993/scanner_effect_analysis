#!/bin/bash
# Exp-06 GPU smoke or full training. PRENORM_CONFIG selects LF16/LF8/LF4.
#SBATCH --job-name=exp06_train
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:a100-40g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/exp06_train_%j.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"
export PYTHONPATH=$PWD/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

extra=()
if [[ "${EXP06_FAST_DEV_RUN:-0}" == "1" ]]; then
  extra+=(--fast-dev-run)
fi
srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  python src/train_exp06.py \
    --config "${PRENORM_CONFIG:-configs/experiments/exp06a_lf16_10slide.yaml}" \
    "${extra[@]}"

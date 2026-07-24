#!/bin/bash
# Exp-02 UNI blur/sharpen trajectories after rotating LF affine alignment.
#SBATCH --job-name=exp02_lf_hf
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/exp02_lf_aligned_hf_%j.out

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
  python src/run_exp02_lf_aligned_hf_trajectory.py \
    --config "${PRENORM_CONFIG:-configs/experiments/exp01a_lf16_10slide.yaml}" \
    --output "${EXP02_LF_HF_OUTPUT:-outputs/exp02_lf_aligned_hf_trajectory}" \
    --locations "${EXP02_LF_HF_LOCATIONS:-3}" \
    --fit-max-locations "${EXP02_LF_HF_FIT_MAX:-600}"

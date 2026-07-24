#!/bin/bash
# Exp-02 high-frequency perturbation UNI trajectory extraction.
#SBATCH --job-name=exp02_hf_traj
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/exp02_hf_trajectory_%j.out

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
  python src/run_exp02_hf_trajectory.py \
    --config "${PRENORM_CONFIG:-configs/experiments/exp01a_lf16_10slide.yaml}" \
    --output "${EXP02_HF_OUTPUT:-outputs/exp02_hf_trajectory}" \
    --locations "${EXP02_HF_LOCATIONS:-3}" \
    --train-mean-locations "${EXP02_HF_TRAIN_MEAN:-256}"

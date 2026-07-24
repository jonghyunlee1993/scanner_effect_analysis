#!/bin/bash
# Exp-02 UNI embedding correction degree (oracle + fingerprint-estimated).
#SBATCH --job-name=exp02_uni
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/exp02_uni_%j.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"
export PYTHONPATH=$PWD/src
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

echo "### Exp-02 UNI embedding correction degree"
srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  python src/eval_exp02_uni.py \
    --config "${PRENORM_CONFIG:-configs/experiments/exp01a_lf16_10slide.yaml}" \
    --max-per-scanner "${EXP02_UNI_MAX:-300}"

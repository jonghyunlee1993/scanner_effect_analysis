#!/bin/bash
# Exp-07 high-frequency probe: AT2 detail transplant oracle + attenuation frontier on UNI.
#SBATCH --job-name=exp07_hf
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/exp07_uni_bands_%j.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"
export PYTHONPATH=$PWD/src
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

echo "### Exp-07 UNI band-manipulation probe (hi-freq oracle + frontier)"
srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  python src/eval_exp07_uni_bands.py \
    --config "${PRENORM_CONFIG:-configs/experiments/exp06a_lf16_10slide.yaml}" \
    --max-per-scanner "${EXP07_UNI_MAX:-300}"

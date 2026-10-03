#!/bin/bash
#SBATCH --job-name=gan-pair-audit
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/gan_encoder_review_2026-09-25/logs/%x_%j.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
python -m scripts.features.review_gan_prediction_audit

#!/bin/bash
#SBATCH --job-name=rf1-pair-agg
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/rf1_paired_robust_aggregate_%j.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

python src/analyze_rf1_paired_robust.py \
  --aggregate \
  --output outputs/rf1_improvement_pilot/paired_robust

#!/bin/bash
#SBATCH --job-name=direct-slide-summary
#SBATCH --partition=defq
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=analysis/paper/results/direct_slide_lmm/logs/aggregate_%j.log

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

python analysis/paper/slide_level_lmm.py aggregate \
  --output-root "${JOB_OUTPUT_ROOT:-analysis/paper/results/direct_slide_lmm}"

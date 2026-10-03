#!/bin/bash
#SBATCH --job-name=direct-slide-input
#SBATCH --partition=defq
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=analysis/paper/results/direct_slide_lmm/logs/prepare_%j.log

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

python analysis/paper/build_direct_slide_contrasts.py \
  --cohort outputs/final_image_study_v1/00_contract/cohort.csv \
  --metrics-root outputs/final_image_study_v1/02_image_phenotypes/shards \
  --bands outputs/final_image_study_v1/03_frequency/band_summary.csv \
  --output "${JOB_INPUT:-analysis/paper/results/direct_slide_lmm/direct_slide_contrasts.csv}"

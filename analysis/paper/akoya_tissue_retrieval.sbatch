#!/bin/bash
#SBATCH --job-name=akoya-tissue-retrieval
#SBATCH --partition=defq
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=analysis/paper/results/frequency_pfm_audit/logs/tissue_retrieval_%j.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

python analysis/paper/akoya_tissue_retrieval.py \
  --cohort "$JOB_PROJECT_ROOT/outputs/scanner_batch_effect_analysis_2026-09-17/00_contract/cohort.csv" \
  --shards "$JOB_PROJECT_ROOT/outputs/scanner_batch_effect_analysis_2026-09-17/03_uni/shards" \
  --output analysis/paper/results/frequency_pfm_audit \
  --replicates 20000

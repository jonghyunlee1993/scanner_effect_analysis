#!/bin/bash
#SBATCH --job-name=akoya-retention-ci
#SBATCH --partition=defq
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=analysis/paper/results/frequency_pfm_audit/logs/retention_bootstrap_%j.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

python analysis/paper/bootstrap_akoya_tissue_retention.py \
  --input analysis/paper/results/frequency_pfm_audit/akoya_tissue_probe_predictions.csv \
  --output analysis/paper/results/frequency_pfm_audit/akoya_tissue_probe_paired_bootstrap.csv \
  --replicates 20000

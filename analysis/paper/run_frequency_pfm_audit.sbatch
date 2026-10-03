#!/usr/bin/env bash
#SBATCH --partition=defq
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --job-name=frequency_pfm_audit

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"
python analysis/paper/frequency_pfm_audit.py

#!/usr/bin/env bash
#SBATCH --partition=defq
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --job-name=akoya_uni_retention

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"
export OPENBLAS_NUM_THREADS=1
python analysis/paper/akoya_uni_retention_audit.py

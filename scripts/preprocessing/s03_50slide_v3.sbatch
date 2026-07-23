#!/bin/bash
# One independent full-cohort slide per task; submit with --array=0-49%24.
#SBATCH --job-name=s03_v3_50
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=logs/s03_v3_50_%A_%a.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"
export PYTHONPATH=$PWD/src
export PRENORM_CONFIG=configs/main/phase_0_preprocessing_50slide_v3.yaml

python src/preprocess/s03_precompute_store.py

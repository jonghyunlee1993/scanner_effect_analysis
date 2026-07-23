#!/bin/bash
# One AT2/S60 slide per task. The 10-slide external cohort uses --array=0-9.
#SBATCH --job-name=s60_s03
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=logs/s60_s03_%A_%a.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"
export PYTHONPATH=$PWD/src
export PRENORM_CONFIG=configs/main/external_s60_preprocessing.yaml
python src/preprocess/s03_precompute_store.py

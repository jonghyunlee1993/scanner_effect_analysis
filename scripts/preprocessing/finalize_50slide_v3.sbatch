#!/bin/bash
# Merge full-cohort sidecars and fit the train-only AT2 stain reference.
#SBATCH --job-name=finalize_v3_50
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/finalize_v3_50_%j.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"
export PYTHONPATH=$PWD/src
export PRENORM_CONFIG=configs/main/phase_0_preprocessing_50slide_v3.yaml

python src/preprocess/s04_build_index.py
python src/fit_stain_reference.py

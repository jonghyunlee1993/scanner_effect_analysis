#!/bin/bash
# Merge external S60 sidecars into the evaluation index.
#SBATCH --job-name=s60_s04
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/s60_s04_%j.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"
export PYTHONPATH=$PWD/src
export PRENORM_CONFIG=configs/main/external_s60_preprocessing.yaml
python src/preprocess/s04_build_index.py

#!/bin/bash
# One independent slide per task. Submit with --array=0-9.
#SBATCH --job-name=s03_v3_10
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/s03_v3_10_%A_%a.out

source ~/.bashrc
conda activate cpath
set -euo pipefail
cd /mnt/isilon/oldridge_lab/leej/prenorm
export PYTHONPATH=$PWD/src
export PRENORM_CONFIG=configs/main/phase_0_preprocessing_10slide_v3.yaml

python src/preprocess/s03_precompute_store.py

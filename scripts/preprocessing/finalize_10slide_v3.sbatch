#!/bin/bash
# Build the flat index and immutable train-split AT2 stain reference.
#SBATCH --job-name=finalize_v3_10
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/finalize_v3_10_%j.out

source ~/.bashrc
conda activate cpath
set -euo pipefail
cd /mnt/isilon/oldridge_lab/leej/prenorm
export PYTHONPATH=$PWD/src
export PRENORM_CONFIG=configs/main/phase_0_preprocessing_10slide_v3.yaml

python src/preprocess/s04_build_index.py
python src/fit_stain_reference.py

#!/bin/bash
# Build v3 grids and candidates for the 10-slide pilot.
#SBATCH --job-name=s01s02_v3_10
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=logs/s01s02_v3_10_%j.out

source ~/.bashrc
conda activate cpath
set -euo pipefail
cd /mnt/isilon/oldridge_lab/leej/prenorm
export PYTHONPATH=$PWD/src
export PRENORM_CONFIG=configs/main/phase_0_preprocessing_10slide_v3.yaml

python src/preprocess/s01_build_grid.py
python src/preprocess/s02_subsample.py

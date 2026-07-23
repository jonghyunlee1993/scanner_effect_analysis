#!/bin/bash
# Build grids and candidates for every registered slide in the current data root.
#SBATCH --job-name=s01s02_v3_109
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=logs/s01s02_v3_109_%j.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"
export PYTHONPATH=$PWD/src
export PRENORM_CONFIG=configs/main/phase_0_preprocessing_109slide_v3.yaml
python src/preprocess/s01_build_grid.py
python src/preprocess/s02_subsample.py

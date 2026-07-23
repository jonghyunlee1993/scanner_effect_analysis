#!/bin/bash
# Build grids and candidates for the full 50-slide 8:1:1 Phase-0 run.
#SBATCH --job-name=s01s02_v3_50
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=logs/s01s02_v3_50_%j.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"
export PYTHONPATH=$PWD/src
export PRENORM_CONFIG=configs/main/phase_0_preprocessing_50slide_v3.yaml

python src/preprocess/s01_build_grid.py
python src/preprocess/s02_subsample.py

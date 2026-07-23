#!/bin/bash
# Build AT2-lattice candidates for the external S60 cohort.
#SBATCH --job-name=s60_s01s02
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=logs/s60_s01s02_%j.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"
export PYTHONPATH=$PWD/src
export PRENORM_CONFIG=configs/main/external_s60_preprocessing.yaml
python src/preprocess/s01_build_grid.py
python src/preprocess/s02_subsample.py

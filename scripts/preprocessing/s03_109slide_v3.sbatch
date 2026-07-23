#!/bin/bash
# One independent slide per task; submit all 109 tasks with --array=0-108%109.
#SBATCH --job-name=s03_v3_109
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/s03_v3_109_%A_%a.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"
export PYTHONPATH=$PWD/src
export PRENORM_CONFIG=configs/main/phase_0_preprocessing_109slide_v3.yaml
python src/preprocess/s03_precompute_store.py

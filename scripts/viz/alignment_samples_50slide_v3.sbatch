#!/bin/bash
# Render ten stored registered tuples per slide; submit with --array=0-49%24.
#SBATCH --job-name=alignviz_v3_50
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=logs/alignviz_v3_50_%A_%a.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"
export PYTHONPATH=$PWD/src

python src/viz_alignment_samples.py --config configs/main/phase_0_preprocessing_50slide_v3.yaml --n-patches 10

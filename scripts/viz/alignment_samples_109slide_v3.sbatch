#!/bin/bash
# Render ten stored registered tuples per slide; submit all 109 tasks with --array=0-108%109.
#SBATCH --job-name=alignviz_v3_109
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --output=logs/alignviz_v3_109_%A_%a.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"
export PYTHONPATH=$PWD/src
python src/viz_alignment_samples.py --config configs/main/phase_0_preprocessing_109slide_v3.yaml --n-patches 10

#!/bin/bash
# Exp-07 qualitative 2-track panel.
#SBATCH --job-name=exp07_viz
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/exp07_viz_%j.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"
export PYTHONPATH=$PWD/src
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/tmp/matplotlib-$SLURM_JOB_ID

echo "### Exp-07 2-track visualization"
python src/viz_exp07_correction.py \
  --config "${PRENORM_CONFIG:-configs/experiments/exp06a_lf16_10slide.yaml}" \
  --per-scanner "${EXP07_PER_SCANNER:-3}"

#!/bin/bash
#SBATCH --job-name=e4-figure4
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/e4_control_figure_%j.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export MPLCONFIGDIR="/tmp/matplotlib-$SLURM_JOB_ID"
export PYTHONNOUSERSITE=1

python src/build_e4_control_figure.py "$@"

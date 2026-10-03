#!/usr/bin/env bash
#SBATCH --partition=defq
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --job-name=plism_axis_transfer

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="$JOB_WORKDIR/analysis/paper/results/scanner_color_frequency_space/.mplconfig"
python analysis/paper/plism_scanner_axis_transfer.py

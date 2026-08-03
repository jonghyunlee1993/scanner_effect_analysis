#!/bin/bash
#SBATCH --job-name=rf1-ms-visual
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/rf1_multiscale_visual_%j.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
export PYTHONPATH="$JOB_WORKDIR/src${PYTHONPATH:+:$PYTHONPATH}"
python src/analyze_rf1_multiscale.py --mode visual

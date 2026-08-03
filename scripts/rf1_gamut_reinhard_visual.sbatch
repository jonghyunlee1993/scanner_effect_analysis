#!/bin/bash
#SBATCH --job-name=rf1-gamut-visual
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/rf1_gamut_reinhard_visual_%j.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
export PYTHONNOUSERSITE=1

python src/render_rf1_gamut_reinhard_audit.py

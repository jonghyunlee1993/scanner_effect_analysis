#!/bin/bash
#SBATCH --job-name=pn-contrast
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --array=0-108%20
#SBATCH --output=logs/pn_contrast_%A_%a.out
source ~/.bashrc
conda activate "${JOB_CONDA_PREFIX:-cpath}"
set -euo pipefail
cd "${JOB_WORKDIR:-/mnt/isilon/oldridge_lab/leej/prenorm}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="$PWD/.mplconfig"; export PYTHONNOUSERSITE=1; export PYTHONPATH="$PWD/src"
python src/audit_pannormal_contrast.py --task-index "$SLURM_ARRAY_TASK_ID"

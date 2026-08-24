#!/bin/bash
#SBATCH --job-name=e9-pfm
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-90
#SBATCH --output=logs/e9_pfm_%A_%a.out

# One task per (section, scanner); 13 x 7 = 91.  The conda environment is chosen
# by encoder because the panel cannot share one -- see fetch_e9_pfm_checkpoints.
source ~/.bashrc
conda activate "$PFM_ENV"
set -euo pipefail
cd /mnt/isilon/oldridge_lab/leej/prenorm
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="$PWD/.mplconfig"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export VIPS_CONCURRENCY="$SLURM_CPUS_PER_TASK"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/src"

python src/extract_plism_core_pfm.py \
  --task-index "$SLURM_ARRAY_TASK_ID" \
  --encoder-id "$PFM_ENCODER" \
  "$@"

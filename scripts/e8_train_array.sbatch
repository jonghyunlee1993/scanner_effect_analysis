#!/bin/bash
#SBATCH --job-name=e8-train
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/e8_train_%A_%a.out

source ~/.bashrc
conda activate "${JOB_CONDA_PREFIX:-cpath}"
set -euo pipefail
cd "${JOB_WORKDIR:-/mnt/isilon/oldridge_lab/leej/prenorm}"

export MPLCONFIGDIR="$PWD/.mplconfig"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/src"

# `--gres=gpu:1` accepts either a100 or a100-40g so the array schedules against
# whichever type frees first; the model needs well under 40 GB.
python src/train_e8_residual.py \
  --stage "${E8_STAGE:-grid}" \
  --target "${E8_TARGET:-gt450}" \
  --task-index "$SLURM_ARRAY_TASK_ID"

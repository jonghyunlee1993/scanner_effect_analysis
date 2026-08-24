#!/bin/bash
#SBATCH --job-name=plism-core-cond
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-363%6
#SBATCH --output=logs/plism_core_cond_%A_%a.out

# One array per encoder, because the panel cannot share a conda environment.
# The index runs over 4 correction conditions x 91 (section, scanner) pairs;
# raw is not re-encoded, since the full feature store already holds it.
#
# Twelve cores per GPU, not six: the correction operators -- Reinhard's gamut
# projection and RF1U's Laplacian pyramid -- run on the CPU between the read and
# the encode, and a smoke run left the GPU idle most of the time waiting for them.
# A GPU node has 48 cores to 4 GPUs, so this takes the share that is already there.
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

CONDITIONS=(reinhard_at2 reinhard_gt450 rf1u_at2 rf1u_gt450)
CONDITION="${CONDITIONS[$((SLURM_ARRAY_TASK_ID / 91))]}"
TASK=$((SLURM_ARRAY_TASK_ID % 91))

python src/extract_plism_core_pfm.py \
  --task-index "$TASK" \
  --encoder-id "$PFM_ENCODER" \
  --condition "$CONDITION" \
  --locations outputs/plism_core_eval_locations \
  --output outputs/plism_core_condition_features \
  "$@"

#!/bin/bash
#SBATCH --job-name=rf1u-energy
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/rf1u_energy_%A_%a.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$JOB_WORKDIR/src${PYTHONPATH:+:$PYTHONPATH}"

TARGETS=(at2 gt450 s60)
TARGET=${TARGETS[$((SLURM_ARRAY_TASK_ID / 3))]}
FOV_TASK=$((SLURM_ARRAY_TASK_ID % 3))

python src/build_rf1m_slide_band_energy.py \
  --task-index "$FOV_TASK" \
  --target "$TARGET" \
  --output outputs/rf1u_multitarget/energy \
  --batch-size 4

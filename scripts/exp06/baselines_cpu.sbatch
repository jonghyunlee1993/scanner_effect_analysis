#!/bin/bash
# Train-only paired affine and fixed-target RGB Reinhard LF16 baselines.
#SBATCH --job-name=exp06_base
#SBATCH --partition=defq
#SBATCH --array=0-2%3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=logs/exp06_baseline_%A_%a.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"
export PYTHONPATH=$PWD/src
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
scanners=(gt450 versa s60)
srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  python src/eval_exp06_baselines.py \
    --config "${PRENORM_CONFIG:-configs/experiments/exp06a_lf16_10slide.yaml}" \
    --scanner "${scanners[$SLURM_ARRAY_TASK_ID]}"

#!/bin/bash
#SBATCH --job-name=e8-tgt-harm
#SBATCH --partition=defq
#SBATCH --array=0-871%120
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/e8_target_harmonization_%A_%a.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$JOB_WORKDIR/src"

# 4 encoders x 2 destinations x 109 slides, flattened by resolve_task().
python src/extract_e8_target_harmonization.py --task-index "$SLURM_ARRAY_TASK_ID" "$@"

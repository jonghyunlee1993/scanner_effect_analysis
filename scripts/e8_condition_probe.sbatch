#!/bin/bash
#SBATCH --job-name=e8-cond-probe
#SBATCH --partition=defq
#SBATCH --array=0-3
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/e8_condition_probe_%A_%a.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$JOB_WORKDIR/src"

# One array task per frozen encoder index (0 resnet50, 1 uni_v1, 2 conch_v1, 3 virchow2).
python src/analyze_e8_condition_probe.py --encoder-index "$SLURM_ARRAY_TASK_ID" "$@"

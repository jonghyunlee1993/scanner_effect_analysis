#!/bin/bash
#SBATCH --job-name=rf1u-probe
#SBATCH --partition=defq
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/rf1u_scanner_probe_%A_%a.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$JOB_WORKDIR/src${PYTHONPATH:+:$PYTHONPATH}"

python src/analyze_rf1u_scanner_probe.py --encoder-index "$SLURM_ARRAY_TASK_ID"

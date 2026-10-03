#!/bin/bash
#SBATCH --job-name=band-crosspfm-sum
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/bandwise_crosspfm_2026-09-25/logs/%x_%j.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"
export PYTHONPATH="$JOB_WORKDIR/src:$JOB_WORKDIR/scripts"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLCONFIGDIR="$JOB_WORKDIR/outputs/bandwise_crosspfm_2026-09-25/mplconfig"
mkdir -p "$MPLCONFIGDIR"

python -m scripts.frequency.review_bandwise_crosspfm_aggregate

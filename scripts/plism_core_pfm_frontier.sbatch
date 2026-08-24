#!/bin/bash
#SBATCH --job-name=plism-core-frontier
#SBATCH --partition=defq
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/plism_core_frontier_%j.out

# Image-space correction endpoints on the core grid.  Reads the raw condition
# out of the full feature store and the four correction conditions out of the
# fused extraction, both restricted to the frozen evaluation locations, so every
# contrast is paired on identical pixels.
source ~/.bashrc
conda activate "${JOB_CONDA_PREFIX:-cpath}"
set -euo pipefail
cd "${JOB_WORKDIR:-/mnt/isilon/oldridge_lab/leej/prenorm}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="$PWD/.mplconfig"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src"

python src/analyze_plism_core_pfm_frontier.py "$@"

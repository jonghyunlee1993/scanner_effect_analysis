#!/bin/bash
#SBATCH --job-name=e0-native-promote
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/e0_native_geometry_promote_%j.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONNOUSERSITE=1

ANALYSIS_PYTHON="/home/leej70/miniconda3/envs/cpath/bin/python"
"$ANALYSIS_PYTHON" src/promote_e0_native_geometry_fallback.py "$@"

#!/bin/bash
#SBATCH --job-name=e0-reg-single
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/e0_registration_single_%j.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONNOUSERSITE=1

REGISTRATION_PYTHON="/home/leej70/miniconda3/envs/cpath/bin/python"
"$REGISTRATION_PYTHON" src/run_e0_registration_sentinel.py "$@"

#!/bin/bash
# Full CPU regression suite for the active scanner-spectrum analysis code.
#SBATCH --job-name=prenorm_tests
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/tests_%j.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"
export PYTHONPATH=$PWD/src
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/tmp/matplotlib-$SLURM_JOB_ID

python -m pytest -p no:cacheprovider -q

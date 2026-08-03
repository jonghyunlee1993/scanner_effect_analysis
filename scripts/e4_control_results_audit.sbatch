#!/bin/bash
#SBATCH --job-name=e4-result-lock
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/e4_control_results_audit_%j.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export PYTHONNOUSERSITE=1

python src/audit_e4_control_results.py "$@"

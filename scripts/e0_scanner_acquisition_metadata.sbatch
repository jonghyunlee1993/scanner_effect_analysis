#!/bin/bash
#SBATCH --job-name=e0-scanner-meta
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=logs/e0_scanner_acquisition_metadata_%j.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python src/audit_e0_scanner_acquisition_metadata.py \
  --workers "$SLURM_CPUS_PER_TASK" \
  "$@"

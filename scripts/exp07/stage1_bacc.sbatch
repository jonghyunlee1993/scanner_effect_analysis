#!/bin/bash
# Exp-07 Stage-1 co-primary: low-band scanner separability (invariance).
#SBATCH --job-name=exp07_bacc
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/exp07_bacc_%j.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"
export PYTHONPATH=$PWD/src
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "### Exp-07 Stage-1 scanner-BACC (invariance co-primary)"
python src/eval_exp07_scanner_bacc.py \
  --config "${PRENORM_CONFIG:-configs/experiments/exp06a_lf16_10slide.yaml}" \
  --max-per-scanner "${EXP07_MAX_PER_SCANNER:-800}"

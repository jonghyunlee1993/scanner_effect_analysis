#!/bin/bash
# Exp-07 zero-shot probe: fingerprint -> scanner params, efficacy + invariance maintained.
#SBATCH --job-name=exp07_par
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=logs/exp07_params_%j.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"
export PYTHONPATH=$PWD/src
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "### Exp-07 scanner-parameter fingerprint probe"
python src/eval_exp07_scanner_params.py \
  --config "${PRENORM_CONFIG:-configs/experiments/exp06a_lf16_10slide.yaml}"

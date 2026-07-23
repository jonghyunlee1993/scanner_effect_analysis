#!/bin/bash
# Exp-07 Stage-1 Piece 1: coefficient-wise (local) range projection vs scalar.
#SBATCH --job-name=exp07_local
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=logs/exp07_local_%j.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"
export PYTHONPATH=$PWD/src
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "### Exp-07 projection unit contract (tests/test_exp07.py)"
python -m pytest -p no:cacheprovider -q tests/test_exp07.py

echo
echo "### Exp-07 Stage-1 local vs scalar projection efficacy"
python src/verify_exp07_local_projection.py \
  --config "${PRENORM_CONFIG:-configs/experiments/exp06a_lf16_10slide.yaml}" \
  --max-images "${EXP07_MAX_IMAGES:-512}" \
  --steps "${EXP07_STEPS:-150}"

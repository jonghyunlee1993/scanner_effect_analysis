#!/bin/bash
# Exp-07 Stage-0: projection unit contract + range-safe affine efficacy audit.
#SBATCH --job-name=exp07_stage0
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/exp07_stage0_%j.out

# Source env BEFORE `set -u`: /etc/bashrc references unbound BASHRCSOURCED.
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
echo "### Exp-07 Stage-0 range-safe affine efficacy audit"
python src/verify_exp07_stage0.py \
  --config "${PRENORM_CONFIG:-configs/experiments/exp06a_lf16_10slide.yaml}" \
  --max-images "${EXP07_MAX_IMAGES:-512}"

#!/bin/bash
# Exp-02 Stage-1A: classical low-band correction baselines on identical tuples.
#SBATCH --job-name=exp02_bl
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=logs/exp02_baselines_%j.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"
export PYTHONPATH=$PWD/src
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "### Exp-02 Stage-1A baselines"
python src/eval_exp02_baselines.py \
  --config "${PRENORM_CONFIG:-configs/experiments/exp01a_lf16_10slide.yaml}" \
  --max-train-images "${EXP02_MAX_TRAIN:-2048}" \
  --max-test-images "${EXP02_MAX_TEST:-512}" \
  --output "${EXP02_BASELINE_OUTPUT:-outputs/exp02_stage1/baselines.json}"

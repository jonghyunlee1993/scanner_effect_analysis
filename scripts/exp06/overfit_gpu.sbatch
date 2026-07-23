#!/bin/bash
# Fixed one-slide Exp-06 overfit gate.
#SBATCH --job-name=exp06_overfit
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:a100-40g:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/exp06_overfit_%j.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"
export PYTHONPATH=$PWD/src
extra=()
output="${OVERFIT_OUTPUT:-}"
if [[ "${OVERFIT_SWEEP:-0}" == "1" ]]; then
  weights=(7.4 8 10)
  weight="${weights[$SLURM_ARRAY_TASK_ID]}"
  if [[ -n "${OVERFIT_WARMUP_STEPS:-}" ]]; then
    output="outputs/exp06a_lf16_10slide/overfit_warm${OVERFIT_WARMUP_STEPS}_idem${weight}"
  else
    output="outputs/exp06a_lf16_10slide/overfit_idem${weight}"
  fi
  extra+=(--idempotence-weight "$weight")
elif [[ -n "${OVERFIT_IDEM_WEIGHT:-}" ]]; then
  extra+=(--idempotence-weight "$OVERFIT_IDEM_WEIGHT")
fi
if [[ -n "$output" ]]; then
  extra+=(--output "$output")
fi
if [[ -n "${OVERFIT_WARMUP_STEPS:-}" ]]; then
  extra+=(--warmup-steps "$OVERFIT_WARMUP_STEPS")
fi
srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  python src/overfit_exp06.py \
    --config "${PRENORM_CONFIG:-configs/experiments/exp06a_lf16_10slide.yaml}" \
    --steps "${OVERFIT_STEPS:-300}" "${extra[@]}"

#!/bin/bash
# RV18 stage 3: endpoints, trends over the AT2 share, arms minus UNI (CPU). JOB_EPOCHS, JOB_OUTPUT_NAME.
#SBATCH --job-name=rv18-metrics
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=analysis/revision/results/uni_continued/logs/metrics_%j.log

export PATH="${JOB_CONDA_PREFIX}/bin:${PATH}"
export PYTHONPATH="${JOB_WORKDIR}:${JOB_WORKDIR}/src:${JOB_WORKDIR}/scripts:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
cd "${JOB_WORKDIR}"
set -euo pipefail
extra=()
[[ -n "${JOB_OUTPUT_ROOT:-}" ]] && extra+=(--output-root "${JOB_OUTPUT_ROOT}")
[[ -n "${JOB_EPOCHS:-}" ]] && extra+=(--epochs "${JOB_EPOCHS}")
[[ -n "${JOB_OUTPUT_NAME:-}" ]] && extra+=(--output-name "${JOB_OUTPUT_NAME}")
[[ -n "${JOB_RUNS:-}" ]] && extra+=(--runs "${JOB_RUNS}")
python analysis/revision/uni_continued_metrics.py "${extra[@]}"

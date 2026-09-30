#!/bin/bash
#SBATCH --job-name=rv14-sens-robust
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=analysis/revision/results/sensitivity_robustness/logs/rv14_%j.log

export PATH="${JOB_CONDA_PREFIX}/bin:${PATH}"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${JOB_WORKDIR}/analysis/revision/results/sensitivity_robustness/logs/.mpl"
cd "${JOB_WORKDIR}"
set -euo pipefail

python analysis/revision/sensitivity_robustness.py

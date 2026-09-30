#!/bin/bash
#SBATCH --job-name=rv07-summary
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=analysis/revision/results/summary_figure/logs/rv07_%j.log

export PATH="${JOB_CONDA_PREFIX}/bin:${PATH}"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${JOB_WORKDIR}/analysis/revision/results/summary_figure/logs/.mpl"
cd "${JOB_WORKDIR}"
set -euo pipefail

python analysis/revision/summary_figure.py

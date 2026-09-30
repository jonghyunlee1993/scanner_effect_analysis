#!/bin/bash
#SBATCH --job-name=rv12-transfer
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=analysis/revision/results/frequency_transfer_figures/logs/rv12_%j.log

export PATH="${JOB_CONDA_PREFIX}/bin:${PATH}"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${JOB_WORKDIR}/analysis/revision/results/frequency_transfer_figures/logs/.mpl"
cd "${JOB_WORKDIR}"
set -euo pipefail

python analysis/revision/frequency_transfer_figures.py

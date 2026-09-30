#!/bin/bash
#SBATCH --job-name=rv11-interaction
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=analysis/revision/results/scanner_tissue_interaction/logs/rv11_%j.log

export PATH="${JOB_CONDA_PREFIX}/bin:${PATH}"
export PYTHONDONTWRITEBYTECODE=1
cd "${JOB_WORKDIR}"
set -euo pipefail

python analysis/revision/scanner_tissue_interaction.py

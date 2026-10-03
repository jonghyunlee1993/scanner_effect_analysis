#!/bin/bash
# RV18 round 3: dose-response of the band ratio over the AT2 share at epoch 1 (CPU).
#SBATCH --job-name=rv18-dose
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=analysis/revision/results/uni_continued/logs/dose_%j.log

export PATH="${JOB_CONDA_PREFIX}/bin:${PATH}"
export PYTHONPATH="${JOB_WORKDIR}:${JOB_WORKDIR}/src:${JOB_WORKDIR}/scripts:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
cd "${JOB_WORKDIR}"
set -euo pipefail
python analysis/revision/uni_continued_dose.py  # env: DOSE_ROOT, DOSE_SHARES, DOSE_MODEL

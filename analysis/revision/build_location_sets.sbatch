#!/bin/bash
#SBATCH --job-name=rv-locsets
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=analysis/revision/results/location_sets/logs/build_%j.log

export PATH="${JOB_CONDA_PREFIX}/bin:${PATH}"
cd "${JOB_WORKDIR}"
set -euo pipefail

python analysis/revision/build_location_sets.py

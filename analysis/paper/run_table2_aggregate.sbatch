#!/usr/bin/env bash
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=table2_aggregate

set -euo pipefail
: "${JOB_CONDA_PREFIX:?JOB_CONDA_PREFIX is required}"
: "${JOB_WORKDIR:?JOB_WORKDIR is required}"
export PATH="${JOB_CONDA_PREFIX}/bin:${PATH}"
export PYTHONPATH="${JOB_WORKDIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${JOB_WORKDIR}"
exec python analysis/paper/table2_benchmark.py aggregate

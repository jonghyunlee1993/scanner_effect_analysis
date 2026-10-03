#!/usr/bin/env bash
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=plism_gt450_agg

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export PYTHONPATH="$JOB_WORKDIR/src${PYTHONPATH:+:$PYTHONPATH}"
python analysis/paper/table2_gt450_extension.py "${JOB_STAGE:-aggregate}"

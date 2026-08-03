#!/bin/bash
#SBATCH --job-name=e5-figure
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/e5_comparator_figure_%j.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-e5-${SLURM_JOB_ID}"
export PYTHONNOUSERSITE=1

python src/build_e5_comparator_figure.py "$@"


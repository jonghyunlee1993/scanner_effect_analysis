#!/bin/bash
#SBATCH --job-name=blur-umap
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/e4_blur_umap_%j.out
source ~/.bashrc; conda activate "$JOB_CONDA_PREFIX"; set -euo pipefail; cd "$JOB_WORKDIR"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$JOB_WORKDIR/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-${SLURM_JOB_ID}"; mkdir -p "$MPLCONFIGDIR"
python src/render_e4_blur_sharpen_umap.py "$@"

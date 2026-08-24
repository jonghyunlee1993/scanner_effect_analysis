#!/bin/bash
#SBATCH --job-name=plism-core-featcorr
#SBATCH --partition=defq
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/plism_core_featcorr_%j.out

# CORAL / Procrustes on the core-grid embeddings, plus the CORAL sample-size
# sweep that settles whether the report's qualification against CORAL was an
# estimation artefact.  Memory is set by the largest encoder: 1536-d features at
# ~8,950 locations x 7 scanners x 13 sections, plus the D x D accumulators.
source ~/.bashrc
conda activate "${JOB_CONDA_PREFIX:-cpath}"
set -euo pipefail
cd "${JOB_WORKDIR:-/mnt/isilon/oldridge_lab/leej/prenorm}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="$PWD/.mplconfig"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src"

python src/analyze_plism_core_feature_correction.py "$@"

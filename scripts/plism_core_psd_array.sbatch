#!/bin/bash
#SBATCH --job-name=plism-core-psd
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --array=0-90%30
#SBATCH --output=logs/plism_core_psd_%A_%a.out

# Native-resolution radial spectra on the core grid.  No resampling; the patch is an integer crop at level 0.
source ~/.bashrc
conda activate "${JOB_CONDA_PREFIX:-cpath}"
set -euo pipefail
cd "${JOB_WORKDIR:-/mnt/isilon/oldridge_lab/leej/prenorm}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="$PWD/.mplconfig"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export VIPS_CONCURRENCY="$SLURM_CPUS_PER_TASK"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/src"

python src/extract_plism_native_psd.py \
  --task-index "$SLURM_ARRAY_TASK_ID" \
  --output outputs/plism_core_native_psd --patches 0 "$@"

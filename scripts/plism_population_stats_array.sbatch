#!/bin/bash
#SBATCH --job-name=plism-pop
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --array=0-90%16
#SBATCH --output=logs/plism_pop_%A_%a.out

# Native-resolution radial spectra for the 91 PLISM WSIs, one task per slide.
# No registration and no resampling; see docs/e9_plism_native_ert_contract.md.

source ~/.bashrc
conda activate "${JOB_CONDA_PREFIX:-cpath}"
set -euo pipefail
cd "${JOB_WORKDIR:-/mnt/isilon/oldridge_lab/leej/prenorm}"

# libvips pulls in conda's libicuuc, which needs GLIBCXX_3.4.30.  The batch nodes
# ship libstdc++ 3.4.29 in /lib64, so the environment's own copy must win.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

export MPLCONFIGDIR="$PWD/.mplconfig"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export VIPS_CONCURRENCY="$SLURM_CPUS_PER_TASK"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/src"

python src/build_plism_population_stats.py --task-index "$SLURM_ARRAY_TASK_ID"

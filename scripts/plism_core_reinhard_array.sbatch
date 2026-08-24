#!/bin/bash
#SBATCH --job-name=plism-core-rein
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --array=0-90%30
#SBATCH --output=logs/plism_core_rein_%A_%a.out

# Raw and post-Reinhard band energies on the core grid.  Needs the population pass first: the Reinhard reference is the pooled AT2 population.
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

python src/extract_plism_reinhard_bands.py --task-index "$SLURM_ARRAY_TASK_ID" "$@"

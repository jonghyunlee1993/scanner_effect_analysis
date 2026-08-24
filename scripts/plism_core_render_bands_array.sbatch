#!/bin/bash
#SBATCH --job-name=plism-core-bands
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-90%24
#SBATCH --output=logs/plism_core_bands_%A_%a.out

# Band energies of the Reinhard-rendered tiles at the frozen evaluation
# locations.  These are what RF1U's gains are fitted from; nothing is kept.
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

python src/measure_plism_core_render_bands.py --task-index "$SLURM_ARRAY_TASK_ID" "$@"

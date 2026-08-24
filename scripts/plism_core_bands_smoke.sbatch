#!/bin/bash
#SBATCH --job-name=plism-bands-smoke
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/plism_core_bands_smoke_%j.out

# Code-path smoke for the RF1U fit input and the gain fitter, before the 91-task
# array goes out.  Two slides is enough to exercise both the identity branch
# (a scanner that is its own destination) and the corrected branch.
source ~/.bashrc
conda activate "${JOB_CONDA_PREFIX:-cpath}"
set -euo pipefail
cd "${JOB_WORKDIR:-/mnt/isilon/oldridge_lab/leej/prenorm}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="$PWD/.mplconfig"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export VIPS_CONCURRENCY="$SLURM_CPUS_PER_TASK"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src"

OUT=outputs/plism_scratch/bands_smoke
for T in 0 6; do
  python src/measure_plism_core_render_bands.py --task-index "$T" --output "$OUT"
done
python src/fit_plism_render_gains.py --band-energy "$OUT" \
  --output outputs/plism_scratch/gainfit_smoke
echo "BANDS SMOKE OK"

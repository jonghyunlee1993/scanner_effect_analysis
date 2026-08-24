#!/bin/bash
#SBATCH --job-name=plism-cov-fold
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-25%26
#SBATCH --output=logs/plism_core_cov_fold_%A_%a.out

# One task per (variant, held-out section).  The folds are independent refits, so
# the serial loop -- which was on course for half a day -- becomes an hour.
source ~/.bashrc
conda activate "${JOB_CONDA_PREFIX:-cpath}"
set -euo pipefail
cd "${JOB_WORKDIR:-/mnt/isilon/oldridge_lab/leej/prenorm}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MPLCONFIGDIR="$PWD/.mplconfig"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src"

SECTIONS=(GIV GIVH GM GMH GV GVH HR HRH KR KRH LM LMH MY)
VARIANTS=(all_blocks focus_ok)
VARIANT="${VARIANTS[$((SLURM_ARRAY_TASK_ID / 13))]}"
SECTION="${SECTIONS[$((SLURM_ARRAY_TASK_ID % 13))]}"
if [ "$VARIANT" = focus_ok ]; then EXCLUDE=default; else EXCLUDE=""; fi

# focus_ok drops the HRH section entirely -- excluding one (section, scanner)
# block removes every location of that section, because the paired design needs
# all seven scanners at a location.  That fold has nothing to hold out.
if [ "$VARIANT" = focus_ok ] && [ "$SECTION" = HRH ]; then
  echo "focus_ok has no $SECTION section to hold out; nothing to do"
  exit 0
fi

python src/analyze_plism_stain_covariate.py \
  --input outputs/plism_core_native_psd --per-core "${PLISM_PER_CORE:-32}" \
  --exclude "$EXCLUDE" --fold-section "$SECTION" \
  --output "outputs/plism_core_stain_covariate/$VARIANT"

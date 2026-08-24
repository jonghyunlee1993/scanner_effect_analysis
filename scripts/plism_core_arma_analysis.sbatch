#!/bin/bash
#SBATCH --job-name=plism-core-arma
#SBATCH --partition=defq
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/plism_core_arma_%j.out

# Arm A analyses on the core grid, in one job because they are all cheap next to
# the per-slide passes that feed them.
#
# Two things are run twice on purpose.
#
# The nesting is fitted with the staining condition on top -- the axis PanNormal
# does not have -- and with the tissue type on top, which is PanNormal section
# 3.2's own model and became runnable here only once the 46 cores were named.
# Same rows, re-nested, so the two cohorts meet on one estimator.
#
# Every sharpness statistic is run with the HRH/S60 block in and out.  That scan
# is out of focus (high-band power down five- to tenfold at a correct 4-11 um
# registration), so a per-(section, scanner) sharpness number that includes it is
# measuring a focus failure.  The exclusion is at block level, never per patch:
# dropping individual patches for looking blurred is the detail-based selection
# section 2.1 forbids.  Reporting both is what makes the choice auditable.
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

PSD=outputs/plism_core_native_psd
REIN=outputs/plism_core_reinhard_bands
PER_CORE="${PLISM_PER_CORE:-32}"

echo "########## native ERT"
python src/analyze_plism_native_ert.py --input "$PSD" \
  --output outputs/plism_core_native_ert --figure

for VARIANT in all_blocks focus_ok; do
  if [ "$VARIANT" = focus_ok ]; then EXCLUDE=default; else EXCLUDE=""; fi
  echo "########## RF1U destinations — $VARIANT"
  python src/analyze_plism_rf1u_destinations.py --input "$REIN" --exclude "$EXCLUDE" \
    --output "outputs/plism_core_rf1u_destinations/$VARIANT"

  echo "########## random slopes, staining condition on top — $VARIANT"
  python src/analyze_plism_random_slopes.py --input "$PSD" --level section \
    --per-core "$PER_CORE" --exclude "$EXCLUDE" \
    --output "outputs/plism_core_random_slopes/section_$VARIANT"

  echo "########## random slopes, tissue type on top — $VARIANT"
  python src/analyze_plism_random_slopes.py --input "$PSD" --level tissue \
    --per-core "$PER_CORE" --exclude "$EXCLUDE" \
    --output "outputs/plism_core_random_slopes/tissue_$VARIANT"

  echo "########## stain covariate — $VARIANT"
  python src/analyze_plism_stain_covariate.py --input "$PSD" \
    --per-core "$PER_CORE" --exclude "$EXCLUDE" \
    --output "outputs/plism_core_stain_covariate/$VARIANT"
done
echo "ARM A ANALYSIS OK"

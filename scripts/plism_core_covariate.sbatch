#!/bin/bash
#SBATCH --job-name=plism-core-cov
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/plism_core_cov_%j.out

# The stain covariate with the leave-one-section-out pass that the sparse
# cohort's conclusion rested on but which never entered the code.
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

for VARIANT in all_blocks focus_ok; do
  if [ "$VARIANT" = focus_ok ]; then EXCLUDE=default; else EXCLUDE=""; fi
  echo "########## stain covariate + leave-one-out — $VARIANT"
  python src/analyze_plism_stain_covariate.py \
    --input outputs/plism_core_native_psd --per-core "${PLISM_PER_CORE:-32}" \
    --exclude "$EXCLUDE" --output "outputs/plism_core_stain_covariate/$VARIANT"
done
echo "COVARIATE OK"

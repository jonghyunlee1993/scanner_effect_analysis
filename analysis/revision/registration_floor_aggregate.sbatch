#!/bin/bash
#SBATCH --job-name=rv08-aggregate
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=analysis/revision/results/registration_floor/logs/aggregate_%j.log

# RV08 stage 3: slide means, slide bootstrap CIs, floor fractions, fraction of
# achievable reduction, qc.json and summary.md. Submit with
# --dependency=afterok:<embed array job>.

export PATH="${JOB_CONDA_PREFIX}/bin:${PATH}"
export PYTHONPATH="${JOB_WORKDIR}:${JOB_WORKDIR}/src:${JOB_WORKDIR}/scripts"
export PYTHONDONTWRITEBYTECODE=1
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
cd "${JOB_WORKDIR}"
set -euo pipefail

extra=()
if [[ -n "${JOB_OUTPUT_DIR:-}" ]]; then
  # smoke test: partial aggregation of a separate output directory
  extra+=(--output-dir "${JOB_OUTPUT_DIR}" --allow-partial)
fi
python analysis/revision/registration_floor.py aggregate "${extra[@]}"

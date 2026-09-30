#!/bin/bash
# RV09 tissue dependence of correction gains. With --array: one task per PFM x method
# (gain contrasts + RQ1 slide-level LMM + parametric-bootstrap RLRT); list with
#   python analysis/revision/gain_tissue_dependence.py list-tasks [--source smoke]
# Without --array: aggregate (BH, Spearman, summary.md). JOB_SOURCE=smoke runs the smoke
# adapter into results/gain_tissue_dependence/smoke; JOB_BOOTSTRAP overrides the 500 replicates.
#SBATCH --job-name=rv09-gain-lmm
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --output=analysis/revision/results/gain_tissue_dependence/logs/rv09_%A_%a.log

export PATH="${JOB_CONDA_PREFIX}/bin:${PATH}"
export PYTHONDONTWRITEBYTECODE=1
export HDF5_USE_FILE_LOCKING=FALSE
# Bootstrap fits run in one process per CPU (as in analysis/paper/run_slide_level_lmm.sbatch).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "${JOB_WORKDIR}"
set -euo pipefail

SOURCE="${JOB_SOURCE:-rv03}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  python analysis/revision/gain_tissue_dependence.py task --task-index "${SLURM_ARRAY_TASK_ID}" \
    --source "${SOURCE}" --bootstrap "${JOB_BOOTSTRAP:-500}" --workers "${SLURM_CPUS_PER_TASK}"
else
  python analysis/revision/gain_tissue_dependence.py aggregate --source "${SOURCE}"
fi

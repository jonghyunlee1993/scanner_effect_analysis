#!/bin/bash
#SBATCH --job-name=rv-p0d-struct-gpu
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-4
#SBATCH --output=analysis/revision/results/provenance_restoration/logs/structure_%x_%A_%a.log

# RV-P0d (iii) GPU stages. JOB_STAGE=predict (default; array task = fold),
# JOB_STAGE=evaluate_uni (submit with --array=0), or JOB_STAGE=train (array task = fold,
# needs --time=03:00:00; not part of the submitted verification).
export PATH="${JOB_CONDA_PREFIX}/bin:${PATH}"
export PYTHONPATH="${JOB_WORKDIR}/src:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLCONFIGDIR="${JOB_WORKDIR}/analysis/revision/results/provenance_restoration/.mplconfig"
mkdir -p "${MPLCONFIGDIR}"
cd "${JOB_WORKDIR}"
set -euo pipefail

stage="${JOB_STAGE:-predict}"
if [[ "${stage}" == "evaluate_uni" ]]; then
  python analysis/revision/restore_structure_ablation.py --stage evaluate_uni
else
  python analysis/revision/restore_structure_ablation.py --stage "${stage}" \
    --fold "${SLURM_ARRAY_TASK_ID}" --workers "${SLURM_CPUS_PER_TASK}"
fi

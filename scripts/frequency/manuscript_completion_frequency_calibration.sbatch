#!/usr/bin/env bash
#SBATCH --job-name=mc-freq-cal
#SBATCH --partition=defq
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/logs/frequency_calibration_%j.out
#SBATCH --error=outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/logs/frequency_calibration_%j.err

source ~/.bashrc
conda activate "${JOB_CONDA_PREFIX}"
set -euo pipefail
cd "${JOB_WORKDIR}"

export PYTHONPATH="${JOB_WORKDIR}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

python -m manuscript_completion.perturbation \
  --output-root "${JOB_OUTPUT_ROOT}" prepare-calibration \
  --workers "${SLURM_CPUS_PER_TASK}"

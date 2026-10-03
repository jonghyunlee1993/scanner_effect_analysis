#!/usr/bin/env bash
#SBATCH --job-name=pa-stain-src-audit
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --array=0-4%5
#SBATCH --output=outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/02_baseline_benchmark/logs/stain_source_audit_%A_%a.out
#SBATCH --error=outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/02_baseline_benchmark/logs/stain_source_audit_%A_%a.err

source ~/.bashrc
conda activate "${JOB_CONDA_PREFIX}"
set -euo pipefail
cd "${JOB_WORKDIR}"
export PYTHONPATH="${JOB_WORKDIR}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
export PANEL_A_STAGE=stain_source_audit
python -c 'from manuscript_completion.panel_a import verify_runtime_contract; verify_runtime_contract()'

mkdir -p "${JOB_OUTPUT_ROOT}/00_contract/${JOB_AUDIT_SUBDIR}" "${JOB_OUTPUT_ROOT}/logs"
python -m manuscript_completion.panel_a survey-vahadane \
  --locked-index "${JOB_LOCKED_INDEX}" \
  --fold "${SLURM_ARRAY_TASK_ID}" \
  --samples-per-fold 100000 \
  --max-iterations 1000,2000,4000 \
  --tolerances 0.001 \
  --alphas 0.001 \
  --solvers mu \
  --output "${JOB_OUTPUT_ROOT}/00_contract/${JOB_AUDIT_SUBDIR}/fold_${SLURM_ARRAY_TASK_ID}.json"

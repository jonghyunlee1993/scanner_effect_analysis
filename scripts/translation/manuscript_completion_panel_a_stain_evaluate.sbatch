#!/usr/bin/env bash
#SBATCH --job-name=pa-stain-eval
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --array=0-102%20
#SBATCH --output=outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/02_baseline_benchmark/logs/stain_eval_%A_%a.out
#SBATCH --error=outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/02_baseline_benchmark/logs/stain_eval_%A_%a.err

source ~/.bashrc
conda activate "${JOB_CONDA_PREFIX}"
set -euo pipefail
cd "${JOB_WORKDIR}"
export PYTHONPATH="${JOB_WORKDIR}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export HDF5_USE_FILE_LOCKING=FALSE
export MPLCONFIGDIR="${JOB_OUTPUT_ROOT}/.mplconfig"
export PANEL_A_STAGE=stain_evaluate
python -c 'from manuscript_completion.panel_a import verify_runtime_contract; verify_runtime_contract()'
stain_subdir="${JOB_STAIN_SUBDIR:-01_stain_v4_mu_convergence}"
mkdir -p "${JOB_OUTPUT_ROOT}/${stain_subdir}/shards" "${JOB_OUTPUT_ROOT}/logs" "${MPLCONFIGDIR}"

python -m manuscript_completion.panel_a evaluate-stain-slide \
  --locked-index "${JOB_LOCKED_INDEX}" \
  --reference-root "${JOB_OUTPUT_ROOT}/${stain_subdir}/references" \
  --target-reference-root "${JOB_TARGET_REFERENCE_ROOT}" \
  --slide-index "${SLURM_ARRAY_TASK_ID}" \
  --output-root "${JOB_OUTPUT_ROOT}/${stain_subdir}" \
  --vahadane-selection-manifest "${JOB_VAHADANE_SELECTION_MANIFEST}"

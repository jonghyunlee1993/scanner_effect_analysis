#!/usr/bin/env bash
#SBATCH --job-name=pa-stain-ref
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=0-24%20
#SBATCH --output=outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/02_baseline_benchmark/logs/stain_ref_%A_%a.out
#SBATCH --error=outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/02_baseline_benchmark/logs/stain_ref_%A_%a.err

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
export PANEL_A_STAGE=stain_reference
python -c 'from manuscript_completion.panel_a import verify_runtime_contract; verify_runtime_contract()'

targets=(versa akoya gt450 s360 s60)
target_index=$((SLURM_ARRAY_TASK_ID / 5))
test_fold=$((SLURM_ARRAY_TASK_ID % 5))
target_scanner="${targets[${target_index}]}"
stain_subdir="${JOB_STAIN_SUBDIR:-01_stain_v4_mu_convergence}"
output="${JOB_OUTPUT_ROOT}/${stain_subdir}/references/fold_${test_fold}/${target_scanner}.json"
mkdir -p "$(dirname "${output}")" "${JOB_OUTPUT_ROOT}/logs" "${MPLCONFIGDIR}"

python -m manuscript_completion.panel_a fit-stain-reference \
  --sample-index "${JOB_SAMPLE_INDEX}" \
  --locked-index "${JOB_LOCKED_INDEX}" \
  --target-scanner "${target_scanner}" \
  --test-fold "${test_fold}" \
  --output "${output}" \
  --vahadane-selection-manifest "${JOB_VAHADANE_SELECTION_MANIFEST}"

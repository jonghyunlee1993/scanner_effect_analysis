#!/usr/bin/env bash
#SBATCH --job-name=pa-gan-image
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-3%4
#SBATCH --output=outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/02_baseline_benchmark/logs/gan_image_%A_%a.out
#SBATCH --error=outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/02_baseline_benchmark/logs/gan_image_%A_%a.err

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
export PANEL_A_STAGE=gan_image_eval
python -c 'from manuscript_completion.panel_a import verify_runtime_contract; verify_runtime_contract()'

targets=(versa akoya s360 s60); target_scanner="${targets[${SLURM_ARRAY_TASK_ID}]}"
method_root="${JOB_OUTPUT_ROOT}/${JOB_METHOD_SUBDIR}"
args=(); for fold in 0 1 2 3 4; do args+=(--prediction-dir "${method_root}/02_predictions/at2_to_${target_scanner}/fold_${fold}"); done
python -m scanner_gan.evaluate_images "${args[@]}" \
  --output-dir "${method_root}/03_image_evaluation/at2_to_${target_scanner}" \
  --bootstrap-replicates 20000
python -m manuscript_completion.panel_a validate-evaluation \
  --evaluation-root "${method_root}/03_image_evaluation/at2_to_${target_scanner}" \
  --space image --method "${JOB_METHOD}" --target-scanner "${target_scanner}"

#!/usr/bin/env bash
#SBATCH --job-name=pa-stain-uni
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-12%4
#SBATCH --output=outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/02_baseline_benchmark/logs/stain_uni_%A_%a.out
#SBATCH --error=outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/02_baseline_benchmark/logs/stain_uni_%A_%a.err

source ~/.bashrc
conda activate "${JOB_CONDA_PREFIX}"
set -euo pipefail
cd "${JOB_WORKDIR}"
export PYTHONPATH="${JOB_WORKDIR}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export HDF5_USE_FILE_LOCKING=FALSE
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR="${JOB_OUTPUT_ROOT}/.mplconfig"
export PANEL_A_STAGE=stain_uni
python -c 'from manuscript_completion.panel_a import verify_runtime_contract; verify_runtime_contract()'
stain_subdir="${JOB_STAIN_SUBDIR:-01_stain_v4_mu_convergence}"
mkdir -p "${JOB_OUTPUT_ROOT}/${stain_subdir}/uni/shards" "${JOB_OUTPUT_ROOT}/${stain_subdir}/uni/task_manifests" "${MPLCONFIGDIR}"

python -m manuscript_completion.panel_a stain-uni-extract \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --task-count 13 \
  --existing-uni-metrics "${JOB_EXISTING_UNI_METRICS}" \
  --locked-index "${JOB_LOCKED_INDEX}" \
  --reference-root "${JOB_OUTPUT_ROOT}/${stain_subdir}/references" \
  --output-root "${JOB_OUTPUT_ROOT}/${stain_subdir}/uni" \
  --vahadane-selection-manifest "${JOB_VAHADANE_SELECTION_MANIFEST}" \
  --batch-size 64

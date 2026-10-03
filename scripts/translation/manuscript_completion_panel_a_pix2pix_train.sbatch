#!/usr/bin/env bash
#SBATCH --job-name=pa-p2p-full
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-19%4
#SBATCH --output=outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/02_baseline_benchmark/logs/pix2pix_train_%A_%a.out
#SBATCH --error=outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/02_baseline_benchmark/logs/pix2pix_train_%A_%a.err

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
export PANEL_A_STAGE=pix2pix_train
python -c 'from manuscript_completion.panel_a import verify_runtime_contract; verify_runtime_contract()'

targets=(versa akoya s360 s60)
target_index=$((SLURM_ARRAY_TASK_ID / 5))
test_fold=$((SLURM_ARRAY_TASK_ID % 5))
target_scanner="${targets[${target_index}]}"
seed=$((20263017 + target_index * 100 + test_fold))
run_dir="${JOB_OUTPUT_ROOT}/03_pix2pix/01_training/runs/at2_to_${target_scanner}/fold_${test_fold}/seed_${seed}"
mkdir -p "${run_dir}" "${JOB_OUTPUT_ROOT}/logs" "${MPLCONFIGDIR}"

python -m manuscript_completion.panel_a train-pix2pix \
  --sample-index "${JOB_SAMPLE_INDEX}" \
  --target-scanner "${target_scanner}" \
  --test-fold "${test_fold}" \
  --output-dir "${run_dir}" \
  --seed "${seed}" \
  --max-passes "${JOB_MAX_PASSES:-200}" \
  --max-steps "${JOB_MAX_STEPS:-0}" \
  --batch-size "${JOB_BATCH_SIZE:-16}" \
  --validation-batch-size "${JOB_VALIDATION_BATCH_SIZE:-32}" \
  --num-workers "${SLURM_CPUS_PER_TASK}" \
  --selection-every "${JOB_SELECTION_EVERY:-5}"

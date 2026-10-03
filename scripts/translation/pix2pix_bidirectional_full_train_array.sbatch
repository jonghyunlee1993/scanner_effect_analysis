#!/usr/bin/env bash
#SBATCH --job-name=p2p-bi-full
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-9%4
#SBATCH --output=outputs/scanner_batch_effect_analysis_2026-09-17/06_learned_baselines/09_bidirectional_full_training/logs/train_%A_%a.out

source ~/.bashrc
conda activate "${JOB_CONDA_PREFIX}"
set -euo pipefail

cd "${JOB_WORKDIR}"
export PYTHONPATH="${JOB_WORKDIR}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export HDF5_USE_FILE_LOCKING=FALSE

sources=(gt450 at2)
targets=(at2 gt450)
direction_index=$((SLURM_ARRAY_TASK_ID / 5))
test_fold=$((SLURM_ARRAY_TASK_ID % 5))
source_scanner="${sources[${direction_index}]}"
target_scanner="${targets[${direction_index}]}"
seed=$((20260917 + direction_index * 100 + test_fold))
run_dir="${JOB_OUTPUT_ROOT}/01_training/runs/${source_scanner}_to_${target_scanner}/fold_${test_fold}/seed_${seed}"
mkdir -p "${run_dir}" "${JOB_OUTPUT_ROOT}/logs"

args=(
  --sample-index "${JOB_SAMPLE_INDEX}"
  --source-scanner "${source_scanner}"
  --target-scanner "${target_scanner}"
  --test-fold "${test_fold}"
  --output-dir "${run_dir}"
  --seed "${seed}"
  --max-passes "${JOB_MAX_PASSES:-200}"
  --locations-per-train-slide 100
  --locations-per-validation-slide 40
  --batch-size 16
  --validation-batch-size 32
  --num-workers "${SLURM_CPUS_PER_TASK}"
  --learning-rate 2e-4
  --lambda-l1 100
  --amp bfloat16
  --constant-passes "${JOB_CONSTANT_PASSES:-100}"
  --selection-policy target_l1
  --selection-every "${JOB_SELECTION_EVERY:-5}"
  --save-every 0
)
if [[ "${JOB_MAX_STEPS:-0}" -gt 0 ]]; then
  args+=(--max-steps "${JOB_MAX_STEPS}")
fi

python -m scanner_gan.train "${args[@]}"

#!/usr/bin/env bash
#SBATCH --job-name=cyc-bi-full
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-4%4
#SBATCH --output=outputs/scanner_batch_effect_analysis_2026-09-17/06_learned_baselines/11_cyclegan_full_training/logs/train_%A_%a.out
#SBATCH --error=outputs/scanner_batch_effect_analysis_2026-09-17/06_learned_baselines/11_cyclegan_full_training/logs/train_%A_%a.err

source ~/.bashrc
conda activate "${JOB_CONDA_PREFIX}"
set -euo pipefail

cd "${JOB_WORKDIR}"
export PYTHONPATH="${JOB_WORKDIR}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export HDF5_USE_FILE_LOCKING=FALSE

test_fold="${SLURM_ARRAY_TASK_ID}"
seed=$((20261917 + test_fold))
run_dir="${JOB_OUTPUT_ROOT}/01_training/runs/gt450_at2/fold_${test_fold}/seed_${seed}"
mkdir -p "${run_dir}" "${JOB_OUTPUT_ROOT}/logs"

args=(
  --sample-index "${JOB_SAMPLE_INDEX}"
  --domain-a gt450
  --domain-b at2
  --test-fold "${test_fold}"
  --output-dir "${run_dir}"
  --seed "${seed}"
  --max-passes "${JOB_MAX_PASSES:-200}"
  --locations-per-train-slide "${JOB_TRAIN_LOCATIONS_PER_SLIDE:-100}"
  --locations-per-validation-slide "${JOB_VALIDATION_LOCATIONS_PER_SLIDE:-40}"
  --batch-size "${JOB_BATCH_SIZE:-16}"
  --validation-batch-size "${JOB_VALIDATION_BATCH_SIZE:-32}"
  --num-workers "${SLURM_CPUS_PER_TASK}"
  --learning-rate 2e-4
  --lambda-cycle 10
  --lambda-identity 5
  --pool-size 50
  --amp bfloat16
  --constant-passes "${JOB_CONSTANT_PASSES:-100}"
  --selection-every "${JOB_SELECTION_EVERY:-5}"
  --save-every 0
)
if [[ "${JOB_MAX_STEPS:-0}" -gt 0 ]]; then
  args+=(--max-steps "${JOB_MAX_STEPS}")
fi

python -m scanner_gan.train_cyclegan "${args[@]}"

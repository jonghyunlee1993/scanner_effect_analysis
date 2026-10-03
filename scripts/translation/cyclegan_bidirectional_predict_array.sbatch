#!/usr/bin/env bash
#SBATCH --job-name=cyc-bi-predict
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-9%4
#SBATCH --output=outputs/scanner_batch_effect_analysis_2026-09-17/06_learned_baselines/11_cyclegan_full_training/logs/predict_%A_%a.out

source ~/.bashrc
conda activate "${JOB_CONDA_PREFIX}"
set -euo pipefail

cd "${JOB_WORKDIR}"
export PYTHONPATH="${JOB_WORKDIR}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export HDF5_USE_FILE_LOCKING=FALSE

sources=(gt450 at2)
targets=(at2 gt450)
direction_index=$((SLURM_ARRAY_TASK_ID / 5))
test_fold=$((SLURM_ARRAY_TASK_ID % 5))
source_scanner="${sources[${direction_index}]}"
target_scanner="${targets[${direction_index}]}"
seed=$((20261917 + test_fold))
run_dir="${JOB_OUTPUT_ROOT}/01_training/runs/gt450_at2/fold_${test_fold}/seed_${seed}"
prediction_dir="${JOB_OUTPUT_ROOT}/02_predictions/${source_scanner}_to_${target_scanner}/fold_${test_fold}"

python -m scanner_gan.predict \
  --method cyclegan \
  --checkpoint "${run_dir}/checkpoints/best_image_only.pt" \
  --run-manifest "${run_dir}/run_manifest.json" \
  --sample-index "${JOB_SAMPLE_INDEX}" \
  --locked-index "${JOB_LOCKED_INDEX}" \
  --source-scanner "${source_scanner}" \
  --target-scanner "${target_scanner}" \
  --test-fold "${test_fold}" \
  --output-dir "${prediction_dir}" \
  --batch-size 32 \
  --num-workers "${SLURM_CPUS_PER_TASK}" \
  --device cuda \
  --amp bfloat16 \
  --overwrite

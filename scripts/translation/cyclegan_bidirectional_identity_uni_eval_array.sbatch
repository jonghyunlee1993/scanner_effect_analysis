#!/usr/bin/env bash
#SBATCH --job-name=cyc-bi-id-uni
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-1%2
#SBATCH --output=outputs/scanner_batch_effect_analysis_2026-09-17/06_learned_baselines/11_cyclegan_full_training/logs/identity_uni_%A_%a.out

source ~/.bashrc
conda activate "${JOB_CONDA_PREFIX}"
set -euo pipefail

cd "${JOB_WORKDIR}"
export PYTHONPATH="${JOB_WORKDIR}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export HDF5_USE_FILE_LOCKING=FALSE
export MPLCONFIGDIR="${JOB_OUTPUT_ROOT}/.mplconfig"
mkdir -p "${MPLCONFIGDIR}"

sources=(gt450 at2)
targets=(at2 gt450)
source_scanner="${sources[${SLURM_ARRAY_TASK_ID}]}"
target_scanner="${targets[${SLURM_ARRAY_TASK_ID}]}"
direction="${source_scanner}_to_${target_scanner}"
args=()
for fold in 0 1 2 3 4; do
  args+=(--prediction-dir "${JOB_OUTPUT_ROOT}/02_identity_predictions/${direction}/fold_${fold}")
done
python -m scanner_gan.evaluate_uni \
  "${args[@]}" \
  --output-dir "${JOB_OUTPUT_ROOT}/04_identity_uni/${direction}" \
  --batch-size 64 \
  --bootstrap-replicates 20000

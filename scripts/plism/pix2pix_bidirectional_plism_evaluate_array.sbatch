#!/usr/bin/env bash
#SBATCH --job-name=p2p-bi-plism-eval
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-9%4
#SBATCH --output=outputs/scanner_batch_effect_analysis_2026-09-17/06_learned_baselines/09_bidirectional_full_training/10_plism_external_bidirectional/logs/evaluate_%A_%a.out

source ~/.bashrc
conda activate "${JOB_CONDA_PREFIX}"
set -euo pipefail

cd "${JOB_WORKDIR}"
export PYTHONPATH="${JOB_WORKDIR}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export HDF5_USE_FILE_LOCKING=FALSE

python -m scanner_gan.plism_bidirectional \
  --output-root "${JOB_OUTPUT_ROOT}/10_plism_external_bidirectional" \
  evaluate-task \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --generator-batch-size 16 \
  --uni-batch-size 32 \
  --device cuda \
  --amp bfloat16

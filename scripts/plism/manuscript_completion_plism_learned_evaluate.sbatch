#!/usr/bin/env bash
#SBATCH --job-name=plism-learned-eval
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-19%5
#SBATCH --output=outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/06_plism_learned_external/logs/evaluate_%A_%a.out

source ~/.bashrc
conda activate "${JOB_CONDA_PREFIX}"
set -euo pipefail

cd "${JOB_WORKDIR}"
export PYTHONPATH="${JOB_WORKDIR}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export HDF5_USE_FILE_LOCKING=FALSE

if (( SLURM_ARRAY_TASK_ID < 10 )); then
  method="pix2pix"
  task_index="${SLURM_ARRAY_TASK_ID}"
else
  method="cyclegan"
  task_index="$((SLURM_ARRAY_TASK_ID - 10))"
fi

python -m scanner_gan.plism_bidirectional \
  --output-root "${JOB_OUTPUT_ROOT}/${method}" \
  evaluate-task \
  --task-index "${task_index}" \
  --generator-batch-size 16 \
  --uni-batch-size 32 \
  --device cuda \
  --amp bfloat16

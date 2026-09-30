#!/bin/bash
#SBATCH --job-name=rv03-images
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-18
#SBATCH --output=analysis/revision/results/corrected_embeddings/logs/images_%A_%a.log

# Tasks 0-5: PanNormal slide chunks (set20); tasks 6-18: PLISM sections.
# Optional smoke-test overrides: JOB_MAX_SLIDES, JOB_MAX_LOCATIONS, JOB_OUTPUT_ROOT, JOB_TASK_INDEX.
source ~/.bashrc
conda activate "${JOB_CONDA_PREFIX}"
set -euo pipefail
cd "${JOB_WORKDIR}"
export PYTHONPATH="${JOB_WORKDIR}:${JOB_WORKDIR}/src:${JOB_WORKDIR}/scripts:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

extra=()
[[ -n "${JOB_MAX_SLIDES:-}" ]] && extra+=(--max-slides "${JOB_MAX_SLIDES}")
[[ -n "${JOB_MAX_LOCATIONS:-}" ]] && extra+=(--max-locations "${JOB_MAX_LOCATIONS}")
[[ -n "${JOB_OUTPUT_ROOT:-}" ]] && extra+=(--output-root "${JOB_OUTPUT_ROOT}")
[[ -n "${JOB_PFMS:-}" ]] && extra+=(--pfms "${JOB_PFMS}")
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python analysis/revision/corrected_embeddings_images.py \
  --task-index "${JOB_TASK_INDEX:-${SLURM_ARRAY_TASK_ID}}" --pan-tasks 6 \
  --workers "${SLURM_CPUS_PER_TASK}" --batch-size "${JOB_BATCH_SIZE:-32}" "${extra[@]}"

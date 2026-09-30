#!/bin/bash
# RV13 stage 1: render every set20 image condition once per slide (raw, RV03 corrections, RV02
# band manipulations; EXAONEPath Macenko) and embed with CONCH, SEAL-CONCH, SEAL-UNI2, PLIP,
# DINOv2 and EXAONEPath. Slides are split across array tasks (task i takes cohort rows i::N).
# Smoke overrides: JOB_MAX_SLIDES, JOB_MAX_LOCATIONS, JOB_OUTPUT_ROOT, JOB_TASK_COUNT.
#SBATCH --job-name=rv13-embed
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --array=0-11
#SBATCH --output=analysis/revision/results/anchor_frequency_diversity/logs/embed_%A_%a.log

export PATH="${JOB_CONDA_PREFIX}/bin:${PATH}"
export PYTHONPATH="${JOB_WORKDIR}:${JOB_WORKDIR}/src:${JOB_WORKDIR}/scripts:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export USE_TF=0
export HDF5_USE_FILE_LOCKING=FALSE
# Rendering and Macenko run in one forked worker per CPU; keep each worker single-threaded.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "${JOB_WORKDIR}"
set -euo pipefail

extra=()
[[ -n "${JOB_MAX_SLIDES:-}" ]] && extra+=(--max-slides "${JOB_MAX_SLIDES}")
[[ -n "${JOB_MAX_LOCATIONS:-}" ]] && extra+=(--max-locations "${JOB_MAX_LOCATIONS}")
[[ -n "${JOB_OUTPUT_ROOT:-}" ]] && extra+=(--output-root "${JOB_OUTPUT_ROOT}")
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python analysis/revision/anchor_frequency_diversity_embed.py \
  --task-index "${JOB_TASK_INDEX:-${SLURM_ARRAY_TASK_ID}}" \
  --task-count "${JOB_TASK_COUNT:-${SLURM_ARRAY_TASK_COUNT}}" \
  --workers "${SLURM_CPUS_PER_TASK}" "${extra[@]}"

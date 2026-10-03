#!/bin/bash
# RV18 stage 2: embed held-out conditions with UNI (epoch 0) and the continued checkpoints (JOB_EPOCHS, default 1,3).
# Smoke overrides: JOB_MAX_SLIDES, JOB_MAX_LOCATIONS, JOB_OUTPUT_ROOT.
#SBATCH --job-name=rv18-embed
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=analysis/revision/results/uni_continued/logs/embed_%j.log

export PATH="${JOB_CONDA_PREFIX}/bin:${PATH}"
export PYTHONPATH="${JOB_WORKDIR}:${JOB_WORKDIR}/src:${JOB_WORKDIR}/scripts:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "${JOB_WORKDIR}"
set -euo pipefail
extra=()
[[ -n "${JOB_MAX_SLIDES:-}" ]] && extra+=(--max-slides "${JOB_MAX_SLIDES}")
[[ -n "${JOB_MAX_LOCATIONS:-}" ]] && extra+=(--max-locations "${JOB_MAX_LOCATIONS}")
[[ -n "${JOB_OUTPUT_ROOT:-}" ]] && extra+=(--output-root "${JOB_OUTPUT_ROOT}")
[[ -n "${JOB_EPOCHS:-}" ]] && extra+=(--epochs "${JOB_EPOCHS}")
[[ -n "${JOB_RUNS:-}" ]] && extra+=(--runs "${JOB_RUNS}")
[[ -n "${JOB_LABEL:-}" ]] && extra+=(--label "${JOB_LABEL}")
[[ -n "${JOB_BASE:-}" ]] && extra+=(--base "${JOB_BASE}")
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python analysis/revision/uni_continued_embed.py --workers "${SLURM_CPUS_PER_TASK}" "${extra[@]}"

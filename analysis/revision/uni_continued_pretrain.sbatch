#!/bin/bash
# RV18 stage 1: continue UNI (ViT-L/16) DINOv2 training, blur off, one array task per AT2 share
# (0=p100, 1=p080, 2=p020, 3=p000). Smoke overrides: JOB_MAX_SLIDES, JOB_MAX_ITERS, JOB_OUTPUT_ROOT.
#SBATCH --job-name=rv18-pretrain
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=analysis/revision/results/uni_continued/logs/pretrain_%A_%a.log

export PATH="${JOB_CONDA_PREFIX}/bin:${PATH}"
export PYTHONPATH="${JOB_WORKDIR}:${JOB_WORKDIR}/src:${JOB_WORKDIR}/scripts:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "${JOB_WORKDIR}"
set -euo pipefail

extra=()
[[ -n "${JOB_MAX_SLIDES:-}" ]] && extra+=(--max-slides "${JOB_MAX_SLIDES}")
[[ -n "${JOB_MAX_ITERS:-}" ]] && extra+=(--max-iters "${JOB_MAX_ITERS}")
[[ -n "${JOB_OUTPUT_ROOT:-}" ]] && extra+=(--output-root "${JOB_OUTPUT_ROOT}")
[[ -n "${JOB_ARMS:-}" ]] && extra+=(--arms "${JOB_ARMS}")
[[ -n "${JOB_SUFFIX:-}" ]] && extra+=(--suffix "${JOB_SUFFIX}")
[[ -n "${JOB_ORDER_SEED:-}" ]] && extra+=(--order-seed "${JOB_ORDER_SEED}")
[[ -n "${JOB_BACKBONE_LR:-}" ]] && extra+=(--backbone-lr "${JOB_BACKBONE_LR}")
[[ -n "${JOB_EPOCHS:-}" ]] && extra+=(--epochs "${JOB_EPOCHS}")
[[ -n "${JOB_BLUR:-}" ]] && extra+=(--blur "${JOB_BLUR}")
[[ -n "${JOB_STOP_EPOCH:-}" ]] && extra+=(--stop-epoch "${JOB_STOP_EPOCH}")
[[ -n "${JOB_BASE:-}" ]] && extra+=(--base "${JOB_BASE}")
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python analysis/revision/uni_continued_pretrain.py \
  --task-index "${JOB_TASK_INDEX:-${SLURM_ARRAY_TASK_ID}}" \
  --workers "$(( SLURM_CPUS_PER_TASK - 2 ))" "${extra[@]}"

#!/bin/bash
# RV13 loader parity: stage 1 runs the official EXAONEPath model-card code (Python >= 3.10,
# JOB_OFFICIAL_PREFIX, default the trident-uni2 env); stage 2 compares our loaders (cpath).
#SBATCH --job-name=rv13-parity
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=analysis/revision/results/anchor_frequency_diversity/logs/parity_%j.log

OFFICIAL_PREFIX="${JOB_OFFICIAL_PREFIX:-/home/leej70/miniconda3/envs/trident-uni2}"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export USE_TF=0
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
cd "${JOB_WORKDIR}"
set -euo pipefail

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"${OFFICIAL_PREFIX}/bin/python" analysis/revision/anchor_frequency_diversity_parity.py official
export PATH="${JOB_CONDA_PREFIX}/bin:${PATH}"
export PYTHONPATH="${JOB_WORKDIR}:${JOB_WORKDIR}/src:${JOB_WORKDIR}/scripts:${PYTHONPATH:-}"
python analysis/revision/anchor_frequency_diversity_parity.py compare

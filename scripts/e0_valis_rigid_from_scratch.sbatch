#!/bin/bash
#SBATCH --job-name=e0-valis-rigid
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:a100-40g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/e0_valis_rigid_%j.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export VIPS_CONCURRENCY="$SLURM_CPUS_PER_TASK"
export PYTHONNOUSERSITE=1

VALIS_PYTHON="/mnt/isilon/oldridge_lab/irinaz/anaconda3/envs/valis310/bin/python"
"$VALIS_PYTHON" src/run_e0_valis_rigid_from_scratch.py "$@"

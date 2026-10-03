#!/bin/bash
#SBATCH --job-name=slide-lmm
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=analysis/paper/results/direct_slide_lmm/logs/slide_%A_%a.log

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python analysis/paper/slide_level_lmm.py run-endpoint \
  --input "${JOB_INPUT:-analysis/paper/results/direct_slide_lmm/direct_slide_contrasts.csv}" \
  --output-root "${JOB_OUTPUT_ROOT:-analysis/paper/results/direct_slide_lmm}" \
  --task-index "$SLURM_ARRAY_TASK_ID" \
  --bootstrap "${SLIDE_LMM_BOOTSTRAP:-500}" \
  --workers "$SLURM_CPUS_PER_TASK"

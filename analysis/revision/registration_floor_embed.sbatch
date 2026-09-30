#!/bin/bash
#SBATCH --job-name=rv08-embed
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=analysis/revision/results/registration_floor/logs/embed_%A_%a.log

# RV08 stage 2: shifted AT2 images (0.25/0.5 px Fourier, 1/2 px WSI, four
# directions), SSIM, LPIPS-VGG16 and four PFM cosine distances; one shard per
# slide. Slides are split over the array as cohort_order[task::task_count].
# LPIPS reads the VGG16 weights recorded in the Table 2 LPIPS contract.

export PATH="${JOB_CONDA_PREFIX}/bin:${PATH}"
export PYTHONPATH="${JOB_WORKDIR}:${JOB_WORKDIR}/src:${JOB_WORKDIR}/scripts"
export PYTHONDONTWRITEBYTECODE=1
export HDF5_USE_FILE_LOCKING=FALSE
export HF_HUB_OFFLINE=1
export TORCH_HOME="${JOB_WORKDIR}/outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/09_table2_benchmark/.torch"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
cd "${JOB_WORKDIR}"
set -euo pipefail

extra=()
if [[ -n "${JOB_SLIDES:-}" ]]; then
  # smoke test: explicit slides into a separate output directory
  read -r -a slides <<< "${JOB_SLIDES}"
  extra+=(--slides "${slides[@]}" --output-dir "${JOB_OUTPUT_DIR:?JOB_OUTPUT_DIR is required with JOB_SLIDES}")
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python analysis/revision/registration_floor.py embed \
  --task-index "${SLURM_ARRAY_TASK_ID:-0}" --task-count "${SLURM_ARRAY_TASK_COUNT:-1}" \
  --batch-size 32 "${extra[@]}"

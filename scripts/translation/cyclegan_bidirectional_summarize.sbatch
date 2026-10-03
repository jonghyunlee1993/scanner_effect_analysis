#!/usr/bin/env bash
#SBATCH --job-name=cyc-bi-summary
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/scanner_batch_effect_analysis_2026-09-17/06_learned_baselines/11_cyclegan_full_training/logs/summary_%j.out

source ~/.bashrc
conda activate "${JOB_CONDA_PREFIX}"
set -euo pipefail

cd "${JOB_WORKDIR}"
export PYTHONPATH="${JOB_WORKDIR}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

python -m scanner_gan.summarize_cyclegan \
  --input-root "${JOB_OUTPUT_ROOT}" \
  --output-root "${JOB_OUTPUT_ROOT}/05_summary"

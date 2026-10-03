#!/usr/bin/env bash
#SBATCH --job-name=cyc-bi-plism-prep
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/scanner_batch_effect_analysis_2026-09-17/06_learned_baselines/11_cyclegan_full_training/10_plism_external_bidirectional/logs/prepare_%j.out

source ~/.bashrc
conda activate "${JOB_CONDA_PREFIX}"
set -euo pipefail

cd "${JOB_WORKDIR}"
export PYTHONPATH="${JOB_WORKDIR}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export HDF5_USE_FILE_LOCKING=FALSE

python -m scanner_gan.plism_bidirectional \
  --output-root "${JOB_OUTPUT_ROOT}/10_plism_external_bidirectional" \
  prepare \
  --method cyclegan \
  --runs-root "${JOB_OUTPUT_ROOT}/01_training/runs" \
  --render-contract-root "${JOB_PLISM_RENDER_CONTRACT_ROOT}"

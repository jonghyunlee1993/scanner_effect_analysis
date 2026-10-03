#!/usr/bin/env bash
#SBATCH --job-name=plism-learned-prep
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/06_plism_learned_external/logs/prepare_%j.out

source ~/.bashrc
conda activate "${JOB_CONDA_PREFIX}"
set -euo pipefail

cd "${JOB_WORKDIR}"
export PYTHONPATH="${JOB_WORKDIR}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export HDF5_USE_FILE_LOCKING=FALSE

python -m scanner_gan.plism_bidirectional \
  --output-root "${JOB_OUTPUT_ROOT}/pix2pix" \
  prepare \
  --method pix2pix \
  --runs-root "${JOB_PANEL_ROOT}/03_pix2pix/01_training/runs" \
  --render-contract-root "${JOB_RENDER_CONTRACT_ROOT}" \
  --direction at2:s360 \
  --direction at2:s60

python -m scanner_gan.plism_bidirectional \
  --output-root "${JOB_OUTPUT_ROOT}/cyclegan" \
  prepare \
  --method cyclegan \
  --runs-root "${JOB_PANEL_ROOT}/04_cyclegan/01_training/runs" \
  --render-contract-root "${JOB_RENDER_CONTRACT_ROOT}" \
  --direction at2:s360 \
  --direction at2:s60

#!/bin/bash
#SBATCH --job-name=mc-plism-uni
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-38%4
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/05_plism_external_correction/logs/external_uni_%A_%a.log

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"
export PYTHONPATH="$JOB_WORKDIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export HDF5_USE_FILE_LOCKING=FALSE
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
sha256sum -c outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/05_plism_external_correction/00_contract_v4/runtime_files.sha256
python -m manuscript_completion.plism_external external-uni-task \
  --task-index "$SLURM_ARRAY_TASK_ID" \
  --batch-size 48 \
  --device cuda \
  --output-root outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/05_plism_external_correction

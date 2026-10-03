#!/usr/bin/env bash
#SBATCH --job-name=pa-final-agg
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/02_baseline_benchmark/logs/final_aggregate_%j.out
#SBATCH --error=outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/02_baseline_benchmark/logs/final_aggregate_%j.err

source ~/.bashrc
conda activate "${JOB_CONDA_PREFIX}"
set -euo pipefail
cd "${JOB_WORKDIR}"
export PYTHONPATH="${JOB_WORKDIR}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLCONFIGDIR="${JOB_PANEL_ROOT}/.mplconfig"
export PANEL_A_STAGE=final_aggregate
python -c 'from manuscript_completion.panel_a import verify_runtime_contract; verify_runtime_contract()'
stain_subdir="${JOB_STAIN_SUBDIR:-01_stain_v4_mu_convergence}"

python -m manuscript_completion.panel_a assemble-panel-a \
  --analysis-root "${JOB_ANALYSIS_ROOT}" \
  --panel-root "${JOB_PANEL_ROOT}" \
  --stain-subdir "${stain_subdir}"

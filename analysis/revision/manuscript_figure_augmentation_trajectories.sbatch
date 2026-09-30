#!/bin/bash
#SBATCH --job-name=mf-augmentation-trajectories
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=analysis/revision/results/manuscript_figures/logs/augmentation_trajectories_%j.log

# Optional: PAPER_FIGURES_DIR redirects output (drafts); default is 00_manuscript/figures.
export PATH="${JOB_CONDA_PREFIX}/bin:${PATH}"
export PYTHONPATH="${JOB_WORKDIR}/src:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${JOB_WORKDIR}/analysis/revision/results/manuscript_figures/logs/.mpl"
mkdir -p "${MPLCONFIGDIR}"
cd "${JOB_WORKDIR}"
set -euo pipefail

python analysis/revision/manuscript_figure_augmentation_trajectories.py

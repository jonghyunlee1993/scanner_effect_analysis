#!/usr/bin/env bash
#SBATCH --partition=defq
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --job-name=manuscript_tex_check
#SBATCH --output=analysis/paper/results/manuscript_build/build_%j.out

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR/00_manuscript"
export PATH="/home/leej70/.local/share/texlive/2026/bin/x86_64-linux:$PATH"
mkdir -p "$JOB_WORKDIR/analysis/paper/results/manuscript_build"
latexmk -xelatex -interaction=nonstopmode -halt-on-error \
  -outdir="$JOB_WORKDIR/analysis/paper/results/manuscript_build" \
  "${JOB_TEX_FILE:-pannormal_scanner_batch_effects.tex}"

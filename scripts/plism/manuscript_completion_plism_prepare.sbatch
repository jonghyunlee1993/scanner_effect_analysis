#!/bin/bash
#SBATCH --job-name=mc-plism-prep
#SBATCH --partition=defq
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/05_plism_external_correction/logs/prepare_%j.log

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"
export PYTHONPATH="$JOB_WORKDIR/src${PYTHONPATH:+:$PYTHONPATH}"
: "${JOB_PLISM_CODE_SHA256:?submitter must freeze plism_external.py hash}"
: "${JOB_PLISM_SCRIPT_SHA256:?submitter must freeze prepare script hash}"
test "$(sha256sum src/manuscript_completion/plism_external.py | awk '{print $1}')" = "$JOB_PLISM_CODE_SHA256"
test "$(sha256sum scripts/manuscript_completion_plism_prepare.sbatch | awk '{print $1}')" = "$JOB_PLISM_SCRIPT_SHA256"
python -m manuscript_completion.plism_external prepare-render \
  --output-root outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/05_plism_external_correction

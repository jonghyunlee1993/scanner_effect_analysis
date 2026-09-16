#!/bin/bash
#SBATCH --job-name=report-v3
#SBATCH --partition=defq
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/dev/null

source ~/.bashrc
conda activate "$JOB_CONDA_PREFIX"
set -euo pipefail
cd "$JOB_WORKDIR"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="/tmp/matplotlib-$SLURM_JOB_ID"

validation_tmp=$(mktemp -d /tmp/pannormal_report_v3.XXXXXX)
trap 'rm -rf -- "$validation_tmp"' EXIT
mkdir -p "$validation_tmp/a" "$validation_tmp/b"

report_dir="presentations/pannormal_scanner_harmonization_report_v3_2026-08-29"
registry="reports/pannormal_story_bilingual_v3/evidence_registry.json"

PYTHONNOUSERSITE=1 python src/build_pannormal_story_report_v3.py \
  --output-ko "$validation_tmp/a/pannormal_scanner_harmonization_evidence_report_ko.html" \
  --output-en "$validation_tmp/a/pannormal_scanner_harmonization_evidence_report_en.html" \
  --registry "$validation_tmp/a/evidence_registry.json" \
  --manifest-ko "$validation_tmp/a/pannormal_scanner_harmonization_evidence_report_ko.manifest.json" \
  --manifest-en "$validation_tmp/a/pannormal_scanner_harmonization_evidence_report_en.manifest.json"
PYTHONNOUSERSITE=1 python src/build_pannormal_story_report_v3.py \
  --output-ko "$validation_tmp/b/pannormal_scanner_harmonization_evidence_report_ko.html" \
  --output-en "$validation_tmp/b/pannormal_scanner_harmonization_evidence_report_en.html" \
  --registry "$validation_tmp/b/evidence_registry.json" \
  --manifest-ko "$validation_tmp/b/pannormal_scanner_harmonization_evidence_report_ko.manifest.json" \
  --manifest-en "$validation_tmp/b/pannormal_scanner_harmonization_evidence_report_en.manifest.json"

cmp "$validation_tmp/a/pannormal_scanner_harmonization_evidence_report_ko.html" "$validation_tmp/b/pannormal_scanner_harmonization_evidence_report_ko.html"
cmp "$validation_tmp/a/pannormal_scanner_harmonization_evidence_report_en.html" "$validation_tmp/b/pannormal_scanner_harmonization_evidence_report_en.html"
cmp "$validation_tmp/a/evidence_registry.json" "$validation_tmp/b/evidence_registry.json"
cmp "$validation_tmp/a/pannormal_scanner_harmonization_evidence_report_ko.manifest.json" "$validation_tmp/b/pannormal_scanner_harmonization_evidence_report_ko.manifest.json"
cmp "$validation_tmp/a/pannormal_scanner_harmonization_evidence_report_en.manifest.json" "$validation_tmp/b/pannormal_scanner_harmonization_evidence_report_en.manifest.json"

# Reproduction is successful only when a clean rebuild is identical to the
# checked-in release. The validator never overwrites the release artifacts.
cmp "$validation_tmp/a/pannormal_scanner_harmonization_evidence_report_ko.html" \
  "$report_dir/pannormal_scanner_harmonization_evidence_report_ko.html"
cmp "$validation_tmp/a/pannormal_scanner_harmonization_evidence_report_en.html" \
  "$report_dir/pannormal_scanner_harmonization_evidence_report_en.html"
cmp "$validation_tmp/a/pannormal_scanner_harmonization_evidence_report_ko.manifest.json" \
  "$report_dir/pannormal_scanner_harmonization_evidence_report_ko.manifest.json"
cmp "$validation_tmp/a/pannormal_scanner_harmonization_evidence_report_en.manifest.json" \
  "$report_dir/pannormal_scanner_harmonization_evidence_report_en.manifest.json"
cmp "$validation_tmp/a/evidence_registry.json" "$registry"

sha256sum \
  "$validation_tmp/a/pannormal_scanner_harmonization_evidence_report_ko.html" \
  "$validation_tmp/a/pannormal_scanner_harmonization_evidence_report_en.html" \
  "$validation_tmp/a/evidence_registry.json"

PYTHONNOUSERSITE=1 python scripts/_verify_report_v3.py

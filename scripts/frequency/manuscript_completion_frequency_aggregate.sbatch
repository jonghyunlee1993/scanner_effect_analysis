#!/usr/bin/env bash
#SBATCH --job-name=mc-freq-agg
#SBATCH --partition=defq
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/logs/frequency_aggregate_%j.out
#SBATCH --error=outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/logs/frequency_aggregate_%j.err

source ~/.bashrc
conda activate "${JOB_CONDA_PREFIX}"
set -euo pipefail
cd "${JOB_WORKDIR}"

export PYTHONPATH="${JOB_WORKDIR}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

implementation_manifest="${JOB_IMPLEMENTATION_MANIFEST:?JOB_IMPLEMENTATION_MANIFEST is required}"
implementation_sha256="${JOB_IMPLEMENTATION_MANIFEST_SHA256:?JOB_IMPLEMENTATION_MANIFEST_SHA256 is required}"
python - "${implementation_manifest}" "${implementation_sha256}" "${JOB_WORKDIR}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1]).resolve()
expected_manifest_sha256 = sys.argv[2].lower()
repository = Path(sys.argv[3]).resolve()

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

observed_manifest_sha256 = digest(manifest_path)
if observed_manifest_sha256 != expected_manifest_sha256:
    raise SystemExit(
        f"implementation manifest mismatch: {observed_manifest_sha256} != {expected_manifest_sha256}"
    )
payload = json.loads(manifest_path.read_text())
if payload.get("status") not in {
    "frozen_before_uni_outcomes",
    "frozen_after_computational_bugfix",
}:
    raise SystemExit("implementation manifest is not frozen")
for relative, expected in payload["files"].items():
    path = (repository / relative).resolve()
    try:
        path.relative_to(repository)
    except ValueError as error:
        raise SystemExit(f"implementation path escapes repository: {relative}") from error
    observed = digest(path) if path.is_file() else "missing"
    if observed != expected:
        raise SystemExit(f"implementation mismatch: {relative}: {observed} != {expected}")
for name, item in payload["inputs"].items():
    path = Path(item["path"]).resolve()
    observed = digest(path) if path.is_file() else "missing"
    if observed != item["sha256"]:
        raise SystemExit(f"input mismatch: {name}: {observed} != {item['sha256']}")
print(json.dumps({"status": "implementation_verified", "sha256": observed_manifest_sha256}))
PY
export FREQUENCY_IMPLEMENTATION_MANIFEST="${implementation_manifest}"
export FREQUENCY_IMPLEMENTATION_MANIFEST_SHA256="${implementation_sha256}"

python -m manuscript_completion.frequency_aggregate \
  --output-root "${JOB_OUTPUT_ROOT}" "${JOB_MODE}"

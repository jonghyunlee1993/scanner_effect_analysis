from pathlib import Path

import pandas as pd
import pytest

from audit_pannormal_core_results import audit_artifact_manifest
from fetch_e0_pfm_checkpoints import sha256


def test_audit_artifact_manifest_accepts_path_and_artifact_schemas(tmp_path):
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("locked\n")
    for column in ("path", "artifact"):
        manifest = tmp_path / f"{column}.csv"
        pd.DataFrame([{column: str(artifact), "sha256": sha256(artifact)}]).to_csv(
            manifest, index=False
        )
        rows = audit_artifact_manifest(tmp_path, manifest)
        assert rows[0]["sha256"] == sha256(artifact)


def test_audit_artifact_manifest_rejects_hash_drift(tmp_path):
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("changed\n")
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([{"artifact": str(artifact), "sha256": "0" * 64}]).to_csv(
        manifest, index=False
    )
    with pytest.raises(ValueError, match="SHA-256 drift"):
        audit_artifact_manifest(tmp_path, manifest)

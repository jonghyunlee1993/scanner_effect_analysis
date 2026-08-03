"""Audit all locked PanNormal-core result manifests and main figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fetch_e0_pfm_checkpoints import sha256


LOCKS = (
    ("e0_e3", "outputs/e0_e3_locked_results/summary.json", "audit_pass"),
    ("e4_controls", "outputs/e4_control_results_lock/summary.json", "result_lock_pass"),
    ("e5_comparators", "outputs/e5_comparator_results_lock/summary.json", "result_lock_pass"),
    ("e5_macenko", "outputs/e5_macenko_supplement_lock/summary.json", "result_lock_pass"),
    ("e6_heterogeneity", "outputs/e6_heterogeneity_results_lock/summary.json", "result_lock_pass"),
    ("e6_spectrum_predictor", "outputs/e6_baseline_spectrum_predictor_lock/summary.json", "result_lock_pass"),
    ("e6_loto", "outputs/e6_loto_results_lock/summary.json", "result_lock_pass"),
    ("e7_tissue_probe", "outputs/e7_tissue_probe_results_lock/summary.json", "result_lock_pass"),
)

MAIN_FIGURES = (
    "outputs/e0_e3_main_figures/figure1_study_estimator_validity",
    "outputs/e0_e3_main_figures/figure2_audited_scanner_spectrum",
    "outputs/e0_e3_main_figures/figure3_scanner_content_structure",
    "outputs/e4_control_figure/figure4_control_bounded_frontier",
    "outputs/e5_comparator_figure/figure5_actual_correction_frontier",
    "outputs/e7_tissue_probe/figure6_content_tissue_evidence",
)

FROZEN_CONTRACTS = (
    ("e5_comparators", "docs/e4_e7_decision_record.md", "decision_record_sha256"),
    ("e5_comparators", "docs/e5_comparator_execution_contract.md", "execution_contract_sha256"),
    ("e5_macenko", "docs/e5_macenko_supplement_contract.md", "supplement_contract_sha256"),
    ("e6_heterogeneity", "docs/e6_heterogeneity_execution_contract.md", "e6_contract_sha256"),
    ("e6_spectrum_predictor", "docs/e6_baseline_spectrum_predictor_contract.md", "contract_sha256"),
    ("e6_loto", "docs/e6_heterogeneity_execution_contract.md", "e6_contract_sha256"),
    ("e7_tissue_probe", "docs/e7_tissue_probe_execution_contract.md", "e7_contract_sha256"),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="outputs/pannormal_core_results_lock")
    return parser.parse_args()


def resolve_artifact(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def audit_artifact_manifest(root: Path, manifest_path: Path):
    frame = pd.read_csv(manifest_path)
    path_column = "path" if "path" in frame.columns else "artifact"
    if path_column not in frame or "sha256" not in frame:
        raise ValueError(f"{manifest_path}: unsupported artifact-manifest schema")
    rows = []
    for record in frame.to_dict("records"):
        path = resolve_artifact(root, str(record[path_column]))
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256(path)
        if observed != record["sha256"]:
            raise ValueError(f"{path}: SHA-256 drift")
        rows.append({
            "artifact": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": observed,
        })
    return rows


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    manifest_rows = []
    lock_rows = []
    summaries = {}
    for lock_id, relative, pass_key in LOCKS:
        summary_path = root / relative
        summary = json.loads(summary_path.read_text())
        if summary.get(pass_key) is not True:
            raise RuntimeError(f"{lock_id}: {pass_key} is not true")
        summaries[lock_id] = summary
        artifact_manifest = summary_path.parent / "artifact_manifest.csv"
        audited = audit_artifact_manifest(root, artifact_manifest)
        manifest_rows.extend(audited)
        lock_rows.append({
            "lock_id": lock_id,
            "summary": str(summary_path),
            "summary_sha256": sha256(summary_path),
            "artifact_manifest": str(artifact_manifest),
            "artifact_manifest_sha256": sha256(artifact_manifest),
            "artifacts_verified": len(audited),
        })
    for lock_id, relative, hash_key in FROZEN_CONTRACTS:
        path = root / relative
        if summaries[lock_id].get(hash_key) != sha256(path):
            raise ValueError(f"{lock_id}: frozen contract drift at {path}")
        manifest_rows.append({
            "artifact": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    for stem in MAIN_FIGURES:
        for suffix in ("png", "pdf"):
            path = root / f"{stem}.{suffix}"
            if path.stat().st_size <= 10_000:
                raise ValueError(f"{path}: implausibly small main figure")
            manifest_rows.append({
                "artifact": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)
    locks = pd.DataFrame(lock_rows)
    artifacts = pd.DataFrame(manifest_rows).drop_duplicates("artifact")
    locks_path = output / "result_locks.csv"
    artifacts_path = output / "artifact_manifest.csv"
    locks.to_csv(locks_path, index=False)
    artifacts.to_csv(artifacts_path, index=False)
    result = {
        "analysis": "pannormal_core_results_lock",
        "experiments": "E0-E7",
        "result_locks": len(locks),
        "main_figures": len(MAIN_FIGURES),
        "artifacts_verified": len(artifacts),
        "artifact_bytes": int(artifacts["bytes"].sum()),
        "result_locks_sha256": sha256(locks_path),
        "artifact_manifest_sha256": sha256(artifacts_path),
        "core_result_lock_pass": True,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

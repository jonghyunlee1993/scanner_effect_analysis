"""Freeze shared inputs for the manuscript-completion experiments.

This module intentionally does not inspect outcome values.  It verifies the
population, folds, locked image/UNI panels, and immutable predecessor artifacts,
then writes a hash-addressed manifest below the dated analysis root.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ROOT = ROOT / "outputs/scanner_batch_effect_analysis_2026-09-17"
OUTPUT_ROOT = ANALYSIS_ROOT / "12_manuscript_completion"
SCANNERS = ("at2", "versa", "akoya", "gt450", "s360", "s60")
TARGET_SCANNERS = SCANNERS[1:]
BANDS_CYC_PER_PIXEL = {
    "low_mid": (0.05052, 0.15156),
    "mid": (0.15156, 0.30312),
    "high": (0.30312, 0.45468),
}
UNI_V1_CHECKPOINT_SHA256 = (
    "56ef09b44a25dc5c7eedc55551b3d47bcd17659a7a33837cf9abc9ec4e2ffb40"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path: Path) -> int:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        return max(sum(1 for _ in csv.reader(handle)) - 1, 0)


def git_revision() -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            args, cwd=ROOT, check=True, text=True, capture_output=True
        )
        return completed.stdout.strip()

    return {
        "head": run("git", "rev-parse", "HEAD"),
        "tracked_dirty": bool(run("git", "status", "--short", "--untracked-files=no")),
    }


def _source(path: Path, *, rows: bool = False) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    if rows:
        result["rows"] = csv_rows(path)
    return result


def _uni_inventory() -> dict[str, Any]:
    root = ANALYSIS_ROOT / "03_uni/shards"
    summaries = sorted(root.glob("*.summary.json"))
    if len(summaries) != 103:
        raise ValueError(f"expected 103 UNI summaries, found {len(summaries)}")
    records = []
    for summary_path in summaries:
        payload = json.loads(summary_path.read_text())
        if payload.get("status") != "pass":
            raise ValueError(f"non-pass UNI shard: {summary_path}")
        shard = summary_path.with_name(summary_path.name.replace(".summary.json", ".h5"))
        if not shard.is_file():
            raise FileNotFoundError(shard)
        if payload.get("output_sha256") != sha256(shard):
            raise ValueError(f"UNI shard hash mismatch: {shard}")
        if payload.get("locations") != 20 or payload.get("feature_dim") != 1024:
            raise ValueError(f"unexpected UNI shape metadata: {summary_path}")
        records.append(
            {
                "slide_id": str(payload["slide_id"]),
                "shard": str(shard.resolve()),
                "shard_sha256": payload["output_sha256"],
                "summary_sha256": sha256(summary_path),
            }
        )
    return {
        "slides": len(records),
        "locations_per_slide": 20,
        "raw_embeddings": len(records) * 20 * len(SCANNERS),
        "records": records,
    }


def prepare(output_root: Path) -> dict[str, Any]:
    directories = (
        "00_contract",
        "01_pfm_problem_definition",
        "02_baseline_benchmark",
        "03_tissue_interaction",
        "04_frequency_mechanism/01_cross_spectral_recoverability",
        "04_frequency_mechanism/02_matched_information_removal",
        "04_frequency_mechanism/03_uni_band_sensitivity",
        "05_plism_external_correction",
        "06_manuscript_tables",
        "logs",
    )
    for relative in directories:
        (output_root / relative).mkdir(parents=True, exist_ok=True)

    cohort = ANALYSIS_ROOT / "00_contract/cohort.csv"
    folds = ROOT / "outputs/augmentation_ood_v1/00_contract/slide_folds.csv"
    locked_image = (
        ANALYSIS_ROOT
        / "06_learned_baselines/00_contract/locked_image_evaluation_index.csv.gz"
    )
    gan_contract = ANALYSIS_ROOT / "06_learned_baselines/00_contract/analysis_contract.json"
    pix2pix_summary = (
        ANALYSIS_ROOT
        / "06_learned_baselines/09_bidirectional_full_training/05_summary/summary.json"
    )
    plism_contract = (
        ANALYSIS_ROOT
        / "06_learned_baselines/09_bidirectional_full_training/10_plism_external_bidirectional/00_contract/analysis_contract.json"
    )
    correction_parameters = ANALYSIS_ROOT / "02_correction/parameters.json"
    sources = {
        "cohort": _source(cohort, rows=True),
        "slide_folds": _source(folds, rows=True),
        "locked_image_evaluation_index": _source(locked_image, rows=True),
        "gan_contract": _source(gan_contract),
        "pix2pix_bidirectional_summary": _source(pix2pix_summary),
        "plism_bidirectional_contract": _source(plism_contract),
        "correction_parameters": _source(correction_parameters),
    }
    if sources["cohort"]["rows"] != 103 or sources["slide_folds"]["rows"] != 103:
        raise ValueError("cohort/fold population is not 103 slides")
    if sources["locked_image_evaluation_index"]["rows"] != 4120:
        raise ValueError("locked image panel is not 4,120 locations")

    payload: dict[str, Any] = {
        "analysis_version": "manuscript_completion_v1",
        "created_utc": utc_now(),
        "status": "prepared",
        "scope": {
            "reference_scanner": "at2",
            "panel_a_targets": list(TARGET_SCANNERS),
            "panel_a_methods": [
                "raw",
                "reinhard",
                "macenko",
                "vahadane",
                "frequency",
                "combined",
                "pix2pix",
                "cyclegan",
            ],
            "cycle_gan_existing_jobs_are_read_only": True,
        },
        "population": {
            "physical_slides": 103,
            "tissues": 37,
            "image_locations_per_slide": 40,
            "uni_locations_per_slide": 20,
            "scanners": list(SCANNERS),
            "folds": 5,
        },
        "inference": {
            "internal_unit": "physical_slide",
            "external_unit": "physical_section",
            "bootstrap_replicates": 20000,
            "bootstrap_seed": 20260917,
        },
        "frequency": {
            "source_axis": "cycles_per_pixel",
            "bands_cyc_per_pixel": BANDS_CYC_PER_PIXEL,
            "legacy_physical_label_mpp": 0.5052,
            "physical_label_status": "provisional_only",
            "alternate_exploratory_predictor_bands_are_not_used": True,
        },
        "uni_v1": {
            "checkpoint_sha256": UNI_V1_CHECKPOINT_SHA256,
            "feature_dim": 1024,
            "historical_shard_checkpoint_attribute_present": False,
            "reuse_gate": "cross-artifact parity or fresh extraction",
        },
        "environment": {
            "job_conda_prefix": os.environ.get("JOB_CONDA_PREFIX") or os.environ.get("CONDA_PREFIX"),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "git": git_revision(),
        "sources": sources,
        "uni_inventory": _uni_inventory(),
    }
    destination = output_root / "00_contract/manifest.json"
    temporary = destination.with_suffix(f".json.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
    print(json.dumps({"status": "pass", "manifest": str(destination)}, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prepare(args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


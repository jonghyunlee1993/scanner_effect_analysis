#!/usr/bin/env python3
"""Select a UNI-blind Pix2Pix checkpoint candidate from validation images.

The result is a candidate for locked outer-test prediction, not a declaration
of formal success.  Full endpoint, invented-edge, and hallucination gates are
evaluated after predictions are materialized.  UNI fields are rejected here.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SELECTION_VERSION = "pix2pix_image_only_selection_v1"
FORBIDDEN_TOKENS = ("uni", "embedding", "feature")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def select_candidate(
    run_dir: str | Path,
    *,
    minimum_source_gradient_ncc: float = 0.90,
    maximum_saturation_fraction: float = 0.10,
) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    manifest_path = root / "run_manifest.json"
    metrics_path = root / "metrics.jsonl"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete":
        raise ValueError(f"training run is not complete: {root}")
    if manifest.get("selection_boundary") != "image-only inner validation; UNI forbidden until lock":
        raise ValueError("training run does not carry the frozen UNI-blind boundary")

    rows = []
    for line in metrics_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        forbidden = [key for key in row if any(token in key.lower() for token in FORBIDDEN_TOKENS)]
        if forbidden:
            raise ValueError(f"selection metrics contain forbidden representation fields: {forbidden}")
        rows.append(row)
    if not rows:
        raise ValueError(f"no validation metrics in {metrics_path}")

    candidates = []
    for row in rows:
        completed_passes = int(row["completed_passes"])
        checkpoint = root / "checkpoints" / f"pass_{completed_passes:02d}.pt"
        if not checkpoint.is_file():
            continue
        l1 = float(row["validation_l1_normalized"])
        ncc = float(row["validation_source_gradient_ncc"])
        saturation = float(row["validation_saturation_fraction"])
        basic_eligible = (
            ncc >= minimum_source_gradient_ncc
            and saturation <= maximum_saturation_fraction
        )
        candidates.append(
            {
                "completed_passes": completed_passes,
                "global_step": int(row["global_step"]),
                "checkpoint": str(checkpoint),
                "validation_l1_normalized": l1,
                "validation_psnr_db": float(row["validation_psnr_db"]),
                "validation_source_gradient_ncc": ncc,
                "validation_saturation_fraction": saturation,
                "basic_fidelity_eligible": basic_eligible,
            }
        )
    if not candidates:
        raise ValueError(f"no retained pass checkpoints in {root / 'checkpoints'}")

    eligible = [row for row in candidates if row["basic_fidelity_eligible"]]
    pool = eligible if eligible else candidates
    selected = min(
        pool,
        key=lambda row: (row["validation_l1_normalized"], row["completed_passes"]),
    )
    payload = {
        "selection_version": SELECTION_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "direction": manifest["direction"],
        "test_fold": int(manifest["config"]["test_fold"]),
        "validation_fold": int(manifest["config"]["validation_fold"]),
        "selection_data": "inner-validation image metrics only",
        "uni_opened": False,
        "gates": {
            "minimum_source_gradient_ncc": minimum_source_gradient_ncc,
            "maximum_saturation_fraction": maximum_saturation_fraction,
            "pending_outer_test_gates": [
                "ten-endpoint residual",
                "invented-edge fraction",
                "hallucination/memorization audit",
            ],
        },
        "basic_eligible_checkpoint_count": len(eligible),
        "selection_status": (
            "provisional_image_candidate"
            if eligible
            else "diagnostic_candidate_after_basic_fidelity_failure"
        ),
        "selected_checkpoint": selected["checkpoint"],
        "selected": selected,
        "candidates": candidates,
        "interpretation_boundary": (
            "This selection permits locked prediction and UNI measurement but cannot be called "
            "a safe or successful correction until all outer-test image gates pass."
        ),
    }
    write_json(root / "checkpoint_selection.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(select_candidate(args.run_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

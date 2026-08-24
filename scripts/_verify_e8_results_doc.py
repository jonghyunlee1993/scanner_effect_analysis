#!/usr/bin/env python
"""Check every number quoted in docs/e8_paired_residual_results.md against its artifact.

The doc carries about eighty numbers copied out of four different JSON/CSV
artifacts.  A number that is wrong there looks exactly like a number that is
right, so the transcription is checked mechanically rather than by reading.

Usage:  PYTHONNOUSERSITE=1 python scripts/_verify_e8_results_doc.py
Exit 0 with "PROBLEMS 0" when every quoted value is present in the document.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "e8_paired_residual_results.md"
OUT = ROOT / "outputs" / "e8_residual"
ENCODERS = ["resnet50", "uni_v1", "conch_v1", "virchow2"]
CONDITIONS = [
    "reinhard_gt450",
    "reinhard_unpaired_band_gt450",
    "e8_free_gt450",
    "coral_gt450",
    "orthogonal_procrustes_gt450",
]
PROBE_CONDITIONS = [
    "raw",
    "rf1u:reinhard_gt450",
    "rf1u:reinhard_unpaired_band_gt450",
    "e8:e8_free_gt450",
    "feature:coral",
    "feature:orthogonal_procrustes",
]


def main() -> int:
    doc = DOC.read_text()
    problems: list[str] = []
    checked = 0

    def want(text: str, label: str) -> None:
        nonlocal checked
        checked += 1
        if text not in doc:
            problems.append(f"{label}: {text!r} not in document")

    probe = pd.concat(
        pd.read_csv(OUT / "probe" / f"{e}.csv").assign(encoder_id=e) for e in ENCODERS
    )
    for name in ("linear", "mlp", "knn"):
        sub = probe[probe.probe == name]
        for condition in PROBE_CONDITIONS:
            for encoder in ENCODERS:
                cell = sub[(sub.condition == condition) & (sub.encoder_id == encoder)]
                if len(cell) != 1:
                    problems.append(f"probe {name}/{condition}/{encoder}: {len(cell)} rows")
                    continue
                # the document tabulates the linear probe in full and the two
                # nonlinear probes for the feature methods only
                if name == "linear" or condition.startswith("feature:"):
                    want(f"{cell.balanced_accuracy.iloc[0]:.3f}", f"{name} {condition} {encoder}")

    frontier = pd.read_csv(OUT / "frontier" / "gt450" / "endpoint_summary.csv")
    for condition in CONDITIONS:
        for encoder in ENCODERS:
            row = frontier[
                (frontier.condition == condition) & (frontier.encoder_id == encoder)
            ]
            if len(row) != 1:
                problems.append(f"frontier {condition}/{encoder}: {len(row)} rows")
                continue
            want(
                f"{abs(row.relative_radius_reduction.iloc[0] * 100):.2f}%",
                f"radius reduction {condition} {encoder}",
            )
    for encoder in ENCODERS:
        rows = frontier[frontier.encoder_id == encoder]
        want(f"{rows.raw_scanner_centroid_rms.iloc[0]:.4f}", f"raw radius {encoder}")
    arm_b = frontier[frontier.condition == "e8_free_gt450"].set_index("encoder_id")
    for encoder in ENCODERS:
        want(
            f"{abs(arm_b.loc[encoder, 'delta_content_ci_lower']):.4f}",
            f"arm B content margin {encoder}",
        )
    want(f"**{int(arm_b.safe_and_invariance_improved.sum())} of 4**", "arm B safe count")

    reading = json.loads((OUT / "reading" / "gt450" / "summary.json").read_text())
    want(reading["section_10"]["verdict"], "section 10 verdict")
    want(f"{min(reading['section_10']['arm_b_probe'].values()):.3f}", "arm B probe floor")

    audit = json.loads((OUT / "audit" / "gt450" / "summary.json").read_text())
    for arm, expected in (("gainfield", 3), ("free", 5)):
        gate = audit["arms"][arm]["gate"]
        if len(gate["folds_passing"]) != expected:
            problems.append(f"{arm} folds passing: {gate['folds_passing']}")
        want(f"{expected} of 5 folds", f"{arm} folds passing")
        detail = audit["arms"][arm]["audit"]
        want(f"{detail['phase_correlation']['mean']:.3f}", f"{arm} phase correlation")
        want(f"{detail['reextracted_sign_change']['mean']:.4f}", f"{arm} sign change")
        want(f"{detail['invented_fraction']['mean']:.2e}", f"{arm} invented fraction")
    want(
        f"{audit['arms']['free']['audit']['phase_correlation_base']['mean']:.3f}",
        "RF1U phase correlation baseline",
    )
    want(f"{audit['gradient_threshold']:.4f}", "gradient threshold")
    want(
        f"{audit['arms']['gainfield']['audit']['residual_exactness']['mean']:.1e}",
        "arm A residual exactness",
    )

    print(f"checked {checked} quoted values")
    if problems:
        print(f"PROBLEMS {len(problems)}")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("PROBLEMS 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())

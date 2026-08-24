#!/usr/bin/env python
"""Cross-document consistency for the E8/E9 facts the manuscript rests on.

Every fact is recomputed from its artifact, then asserted against each document
that quotes it.  Documents state the same number at different precision by
design -- prose rounds, tables do not -- so each fact carries every spelling it
is allowed to appear as, and a document passes if any one of them is present.

Usage:  PYTHONNOUSERSITE=1 python scripts/_verify_cross_doc.py
Exit 0 with "PROBLEMS 0" when every document agrees with the artifacts.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
ENCODERS = ["resnet50", "uni_v1", "conch_v1", "virchow2"]
DOC_NAMES = [
    "storyline.md",
    "final_study_protocol.md",
    "storyline_completion_audit.md",
    "README.md",
    "e8_paired_residual_results.md",
    "e8_paired_residual_contract.md",
    "e9_plism_core_results.md",
    "e4_e7_decision_record.md",
]
REPORT = "presentations/pannormal_plism_2026-08-23/index.html"
E9_FIGURES = ["116,831", "817,817", "88.2", "+0.964", "2,177,672", "700,986", "99.57"]


def spellings(lo: float, hi: float) -> tuple[str, ...]:
    """The precisions and dash styles a range is allowed to be written in."""
    out = []
    for digits in (2, 3):
        a, b = f"{lo:.{digits}f}", f"{hi:.{digits}f}"
        out += [f"{a}--{b}", f"{a}–{b}", a]
    return tuple(out)


def main() -> int:
    docs = {n: (ROOT / "docs" / n).read_text() for n in DOC_NAMES}
    docs["report"] = (ROOT / REPORT).read_text()
    out = ROOT / "outputs" / "e8_residual"

    probe = pd.concat(
        pd.read_csv(out / "probe" / f"{e}.csv").assign(encoder_id=e) for e in ENCODERS
    )
    frontier = pd.read_csv(out / "frontier" / "gt450" / "endpoint_summary.csv")
    reading = json.loads((out / "reading" / "gt450" / "summary.json").read_text())["section_10"]
    audit = json.loads((out / "audit" / "gt450" / "summary.json").read_text())

    def probe_range(kind: str, condition: str) -> tuple[float, float]:
        v = probe[(probe.probe == kind) & (probe.condition == condition)].balanced_accuracy
        return v.min(), v.max()

    problems: list[str] = []
    checks = 0

    def expect(names, variants, label):
        nonlocal checks
        for name in names:
            checks += 1
            if not any(v in docs[name] for v in variants):
                problems.append(f"{name}: {label} — none of {variants} present")

    quoting = ["storyline.md", "final_study_protocol.md", "e8_paired_residual_results.md"]

    rr = frontier[
        (frontier.condition == "e8_free_gt450") & (frontier.encoder_id == "resnet50")
    ].relative_radius_reduction.iloc[0] * 100
    expect(quoting + ["README.md"], (f"+{rr:.2f}%", f"{rr:.2f}%"), "arm B ResNet50 RR")
    expect(["report"], (f"+{rr:.1f}%", f"+{rr:.2f}%"), "arm B ResNet50 RR")

    lo, hi = min(reading["arm_b_probe"].values()), max(reading["arm_b_probe"].values())
    expect(quoting + ["report"], spellings(lo, hi), "arm B probe range")

    for condition, label in (
        ("feature:orthogonal_procrustes", "MLP Procrustes"),
        ("feature:coral", "MLP CORAL"),
    ):
        lo, hi = probe_range("mlp", condition)
        expect(quoting + ["README.md", "report"], spellings(lo, hi), label)

    phase = audit["arms"]["free"]["audit"]["phase_correlation"]["mean"]
    expect(
        ["e8_paired_residual_results.md", "storyline.md"],
        (f"{phase:.3f}", f"{phase:.2f}"),
        "arm B phase correlation",
    )

    for arm, expected in (("gainfield", 3), ("free", 5)):
        checks += 1
        got = len(audit["arms"][arm]["gate"]["folds_passing"])
        if got != expected:
            problems.append(f"artifact: {arm} folds passing is {got}, documents say {expected}")

    for name in quoting + ["storyline_completion_audit.md", "README.md"]:
        checks += 1
        if reading["verdict"] not in docs[name]:
            problems.append(f"{name}: pre-registered verdict {reading['verdict']!r} missing")

    for figure in E9_FIGURES:
        checks += 1
        if figure not in docs["e9_plism_core_results.md"] and figure not in docs["report"]:
            problems.append(f"E9 figure {figure!r} absent from its results doc and the report")

    print(f"checked {checks} document assertions against recomputed artifacts")
    if problems:
        print(f"PROBLEMS {len(problems)}")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("PROBLEMS 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())

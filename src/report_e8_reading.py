"""Evaluate the E8 result against the reading pre-registered in contract section 10.

The point of writing section 10 before any endpoint existed was that a negative
result is worth exactly what its pre-registration is worth.  This script applies
that section mechanically rather than by argument, and prints the verdict it
reaches together with every number the verdict rests on.

The four admissible readings, verbatim from the contract:

- **supported** -- arm B passes content non-inferiority and collapse, and leaves
  the scanner probe above 0.50 in at least three of four PFMs while the
  feature-space methods reach at most 0.17.
- **falsified** -- arm B passes content and collapse and drops the probe below
  0.30 in at least two PFMs.  Section 12 step 4 and section 13's "fully optimized
  image correction" clause are then rewritten before submission.
- **degeneracy** -- arm B lowers the probe substantially but fails content or
  collapse.  Neither support nor falsification: the invariance-fidelity
  degeneracy for a third time, now in a learned method.
- **inconclusive** -- anything else, including arm B never clearing the gate.

Arm A is read separately and independently: whether the scanner x tissue
interaction is worth modelling over RF1U's three numbers per scanner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from rf1u_unpaired import RF1U_CONDITION


ANALYSIS = "e8_paired_residual_reading"
PROBE_SUPPORT_FLOOR = 0.50
PROBE_FALSIFY_CEILING = 0.30
PROBE_FEATURE_CEILING = 0.17
SUPPORT_MIN_MODELS = 3
FALSIFY_MIN_MODELS = 2
PROBE_CHANCE = 1.0 / 6.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="gt450")
    parser.add_argument("--audit", default="outputs/e8_residual/audit")
    parser.add_argument("--frontier", default="outputs/e8_residual/frontier")
    parser.add_argument("--probe", default="outputs/e8_residual/probe")
    parser.add_argument("--output", default="outputs/e8_residual/reading")
    parser.add_argument("--probe-estimator", default="linear")
    return parser.parse_args()


def load_probe(root: Path, estimator: str) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in sorted(Path(root).glob("*.csv"))
              if not path.name.endswith(".confusion.csv")]
    if not frames:
        raise RuntimeError(f"no probe results under {root}")
    probe = pd.concat(frames, ignore_index=True)
    return probe[probe["probe"] == estimator]


def strip_label(condition: str) -> str:
    """Drop the probe's `<source label>:` prefix, if it carries one.

    `analyze_e8_condition_probe.py` namespaces every condition it reads by the
    `--source LABEL=ROOT` label, so `e8_free_gt450` arrives as `e8:e8_free_gt450`
    and the retargeted comparators arrive as `feature:coral`.  Matching on the
    bare name keeps this reading independent of how the roots were labelled.
    """
    return condition.split(":", 1)[-1]


def probe_by_model(probe: pd.DataFrame, condition: str, target: str = "") -> dict:
    """Balanced accuracy per encoder for one condition, however it was labelled.

    The feature-space shards store `coral` without a destination suffix because
    they already live under a per-target directory, so a request for
    `coral_gt450` has to also accept a bare `coral`.
    """
    wanted = {condition}
    if target and condition.endswith(f"_{target}"):
        wanted.add(condition[: -len(f"_{target}")])
    names = probe["condition"].map(strip_label)
    selected = probe[names.isin(wanted)]
    return dict(zip(selected["encoder_id"], selected["balanced_accuracy"]))


def arm_condition(arm: str, target: str) -> str:
    return f"e8_{arm}_{target}"


def read_section_ten(endpoints: pd.DataFrame, probe: pd.DataFrame, target: str) -> dict:
    free = arm_condition("free", target)
    rows = endpoints[endpoints["condition"] == free]
    if rows.empty:
        return {"verdict": "inconclusive", "reason": "arm B produced no endpoints"}

    fidelity = {
        row["encoder_id"]: bool(
            row["content_noninferiority_pass"] and row["collapse_every_scanner_pass"]
        )
        for _, row in rows.iterrows()
    }
    arm_b = probe_by_model(probe, free, target)
    raw = probe_by_model(probe, "raw", target)
    feature = {
        name: probe_by_model(probe, f"{name}_{target}", target)
        for name in ("coral", "orthogonal_procrustes")
    }
    if not arm_b:
        return {
            "verdict": "inconclusive",
            "reason": f"the probe holds no condition matching {free}; "
                      f"observed: {sorted(set(probe['condition'].map(strip_label)))}",
        }

    passes_fidelity = all(fidelity.values())
    above_floor = [m for m, v in arm_b.items() if v > PROBE_SUPPORT_FLOOR]
    below_ceiling = [m for m, v in arm_b.items() if v < PROBE_FALSIFY_CEILING]
    feature_low = all(
        value <= PROBE_FEATURE_CEILING
        for values in feature.values()
        for value in values.values()
    ) and any(values for values in feature.values())

    if passes_fidelity and len(below_ceiling) >= FALSIFY_MIN_MODELS:
        verdict = "falsified"
        reason = (
            f"arm B cleared content and collapse and drove the probe below "
            f"{PROBE_FALSIFY_CEILING} in {len(below_ceiling)} PFMs "
            f"({', '.join(sorted(below_ceiling))}); section 12 step 4 and section 13's "
            f"'fully optimized image correction' clause must be rewritten"
        )
    elif passes_fidelity and len(above_floor) >= SUPPORT_MIN_MODELS and feature_low:
        verdict = "supported"
        reason = (
            f"arm B cleared content and collapse yet stayed above "
            f"{PROBE_SUPPORT_FLOOR} in {len(above_floor)} PFMs while the feature-space "
            f"methods reached at most {PROBE_FEATURE_CEILING}"
        )
    elif not passes_fidelity and any(
        raw.get(model, 1.0) - value > 0.10 for model, value in arm_b.items()
    ):
        failed = sorted(m for m, v in fidelity.items() if not v)
        verdict = "degeneracy"
        reason = (
            f"arm B moved the probe substantially but failed content or collapse in "
            f"{', '.join(failed)} -- the invariance-fidelity degeneracy again, this "
            f"time in a learned method, which strengthens the study's central "
            f"negative result rather than settling the ceiling question"
        )
    else:
        verdict = "inconclusive"
        reason = "no pre-registered condition is met"

    return {
        "verdict": verdict,
        "reason": reason,
        "arm_b_probe": arm_b,
        "raw_probe": raw,
        "feature_probe": feature,
        "arm_b_fidelity_pass": fidelity,
        "models_above_support_floor": sorted(above_floor),
        "models_below_falsify_ceiling": sorted(below_ceiling),
    }


def read_arm_a(endpoints: pd.DataFrame, comparisons: pd.DataFrame, target: str) -> dict:
    condition = arm_condition("gainfield", target)
    rows = endpoints[endpoints["condition"] == condition]
    if rows.empty:
        return {"verdict": "not run", "reason": "arm A produced no endpoints"}
    reference = f"{RF1U_CONDITION}_{target}"
    against = comparisons[
        (comparisons["condition"] == condition)
        & (comparisons["reference"] == reference)
    ]
    safe = {
        row["encoder_id"]: bool(row["safe_and_invariance_improved"])
        for _, row in rows.iterrows()
    }
    better = {
        row["encoder_id"]: bool(row["improved_over_reference"])
        for _, row in against.iterrows()
    }
    beats = [model for model in better if better[model] and safe.get(model)]
    return {
        "verdict": "adds over RF1U" if len(beats) >= 3 else "no gain over RF1U",
        "reason": (
            f"safe and tighter than RF1U in {len(beats)} of {len(safe)} PFMs "
            f"({', '.join(sorted(beats)) or 'none'})"
        ),
        "safe_and_improved_over_raw": safe,
        "tighter_than_rf1u": better,
    }


def main():
    args = parse_args()
    target = args.target
    frontier = Path(args.frontier) / target
    endpoints = pd.read_csv(frontier / "endpoint_summary.csv")
    comparisons = pd.read_csv(frontier / "paired_comparisons.csv")
    probe = load_probe(Path(args.probe), args.probe_estimator)
    audit = json.loads((Path(args.audit) / target / "summary.json").read_text())

    section_ten = read_section_ten(endpoints, probe, target)
    arm_a = read_arm_a(endpoints, comparisons, target)

    report = {
        "analysis": ANALYSIS,
        "target": target,
        "probe_estimator": args.probe_estimator,
        "probe_chance": PROBE_CHANCE,
        "section_10": section_ten,
        "arm_a": arm_a,
        "gate": {
            arm: {
                "gate_pass": value["gate"]["gate_pass"],
                "folds_passing": value["gate"]["folds_passing"],
                "winner": value["gate"]["winner"],
            }
            for arm, value in audit["arms"].items()
        },
        "hallucination": {
            arm: {
                "invented_fraction_mean": (value.get("audit") or {})
                .get("invented_fraction", {})
                .get("mean"),
                "within_budget": value.get("invented_within_budget"),
                "phase_correlation_mean": (value.get("audit") or {})
                .get("phase_correlation", {})
                .get("mean"),
                "phase_correlation_rf1u": (value.get("audit") or {})
                .get("phase_correlation_base", {})
                .get("mean"),
                "exact_properties_hold": value.get("exact_properties_hold"),
            }
            for arm, value in audit["arms"].items()
        },
    }
    output = Path(args.output) / target
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(report, indent=1))

    print(f"=== E8 reading, target {target} ===\n")
    print("Scanner probe, balanced accuracy (chance 0.167):")
    columns = ["condition", "encoder_id", "balanced_accuracy"]
    table = probe.assign(condition=probe["condition"].map(strip_label))
    print(table[columns].pivot(index="condition", columns="encoder_id",
                               values="balanced_accuracy").round(3).to_string())
    print(f"\nSection 10 verdict: {section_ten['verdict'].upper()}")
    print(f"  {section_ten['reason']}")
    print(f"\nArm A: {arm_a['verdict']}")
    print(f"  {arm_a['reason']}")


if __name__ == "__main__":
    main()

"""Score the E8 paired residual arms with the frozen E4--E7 fidelity gates.

Nothing about the endpoints is new.  `score` is imported from the RF1U frontier
so the invariance, content and collapse definitions, the thresholds, the seed and
the replicate count are literally the same function, not a re-implementation.
The content anchor is the condition's own target, as RF1U's contract generalised
it.

What is new is the comparison set.  Every E8 arm is reported three ways:

- against **raw**, which is whether it qualifies at all;
- against **Reinhard toward T**, which places it in the report's table;
- against **RF1U toward T**, which is the question arm A exists to answer -- does
  a learned spatially varying correction add anything over three numbers per
  scanner?
- against **CORAL and orthogonal Procrustes retargeted to T**, which is the
  matched image-versus-feature comparison the study report does not currently
  have: its image methods aim at GT450 while its feature methods aim at AT2, and
  the destination result says that difference is not innocent.

Frozen rules: `docs/e8_paired_residual_contract.md`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from analyze_e4_control_frontier import bootstrap_indices
from analyze_rf1u_frontier import score
from e4_primary_metrics import (
    COLLAPSE_METRICS,
    bootstrap_mean_ci,
    collapse_metric_values,
    l2_normalize,
    scanner_centroid_rms,
    unmatched_q95,
)
from e5_comparator_population import SCANNERS
from rf1u_unpaired import RF1U_CONDITION, source_indices, target_index


ANALYSIS = "e8_paired_residual_frontier"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="gt450")
    parser.add_argument("--contract", default="outputs/e0_pfm_contract/checkpoint_manifest.json")
    parser.add_argument("--raw", default="outputs/e0_pfm_features")
    parser.add_argument("--rf1u", default="outputs/rf1u_multitarget/features")
    parser.add_argument(
        "--e8",
        default="outputs/e8_residual/features",
        help="E8 arm features; pass an empty string to score the comparators alone",
    )
    parser.add_argument(
        "--harmonization",
        default="outputs/e8_target_harmonization",
        help="retargeted CORAL/Procrustes; pass an empty string to omit",
    )
    parser.add_argument("--output", default="outputs/e8_residual/frontier")
    return parser.parse_args()


def read_conditions(path: Path, feature_dim: int, target: str):
    """Condition name to features, with every name carrying its destination.

    The feature-space shards store `coral` and `orthogonal_procrustes` without a
    target suffix because they live under a per-target directory.  Appending it
    here keeps one flat namespace in which every condition states where it aims,
    which is the whole point of comparing them.
    """
    with h5py.File(path, "r") as source:
        values = np.asarray(source["features"][:], dtype=np.float32)
        names = [value.decode() for value in source["condition"][:]]
    if values.shape[1:] != (len(SCANNERS), 100, feature_dim):
        raise ValueError(f"{path}: unexpected feature shape {values.shape}")
    return {
        (name if name.endswith(f"_{target}") else f"{name}_{target}"): value
        for name, value in zip(names, values)
    }


def slide_rows(model: dict, target: str, raw_root: Path, roots: dict):
    """Per-slide invariance, content and collapse rows over every condition."""
    model_id = model["encoder_id"]
    feature_dim = int(model["feature_dim"])
    reference = target_index(target)
    sources = source_indices(target)

    available = {
        label: {path.stem for path in (root / target / model_id / "shards").glob("*.h5")}
        for label, root in roots.items()
    }
    slides = sorted(set.intersection(*available.values()))
    if not slides:
        raise RuntimeError(f"{target}/{model_id}: no slide is present in every root")

    invariance_rows, content_rows, collapse_rows = [], [], []
    for slide_id in slides:
        with h5py.File(raw_root / model_id / "shards" / f"{slide_id}.h5", "r") as source:
            raw = np.asarray(source["features"][:], dtype=np.float32)
        if raw.shape != (len(SCANNERS), 100, feature_dim):
            raise ValueError(f"{model_id}/{slide_id}: invalid raw features")
        conditions = {"raw": raw}
        for label, root in roots.items():
            conditions.update(
                read_conditions(
                    root / target / model_id / "shards" / f"{slide_id}.h5",
                    feature_dim,
                    target,
                )
            )

        raw_reference = raw[reference]
        q95 = unmatched_q95(raw_reference)
        reference_unit = l2_normalize(raw_reference)
        raw_collapse = {index: collapse_metric_values(raw[index]) for index in sources}

        for condition, values in conditions.items():
            unit = l2_normalize(values)
            invariance_rows.append(
                {
                    "target": target,
                    "encoder_id": model_id,
                    "slide_id": slide_id,
                    "condition": condition,
                    "scanner_centroid_rms": scanner_centroid_rms(values),
                }
            )
            for index in sources:
                margin = float(
                    (np.sum(unit[index] * reference_unit, axis=-1) - q95).mean()
                )
                content_rows.append(
                    {
                        "target": target,
                        "encoder_id": model_id,
                        "slide_id": slide_id,
                        "condition": condition,
                        "scanner": SCANNERS[index],
                        "content_margin": margin,
                    }
                )
                measured = (
                    raw_collapse[index]
                    if condition == "raw"
                    else collapse_metric_values(values[index])
                )
                for metric in COLLAPSE_METRICS:
                    collapse_rows.append(
                        {
                            "target": target,
                            "encoder_id": model_id,
                            "slide_id": slide_id,
                            "condition": condition,
                            "scanner": SCANNERS[index],
                            "metric": metric,
                            "ratio": measured[metric] / raw_collapse[index][metric],
                        }
                    )
    return invariance_rows, content_rows, collapse_rows, slides


def paired_comparison(radii: dict, condition: str, reference: str, indices, context: dict):
    """Paired per-slide radius difference, positive when `condition` is tighter."""
    if reference not in radii:
        return None
    difference = radii[reference] - radii[condition]
    point, lower, upper = bootstrap_mean_ci(difference, indices)
    return {
        **context,
        "condition": condition,
        "reference": reference,
        "radius_difference": point,
        "difference_ci_lower": lower,
        "difference_ci_upper": upper,
        "improved_over_reference": bool(lower > 0),
        "worse_than_reference": bool(upper < 0),
    }


def main():
    args = parse_args()
    output = Path(args.output) / args.target
    output.mkdir(parents=True, exist_ok=True)
    contract = json.loads(Path(args.contract).read_text())
    models = contract["models"] if isinstance(contract, dict) else contract
    target = args.target
    roots = {"rf1u": Path(args.rf1u)}
    if args.e8 and (Path(args.e8) / target).is_dir():
        roots["e8"] = Path(args.e8)
    if args.harmonization:
        harmonization = Path(args.harmonization)
        if (harmonization / target).is_dir():
            roots["feature"] = harmonization

    endpoint_rows, collapse_detail, comparison_rows = [], [], []
    for model in models:
        model_id = model["encoder_id"]
        invariance, content, collapse, slides = slide_rows(
            model, target, Path(args.raw), roots
        )
        indices = bootstrap_indices(len(slides))
        invariance = pd.DataFrame(invariance)
        content = pd.DataFrame(content)
        collapse = pd.DataFrame(collapse)

        radii = {}
        for condition in sorted(set(invariance["condition"]) - {"raw"}):
            endpoint, rows, radius = score(
                invariance, content, collapse, target, model_id, condition, indices
            )
            endpoint["slides"] = len(slides)
            endpoint_rows.append(endpoint)
            collapse_detail.extend(rows)
            radii[condition] = radius

        context = {"target": target, "encoder_id": model_id, "slides": len(slides)}
        for condition in sorted(radii):
            if not condition.startswith("e8_"):
                continue
            references = (
                f"reinhard_{target}",
                f"{RF1U_CONDITION}_{target}",
                f"coral_{target}",
                f"orthogonal_procrustes_{target}",
            )
            for reference in references:
                row = paired_comparison(radii, condition, reference, indices, context)
                if row is not None:
                    comparison_rows.append(row)

    endpoints = pd.DataFrame(endpoint_rows)
    endpoints.to_csv(output / "endpoint_summary.csv", index=False)
    pd.DataFrame(collapse_detail).to_csv(output / "collapse_detail.csv", index=False)
    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(output / "paired_comparisons.csv", index=False)

    summary = {
        "analysis": ANALYSIS,
        "target": target,
        "models": [model["encoder_id"] for model in models],
        "conditions": sorted(endpoints["condition"].unique()),
        "safe_and_improved_over_raw": {
            condition: int(
                endpoints[endpoints["condition"] == condition][
                    "safe_and_invariance_improved"
                ].sum()
            )
            for condition in sorted(endpoints["condition"].unique())
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=1))

    columns = [
        "encoder_id",
        "condition",
        "relative_radius_reduction",
        "delta_content_margin",
        "content_noninferiority_pass",
        "collapse_every_scanner_pass",
        "safe_and_invariance_improved",
    ]
    print(endpoints[columns].to_string(index=False))
    if not comparisons.empty:
        print()
        print(
            comparisons[
                [
                    "encoder_id",
                    "condition",
                    "reference",
                    "radius_difference",
                    "difference_ci_lower",
                    "difference_ci_upper",
                    "improved_over_reference",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()

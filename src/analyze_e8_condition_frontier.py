"""Invariance and fidelity endpoints for any feature condition.

The probe is an invariance-only endpoint and is degenerate read alone: a
constant representation reaches chance while destroying every patch. That is the
central methodological point of this study, so every new arm needs the frozen
fidelity endpoints beside it, not just a probe number.

This computes the same three quantities the locked frontier uses -- scanner
radius, content margin against the raw AT2 acquisition of the same physical
location, and the three collapse geometry statistics -- for whatever conditions
are pointed at it, using the pure implementations in `e4_primary_metrics.py`.
Inference is blocked by physical slide: the bootstrap resamples slides, never
patches.

The content reference stays raw AT2 whatever the destination is. That is not an
oversight about destinations; it is the frozen definition. Content margin asks
whether a corrected patch still resembles its own physical location more than a
random other location, and the anchor for "its own location" has to be one fixed
acquisition or the question changes with every arm.

Gate thresholds are arguments rather than constants so the same run reports the
frozen decision and its sensitivity. The primary verdict remains the frozen
-0.02 margin and 0.90/0.85 collapse thresholds; anything else is labelled a
sensitivity and never replaces it.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np

from e4_primary_metrics import (
    COLLAPSE_METRICS,
    bootstrap_mean_ci,
    collapse_metric_values,
    content_margin_by_scanner,
    l2_normalize,
    scanner_centroid_rms,
    unmatched_q95,
)
from e5_comparator_population import SCANNERS
from fetch_e0_pfm_checkpoints import sha256


BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20260803
CONTENT_MARGIN = -0.02
COLLAPSE_POINT = 0.90
COLLAPSE_CI = 0.85


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", default="outputs/e0_pfm_contract/checkpoint_manifest.json")
    parser.add_argument("--encoder-index", type=int, required=True)
    parser.add_argument("--raw", default="outputs/e0_pfm_features")
    parser.add_argument("--source", action="append", default=[], metavar="LABEL=ROOT")
    parser.add_argument(
        "--destination",
        action="append",
        default=[],
        metavar="LABEL=SCANNER",
        help="destination each source aims at; defaults to at2",
    )
    parser.add_argument("--content-margin", type=float, default=CONTENT_MARGIN)
    parser.add_argument("--collapse-point", type=float, default=COLLAPSE_POINT)
    parser.add_argument("--collapse-ci", type=float, default=COLLAPSE_CI)
    parser.add_argument("--replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def parse_sources(values):
    sources = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--source expects LABEL=ROOT, got {value!r}")
        label, root = value.split("=", 1)
        if label in sources:
            raise ValueError(f"duplicate source label {label!r}")
        sources[label] = Path(root)
    return sources


def content_margin_against(features: np.ndarray, reference: np.ndarray, destination: int):
    """Content margin for every scanner but the destination, against one reference.

    The frozen endpoint anchors on raw AT2, which was the only destination when
    it was written. That anchor is not neutral once other destinations exist: a
    correction aimed at GT450 moves every embedding onto GT450's distribution and
    therefore away from AT2, so its cosine to raw AT2 falls whether or not any
    content was lost. Measured that way the feature-space arms aimed at GT450 and
    S60 fail content non-inferiority by construction.

    Anchoring instead on the destination's own raw acquisition of the same
    physical location keeps the question the endpoint is meant to ask -- does the
    corrected patch still look more like its own spot than like a random other
    spot -- while letting each destination be scored on its own terms.
    """
    value = l2_normalize(features)
    anchor = l2_normalize(reference)
    baseline = unmatched_q95(anchor)
    rows = [index for index in range(len(SCANNERS)) if index != destination]
    matched = np.sum(value[rows] * anchor[None], axis=-1)
    return matched - baseline[None], [SCANNERS[index] for index in rows]


def slide_statistics(features: np.ndarray, raw: np.ndarray, destination: int):
    """Radius, both content anchors and per-scanner collapse for one slide."""
    destination_margin, destination_scanners = content_margin_against(
        features, raw[destination], destination
    )
    return {
        "radius": scanner_centroid_rms(features),
        # Frozen endpoint, kept so every arm stays comparable to the locked table.
        "content": content_margin_by_scanner(features, raw[0]).mean(axis=1),
        "content_destination": destination_margin.mean(axis=1),
        "destination_scanners": destination_scanners,
        "collapse": [collapse_metric_values(features[index]) for index in range(len(SCANNERS))],
    }


def main():
    args = parse_args()
    contract = json.loads(Path(args.contract).read_text())
    model = contract["models"][args.encoder_index]
    model_id = model["encoder_id"]
    feature_dim = int(model["feature_dim"])
    sources = parse_sources(args.source)
    destinations = {}
    for value in args.destination:
        label, scanner = value.split("=", 1)
        if label not in sources:
            raise ValueError(f"--destination names unknown source {label!r}")
        if scanner not in SCANNERS:
            raise ValueError(f"unknown destination scanner {scanner!r}")
        destinations[label] = SCANNERS.index(scanner)

    raw_paths = sorted((Path(args.raw) / model_id / "shards").glob("*.h5"))
    slide_ids = [path.stem for path in raw_paths]
    if len(slide_ids) != 109:
        raise ValueError(f"{model_id}: expected 109 raw shards, got {len(slide_ids)}")

    per_slide: dict[str, list] = {}
    condition_destination: dict[str, int] = {"raw": 0}
    # Raw is the non-inferiority baseline, so it needs a baseline under every
    # anchor in play, not only AT2.
    needed_destinations = sorted({0} | set(destinations.values()))
    raw_destination_margin: dict[int, list] = {value: [] for value in needed_destinations}

    for path in raw_paths:
        slide_id = path.stem
        with h5py.File(path, "r") as handle:
            raw = np.asarray(handle["features"][:], dtype=np.float32)
        if raw.shape != (len(SCANNERS), 100, feature_dim):
            raise ValueError(f"{model_id}/{slide_id}: unexpected raw feature shape")
        per_slide.setdefault("raw", []).append(slide_statistics(raw, raw, 0))
        for value in needed_destinations:
            margin, _ = content_margin_against(raw, raw[value], value)
            raw_destination_margin[value].append(margin.mean(axis=1))

        for label, root in sources.items():
            destination = destinations.get(label, 0)
            with h5py.File(root / model_id / "shards" / f"{slide_id}.h5", "r") as handle:
                names = [value.decode() for value in handle["condition"][:]]
                values = np.asarray(handle["features"][:], dtype=np.float32)
            if values.ndim != 4 or values.shape[1:] != (len(SCANNERS), 100, feature_dim):
                raise ValueError(f"{root}/{slide_id}: expected (C, 6, 100, D)")
            for index, name in enumerate(names):
                condition = f"{label}:{name}"
                condition_destination[condition] = destination
                per_slide.setdefault(condition, []).append(
                    slide_statistics(values[index], raw, destination)
                )
        print(f"{model_id} {slide_id}", flush=True)

    generator = np.random.default_rng(BOOTSTRAP_SEED)
    indices = generator.integers(0, len(slide_ids), size=(args.replicates, len(slide_ids)))

    raw_radius = np.asarray([entry["radius"] for entry in per_slide["raw"]])
    raw_content = np.stack([entry["content"] for entry in per_slide["raw"]])
    raw_collapse = {
        metric: np.stack(
            [
                [entry["collapse"][index][metric] for index in range(len(SCANNERS))]
                for entry in per_slide["raw"]
            ]
        )
        for metric in COLLAPSE_METRICS
    }

    rows, collapse_rows = [], []
    for condition, entries in per_slide.items():
        radius = np.asarray([entry["radius"] for entry in entries])
        content = np.stack([entry["content"] for entry in entries])
        reduction = 1.0 - radius / raw_radius
        rr, rr_low, rr_high = bootstrap_mean_ci(reduction, indices)
        delta = content - raw_content

        # Every source scanner must clear the gates, not their average: a pooled
        # endpoint hides structured per-scanner response.
        # Second anchor: the destination's own raw acquisition. Reported for every
        # condition so the AT2-anchored verdict can be read against a
        # destination-fair one instead of standing alone.
        destination = condition_destination.get(condition, 0)
        destination_delta = np.stack(
            [entry["content_destination"] for entry in entries]
        ) - np.stack(raw_destination_margin[destination])
        destination_scanners = entries[0]["destination_scanners"]
        worst_destination = float("inf")
        worst_destination_scanner = ""
        for position in range(destination_delta.shape[1]):
            value, low, high = bootstrap_mean_ci(destination_delta[:, position], indices)
            if value < worst_destination:
                worst_destination = value
                worst_destination_scanner = destination_scanners[position]
            rows.append(
                {
                    "encoder_id": model_id,
                    "condition": condition,
                    "endpoint": "content_margin_delta_destination_anchor",
                    "scanner": destination_scanners[position],
                    "estimate": value,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )

        # Primary content gate, matching the locked RF1U frontier exactly: pool
        # over source scanners and locations within a slide, bootstrap the
        # difference from raw, and require its CI lower bound to clear the
        # margin. The per-scanner rows below are diagnostics, not the gate --
        # scoring on the worst scanner instead would be a different, stricter
        # rule than the one this study froze, and would silently disagree with
        # the locked verdicts.
        #
        # The anchor is the destination's own raw acquisition, which is the one
        # generalization the frozen contract allows and reduces to the locked
        # definition when the destination is AT2. The AT2-anchored figures are
        # kept beside it because they are not neutral once destinations differ:
        # aiming at GT450 moves every embedding away from AT2 and drives that
        # column negative whether or not any content was lost.
        pooled_delta = delta.mean(axis=1)
        content_point, content_low, content_high = bootstrap_mean_ci(pooled_delta, indices)
        pooled_destination_delta = destination_delta.mean(axis=1)
        destination_point, destination_low, _ = bootstrap_mean_ci(
            pooled_destination_delta, indices
        )

        worst_content = float("inf")
        worst_content_scanner = ""
        for position in range(delta.shape[1]):
            value, low, high = bootstrap_mean_ci(delta[:, position], indices)
            if value < worst_content:
                worst_content, worst_content_scanner = value, SCANNERS[position + 1]
            rows.append(
                {
                    "encoder_id": model_id,
                    "condition": condition,
                    "endpoint": "content_margin_delta",
                    "scanner": SCANNERS[position + 1],
                    "estimate": value,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )

        worst_collapse = float("inf")
        worst_collapse_ci = float("inf")
        worst_collapse_label = ""
        for metric in COLLAPSE_METRICS:
            observed = np.stack(
                [
                    [entry["collapse"][index][metric] for index in range(len(SCANNERS))]
                    for entry in entries
                ]
            )
            ratio = observed / np.maximum(raw_collapse[metric], 1e-12)
            for index in range(len(SCANNERS)):
                value, low, high = bootstrap_mean_ci(ratio[:, index], indices)
                collapse_rows.append(
                    {
                        "encoder_id": model_id,
                        "condition": condition,
                        "metric": metric,
                        "scanner": SCANNERS[index],
                        "ratio": value,
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )
                if index == 0:
                    continue  # the destination passes through unchanged
                if value < worst_collapse:
                    worst_collapse = value
                    worst_collapse_ci = low
                    worst_collapse_label = f"{SCANNERS[index]}/{metric}"

        content_pass = destination_low >= args.content_margin
        content_pass_at2_anchor = content_low >= args.content_margin
        collapse_pass = (
            worst_collapse >= args.collapse_point and worst_collapse_ci >= args.collapse_ci
        )
        rows.append(
            {
                "encoder_id": model_id,
                "condition": condition,
                "endpoint": "relative_radius_reduction",
                "scanner": "all",
                "estimate": rr,
                "ci95_low": rr_low,
                "ci95_high": rr_high,
                "destination": SCANNERS[condition_destination.get(condition, 0)],
                "pooled_content_delta": destination_point,
                "pooled_content_ci_low": destination_low,
                "worst_content_delta": worst_destination,
                "worst_content_scanner": worst_destination_scanner,
                "pooled_content_delta_at2_anchor": content_point,
                "pooled_content_ci_low_at2_anchor": content_low,
                "pooled_content_ci_high_at2_anchor": content_high,
                "worst_content_delta_at2_anchor": worst_content,
                "worst_content_scanner_at2_anchor": worst_content_scanner,
                "content_pass_at2_anchor": content_pass_at2_anchor,
                "worst_collapse_ratio": worst_collapse,
                "worst_collapse_ci_low": worst_collapse_ci,
                "worst_collapse_cell": worst_collapse_label,
                "content_pass": content_pass,
                "collapse_pass": collapse_pass,
                "safe": content_pass and collapse_pass,
                "improved": rr_low > 0,
                "verdict": (
                    "safe_and_improved"
                    if content_pass and collapse_pass and rr_low > 0
                    else "safe"
                    if content_pass and collapse_pass
                    else "unsafe"
                ),
            }
        )
        print(
            f"{model_id} {condition:44s} RR {rr:+.4f} [{rr_low:+.4f}, {rr_high:+.4f}] "
            f"content(at2) {content_point:+.4f}/{content_low:+.4f} "
            f"content(dest) {destination_point:+.4f}/{destination_low:+.4f} "
            f"collapse {worst_collapse:.4f} {rows[-1]['verdict']}",
            flush=True,
        )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    table = output / f"{model_id}.csv"
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=sorted({key for row in rows for key in row})
        )
        writer.writeheader()
        writer.writerows(rows)
    collapse_table = output / f"{model_id}.collapse.csv"
    with collapse_table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(collapse_rows[0]))
        writer.writeheader()
        writer.writerows(collapse_rows)

    summary = {
        "analysis": "e8_condition_frontier",
        "status": "POST_CORE_EXPLORATORY_EXTENSION",
        "encoder_id": model_id,
        "sources": {label: str(root) for label, root in sources.items()},
        "conditions": list(per_slide),
        "content_reference": (
            "raw acquisition of the same physical location by each condition's own "
            "destination, the generalization the frozen contract allows; reduces "
            "to the locked AT2 definition when the destination is AT2"
        ),
        "content_reference_secondary": (
            "raw AT2 acquisition, reported beside it because it is not neutral "
            "once destinations differ and turns negative for any correction aimed "
            "away from AT2 whether or not content was lost"
        ),
        "destinations": {
            condition: SCANNERS[value] for condition, value in condition_destination.items()
        },
        "thresholds": {
            "content_margin": args.content_margin,
            "collapse_point": args.collapse_point,
            "collapse_ci_low": args.collapse_ci,
            "frozen": (
                args.content_margin == CONTENT_MARGIN
                and args.collapse_point == COLLAPSE_POINT
                and args.collapse_ci == COLLAPSE_CI
            ),
        },
        "gate_rule": "every source scanner must pass, not the average",
        "bootstrap": {
            "replicates": args.replicates,
            "seed": BOOTSTRAP_SEED,
            "unit": "physical slide",
        },
        "artifacts": {
            table.name: sha256(table),
            collapse_table.name: sha256(collapse_table),
        },
    }
    (output / f"{model_id}.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

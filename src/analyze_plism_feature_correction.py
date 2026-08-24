"""CORAL and orthogonal Procrustes on PLISM embeddings — the feature-space arm.

The report ends on a feature-space conclusion: image correction caps out, and the
headroom is in a per-scanner linear map on the embedding.  That conclusion has not
been tested outside our own cohort, and it can be, cheaply, because the PLISM
embeddings already exist — both methods are linear algebra on features rather than
another encoding pass.

Both comparators are fitted leave-one-section-out, so a section never contributes
to the transform applied to it.  CORAL needs no pairing; orthogonal Procrustes is
fitted on the paired same-location embeddings, which PLISM supplies and which is
why it is reported as the paired upper bound rather than as a deployable method.

Endpoints and thresholds come from the same restated-and-checked helpers used for
the image arm.

Contract: docs/e9_plism_native_ert_contract.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_e4_control_frontier import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    COLLAPSE_POINT_THRESHOLD,
    CONTENT_NONINFERIORITY_MARGIN,
)
from analyze_plism_pfm_frontier import (
    REFERENCE,
    aligned_stack,
    centroid_rms,
    collapse_values,
    load_condition,
    unmatched_quantile,
)
from analyze_rf1u_scanner_probe import probe_balanced_accuracy
from e4_primary_metrics import COLLAPSE_METRICS, l2_normalize
from e5_reinhard_residual_frequency import RF1_FOLDS

RIDGE = 1e-6


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="outputs/plism_pfm_features")
    parser.add_argument("--condition", default="raw")
    parser.add_argument("--output", default="outputs/plism_feature_correction")
    parser.add_argument("--destination", default="AT2")
    return parser.parse_args()


def coral_transform(source: np.ndarray, target: np.ndarray):
    """Whiten the source second moment and recolour it with the target's."""
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    dimension = source.shape[1]
    source_cov = np.cov(source - source_mean, rowvar=False) + RIDGE * np.eye(dimension)
    target_cov = np.cov(target - target_mean, rowvar=False) + RIDGE * np.eye(dimension)

    def root(matrix, power):
        values, vectors = np.linalg.eigh(matrix)
        values = np.clip(values, 1e-12, None)
        return (vectors * values**power) @ vectors.T

    matrix = root(source_cov, -0.5) @ root(target_cov, 0.5)
    return source_mean, target_mean, matrix


def procrustes_transform(source: np.ndarray, target: np.ndarray):
    """Orthogonal map minimising |source R - target| on paired rows."""
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_mean).T @ (target - target_mean),
                             full_matrices=False)
    return source_mean, target_mean, u @ vt


def main() -> None:
    args = parse_args()
    root = Path(args.features)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    encoders = sorted(p.name for p in (root / args.condition).iterdir() if p.is_dir())
    rows, probe_rows = [], []

    for encoder in encoders:
        store = load_condition(root, args.condition, encoder)
        sections = sorted(store)
        scanners = sorted(store[sections[0]].keys())
        stacks = {}
        for section in sections:
            if set(store[section]) != set(scanners):
                continue
            stacks[section], _ = aligned_stack(store[section], scanners)
        sections = sorted(stacks)
        destination_index = scanners.index(args.destination)

        for method, fit in (("coral", coral_transform), ("procrustes", procrustes_transform)):
            for held_out in sections:
                train = [s for s in sections if s != held_out]
                corrected = []
                for index, scanner in enumerate(scanners):
                    if index == destination_index:
                        corrected.append(stacks[held_out][index])
                        continue
                    source = np.concatenate([stacks[s][index] for s in train])
                    target = np.concatenate([stacks[s][destination_index] for s in train])
                    source_mean, target_mean, matrix = fit(source, target)
                    corrected.append((stacks[held_out][index] - source_mean) @ matrix + target_mean)
                corrected = np.stack(corrected)
                raw = stacks[held_out]

                anchor = raw[destination_index]
                threshold = unmatched_quantile(anchor)
                anchor_unit = l2_normalize(anchor)
                probe_rows.append({"encoder": encoder, "method": method,
                                   "section": held_out, "features": corrected})
                for index, scanner in enumerate(scanners):
                    margin = (l2_normalize(corrected[index]) * anchor_unit).sum(axis=1) - threshold
                    raw_margin = (l2_normalize(raw[index]) * anchor_unit).sum(axis=1) - threshold
                    collapse = collapse_values(corrected[index])
                    raw_collapse = collapse_values(raw[index])
                    rows.append({
                        "encoder": encoder, "method": method, "section": held_out,
                        "scanner": scanner,
                        "radius": centroid_rms(corrected), "raw_radius": centroid_rms(raw),
                        "content_delta": float(margin.mean() - raw_margin.mean()),
                        **{f"collapse_{m}": collapse[m] / raw_collapse[m]
                           if raw_collapse[m] > 0 else np.nan for m in COLLAPSE_METRICS},
                    })

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "per_section.csv", index=False)

    probe_scores = {}
    for encoder in encoders:
        for method in ("coral", "procrustes"):
            block = [r for r in probe_rows if r["encoder"] == encoder and r["method"] == method]
            sections = sorted({r["section"] for r in block})
            features, groups, folds = [], [], []
            for entry in block:
                stack = entry["features"]
                fold = sections.index(entry["section"]) % RF1_FOLDS
                for index in range(stack.shape[0]):
                    features.append(stack[index])
                    groups.append(np.full(stack.shape[1], index))
                    folds.append(np.full(stack.shape[1], fold))
            scores = probe_balanced_accuracy(np.concatenate(features), np.concatenate(groups),
                                             np.concatenate(folds))
            probe_scores[(encoder, method)] = float(np.mean(scores))
            print(f"  probe {encoder:10s} {method:11s} {np.mean(scores):.4f}")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summary = []
    for (encoder, method), block in frame.groupby(["encoder", "method"]):
        sections = sorted(block["section"].unique())
        draws = rng.integers(0, len(sections), size=(BOOTSTRAP_REPLICATES, len(sections)))
        by_section = block.groupby("section")
        radius = np.array([by_section.get_group(s)["radius"].iloc[0] for s in sections])
        raw_radius = np.array([by_section.get_group(s)["raw_radius"].iloc[0] for s in sections])
        reduction = 1.0 - radius / raw_radius
        samples = reduction[draws].mean(axis=1)
        delta = block.groupby("section")["content_delta"].mean().reindex(sections).to_numpy()
        delta_samples = delta[draws].mean(axis=1)
        by_scanner = block.groupby("scanner")[[f"collapse_{m}" for m in COLLAPSE_METRICS]].mean()
        worst = {m: float(by_scanner[f"collapse_{m}"].min()) for m in COLLAPSE_METRICS}
        summary.append({
            "encoder": encoder, "method": method,
            "rr": float(reduction.mean()),
            "rr_ci_low": float(np.percentile(samples, 2.5)),
            "rr_ci_high": float(np.percentile(samples, 97.5)),
            "content_delta": float(delta.mean()),
            "content_ci_low": float(np.percentile(delta_samples, 2.5)),
            **{f"worst_{m}": v for m, v in worst.items()},
            "probe": probe_scores.get((encoder, method), np.nan),
            "safe": bool(np.percentile(delta_samples, 2.5) > CONTENT_NONINFERIORITY_MARGIN
                         and all(v >= COLLAPSE_POINT_THRESHOLD for v in worst.values())),
            "improved": bool(np.percentile(samples, 2.5) > 0),
        })

    table = pd.DataFrame(summary)
    table.to_csv(output_dir / "frontier.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n=== feature-space correction toward "
          f"{args.destination}, leave-one-section-out ===")
    print(table[["encoder", "method", "rr", "rr_ci_low", "rr_ci_high", "content_delta",
                 "worst_variance_trace", "probe", "safe", "improved"]].round(4).to_string(index=False))
    (output_dir / "summary.json").write_text(json.dumps({
        "condition": args.condition, "destination": args.destination,
        "bootstrap": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
        "chance": 1.0 / frame["scanner"].nunique(), "ridge": RIDGE}, indent=2))
    print(f"\nwrote {output_dir}")


if __name__ == "__main__":
    main()

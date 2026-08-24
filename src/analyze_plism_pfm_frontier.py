"""Representation-level endpoints for the PLISM correction replication — Arm B.

The four locked endpoints, computed on PLISM: scanner radius, content margin,
the collapse guardrail, and the linear scanner probe.

The locked helpers in ``e4_primary_metrics`` hard-code PanNormal's shape — six
scanners by a hundred locations — so the formulas are restated here for seven
scanners and a variable location count.  Nothing else changes, and ``--self-check``
runs both implementations on a 6x100 input and reports the maximum disagreement
before any PLISM number is read.  Thresholds, bootstrap settings and the
non-inferiority margin are imported from the locked modules rather than retyped.

Two cohort facts change the reading and are surfaced in the output.  Chance for
the probe is 1/7 = 0.143, not 0.167.  And the bootstrap replicate unit is the
stain section, of which there are 13 against PanNormal's 109 slides, so intervals
are wide by construction.

Contract: docs/e9_plism_native_ert_contract.md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from analyze_e4_control_frontier import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    COLLAPSE_LOWER_THRESHOLD,
    COLLAPSE_POINT_THRESHOLD,
    CONTENT_NONINFERIORITY_MARGIN,
)
from analyze_rf1u_scanner_probe import probe_balanced_accuracy
from e4_primary_metrics import COLLAPSE_METRICS, l2_normalize
from e5_reinhard_residual_frequency import RF1_FOLDS

REFERENCE = "AT2"
# Content is anchored on the destination's own raw acquisition, the one
# generalisation the frozen contract allows.  An AT2 anchor would flatter every
# correction aimed at AT2 and penalise every correction aimed away from it.
CONDITION_DESTINATION = {"raw": None, "reinhard_at2": "AT2", "rf1u_at2": "AT2",
                         "reinhard_gt450": "GT450", "rf1u_gt450": "GT450"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="outputs/plism_pfm_features")
    parser.add_argument("--output", default="outputs/plism_pfm_frontier")
    parser.add_argument("--raw-condition", default="raw")
    parser.add_argument("--self-check", action="store_true", default=True)
    return parser.parse_args()


def centroid_rms(features: np.ndarray) -> float:
    """RMS distance of each scanner's embedding from the per-location centroid."""
    value = l2_normalize(features)
    centroid = value.mean(axis=0)
    per_location = np.sqrt(np.mean(np.sum((value - centroid[None]) ** 2, axis=-1), axis=0))
    return float(per_location.mean())


def unmatched_quantile(reference: np.ndarray, quantile: float = 0.95) -> np.ndarray:
    """For each location, the q-th percentile cosine to *other* locations."""
    value = l2_normalize(reference)
    similarities = value @ value.T
    return np.asarray([np.quantile(np.delete(similarities[i], i), quantile)
                       for i in range(len(value))], dtype=np.float64)


def collapse_values(features: np.ndarray) -> dict:
    value = l2_normalize(features)
    centered = value - value.mean(axis=0, keepdims=True)
    gram = centered @ centered.T / (len(value) - 1)
    eigenvalues = np.clip(np.linalg.eigvalsh(gram), 0.0, None)
    trace = float(eigenvalues.sum())
    if trace <= 0:
        rank = 0.0
    else:
        probabilities = eigenvalues[eigenvalues > 0] / trace
        rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    similarity = np.clip(value @ value.T, -1.0, 1.0)
    upper = np.triu_indices(len(value), k=1)
    pairwise = np.sqrt(np.maximum(2.0 - 2.0 * similarity[upper], 0.0))
    return {"variance_trace": trace, "entropy_effective_rank": rank,
            "median_pairwise_distance": float(np.median(pairwise))}


def self_check() -> dict:
    """Restated formulas against the locked ones, on PanNormal's own shape."""
    from e4_primary_metrics import (
        collapse_metric_values,
        scanner_centroid_rms,
        unmatched_q95,
    )

    rng = np.random.default_rng(0)
    block = rng.normal(size=(6, 100, 32))
    differences = {
        "scanner_centroid_rms": abs(centroid_rms(block) - scanner_centroid_rms(block)),
        "unmatched_q95": float(np.abs(unmatched_quantile(block[0]) - unmatched_q95(block[0])).max()),
    }
    locked = collapse_metric_values(block[0])
    mine = collapse_values(block[0])
    for metric in COLLAPSE_METRICS:
        differences[metric] = abs(locked[metric] - mine[metric])
    return differences


def load_condition(root: Path, condition: str, encoder: str) -> dict:
    """features[section][scanner] -> (locations, dim) aligned on location id."""
    store: dict = defaultdict(dict)
    directory = root / condition / encoder
    for path in sorted(directory.glob("*.h5")):
        with h5py.File(path, "r") as handle:
            meta = json.loads(handle.attrs["meta"])
            order = np.argsort(handle["location"][:])
            store[meta["stain"]][meta["scanner"]] = {
                "features": handle["features"][:][order],
                "location": handle["location"][:][order],
            }
    return store


def aligned_stack(section: dict, scanners: list[str]) -> tuple[np.ndarray, np.ndarray]:
    shared = None
    for scanner in scanners:
        ids = section[scanner]["location"]
        shared = ids if shared is None else np.intersect1d(shared, ids)
    stack = []
    for scanner in scanners:
        entry = section[scanner]
        mask = np.isin(entry["location"], shared)
        stack.append(entry["features"][mask])
    return np.stack(stack), shared


def main() -> None:
    args = parse_args()
    root = Path(args.features)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    checks = self_check()
    worst = max(checks.values())
    print("self-check against the locked estimators on a 6x100 block:")
    for name, value in checks.items():
        print(f"  {name:26s} |difference| = {value:.3e}")
    if worst > 1e-9:
        raise RuntimeError(f"restated metric disagrees with the locked one by {worst:.3e}")
    print(f"  worst {worst:.3e} — restatement is exact\n")

    conditions = sorted(p.name for p in root.iterdir() if p.is_dir())
    encoders = sorted(p.name for p in (root / args.raw_condition).iterdir() if p.is_dir())
    print(f"conditions {conditions}\nencoders {encoders}\n")

    rows = []
    probe_rows = []
    for encoder in encoders:
        raw_store = load_condition(root, args.raw_condition, encoder)
        scanners = sorted(next(iter(raw_store.values())).keys())
        for condition in conditions:
            store = load_condition(root, condition, encoder)
            for section in sorted(store):
                if set(store[section]) != set(scanners):
                    continue
                corrected, shared = aligned_stack(store[section], scanners)
                raw, _ = aligned_stack(raw_store[section], scanners)
                probe_rows.append({"encoder": encoder, "condition": condition,
                                   "section": section, "features": corrected})

                radius = centroid_rms(corrected)
                raw_radius = centroid_rms(raw)
                destination = CONDITION_DESTINATION.get(condition) or REFERENCE
                anchor = raw[scanners.index(destination)]
                threshold = unmatched_quantile(anchor)
                anchor_unit = l2_normalize(anchor)

                for index, scanner in enumerate(scanners):
                    margin = (l2_normalize(corrected[index]) * anchor_unit).sum(axis=1) - threshold
                    raw_margin = (l2_normalize(raw[index]) * anchor_unit).sum(axis=1) - threshold
                    collapse = collapse_values(corrected[index])
                    raw_collapse = collapse_values(raw[index])
                    rows.append({
                        "encoder": encoder, "condition": condition, "section": section,
                        "scanner": scanner, "locations": len(shared),
                        "anchor": destination,
                        "radius": radius, "raw_radius": raw_radius,
                        "content_margin": float(margin.mean()),
                        "content_delta": float(margin.mean() - raw_margin.mean()),
                        **{f"collapse_{m}": collapse[m] / raw_collapse[m]
                           if raw_collapse[m] > 0 else np.nan for m in COLLAPSE_METRICS},
                    })

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "per_section.csv", index=False)

    # Scanner probe, blocked by stain section.  The locked routine runs five folds,
    # so the 13 sections are dealt round-robin into them; a section is therefore
    # never in both the fit and the test.
    probe_scores = {}
    for encoder in encoders:
        for condition in conditions:
            block = [r for r in probe_rows if r["encoder"] == encoder
                     and r["condition"] == condition]
            if not block:
                continue
            sections = sorted({r["section"] for r in block})
            features, groups, folds = [], [], []
            for entry in block:
                stack = entry["features"]
                fold = sections.index(entry["section"]) % RF1_FOLDS
                for index in range(stack.shape[0]):
                    features.append(stack[index])
                    groups.append(np.full(stack.shape[1], index))
                    folds.append(np.full(stack.shape[1], fold))
            scores = probe_balanced_accuracy(np.concatenate(features),
                                             np.concatenate(groups),
                                             np.concatenate(folds))
            probe_scores[(encoder, condition)] = float(np.mean(scores))
            print(f"  probe {encoder:10s} {condition:16s} {np.mean(scores):.4f}")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summary = []
    for (encoder, condition), block in frame.groupby(["encoder", "condition"]):
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
        worst_collapse = {m: float(by_scanner[f"collapse_{m}"].min()) for m in COLLAPSE_METRICS}
        worst_scanner = str(by_scanner["collapse_variance_trace"].idxmin())
        safe = (np.percentile(delta_samples, 2.5) > CONTENT_NONINFERIORITY_MARGIN
                and all(v >= COLLAPSE_POINT_THRESHOLD for v in worst_collapse.values()))
        summary.append({
            "encoder": encoder, "condition": condition,
            "rr": float(reduction.mean()),
            "rr_ci_low": float(np.percentile(samples, 2.5)),
            "rr_ci_high": float(np.percentile(samples, 97.5)),
            "content_delta": float(delta.mean()),
            "content_ci_low": float(np.percentile(delta_samples, 2.5)),
            **{f"worst_{m}": float(v) for m, v in worst_collapse.items()},
            "worst_scanner": worst_scanner,
            "safe": bool(safe),
            "improved": bool(np.percentile(samples, 2.5) > 0),
            "sections": len(sections),
            "probe": probe_scores.get((encoder, condition), float("nan")),
            "probe_chance": 1.0 / block["scanner"].nunique(),
        })

    table = pd.DataFrame(summary)
    table.to_csv(output_dir / "frontier.csv", index=False)
    pd.set_option("display.width", 230)
    print("=== relative radius reduction and safety, bootstrap over 13 sections ===")
    display = table[["encoder", "condition", "rr", "rr_ci_low", "rr_ci_high", "content_delta",
                     "content_ci_low", "worst_variance_trace", "probe", "safe", "improved"]]
    print(display.round(4).to_string(index=False))
    (output_dir / "summary.json").write_text(json.dumps({
        "self_check_worst": float(worst),
        "chance_level": 1.0 / len(next(iter(load_condition(root, args.raw_condition,
                                                           encoders[0]).values()))),
        "bootstrap": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
        "content_margin": CONTENT_NONINFERIORITY_MARGIN,
        "collapse_thresholds": [COLLAPSE_POINT_THRESHOLD, COLLAPSE_LOWER_THRESHOLD],
        "replicate_unit": "stain section",
    }, indent=2))
    print(f"\nwrote {output_dir}")


if __name__ == "__main__":
    main()

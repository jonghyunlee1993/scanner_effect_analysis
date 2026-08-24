"""Image-space correction endpoints on the PLISM core grid — Arm B.

Replaces `analyze_plism_pfm_frontier.py`, which ran on the sparse sampling and
the E0 encoder panel.  The estimator is unchanged and is imported rather than
retyped; what changes is the cohort underneath it — ~8,950 refined locations per
section instead of a few hundred, the SQ file-label correction applied, and the
E9 panel of encoders a 2026 study would deploy.

The four locked endpoints are read on the frozen evaluation subsample, which is
also the only place the correction conditions were encoded.  The raw condition is
not re-encoded: the full feature store already holds those vectors, produced by
the same pass with the same crop and the same eval transform, so it is taken as a
subset.  Every condition of a section therefore sits on identical physical
locations and the contrasts are exactly paired.

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
    COLLAPSE_LOWER_THRESHOLD,
    COLLAPSE_POINT_THRESHOLD,
    CONTENT_NONINFERIORITY_MARGIN,
)
from analyze_plism_core_feature_correction import PROBE_STRIDE, align, load_encoder
from analyze_plism_pfm_frontier import (
    CONDITION_DESTINATION,
    REFERENCE,
    centroid_rms,
    collapse_values,
    self_check,
    unmatched_quantile,
)
from analyze_rf1u_scanner_probe import probe_balanced_accuracy
from e4_primary_metrics import COLLAPSE_METRICS, l2_normalize
from e5_reinhard_residual_frequency import RF1_FOLDS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/PLISM_dataset/features")
    parser.add_argument("--conditions", default="outputs/plism_core_condition_features")
    parser.add_argument("--eval-locations", default="outputs/plism_core_eval_locations")
    parser.add_argument("--output", default="outputs/plism_core_pfm_frontier")
    parser.add_argument("--encoders", default="")
    return parser.parse_args()


def evaluated(store: dict, scanners: list[str], encoder: str, eval_dir: Path) -> dict:
    """section -> (scanners, evaluated locations, dim), on the frozen subsample."""
    out = {}
    for section in sorted(store):
        if set(store[section]) != set(scanners):
            continue
        stack, shared = align(store[section], scanners)
        wanted = json.loads((eval_dir / encoder / f"{section}.json").read_text())
        mask = np.isin(shared, np.asarray(wanted["locations"]))
        out[section] = (np.asarray(stack[:, mask], dtype=np.float64), shared[mask])
    return out


def main() -> None:
    args = parse_args()
    raw_root, condition_root = Path(args.raw), Path(args.conditions)
    eval_dir = Path(args.eval_locations)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    checks = self_check()
    print("self-check against the locked estimators on a 6x100 block:")
    for name, value in checks.items():
        print(f"  {name:26s} |difference| = {value:.3e}")
    worst = max(checks.values())
    if worst > 1e-9:
        raise RuntimeError(f"restated metric disagrees with the locked one by {worst:.3e}")
    print(f"  worst {worst:.3e} — restatement is exact\n")

    encoders = ([e.strip() for e in args.encoders.split(",") if e.strip()]
                or sorted(p.name for p in raw_root.iterdir()
                          if p.is_dir() and p.name != "analysis"))
    if not condition_root.exists():
        raise FileNotFoundError(
            f"{condition_root} — the correction conditions have not been encoded; "
            "sections 17 and 18 cannot be written from the raw condition alone")
    conditions = ["raw"] + sorted(p.name for p in condition_root.iterdir() if p.is_dir())
    print(f"encoders {encoders}\nconditions {conditions}\n")

    rows, probe_scores = [], {}
    for encoder in encoders:
        raw_store = load_encoder(raw_root / encoder)
        scanners = sorted(next(iter(raw_store.values())))
        raw = evaluated(raw_store, scanners, encoder, eval_dir)
        del raw_store
        print(f"{encoder}: {len(raw)} sections x {len(scanners)} scanners, "
              f"{raw[next(iter(raw))][0].shape[1]:,} evaluated locations")

        probe_features = {}
        for condition in conditions:
            if condition == "raw":
                current = raw
            else:
                directory = condition_root / condition / encoder
                if not directory.exists():
                    print(f"  {condition}: absent for {encoder}, skipped")
                    continue
                store = load_encoder(directory)
                current = {}
                for section, (stack, shared) in raw.items():
                    if section not in store or set(store[section]) != set(scanners):
                        continue
                    block, ids = align(store[section], scanners)
                    mask = np.isin(ids, shared)
                    keep = np.isin(shared, ids[mask])
                    current[section] = (np.asarray(block[:, mask], dtype=np.float64),
                                        ids[mask], keep)
                del store

            anchor_scanner = CONDITION_DESTINATION.get(condition) or REFERENCE
            for section in sorted(current):
                entry = current[section]
                stack = entry[0]
                raw_stack = raw[section][0] if condition == "raw" else raw[section][0][:, entry[2]]
                if stack.shape[1] != raw_stack.shape[1]:
                    raise RuntimeError(
                        f"{encoder}/{condition}/{section}: {stack.shape[1]} locations "
                        f"against {raw_stack.shape[1]} raw")
                probe_features[(condition, section)] = stack.astype(np.float32)

                anchor = raw_stack[scanners.index(anchor_scanner)]
                threshold = unmatched_quantile(anchor)
                anchor_unit = l2_normalize(anchor)
                radius, raw_radius = centroid_rms(stack), centroid_rms(raw_stack)
                for index, scanner in enumerate(scanners):
                    margin = (l2_normalize(stack[index]) * anchor_unit).sum(axis=1) - threshold
                    raw_margin = (l2_normalize(raw_stack[index]) * anchor_unit).sum(axis=1) \
                        - threshold
                    collapse = collapse_values(stack[index])
                    base = collapse_values(raw_stack[index])
                    rows.append({
                        "encoder": encoder, "condition": condition, "section": section,
                        "scanner": scanner, "locations": stack.shape[1],
                        "anchor": anchor_scanner,
                        "radius": radius, "raw_radius": raw_radius,
                        "content_margin": float(margin.mean()),
                        "content_delta": float(margin.mean() - raw_margin.mean()),
                        **{f"collapse_{m}": collapse[m] / base[m] if base[m] > 0 else np.nan
                           for m in COLLAPSE_METRICS}})
            if not current:
                print(f"  {condition:16s} no complete section, skipped")
                continue
            print(f"  {condition:16s} {len(current)} sections  "
                  f"rr={1 - rows[-1]['radius'] / rows[-1]['raw_radius']:+.4f} (last section)")

        print(f"  scanner probe, every {PROBE_STRIDE}th evaluated location")
        for condition in sorted({c for c, _ in probe_features}):
            sections = sorted(s for c, s in probe_features if c == condition)
            features, groups, folds = [], [], []
            for section in sections:
                stack = probe_features[(condition, section)][:, ::PROBE_STRIDE]
                fold = sections.index(section) % RF1_FOLDS
                for index in range(stack.shape[0]):
                    features.append(stack[index])
                    groups.append(np.full(stack.shape[1], index))
                    folds.append(np.full(stack.shape[1], fold))
            scores = probe_balanced_accuracy(np.concatenate(features), np.concatenate(groups),
                                             np.concatenate(folds))
            probe_scores[(encoder, condition)] = float(np.mean(scores))
            print(f"    {condition:16s} {np.mean(scores):.4f}")
        del raw, probe_features
        print()

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "per_section.csv", index=False)

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
        summary.append({
            "encoder": encoder, "condition": condition,
            "rr": float(reduction.mean()),
            "rr_ci_low": float(np.percentile(samples, 2.5)),
            "rr_ci_high": float(np.percentile(samples, 97.5)),
            "content_delta": float(delta.mean()),
            "content_ci_low": float(np.percentile(delta_samples, 2.5)),
            **{f"worst_{m}": float(v) for m, v in worst_collapse.items()},
            "worst_scanner": str(by_scanner["collapse_variance_trace"].idxmin()),
            "safe": bool(np.percentile(delta_samples, 2.5) > CONTENT_NONINFERIORITY_MARGIN
                         and all(v >= COLLAPSE_POINT_THRESHOLD
                                 for v in worst_collapse.values())),
            "improved": bool(np.percentile(samples, 2.5) > 0),
            "sections": len(sections),
            "probe": probe_scores.get((encoder, condition), float("nan")),
            "probe_chance": 1.0 / block["scanner"].nunique()})

    table = pd.DataFrame(summary)
    table.to_csv(output_dir / "frontier.csv", index=False)
    pd.set_option("display.width", 240)
    print("=== relative radius reduction and safety, bootstrap over sections ===")
    print(table[["encoder", "condition", "rr", "rr_ci_low", "rr_ci_high", "content_delta",
                 "content_ci_low", "worst_variance_trace", "probe", "safe", "improved"]]
          .round(4).to_string(index=False))
    (output_dir / "summary.json").write_text(json.dumps({
        "analysis": "plism_core_pfm_frontier",
        "self_check_worst": float(worst),
        "encoders": encoders, "conditions": conditions,
        "probe_stride": PROBE_STRIDE,
        "bootstrap": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
        "content_margin": CONTENT_NONINFERIORITY_MARGIN,
        "collapse_thresholds": [COLLAPSE_POINT_THRESHOLD, COLLAPSE_LOWER_THRESHOLD],
        "replicate_unit": "stain section"}, indent=2))
    print(f"\nwrote {output_dir}")


if __name__ == "__main__":
    main()

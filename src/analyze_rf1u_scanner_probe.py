"""Can a linear probe still tell the six scanners apart after correction?

Scanner radius measures how far the six acquisitions of one location drift from
their own centroid. A probe asks the adversarial version of the same question:
fit a classifier on training slides and see whether it can still name the scanner
on slides it has never seen. Chance is 1/6.

This is deliberately an invariance-only endpoint, and invariance-only endpoints
are degenerate on their own — a constant representation drives the probe to
chance while destroying everything. It is therefore reported next to the content
and collapse verdicts the frozen contract already produced, never alone.

The estimator matches the earlier Exp-02 separability probe: standardize, then
multinomial logistic regression with balanced class weights at C=1. Splits are
the frozen RF1 physical-slide folds, so no slide contributes to the fit that
scores it.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from e4_primary_metrics import l2_normalize
from e5_comparator_population import SCANNERS
from e5_reinhard_residual_frequency import RF1_FOLDS, fold_assignments
from extract_rf1u_features import ANALYSIS as SHARD_ANALYSIS
from fetch_e0_pfm_checkpoints import sha256
from rf1u_unpaired import RF1U_TARGETS, RF1U_VERSION


LOCATION_STRIDE = 4
PROBE_SEED = 20260803


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", default="outputs/e0_pfm_contract/checkpoint_manifest.json")
    parser.add_argument("--encoder-index", type=int, required=True)
    parser.add_argument("--raw", default="outputs/e0_pfm_features")
    parser.add_argument("--features", default="outputs/rf1u_multitarget/features")
    parser.add_argument(
        "--feature-harmonization", default="outputs/e5_feature_harmonization"
    )
    parser.add_argument("--output", default="outputs/rf1u_multitarget/scanner_probe")
    parser.add_argument("--location-stride", type=int, default=LOCATION_STRIDE)
    return parser.parse_args()


def probe_balanced_accuracy(features: np.ndarray, groups: np.ndarray, folds: np.ndarray):
    """Slide-blocked balanced accuracy of a linear scanner classifier.

    `features` is samples by dimension, `groups` the scanner label of each
    sample and `folds` the physical-slide fold, so a slide is never in both the
    fit and the score of the same split.
    """
    if features.ndim != 2 or len(groups) != len(features) or len(folds) != len(features):
        raise ValueError("probe inputs must be aligned")
    scores = []
    for fold in range(RF1_FOLDS):
        test = folds == fold
        train = ~test
        if not test.any() or not train.any():
            raise ValueError(f"fold {fold} is empty")
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                C=1.0,
                random_state=PROBE_SEED,
            ),
        )
        model.fit(features[train], groups[train])
        scores.append(
            float(balanced_accuracy_score(groups[test], model.predict(features[test])))
        )
    return scores


def main():
    args = parse_args()
    contract = json.loads(Path(args.contract).read_text())
    model = contract["models"][args.encoder_index]
    model_id = model["encoder_id"]
    feature_dim = int(model["feature_dim"])
    positions = list(range(0, 100, args.location_stride))

    raw_paths = sorted((Path(args.raw) / model_id / "shards").glob("*.h5"))
    slide_ids = [path.stem for path in raw_paths]
    if len(slide_ids) != 109:
        raise ValueError(f"{model_id}: expected 109 raw shards, got {len(slide_ids)}")
    assignments = fold_assignments(slide_ids)

    harmonization = Path(args.feature_harmonization) / model_id / "shards"
    feature_methods = []
    if harmonization.exists():
        with h5py.File(next(harmonization.glob("*.h5")), "r") as source:
            feature_methods = [value.decode() for value in source["condition"][:]]

    stacks = {"raw": []}
    for target in RF1U_TARGETS:
        stacks[f"reinhard_{target}"] = []
        stacks[f"ours_{target}"] = []
    for method in feature_methods:
        stacks[method] = []
    groups, folds = [], []

    for path in raw_paths:
        slide_id = path.stem
        with h5py.File(path, "r") as source:
            raw = np.asarray(source["features"][:], dtype=np.float32)
        if raw.shape != (len(SCANNERS), 100, feature_dim):
            raise ValueError(f"{model_id}/{slide_id}: unexpected raw feature shape")
        stacks["raw"].append(l2_normalize(raw[:, positions]).reshape(-1, feature_dim))
        for target in RF1U_TARGETS:
            shard = (
                Path(args.features) / target / model_id / "shards" / f"{slide_id}.h5"
            )
            summary = json.loads(shard.with_suffix(".summary.json").read_text())
            if not (
                summary.get("analysis") == SHARD_ANALYSIS
                and summary.get("target") == target
                and summary.get("shard_gate_pass") is True
            ):
                raise RuntimeError(f"invalid RF1U shard: {shard}")
            with h5py.File(shard, "r") as source:
                names = [value.decode() for value in source["condition"][:]]
                values = np.asarray(source["features"][:], dtype=np.float32)
            for index, name in enumerate(names):
                key = (
                    f"reinhard_{target}"
                    if name.startswith("reinhard_" + target)
                    else f"ours_{target}"
                )
                stacks[key].append(
                    l2_normalize(values[index][:, positions]).reshape(-1, feature_dim)
                )
        if feature_methods:
            with h5py.File(harmonization / f"{slide_id}.h5", "r") as source:
                names = [value.decode() for value in source["condition"][:]]
                harmonized = np.asarray(source["features"][:], dtype=np.float32)
            if names != feature_methods:
                raise RuntimeError(f"{slide_id}: feature-method order differs")
            for index, method in enumerate(names):
                stacks[method].append(
                    l2_normalize(harmonized[index][:, positions]).reshape(-1, feature_dim)
                )
        groups.append(
            np.repeat(np.arange(len(SCANNERS)), len(positions)).astype(np.int64)
        )
        folds.append(np.full(len(SCANNERS) * len(positions), assignments[slide_id]))

    groups = np.concatenate(groups)
    folds = np.concatenate(folds)
    rows = []
    for condition, parts in stacks.items():
        features = np.concatenate(parts).astype(np.float64)
        scores = probe_balanced_accuracy(features, groups, folds)
        rows.append(
            {
                "encoder_id": model_id,
                "condition": condition,
                "samples": int(len(features)),
                "locations_per_slide": len(positions),
                "balanced_accuracy": float(np.mean(scores)),
                "fold_min": float(np.min(scores)),
                "fold_max": float(np.max(scores)),
                **{f"fold_{index}": value for index, value in enumerate(scores)},
            }
        )
        print(
            f"{model_id} {condition:28s} BACC {np.mean(scores):.4f} "
            f"[{np.min(scores):.4f}, {np.max(scores):.4f}]",
            flush=True,
        )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    table = output / f"{model_id}.csv"
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "analysis": "rf1u_scanner_probe",
        "rf1u_version": RF1U_VERSION,
        "status": "SECONDARY_INVARIANCE_ONLY_ENDPOINT",
        "caveat": (
            "An invariance-only endpoint is degenerate on its own: a constant "
            "representation reaches chance while destroying all content. Read this "
            "next to the frozen content and collapse verdicts, never alone."
        ),
        "encoder_id": model_id,
        "estimator": "StandardScaler + multinomial logistic regression, balanced, C=1",
        "splits": "frozen RF1 physical-slide folds",
        "chance_balanced_accuracy": 1.0 / len(SCANNERS),
        "locations_per_slide": len(positions),
        "location_stride": args.location_stride,
        "seed": PROBE_SEED,
        "conditions": list(stacks),
        "feature_space_methods": feature_methods,
        "feature_space_note": (
            "CORAL and orthogonal Procrustes are the locked E5 feature-space "
            "comparators, already cross-fitted by 109-fold leave-one-slide-out; "
            "only the probe is refitted here."
        ),
        "artifacts": {table.name: sha256(table)},
    }
    (output / f"{model_id}.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

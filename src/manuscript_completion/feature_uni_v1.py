"""Held-out UNI feature correction on the manuscript's 103-slide paired panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prenorm.feature_correction import (
    apply_affine, fit_affine, fit_combat, fit_procrustes, macro_retrieval,
    paired_distance, unit,
)

from manuscript_completion.pfm_problem import (
    DEFAULT_COHORT,
    DEFAULT_FOLDS,
    DEFAULT_INPUT_ROOT,
    TARGET_SCANNERS,
    audit_and_load_raw_embeddings,
    load_contract,
)
from manuscript_completion.feature_panel40 import load_panel40


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--shards", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--direction", choices=("at2_to_scanner", "scanner_to_at2"), default="at2_to_scanner")
    parser.add_argument("--train-panel", type=int, choices=(20, 40), default=20)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cohort = load_contract(args.cohort, args.folds)
    embeddings, _, audit = audit_and_load_raw_embeddings(cohort, args.shards)
    train_embeddings = embeddings if args.train_panel == 20 else load_panel40(cohort)[0]
    audit.to_csv(args.output / "input_audit.csv", index=False)
    (args.output / "design.json").write_text(json.dumps({
        "direction": args.direction, "training_locations_per_slide": args.train_panel,
        "evaluation_locations_per_slide": 20,
        "outer_folds": "locked physical-slide 5-fold",
    }, indent=2) + "\n")
    folds = cohort["fold"].to_numpy(dtype=int)
    tissues = cohort["tissue_type"].to_numpy()
    rows = []
    retrieval_rows = []
    alpha_rows = []
    for scanner_index, scanner in enumerate(TARGET_SCANNERS, start=1):
        if args.direction == "at2_to_scanner":
            source = embeddings[:, :, 0, :].astype(np.float64)
            target = embeddings[:, :, scanner_index, :].astype(np.float64)
        else:
            source = embeddings[:, :, scanner_index, :].astype(np.float64)
            target = embeddings[:, :, 0, :].astype(np.float64)
        predictions = {
            "raw": source.copy(),
            "featmap_ridge": np.empty_like(source),
            "featmap_ols": np.empty_like(source),
            "procrustes": np.empty_like(source),
            "combat": np.empty_like(source),
        }
        for outer_fold in range(5):
            train = folds != outer_fold
            test = folds == outer_fold
            validation = folds == (outer_fold + 1) % 5
            inner_train = train & ~validation
            train_source_index, train_target_index = ((0, scanner_index)
                if args.direction == "at2_to_scanner" else (scanner_index, 0))
            xin = train_embeddings[inner_train, :, train_source_index, :].reshape(-1, source.shape[-1]).astype(np.float64)
            yin = train_embeddings[inner_train, :, train_target_index, :].reshape(-1, target.shape[-1]).astype(np.float64)
            xval = source[validation].reshape(-1, source.shape[-1])
            yval = target[validation].reshape(-1, target.shape[-1])
            candidates = (0.01, 0.1, 1.0)
            losses = []
            for alpha in candidates:
                mapped = apply_affine(fit_affine(xin, yin, alpha), xval)
                losses.append(float(paired_distance(mapped, yval).mean()))
            chosen = candidates[int(np.argmin(losses))]
            alpha_rows.append({
                "scanner": scanner, "outer_fold": outer_fold,
                "chosen_relative_alpha": chosen, "inner_losses": json.dumps(losses),
            })
            xtrain = train_embeddings[train, :, train_source_index, :].reshape(-1, source.shape[-1]).astype(np.float64)
            ytrain = train_embeddings[train, :, train_target_index, :].reshape(-1, target.shape[-1]).astype(np.float64)
            xtest = source[test].reshape(-1, source.shape[-1])
            test_shape = source[test].shape
            predictions["featmap_ridge"][test] = apply_affine(
                fit_affine(xtrain, ytrain, chosen), xtest
            ).reshape(test_shape)
            predictions["featmap_ols"][test] = apply_affine(
                fit_affine(xtrain, ytrain, 0.0), xtest
            ).reshape(test_shape)
            predictions["procrustes"][test] = apply_affine(
                fit_procrustes(xtrain, ytrain), xtest
            ).reshape(test_shape)
            predictions["combat"][test] = fit_combat(xtrain, ytrain, xtest).reshape(test_shape)
        raw_distance = paired_distance(source, target).mean(axis=1)
        target_slide = unit(target.mean(axis=1))
        for method, corrected in predictions.items():
            distances = paired_distance(corrected, target).mean(axis=1)
            corrected_slide = unit(corrected.mean(axis=1))
            retrieval_rows.append({
                "scanner": scanner, "method": method,
                "macro_tissue_retrieval": macro_retrieval(corrected_slide, target_slide, tissues),
                "target_self_retrieval": macro_retrieval(target_slide, target_slide, tissues),
                "variance_trace_ratio": float(np.var(corrected_slide, axis=0).sum() / np.var(target_slide, axis=0).sum()),
            })
            for i, cohort_row in enumerate(cohort.itertuples(index=False)):
                rows.append({
                    "slide_id": cohort_row.slide_id, "tissue_type": cohort_row.tissue_type,
                    "fold": int(cohort_row.fold), "scanner": scanner, "method": method,
                    "raw_distance": float(raw_distance[i]),
                    "corrected_distance": float(distances[i]),
                    "target_gain": float(raw_distance[i] - distances[i]),
                })
        print(json.dumps({"scanner_completed": scanner}), flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(args.output / "per_slide.csv", index=False)
    pd.DataFrame(retrieval_rows).to_csv(args.output / "content_summary.csv", index=False)
    pd.DataFrame(alpha_rows).to_csv(args.output / "ridge_selection.csv", index=False)
    scanner_summary = result.groupby(["scanner", "method"], sort=True)[
        ["raw_distance", "corrected_distance", "target_gain"]
    ].mean().reset_index()
    scanner_summary.to_csv(args.output / "per_scanner_summary.csv", index=False)
    pooled = result.groupby(["slide_id", "method"], sort=True)["target_gain"].mean().unstack()
    rng = np.random.default_rng(20260925)
    sampled = rng.integers(0, len(pooled), size=(5000, len(pooled)))
    summary_rows = []
    for method in pooled.columns:
        values = pooled[method].to_numpy()
        boot = values[sampled].mean(axis=1)
        summary_rows.append({
            "method": method, "mean_target_gain": float(values.mean()),
            "ci_low": float(np.quantile(boot, 0.025)),
            "ci_high": float(np.quantile(boot, 0.975)),
        })
    pd.DataFrame(summary_rows).to_csv(args.output / "pooled_summary.csv", index=False)


if __name__ == "__main__":
    main()

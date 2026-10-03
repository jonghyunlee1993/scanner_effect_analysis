"""Apply PanNormal-fitted UNI feature maps to PLISM without refitting."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from prenorm.feature_correction import (
    apply_affine, fit_affine, fit_combat, fit_procrustes, paired_distance,
)
from manuscript_completion.pfm_problem import (
    DEFAULT_COHORT, DEFAULT_FOLDS, DEFAULT_INPUT_ROOT, TARGET_SCANNERS,
    audit_and_load_raw_embeddings, load_contract,
)
from manuscript_completion.feature_panel40 import load_panel40


PLISM = Path(
    "outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/"
    "05_plism_external_correction/04_external_uni/features"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--internal-output", type=Path, required=True)
    parser.add_argument("--direction", choices=("at2_to_scanner", "scanner_to_at2"), default="at2_to_scanner")
    parser.add_argument("--train-panel", type=int, choices=(20, 40), default=20)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cohort = load_contract(DEFAULT_COHORT, DEFAULT_FOLDS)
    embeddings, _, _ = audit_and_load_raw_embeddings(cohort, DEFAULT_INPUT_ROOT)
    train_embeddings = embeddings if args.train_panel == 20 else load_panel40(cohort)[0]
    folds = cohort["fold"].to_numpy(dtype=int)
    choices = pd.read_csv(args.internal_output / "ridge_selection.csv")
    records = []
    for scanner in ("gt450", "s360", "s60"):
        scanner_index = TARGET_SCANNERS.index(scanner) + 1
        files = sorted((PLISM / scanner).glob("*.h5"))
        if len(files) != 13:
            raise ValueError(f"expected 13 PLISM sections for {scanner}, got {len(files)}")
        sections = []
        for path in files:
            with h5py.File(path, "r") as store:
                at2 = np.asarray(store["target_features"], dtype=np.float64)
                scanner_features = np.asarray(store["source_features"], dtype=np.float64)
            sections.append((path.stem, at2, scanner_features))
        for fold in range(5):
            train = folds != fold
            x_index, y_index = ((0, scanner_index) if args.direction == "at2_to_scanner"
                                else (scanner_index, 0))
            x = train_embeddings[train, :, x_index, :].reshape(-1, train_embeddings.shape[-1]).astype(np.float64)
            y = train_embeddings[train, :, y_index, :].reshape(-1, train_embeddings.shape[-1]).astype(np.float64)
            alpha = float(choices.loc[
                choices["scanner"].eq(scanner) & choices["outer_fold"].eq(fold),
                "chosen_relative_alpha",
            ].iloc[0])
            affine = fit_affine(x, y, alpha)
            affine_ols = fit_affine(x, y, 0.0)
            procrustes = fit_procrustes(x, y)
            for section, at2, scanner_features in sections:
                source, target = ((at2, scanner_features) if args.direction == "at2_to_scanner"
                                  else (scanner_features, at2))
                raw = paired_distance(source, target)
                corrected = {
                    "raw": source,
                    "featmap_ridge": apply_affine(affine, source),
                    "featmap_ols": apply_affine(affine_ols, source),
                    "procrustes": apply_affine(procrustes, source),
                    "combat": fit_combat(x, y, source),
                }
                for method, mapped in corrected.items():
                    distance = paired_distance(mapped, target)
                    records.append({
                        "scanner": scanner, "section": section, "fold": fold,
                        "method": method, "locations": len(raw),
                        "raw_distance": float(raw.mean()),
                        "corrected_distance": float(distance.mean()),
                        "target_gain": float((raw - distance).mean()),
                    })
        print(f"completed {scanner}", flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(args.output / "per_section_fold.csv", index=False)
    section = frame.groupby(["scanner", "section", "method"], sort=True)[
        ["raw_distance", "corrected_distance", "target_gain"]
    ].mean().reset_index()
    section.to_csv(args.output / "per_section.csv", index=False)
    section.groupby(["scanner", "method"], sort=True)[
        ["raw_distance", "corrected_distance", "target_gain"]
    ].mean().to_csv(args.output / "per_scanner_summary.csv")
    pooled = section.groupby(["section", "method"], sort=True)["target_gain"].mean().unstack()
    rng = np.random.default_rng(20260925)
    sampled = rng.integers(0, len(pooled), size=(5000, len(pooled)))
    summaries = []
    for method in pooled.columns:
        values = pooled[method].to_numpy()
        boot = values[sampled].mean(axis=1)
        summaries.append({
            "method": method, "mean_target_gain": float(values.mean()),
            "ci_low": float(np.quantile(boot, 0.025)),
            "ci_high": float(np.quantile(boot, 0.975)),
        })
    pd.DataFrame(summaries).to_csv(args.output / "pooled_summary.csv", index=False)


if __name__ == "__main__":
    main()

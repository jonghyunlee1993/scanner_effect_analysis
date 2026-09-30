#!/usr/bin/env python3
"""RV-P0d (iii): compare matched GT450 Pix2Pix runs with and without source-gradient loss.

Recovered verbatim from `.Trash/2026-09-29_paper_code_cleanup/src/manuscript_completion/discussion_structure_compare.py`
(RV-P0d iii). Run-time byte-code cache: `.Trash/2026-09-26_paper_refactor/generated/src/
manuscript_completion/__pycache__/discussion_structure_compare.cpython-39.pyc` (same source size and mtime).
Paths are relative to the repository root. The caller passes an output location under
`analysis/revision/results/provenance_restoration/`; see `restore_structure_ablation.py`.
Changes: module docstring, and `macro_retrieval`/`unit` are imported from their current
home `prenorm.feature_correction` (byte code identical to the run-time
`discussion_feature_compare` versions). Output goes to `<ablation-root>/05_comparison`.
"""


from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from prenorm.feature_correction import macro_retrieval, unit


BASE = Path(
    "outputs/scanner_batch_effect_analysis_2026-09-17/06_learned_baselines/"
    "09_bidirectional_full_training"
)


def bootstrap(values: np.ndarray) -> tuple[float, float, float]:
    rng = np.random.default_rng(20260925)
    sampled = rng.integers(0, len(values), size=(5000, len(values)))
    boot = values[sampled].mean(axis=1)
    return float(values.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-root", type=Path, required=True)
    args = parser.parse_args()
    target = args.ablation_root / "05_comparison"
    target.mkdir(parents=True, exist_ok=True)
    metrics = (
        ("image", "03_image_evaluation/at2_to_gt450"),
        ("uni", "04_uni/at2_to_gt450"),
    )
    comparisons = []
    for space, relative in metrics:
        baseline = pd.read_csv(BASE / relative / "slide_metrics.csv")
        regularized = pd.read_csv(args.ablation_root / relative / "slide_metrics.csv")
        if set(baseline.slide_id) != set(regularized.slide_id):
            raise ValueError(f"{space} slide sets differ")
        shared = sorted(set(baseline.columns) & set(regularized.columns) - {
            "slide_id", "source_scanner", "method", "n_locations",
        })
        pair = baseline[["slide_id"] + shared].merge(
            regularized[["slide_id"] + shared], on="slide_id",
            suffixes=("_baseline", "_regularized"), validate="one_to_one",
        )
        for metric in shared:
            values = (
                pair[f"{metric}_regularized"] - pair[f"{metric}_baseline"]
            ).to_numpy(dtype=float)
            mean, low, high = bootstrap(values)
            comparisons.append({
                "space": space, "metric": metric,
                "baseline_mean": float(pair[f"{metric}_baseline"].mean()),
                "regularized_mean": float(pair[f"{metric}_regularized"].mean()),
                "regularized_minus_baseline": mean, "ci_low": low, "ci_high": high,
            })
    pd.DataFrame(comparisons).to_csv(target / "paired_slide_contrasts.csv", index=False)

    cohort = pd.read_csv(
        "outputs/scanner_batch_effect_analysis_2026-09-17/00_contract/cohort.csv",
        dtype={"slide_id": str},
    )
    slides = sorted(cohort.slide_id.tolist())
    tissues = cohort.set_index("slide_id").loc[slides, "tissue_type"].to_numpy()
    retrieval = {}
    target_means = None
    for method, root in (("baseline", BASE), ("regularized", args.ablation_root)):
        means = []
        targets = []
        for slide_id in slides:
            with h5py.File(root / "04_uni/at2_to_gt450/feature_shards" / f"{slide_id}.h5", "r") as store:
                means.append(np.asarray(store["generated"], dtype=np.float64).mean(axis=0))
                targets.append(np.asarray(store["real_target"], dtype=np.float64).mean(axis=0))
        means = unit(np.stack(means))
        targets = unit(np.stack(targets))
        if target_means is None:
            target_means = targets
        elif not np.allclose(target_means, targets, atol=5e-4):
            raise ValueError("paired target embeddings differ between conditions")
        retrieval[method] = macro_retrieval(means, targets, tissues)
    retrieval["target_self"] = macro_retrieval(target_means, target_means, tissues)
    pd.DataFrame([retrieval]).to_csv(target / "tissue_retrieval.csv", index=False)


if __name__ == "__main__":
    main()

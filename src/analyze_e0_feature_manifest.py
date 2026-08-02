"""Aggregate and verify the 109-slide six-scanner feature-location manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCANNERS = ("gt450", "versa", "akoya", "s60", "s360")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cohort-manifest",
        default="outputs/e0_registration_cohort_109/cohort_manifest.csv",
    )
    parser.add_argument("--shards", default="outputs/e0_feature_manifest_109/shards")
    parser.add_argument("--output", default="outputs/e0_feature_manifest_109")
    return parser.parse_args()


def main():
    args = parse_args()
    cohort = pd.read_csv(args.cohort_manifest, dtype={"slide_id": str})
    if len(cohort) != 109:
        raise ValueError(f"expected 109 cohort slides, got {len(cohort)}")
    root = Path(args.shards)
    manifests = []
    slide_rows = []
    failure_counts = Counter()
    missing = []
    for row in cohort.itertuples(index=False):
        slide_id = str(row.slide_id)
        shard = root / slide_id
        paths = {
            "manifest": shard / "feature_manifest.csv",
            "audit": shard / "candidate_audit.csv",
            "summary": shard / "summary.json",
        }
        absent = [str(path) for path in paths.values() if not path.exists()]
        if absent:
            missing.extend(absent)
            continue
        summary = json.loads(paths["summary"].read_text())
        if not summary["complete"]:
            raise ValueError(f"incomplete slide manifest: {slide_id}: {summary}")
        frame = pd.read_csv(paths["manifest"], dtype={"slide_id": str})
        if len(frame) != 100:
            raise ValueError(f"{slide_id}: expected 100 feature locations, got {len(frame)}")
        if frame["location_id"].astype(int).tolist() != list(range(100)):
            raise ValueError(f"{slide_id}: location_id contract is not 0..99")
        if frame.duplicated(["x", "y"]).any():
            raise ValueError(f"{slide_id}: duplicate feature coordinates")
        replicate_counts = frame["replicate_id"].astype(int).value_counts().sort_index()
        if replicate_counts.to_dict() != {0: 20, 1: 20, 2: 20, 3: 20, 4: 20}:
            raise ValueError(f"{slide_id}: replicate contract failed: {replicate_counts}")
        if not frame["eligible"].astype(bool).all():
            raise ValueError(f"{slide_id}: ineligible row reached final manifest")
        if not frame["max_model_fov_px"].eq(512).all():
            raise ValueError(f"{slide_id}: max FOV contract failed")
        for scanner in SCANNERS:
            if not frame[f"{scanner}_geometry_pass"].astype(bool).all():
                raise ValueError(f"{slide_id}/{scanner}: geometry failure in final manifest")
            if not frame[f"{scanner}_fov_512_pass"].astype(bool).all():
                raise ValueError(f"{slide_id}/{scanner}: 512 px FOV failure")
            if frame[f"{scanner}_pad_512"].gt(0.01 + 1e-12).any():
                raise ValueError(f"{slide_id}/{scanner}: 512 px padding exceeds 1%")
        frame["tissue_type"] = str(row.tissue_type)
        manifests.append(frame)
        audit = pd.read_csv(paths["audit"])
        failure_counts.update(audit.loc[~audit["eligible"].astype(bool), "failure_reason"])
        slide_rows.append(
            {
                "slide_id": slide_id,
                "tissue_type": str(row.tissue_type),
                **{key: summary[key] for key in (
                    "retained_original",
                    "replacements_needed",
                    "replacements_found",
                    "common_coordinate_pool",
                    "replacement_candidates_evaluated",
                )},
                "replacement_acceptance_fraction": (
                    summary["replacements_found"]
                    / max(summary["replacement_candidates_evaluated"], 1)
                ),
            }
        )
    if missing:
        raise FileNotFoundError("missing feature-manifest shards:\n" + "\n".join(missing))

    combined = pd.concat(manifests, ignore_index=True)
    slides = pd.DataFrame(slide_rows)
    if len(combined) != 10_900:
        raise ValueError(f"expected 10,900 combined rows, got {len(combined)}")
    if combined.groupby("slide_id").size().ne(100).any():
        raise ValueError("combined slide size contract failed")
    if combined.groupby(["slide_id", "location_id"]).size().ne(1).any():
        raise ValueError("sample identity is not unique")

    scanner_rows = []
    for scanner in SCANNERS:
        scanner_rows.append(
            {
                "scanner": scanner,
                "median_abs_dy": float(combined[f"{scanner}_dy"].abs().median()),
                "q95_abs_dy": float(combined[f"{scanner}_dy"].abs().quantile(0.95)),
                "median_abs_dx": float(combined[f"{scanner}_dx"].abs().median()),
                "q95_abs_dx": float(combined[f"{scanner}_dx"].abs().quantile(0.95)),
                "median_ncc": float(combined[f"{scanner}_ncc"].median()),
                "median_residual_shift": float(
                    combined[f"{scanner}_residual_shift"].median()
                ),
                "q95_pad_512": float(combined[f"{scanner}_pad_512"].quantile(0.95)),
            }
        )
    scanner_summary = pd.DataFrame(scanner_rows)
    failure_summary = pd.DataFrame(
        sorted(failure_counts.items(), key=lambda item: (-item[1], item[0])),
        columns=["failure_reason", "count"],
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output / "feature_manifest.csv", index=False)
    slides.to_csv(output / "slide_summary.csv", index=False)
    scanner_summary.to_csv(output / "scanner_summary.csv", index=False)
    failure_summary.to_csv(output / "failure_reason_summary.csv", index=False)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].hist(slides["replacements_needed"], bins=np.arange(0, 102, 5), color="#2563EB")
    axes[0].set_xlabel("Replacements per slide")
    axes[0].set_ylabel("Slides")
    axes[0].set_title("A  Original-location replacement")
    axes[1].hist(
        slides["replacement_candidates_evaluated"],
        bins=20,
        color="#0F766E",
    )
    axes[1].set_xlabel("Candidate locations evaluated")
    axes[1].set_title("B  Candidate search cost")
    axes[2].hist(
        combined["reference_tissue_fraction_512"],
        bins=np.linspace(0, 1, 21),
        color="#7C3AED",
    )
    axes[2].set_xlabel("AT2 tissue fraction in 512 px FOV")
    axes[2].set_title("C  Native-FOV content")
    figure.tight_layout()
    figure.savefig(output / "figure_e0_feature_manifest.png", dpi=200)
    figure.savefig(output / "figure_e0_feature_manifest.pdf")
    plt.close(figure)

    summary = {
        "analysis": "e0_six_scanner_feature_manifest_aggregate",
        "slides": int(len(slides)),
        "locations": int(len(combined)),
        "tissue_types": int(combined["tissue_type"].nunique()),
        "retained_original": int(slides["retained_original"].sum()),
        "replacements": int(slides["replacements_found"].sum()),
        "median_replacements_per_slide": float(slides["replacements_found"].median()),
        "maximum_replacements_per_slide": int(slides["replacements_found"].max()),
        "median_candidates_evaluated": float(
            slides["replacement_candidates_evaluated"].median()
        ),
        "maximum_candidates_evaluated": int(
            slides["replacement_candidates_evaluated"].max()
        ),
        "all_slides_complete": True,
        "all_rows_six_scanner_geometry_pass": True,
        "all_rows_512_fov_pass": True,
        "manifest_version": "e0_integer_512_v1",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(slides.describe().to_string())
    print(scanner_summary.to_string(index=False))
    print(failure_summary.head(30).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Build the 109-slide by 6-scanner raw-background task manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from run_exp05_spectral_pilot import SCANNERS
from run_exp06_raw_nps_pilot import RAW_EXTENSIONS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotation",
        default=(
            "/mnt/isilon/oldridge_lab/batch_effects/metadata/"
            "pan_normal_annotation.csv"
        ),
    )
    parser.add_argument(
        "--raw-root",
        default=(
            "/mnt/isilon/oldridge_lab/batch_effects/pan_normal/raw/raw_images"
        ),
    )
    parser.add_argument(
        "--output-root", default="outputs/exp07_raw_background_109x6"
    )
    parser.add_argument("--expected-common-slides", type=int, default=109)
    return parser.parse_args()


def main():
    args = parse_args()
    annotation_path = Path(args.annotation)
    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "cells").mkdir(exist_ok=True)
    (output_root / "incomplete").mkdir(exist_ok=True)
    (output_root / "_tmp").mkdir(exist_ok=True)

    annotation = pd.read_csv(annotation_path).rename(
        columns={"AnonSlideID": "slide_id", "Tissue Type": "tissue_type"}
    )
    annotation["slide_id"] = annotation["slide_id"].astype(str)
    if annotation["slide_id"].duplicated().any():
        raise ValueError("annotation contains duplicate slide IDs")

    complete_slides = []
    exclusions = []
    for row in annotation.itertuples(index=False):
        paths = {
            scanner: raw_root
            / scanner
            / f"{row.slide_id}{RAW_EXTENSIONS[scanner]}"
            for scanner in SCANNERS
        }
        missing = [scanner for scanner, path in paths.items() if not path.exists()]
        if missing:
            exclusions.append(
                {
                    "slide_id": row.slide_id,
                    "tissue_type": row.tissue_type,
                    "missing_scanners": missing,
                }
            )
        else:
            complete_slides.append((row.slide_id, row.tissue_type, paths))

    if len(complete_slides) != args.expected_common_slides:
        raise ValueError(
            f"expected {args.expected_common_slides} complete slides, "
            f"found {len(complete_slides)}"
        )

    rows = []
    for slide_id, tissue_type, paths in complete_slides:
        for scanner in SCANNERS:
            rows.append(
                {
                    "task_id": len(rows),
                    "slide_id": slide_id,
                    "tissue_type": tissue_type,
                    "scanner": scanner,
                    "raw_path": str(paths[scanner]),
                }
            )
    manifest = pd.DataFrame(rows)
    expected_tasks = args.expected_common_slides * len(SCANNERS)
    if len(manifest) != expected_tasks:
        raise AssertionError((len(manifest), expected_tasks))
    manifest.to_csv(output_root / "manifest.csv", index=False)
    with (output_root / "manifest_summary.json").open("w") as handle:
        json.dump(
            {
                "annotation": str(annotation_path),
                "raw_root": str(raw_root),
                "annotation_slides": len(annotation),
                "complete_slides": len(complete_slides),
                "scanners": list(SCANNERS),
                "tasks": len(manifest),
                "excluded_slides": exclusions,
            },
            handle,
            indent=2,
        )
    print(
        manifest.groupby("scanner").size().rename("tasks").to_string(),
        flush=True,
    )
    print(f"wrote {output_root / 'manifest.csv'}", flush=True)


if __name__ == "__main__":
    main()

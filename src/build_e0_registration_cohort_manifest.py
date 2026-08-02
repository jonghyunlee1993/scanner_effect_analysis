"""Build the frozen 109-slide current-route E0 audit manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selected-patches",
        default="outputs/exp05_spectral_cohort_109/selected_patches.csv",
    )
    parser.add_argument(
        "--tissue",
        default="outputs/exp05_spectral_cohort_109/slide_tissue_annotation_used.csv",
    )
    parser.add_argument(
        "--output",
        default="outputs/e0_registration_cohort_109/cohort_manifest.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    patches = pd.read_csv(args.selected_patches, dtype={"slide_id": str})
    tissue = pd.read_csv(args.tissue, dtype={"slide_id": str})
    counts = patches.groupby("slide_id", as_index=False).agg(
        selected_locations=("patch_index", "size"),
        common_coordinate_pool=("common_coordinate_pool", "first"),
    )
    manifest = counts.merge(tissue, on="slide_id", how="left", validate="one_to_one")
    manifest = manifest.sort_values("slide_id").reset_index(drop=True)
    manifest.insert(0, "cohort_index", range(len(manifest)))
    if len(manifest) != 109:
        raise ValueError(f"expected 109 slides, got {len(manifest)}")
    if not manifest["selected_locations"].eq(100).all():
        raise ValueError("every slide must have exactly 100 selected locations")
    if manifest["tissue_type"].isna().any():
        raise ValueError("missing tissue labels in cohort manifest")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)
    print(
        f"wrote {len(manifest)} slides, {manifest.tissue_type.nunique()} tissues -> {output}"
    )


if __name__ == "__main__":
    main()

"""Build the frozen E0 registration sentinel manifest from Exp05 diagnostics.

Selection is outcome-blind with respect to the new registration audit.  Slides are
stratified only by the already-observed AKOYA saturation rate from the obsolete
zero-centred +/-16 px search.  Within each stratum we greedily retain distinct tissue
types and require both existing VALIS rigid branches to be available.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SCANNERS_RIGID_ALL = ("at2", "gt450", "versa", "s60", "s360")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alignment",
        default="outputs/exp05_spectral_cohort_109/patch_alignment.csv",
    )
    parser.add_argument(
        "--tissue",
        default="outputs/exp05_spectral_cohort_109/slide_tissue_annotation_used.csv",
    )
    parser.add_argument(
        "--valis-root",
        default=(
            "/mnt/isilon/oldridge_lab/batch_effects/pan_normal/"
            "registered_ref_at2_all/registered_images"
        ),
    )
    parser.add_argument(
        "--output",
        default="outputs/e0_registration_sentinel/sentinel_manifest.csv",
    )
    parser.add_argument("--per-stratum", type=int, default=5)
    return parser.parse_args()


def _has_valis(root: Path, slide_id: str) -> bool:
    akoya = all(
        (root / "rigid_akoya" / scanner / f"{slide_id}.ome.tiff").exists()
        for scanner in ("at2", "akoya")
    )
    rigid_all = all(
        (root / "rigid_all" / scanner / f"{slide_id}.ome.tiff").exists()
        for scanner in SCANNERS_RIGID_ALL
    )
    return akoya and rigid_all


def _greedy_distinct(frame: pd.DataFrame, count: int, used_tissues: set[str]):
    selected = []
    selected_ids = set()
    for row in frame.itertuples(index=False):
        tissue = str(row.tissue_type)
        if tissue in used_tissues:
            continue
        selected.append(row)
        selected_ids.add(str(row.slide_id))
        used_tissues.add(tissue)
        if len(selected) == count:
            return selected
    for row in frame.itertuples(index=False):
        if str(row.slide_id) in selected_ids:
            continue
        selected.append(row)
        if len(selected) == count:
            return selected
    raise ValueError(f"only {len(selected)} eligible slides for requested stratum size {count}")


def main():
    args = parse_args()
    alignment = pd.read_csv(args.alignment, dtype={"slide_id": str})
    tissue = pd.read_csv(args.tissue, dtype={"slide_id": str})
    akoya = alignment[alignment["scanner"].eq("akoya")].copy()
    akoya["boundary"] = akoya["dy"].abs().ge(16) | akoya["dx"].abs().ge(16)
    akoya["low_ncc"] = akoya["ncc"].lt(0.5)
    metrics = (
        akoya.groupby("slide_id", as_index=False)
        .agg(
            old_boundary_fraction=("boundary", "mean"),
            old_low_ncc_fraction=("low_ncc", "mean"),
            old_median_ncc=("ncc", "median"),
        )
        .merge(tissue, on="slide_id", how="left", validate="one_to_one")
    )
    if metrics["tissue_type"].isna().any():
        missing = metrics.loc[metrics["tissue_type"].isna(), "slide_id"].tolist()
        raise ValueError(f"missing tissue labels: {missing}")

    valis_root = Path(args.valis_root)
    metrics["valis_complete"] = metrics["slide_id"].map(
        lambda value: _has_valis(valis_root, str(value))
    )
    metrics = metrics[metrics["valis_complete"]].copy()
    if len(metrics) < 3 * args.per_stratum:
        raise ValueError(f"only {len(metrics)} slides have complete VALIS rigid outputs")

    overall_median = float(metrics["old_boundary_fraction"].median())
    n_pool = max(args.per_stratum * 4, args.per_stratum)
    ordered = {
        "worst": metrics.sort_values(
            ["old_boundary_fraction", "old_low_ncc_fraction", "slide_id"],
            ascending=[False, False, True],
        ).head(n_pool),
        "median": metrics.assign(
            distance_to_median=(metrics["old_boundary_fraction"] - overall_median).abs()
        ).sort_values(
            ["distance_to_median", "old_low_ncc_fraction", "slide_id"],
            ascending=[True, False, True],
        ).head(n_pool),
        "best": metrics.sort_values(
            ["old_boundary_fraction", "old_low_ncc_fraction", "slide_id"],
            ascending=[True, True, True],
        ).head(n_pool),
    }

    rows = []
    used_slides: set[str] = set()
    used_tissues: set[str] = set()
    for stratum in ("worst", "median", "best"):
        candidates = ordered[stratum][
            ~ordered[stratum]["slide_id"].astype(str).isin(used_slides)
        ]
        chosen = _greedy_distinct(candidates, args.per_stratum, used_tissues)
        for within_rank, row in enumerate(chosen, start=1):
            record = row._asdict()
            record.update(
                sentinel_index=len(rows),
                stratum=stratum,
                within_stratum_rank=within_rank,
                selection_source="exp05_akoya_zero_centered_16px_diagnostic",
                selection_boundary_threshold_px=16,
            )
            record.pop("valis_complete", None)
            record.pop("distance_to_median", None)
            rows.append(record)
            used_slides.add(str(row.slide_id))

    manifest = pd.DataFrame(rows).sort_values("sentinel_index")
    if manifest["slide_id"].duplicated().any():
        raise AssertionError("sentinel slide IDs are not unique")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)
    print(manifest.to_string(index=False))
    print(f"wrote {len(manifest)} sentinel slides across {manifest.tissue_type.nunique()} tissues -> {output}")


if __name__ == "__main__":
    main()

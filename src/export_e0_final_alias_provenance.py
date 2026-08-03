"""Export final native-to-target linear maps for the post-recovery alias audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_MPP = 0.5052
SCANNERS = ("gt450", "versa")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cells", default="outputs/e0_native_geometry_final/cell_qc.csv"
    )
    parser.add_argument(
        "--output",
        default="outputs/e0_alias_audit_final_geometry/transform_provenance.csv",
    )
    return parser.parse_args()


def export_provenance(cells: pd.DataFrame):
    selected = cells[cells["scanner"].isin(SCANNERS)].copy()
    if len(selected) != 109 * len(SCANNERS):
        raise ValueError(f"expected 218 GT450/VERSA cells, got {len(selected)}")
    if not selected["cell_pass"].astype(str).str.lower().eq("true").all():
        raise ValueError("failed geometry cell reached final alias provenance")
    rows = []
    for row in selected.itertuples(index=False):
        route_label = (
            "primary"
            if pd.isna(row.geometry_route_label)
            else str(row.geometry_route_label)
        )
        matrix = np.asarray(
            [
                [row.native_to_target_m00, row.native_to_target_m01],
                [row.native_to_target_m10, row.native_to_target_m11],
            ],
            dtype=float,
        )
        singular = np.linalg.svd(matrix, compute_uv=False)
        polar_rotation = np.degrees(
            np.arctan2(
                matrix[1, 0] - matrix[0, 1],
                matrix[0, 0] + matrix[1, 1],
            )
        )
        expected_native_px_per_target = float(row.expected_native_px_per_target_px)
        if expected_native_px_per_target <= 0:
            raise ValueError(f"{row.slide_id}/{row.scanner}: invalid expected scale")
        rows.append(
            {
                "slide_id": str(row.slide_id),
                "scanner": str(row.scanner),
                "transform_route": route_label,
                "native_path": str(row.native_path),
                "native_width": int(row.native_width),
                "native_height": int(row.native_height),
                "native_mpp": TARGET_MPP / expected_native_px_per_target,
                "native_mpp_source": "frozen scanner MPP expectation in final cell gate",
                "target_mpp": TARGET_MPP,
                "target_mpp_source": "final native-AA grid contract",
                "nominal_native_px_per_target_px": expected_native_px_per_target,
                "interpolation": "Lanczos3 reduction then residual libvips bicubic",
                "explicit_antialias_prefilter": True,
                "forward_xx": float(matrix[0, 0]),
                "forward_xy": float(matrix[0, 1]),
                "forward_yx": float(matrix[1, 0]),
                "forward_yy": float(matrix[1, 1]),
                "forward_det": float(np.linalg.det(matrix)),
                "output_px_per_native_px_min": float(singular.min()),
                "output_px_per_native_px_max": float(singular.max()),
                "output_px_per_native_px_geom": float(np.sqrt(abs(np.linalg.det(matrix)))),
                "native_px_per_output_px": float(1.0 / np.sqrt(abs(np.linalg.det(matrix)))),
                "affine_anisotropy_ratio": float(singular.max() / singular.min()),
                "affine_rotation_deg": float(polar_rotation),
                "explicit_aa_pre_scale": float(row.explicit_aa_pre_scale),
                "alignment_version": str(row.alignment_version),
            }
        )
    result = pd.DataFrame(rows).sort_values(["scanner", "slide_id"]).reset_index(drop=True)
    if result.duplicated(["scanner", "slide_id"]).any():
        raise ValueError("final alias provenance contains duplicate cells")
    if not np.allclose(
        result["explicit_aa_pre_scale"],
        result["output_px_per_native_px_min"].clip(upper=1.0),
        atol=1e-10,
    ):
        raise ValueError("renderer pre-scale differs from final matrix singular values")
    return result


def main():
    args = parse_args()
    result = export_provenance(pd.read_csv(args.cells, dtype={"slide_id": str}))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    summary = {
        "analysis": "e0_final_native_alias_transform_provenance",
        "cells": len(result),
        "slides": int(result["slide_id"].nunique()),
        "scanners": list(SCANNERS),
        "routes": result.groupby(["scanner", "transform_route"]).size().reset_index(name="cells").to_dict("records"),
        "selection_uses": "final registration geometry only; no spectrum, PFM, tissue, or outcome",
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

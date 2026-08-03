"""Combine audited rigid-route offsets with a native-WSI transform.

The cross-scanner integer offsets come from ``run_e0_registration_sentinel`` on the
frozen 100 centers. This script recovers the same-scanner native -> rigid VALIS target
matrix, verifies native 512 px bounds, and writes rows compatible with the primary E0
native-geometry manifest. Historical rigid pixels remain geometry/QC only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_e0_native_geometry_cohort import (
    base_location_record,
    corners_within_native,
    find_raw_path,
    make_reconstructed_view,
    matrix_manifest_fields,
    recover_or_load_transform,
    target_square_native_corners,
    transform_gate,
)
from run_e0_primary_transform_recovery_pilot import extract_rgb, residual_integer_offset


MOVING_SCANNERS = ("gt450", "versa", "akoya", "s60", "s360")


def rigid_branch(scanner: str) -> str:
    """Return the preserved VALIS rigid branch for one moving scanner."""

    if scanner not in MOVING_SCANNERS:
        raise ValueError(f"unsupported moving scanner: {scanner}")
    return "rigid_akoya" if scanner == "akoya" else "rigid_all"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide-id", required=True)
    parser.add_argument("--scanner", choices=MOVING_SCANNERS, default="akoya")
    parser.add_argument(
        "--manifest", default="outputs/e0_feature_manifest_109/feature_manifest.csv"
    )
    parser.add_argument("--alignment-root")
    parser.add_argument(
        "--raw-root",
        default="/mnt/isilon/oldridge_lab/batch_effects/pan_normal/raw/raw_images",
    )
    parser.add_argument(
        "--rigid-root",
        default=(
            "/mnt/isilon/oldridge_lab/batch_effects/pan_normal/"
            "registered_ref_at2_all/registered_images"
        ),
    )
    parser.add_argument("--output", default="outputs/e0_rigid_native_fallback")
    parser.add_argument(
        "--transform-cache-root",
        help="Reuse frozen native-to-rigid transforms from ROOT/<scanner>/shards/<slide>.",
    )
    parser.add_argument("--thumbnail-size", type=int, default=4096)
    parser.add_argument("--minimum-sift-inliers", type=int, default=80)
    parser.add_argument("--maximum-thumbnail-q95-px", type=float, default=4.0)
    parser.add_argument("--maximum-affine-anisotropy", type=float, default=1.02)
    parser.add_argument("--maximum-scale-relative-error", type=float, default=0.05)
    parser.add_argument("--match-patch-size", type=int, default=256)
    parser.add_argument("--search-margin", type=int, default=120)
    parser.add_argument("--minimum-location-ncc", type=float, default=0.75)
    parser.add_argument("--max-model-fov", type=int, default=512)
    return parser.parse_args()


def truth(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() == "true"


def fallback_location_checks(
    *,
    global_transform_pass: bool,
    cross_scanner_geometry_pass: bool,
    search_in_bounds: bool,
    same_scanner_ncc: float,
    same_scanner_boundary: bool,
    target_bounds: bool,
    native_bounds: bool,
    minimum_location_ncc: float,
):
    """Evaluate the prespecified per-location fallback gate."""

    return {
        "global_transform_gate": bool(global_transform_pass),
        "rigid_cross_scanner_geometry": bool(cross_scanner_geometry_pass),
        "same_scanner_local_search_out_of_bounds": bool(search_in_bounds),
        "same_scanner_low_ncc": bool(
            np.isfinite(same_scanner_ncc)
            and same_scanner_ncc >= minimum_location_ncc
        ),
        "same_scanner_search_boundary": not bool(same_scanner_boundary),
        "target_fov_bounds": bool(target_bounds),
        "native_fov_bounds": bool(native_bounds),
    }


def main():
    import pyvips

    args = parse_args()
    slide_id = str(args.slide_id)
    scanner = str(args.scanner)
    fallback_version = f"valis_rigid_{scanner}_native_local_integer_v2"
    manifest = pd.read_csv(args.manifest, dtype={"slide_id": str})
    locations = (
        manifest[manifest["slide_id"].eq(slide_id)]
        .sort_values("location_id")
        .reset_index(drop=True)
    )
    if len(locations) != 100 or locations["location_id"].nunique() != 100:
        raise ValueError(f"{slide_id}: expected 100 frozen locations")
    alignment_root = (
        Path(args.alignment_root)
        if args.alignment_root
        else Path("outputs/e0_rigid_alignment") / scanner / "shards"
    )
    metrics_path = alignment_root / slide_id / "alignment_patch_metrics.csv"
    metrics = pd.read_csv(metrics_path, dtype={"slide_id": str})
    metrics = metrics[
        metrics["slide_id"].eq(slide_id)
        & metrics["scanner"].eq(scanner)
        & metrics["branch"].eq("valis")
    ].copy()
    if len(metrics) != 100 or metrics["patch_index"].nunique() != 100:
        raise ValueError(f"{slide_id}/{scanner}: expected 100 rigid alignment rows")
    metrics = metrics.rename(columns={"patch_index": "location_id"})
    joined = locations.merge(metrics, on=["slide_id", "location_id"], validate="one_to_one")

    native_path = find_raw_path(Path(args.raw_root), scanner, slide_id)
    branch = rigid_branch(scanner)
    rigid_path = (
        Path(args.rigid_root) / branch / scanner / f"{slide_id}.ome.tiff"
    ).resolve()
    native_image = pyvips.Image.new_from_file(str(native_path), access="random")
    rigid_image = pyvips.Image.new_from_file(str(rigid_path), access="random")
    output = Path(args.output) / scanner / "shards" / slide_id
    output.mkdir(parents=True, exist_ok=True)
    transform_output = (
        Path(args.transform_cache_root) / scanner / "shards" / slide_id
        if args.transform_cache_root
        else output
    )
    if args.transform_cache_root and not transform_output.is_dir():
        raise FileNotFoundError(transform_output)
    affine, sift_metrics, cache_hit = recover_or_load_transform(
        native_image,
        rigid_image,
        native_path,
        rigid_path,
        scanner,
        transform_output,
        args,
        require_cache=bool(args.transform_cache_root),
    )
    gate = transform_gate(sift_metrics, affine, scanner, args)
    reconstructed = make_reconstructed_view(
        native_image,
        affine,
        np.array([rigid_image.height, rigid_image.width]),
    )
    matrix_path = transform_output / f"native_to_target_{scanner}.npz"
    matrix_fields = matrix_manifest_fields(affine, matrix_path)

    rows = []
    for row in joined.itertuples(index=False):
        cross_dx = int(row.corrected_dx)
        cross_dy = int(row.corrected_dy)
        match_x = int(row.center_x - args.match_patch_size // 2 + cross_dx)
        match_y = int(row.center_y - args.match_patch_size // 2 + cross_dy)
        margin = int(args.search_margin)
        size = int(args.match_patch_size)
        search_in_bounds = bool(
            match_x >= 0
            and match_y >= 0
            and match_x + size <= rigid_image.width
            and match_y + size <= rigid_image.height
            and match_x - margin >= 0
            and match_y - margin >= 0
            and match_x + size + margin <= reconstructed.width
            and match_y + size + margin <= reconstructed.height
        )
        if search_in_bounds:
            rigid_patch = extract_rgb(rigid_image, match_x, match_y, size)
            reconstructed_ext = extract_rgb(
                reconstructed,
                match_x - margin,
                match_y - margin,
                size + 2 * margin,
            )
            residual_dy, residual_dx, same_scanner_ncc = residual_integer_offset(
                reconstructed_ext, rigid_patch, margin
            )
            reconstruction = reconstructed_ext[
                margin + residual_dy : margin + residual_dy + size,
                margin + residual_dx : margin + residual_dx + size,
            ]
            reconstruction_mae = float(
                np.mean(np.abs(reconstruction.astype(float) - rigid_patch.astype(float)))
            )
            same_scanner_boundary = bool(
                abs(residual_dx) >= margin or abs(residual_dy) >= margin
            )
        else:
            residual_dx = residual_dy = 0
            same_scanner_ncc = np.nan
            reconstruction_mae = np.nan
            same_scanner_boundary = True
        total_dx = cross_dx + residual_dx
        total_dy = cross_dy + residual_dy
        top_left_x = int(row.center_x - args.max_model_fov // 2 + total_dx)
        top_left_y = int(row.center_y - args.max_model_fov // 2 + total_dy)
        target_bounds = bool(
            top_left_x >= 0
            and top_left_y >= 0
            and top_left_x + args.max_model_fov <= rigid_image.width
            and top_left_y + args.max_model_fov <= rigid_image.height
        )
        corners = target_square_native_corners(
            affine, top_left_x, top_left_y, args.max_model_fov
        )
        native_bounds = corners_within_native(
            corners, native_image.width, native_image.height
        )
        alignment_pass = truth(row.geometry_pass)
        checks = fallback_location_checks(
            global_transform_pass=gate["global_transform_pass"],
            cross_scanner_geometry_pass=alignment_pass,
            search_in_bounds=search_in_bounds,
            same_scanner_ncc=same_scanner_ncc,
            same_scanner_boundary=same_scanner_boundary,
            target_bounds=target_bounds,
            native_bounds=native_bounds,
            minimum_location_ncc=args.minimum_location_ncc,
        )
        location_pass = bool(all(checks.values()))
        failures = [reason for reason, passed in checks.items() if not passed]
        record = base_location_record(row, scanner, native_path, rigid_path)
        record.update(
            {
                "manifest_version": "e0_native_geometry_v1",
                "alignment_version": fallback_version,
                "legacy_dx": cross_dx,
                "legacy_dy": cross_dy,
                "cross_scanner_dx": cross_dx,
                "cross_scanner_dy": cross_dy,
                "cross_scanner_ncc": float(row.corrected_ncc),
                "cross_scanner_search_boundary": truth(row.corrected_boundary),
                "cross_scanner_geometry_pass": alignment_pass,
                "native_recovery_dx": residual_dx if search_in_bounds else np.nan,
                "native_recovery_dy": residual_dy if search_in_bounds else np.nan,
                "total_target_dx": total_dx,
                "total_target_dy": total_dy,
                "native_recovery_ncc": same_scanner_ncc,
                "reconstruction_rgb_mae": reconstruction_mae,
                "q_reg": float(row.q_reg),
                "ncc_lp": float(row.ncc_lp),
                "phase_response": float(row.phase_response),
                "residual_shift": float(row.residual_shift),
                "pad_fraction": float(row.pad_fraction),
                "search_boundary": same_scanner_boundary,
                "target_fov_top_left_x": top_left_x,
                "target_fov_top_left_y": top_left_y,
                **{
                    f"native_corner_{axis}{index}": float(corners[index, axis_index])
                    for index in range(4)
                    for axis_index, axis in enumerate(("x", "y"))
                },
                "target_fov_bounds_pass": target_bounds,
                "native_fov_bounds_pass": native_bounds,
                "geometry_pass": location_pass,
                "failure_reason": ";".join(failures) if failures else "pass",
                **matrix_fields,
            }
        )
        rows.append(record)

    frame = pd.DataFrame(rows)
    failures = frame[~frame["geometry_pass"].astype(bool)]
    failure_reasons = sorted(
        {
            reason
            for value in failures["failure_reason"].astype(str)
            for reason in value.split(";")
            if reason and reason != "pass"
        }
    )
    cell_pass = bool(len(frame) == 100 and failures.empty)
    cell = {
        "slide_id": slide_id,
        "scanner": scanner,
        "status": "ok",
        "alignment_version": fallback_version,
        "native_path": str(native_path),
        "historical_registered_path": str(rigid_path),
        "transform_cache_hit": cache_hit,
        **sift_metrics,
        "native_width": int(native_image.width),
        "native_height": int(native_image.height),
        "registered_width": int(rigid_image.width),
        "registered_height": int(rigid_image.height),
        **{key: value for key, value in gate.items() if key != "global_transform_checks"},
        **{
            f"global_check_{key}": value
            for key, value in gate["global_transform_checks"].items()
        },
        **matrix_fields,
        "locations_expected": 100,
        "locations_measured": int(len(frame)),
        "location_ncc_min": float(frame["native_recovery_ncc"].min()),
        "location_ncc_q05": float(frame["native_recovery_ncc"].quantile(0.05)),
        "location_ncc_median": float(frame["native_recovery_ncc"].median()),
        "residual_shift_q95": float(frame["residual_shift"].quantile(0.95)),
        "search_boundary_failures": int(frame["search_boundary"].astype(bool).sum()),
        "target_fov_bounds_failures": int((~frame["target_fov_bounds_pass"].astype(bool)).sum()),
        "native_fov_bounds_failures": int((~frame["native_fov_bounds_pass"].astype(bool)).sum()),
        "location_geometry_failures": int(len(failures)),
        "cell_pass": cell_pass,
        "transform_cache_root": str(Path(args.transform_cache_root).resolve())
        if args.transform_cache_root
        else None,
        "failure_reason": ";".join(failure_reasons) if failure_reasons else "pass",
    }
    frame.to_csv(output / "native_geometry_locations.csv", index=False)
    pd.DataFrame([cell]).to_csv(output / "native_geometry_cells.csv", index=False)
    summary = {
        "analysis": "e0_rigid_native_fallback",
        "alignment_version": fallback_version,
        "slide_id": slide_id,
        "scanner": scanner,
        "locations_written": int(len(frame)),
        "locations_passing": int(frame["geometry_pass"].astype(bool).sum()),
        "cell_pass": cell_pass,
        "transform_cache_root": str(Path(args.transform_cache_root).resolve())
        if args.transform_cache_root
        else None,
        "pixel_source": "native_wsi",
        "historical_rigid_pixels_used_for": "geometry and QC only",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(pd.DataFrame([cell]).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

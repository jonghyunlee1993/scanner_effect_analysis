"""Recover native-WSI geometry for the frozen PanNormal feature manifest.

Each array task handles one physical slide. Historical registered TIFFs are used
only to recover same-scanner geometry and to measure a low-pass local residual.
The resulting manifest points back to native WSIs; it never treats registered
pixels as primary RGB data.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from run_e0_primary_transform_recovery_pilot import (
    extract_rgb,
    find_raw_path,
    recover_similarity,
    residual_integer_offset,
    scale_thumbnail_affine_to_full,
    thumbnail,
)


SCANNERS = ("at2", "gt450", "versa", "akoya", "s60", "s360")
MOVING_SCANNERS = SCANNERS[1:]
TARGET_MPP = 0.5052
NATIVE_MPP = {
    "at2": 0.5052,
    "gt450": 0.262407,
    "versa": 0.2742,
    "akoya": 0.4999,
    "s60": 0.442595,
    "s360": 0.46032,
}
ALGORITHM_VERSION = "same_scanner_similarity_local_integer_v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide-id")
    parser.add_argument("--slide-index", type=int)
    parser.add_argument(
        "--raw-root",
        default="/mnt/isilon/oldridge_lab/batch_effects/pan_normal/raw/raw_images",
    )
    parser.add_argument(
        "--registry",
        default=(
            "/mnt/isilon/oldridge_lab/batch_effects/pan_normal/registered_ref_at2_all/"
            "registered_images/qc_registered_eval/registered_pair_summary.csv"
        ),
    )
    parser.add_argument(
        "--manifest", default="outputs/e0_feature_manifest_109/feature_manifest.csv"
    )
    parser.add_argument("--output", default="outputs/e0_native_geometry_cohort")
    parser.add_argument("--thumbnail-size", type=int, default=4096)
    parser.add_argument("--match-patch-size", type=int, default=256)
    parser.add_argument("--max-model-fov", type=int, default=512)
    parser.add_argument("--search-margin", type=int, default=120)
    parser.add_argument("--minimum-sift-inliers", type=int, default=80)
    parser.add_argument("--maximum-thumbnail-q95-px", type=float, default=4.0)
    parser.add_argument("--maximum-affine-anisotropy", type=float, default=1.02)
    parser.add_argument("--maximum-scale-relative-error", type=float, default=0.05)
    parser.add_argument("--minimum-location-ncc", type=float, default=0.75)
    parser.add_argument("--maximum-at2-identity-mae", type=float, default=0.5)
    return parser.parse_args()


def resolve_slide_id(manifest: pd.DataFrame, slide_id: str | None, slide_index: int | None):
    slides = sorted(manifest["slide_id"].astype(str).unique())
    if (slide_id is None) == (slide_index is None):
        raise ValueError("provide exactly one of --slide-id or --slide-index")
    if slide_index is not None:
        if slide_index < 0 or slide_index >= len(slides):
            raise IndexError(f"slide index {slide_index} outside 0..{len(slides) - 1}")
        return slides[slide_index], slides
    if str(slide_id) not in slides:
        raise KeyError(f"slide {slide_id} absent from frozen manifest")
    return str(slide_id), slides


def file_fingerprint(path: Path):
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def make_reconstructed_view(native_image, native_to_registered, registered_shape_rc):
    from valis import warp_tools

    inverse = np.linalg.inv(native_to_registered)
    native_shape_rc = np.array([native_image.height, native_image.width])
    return warp_tools.warp_img(
        native_image,
        M=inverse,
        transformation_src_shape_rc=native_shape_rc,
        transformation_dst_shape_rc=registered_shape_rc,
        out_shape_rc=registered_shape_rc,
        interp_method="bicubic",
        bg_color=[0] * native_image.bands,
    )


def target_square_native_corners(
    native_to_target: np.ndarray, top_left_x: float, top_left_y: float, size: int
):
    inverse = np.linalg.inv(native_to_target)
    target = np.array(
        [
            [top_left_x, top_left_y, 1.0],
            [top_left_x + size, top_left_y, 1.0],
            [top_left_x + size, top_left_y + size, 1.0],
            [top_left_x, top_left_y + size, 1.0],
        ],
        dtype=float,
    )
    native = (inverse @ target.T).T
    return native[:, :2] / native[:, 2:3]


def corners_within_native(corners, width: int, height: int, interpolation_margin=4.0):
    corners = np.asarray(corners, dtype=float)
    return bool(
        np.isfinite(corners).all()
        and corners[:, 0].min() >= interpolation_margin
        and corners[:, 1].min() >= interpolation_margin
        and corners[:, 0].max() <= width - interpolation_margin
        and corners[:, 1].max() <= height - interpolation_margin
    )


def transform_gate(metrics, affine, scanner: str, args):
    singular = np.linalg.svd(affine[:2, :2], compute_uv=False)
    recovered_scale = float(1.0 / np.sqrt(np.prod(singular)))
    expected_scale = float(TARGET_MPP / NATIVE_MPP[scanner])
    relative_error = abs(recovered_scale / expected_scale - 1.0)
    anisotropy = float(singular.max() / singular.min())
    checks = {
        "minimum_sift_inliers": int(metrics["sift_inliers"])
        >= args.minimum_sift_inliers,
        "maximum_thumbnail_q95_px": float(metrics["thumbnail_reprojection_q95_px"])
        <= args.maximum_thumbnail_q95_px,
        "maximum_affine_anisotropy": anisotropy <= args.maximum_affine_anisotropy,
        "maximum_scale_relative_error": relative_error
        <= args.maximum_scale_relative_error,
    }
    return {
        "native_px_per_target_px": recovered_scale,
        "expected_native_px_per_target_px": expected_scale,
        "scale_relative_error": relative_error,
        "affine_anisotropy_ratio": anisotropy,
        "explicit_aa_pre_scale": float(min(1.0, singular.min())),
        "global_transform_pass": bool(all(checks.values())),
        "global_transform_checks": checks,
    }


def recover_or_load_transform(
    native_image,
    registered_image,
    native_path: Path,
    registered_path: Path,
    scanner: str,
    output: Path,
    args,
):
    matrix_path = output / f"native_to_target_{scanner}.npz"
    metrics_path = output / f"native_to_target_{scanner}.json"
    fingerprint = {
        "algorithm_version": ALGORITHM_VERSION,
        "thumbnail_size": args.thumbnail_size,
        "native": file_fingerprint(native_path),
        "registered": file_fingerprint(registered_path),
    }
    if matrix_path.exists() and metrics_path.exists():
        cached_metrics = json.loads(metrics_path.read_text())
        if cached_metrics.get("fingerprint") == fingerprint:
            matrix = np.asarray(np.load(matrix_path)["native_to_target"], dtype=float)
            return matrix, cached_metrics["sift_metrics"], True

    native_thumb, _ = thumbnail(native_image, args.thumbnail_size)
    registered_thumb, _ = thumbnail(registered_image, args.thumbnail_size)
    thumbnail_affine, sift_metrics, source_inliers, destination_inliers = recover_similarity(
        native_thumb, registered_thumb, args.minimum_sift_inliers
    )
    full_affine = scale_thumbnail_affine_to_full(
        thumbnail_affine,
        (native_image.width, native_image.height),
        (native_thumb.shape[1], native_thumb.shape[0]),
        (registered_image.width, registered_image.height),
        (registered_thumb.shape[1], registered_thumb.shape[0]),
    )
    np.savez_compressed(
        matrix_path,
        native_to_target=full_affine,
        thumbnail_affine=thumbnail_affine,
        source_inliers=source_inliers,
        destination_inliers=destination_inliers,
    )
    metrics_path.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "sift_metrics": sift_metrics,
                "native_to_target": full_affine.tolist(),
            },
            indent=2,
        )
        + "\n"
    )
    return full_affine, sift_metrics, False


def base_location_record(row, scanner: str, native_path: Path, registered_path: Path):
    return {
        "manifest_version": "e0_native_geometry_v1",
        "alignment_version": ALGORITHM_VERSION,
        "slide_id": str(row.slide_id),
        "tissue_type": str(row.tissue_type),
        "location_id": int(row.location_id),
        "replicate_id": int(row.replicate_id),
        "scanner": scanner,
        "canonical_center_x": int(row.center_x),
        "canonical_center_y": int(row.center_y),
        "native_path": str(native_path),
        "historical_registered_path": str(registered_path),
        "pixel_source": "native_wsi",
        "historical_pixels_primary": False,
    }


def failure_location_records(locations, scanner, native_path, registered_path, error):
    rows = []
    for row in locations.itertuples(index=False):
        record = base_location_record(row, scanner, native_path, registered_path)
        record.update(
            {
                "legacy_dx": int(getattr(row, f"{scanner}_dx", 0)),
                "legacy_dy": int(getattr(row, f"{scanner}_dy", 0)),
                "native_recovery_dx": np.nan,
                "native_recovery_dy": np.nan,
                "total_target_dx": np.nan,
                "total_target_dy": np.nan,
                "native_recovery_ncc": np.nan,
                "reconstruction_rgb_mae": np.nan,
                "search_boundary": True,
                "target_fov_bounds_pass": False,
                "native_fov_bounds_pass": False,
                "geometry_pass": False,
                "failure_reason": error,
            }
        )
        rows.append(record)
    return rows


def process_reference(locations, native_path, registered_path, native_image, registered_image, args):
    rows = []
    dimensions_match = (
        native_image.width == registered_image.width
        and native_image.height == registered_image.height
    )
    for row in locations.itertuples(index=False):
        identity_in_bounds = bool(
            row.x >= 0
            and row.y >= 0
            and row.x + args.match_patch_size <= native_image.width
            and row.y + args.match_patch_size <= native_image.height
            and row.x + args.match_patch_size <= registered_image.width
            and row.y + args.match_patch_size <= registered_image.height
        )
        if identity_in_bounds:
            native_patch = extract_rgb(
                native_image, int(row.x), int(row.y), args.match_patch_size
            )
            registered_patch = extract_rgb(
                registered_image, int(row.x), int(row.y), args.match_patch_size
            )
            identity_mae = float(
                np.mean(np.abs(native_patch.astype(float) - registered_patch.astype(float)))
            )
        else:
            identity_mae = np.nan
        identity_pass = bool(
            identity_in_bounds
            and identity_mae <= args.maximum_at2_identity_mae
        )
        top_left_x = int(row.center_x - args.max_model_fov // 2)
        top_left_y = int(row.center_y - args.max_model_fov // 2)
        target_pass = bool(
            top_left_x >= 0
            and top_left_y >= 0
            and top_left_x + args.max_model_fov <= registered_image.width
            and top_left_y + args.max_model_fov <= registered_image.height
        )
        native_pass = bool(
            top_left_x >= 4
            and top_left_y >= 4
            and top_left_x + args.max_model_fov <= native_image.width - 4
            and top_left_y + args.max_model_fov <= native_image.height - 4
        )
        record = base_location_record(row, "at2", native_path, registered_path)
        record.update(
            {
                "legacy_dx": 0,
                "legacy_dy": 0,
                "native_recovery_dx": 0,
                "native_recovery_dy": 0,
                "total_target_dx": 0,
                "total_target_dy": 0,
                "native_recovery_ncc": 1.0,
                "reconstruction_rgb_mae": identity_mae,
                "search_boundary": False,
                "target_fov_top_left_x": top_left_x,
                "target_fov_top_left_y": top_left_y,
                "native_corner_x0": float(top_left_x),
                "native_corner_y0": float(top_left_y),
                "native_corner_x1": float(top_left_x + args.max_model_fov),
                "native_corner_y1": float(top_left_y),
                "native_corner_x2": float(top_left_x + args.max_model_fov),
                "native_corner_y2": float(top_left_y + args.max_model_fov),
                "native_corner_x3": float(top_left_x),
                "native_corner_y3": float(top_left_y + args.max_model_fov),
                "target_fov_bounds_pass": target_pass,
                "native_fov_bounds_pass": native_pass,
                "geometry_pass": bool(
                    dimensions_match and identity_pass and target_pass and native_pass
                ),
                "failure_reason": "pass"
                if dimensions_match and identity_pass and target_pass and native_pass
                else "at2_identity_or_bounds_failure",
            }
        )
        rows.append(record)
    cell_pass = bool(dimensions_match and all(row["geometry_pass"] for row in rows))
    cell = {
        "slide_id": str(locations.iloc[0]["slide_id"]),
        "scanner": "at2",
        "status": "ok",
        "native_path": str(native_path),
        "historical_registered_path": str(registered_path),
        "native_width": int(native_image.width),
        "native_height": int(native_image.height),
        "registered_width": int(registered_image.width),
        "registered_height": int(registered_image.height),
        "dimensions_match": dimensions_match,
        "sift_inliers": np.nan,
        "thumbnail_reprojection_q95_px": 0.0,
        "native_px_per_target_px": 1.0,
        "expected_native_px_per_target_px": 1.0,
        "scale_relative_error": 0.0,
        "affine_anisotropy_ratio": 1.0,
        "explicit_aa_pre_scale": 1.0,
        "global_transform_pass": dimensions_match,
        "locations_expected": int(len(rows)),
        "locations_measured": int(len(rows)),
        "location_ncc_min": 1.0,
        "location_ncc_q05": 1.0,
        "location_ncc_median": 1.0,
        "reconstruction_rgb_mae_median": float(
            pd.to_numeric(
                pd.DataFrame(rows)["reconstruction_rgb_mae"], errors="coerce"
            ).median()
        ),
        "search_boundary_failures": 0,
        "target_fov_bounds_failures": int(sum(not row["target_fov_bounds_pass"] for row in rows)),
        "native_fov_bounds_failures": int(sum(not row["native_fov_bounds_pass"] for row in rows)),
        "location_geometry_failures": int(sum(not row["geometry_pass"] for row in rows)),
        "cell_pass": cell_pass,
        "failure_reason": "pass" if cell_pass else "at2_identity_or_bounds_failure",
    }
    return rows, cell, np.eye(3, dtype=float)


def process_moving(
    locations,
    scanner,
    native_path,
    registered_path,
    native_image,
    registered_image,
    output,
    args,
):
    affine, sift_metrics, cache_hit = recover_or_load_transform(
        native_image,
        registered_image,
        native_path,
        registered_path,
        scanner,
        output,
        args,
    )
    gate = transform_gate(sift_metrics, affine, scanner, args)
    reconstructed = make_reconstructed_view(
        native_image,
        affine,
        np.array([registered_image.height, registered_image.width]),
    )
    rows = []
    margin = args.search_margin
    size = args.match_patch_size
    for row in locations.itertuples(index=False):
        legacy_dx = int(getattr(row, f"{scanner}_dx"))
        legacy_dy = int(getattr(row, f"{scanner}_dy"))
        match_x = int(row.x + legacy_dx)
        match_y = int(row.y + legacy_dy)
        record = base_location_record(row, scanner, native_path, registered_path)
        record.update({"legacy_dx": legacy_dx, "legacy_dy": legacy_dy})
        search_in_bounds = bool(
            match_x >= 0
            and match_y >= 0
            and match_x + size <= registered_image.width
            and match_y + size <= registered_image.height
            and match_x - margin >= 0
            and match_y - margin >= 0
            and match_x + size + margin <= reconstructed.width
            and match_y + size + margin <= reconstructed.height
        )
        if not search_in_bounds:
            record.update(
                {
                    "native_recovery_dx": np.nan,
                    "native_recovery_dy": np.nan,
                    "total_target_dx": np.nan,
                    "total_target_dy": np.nan,
                    "native_recovery_ncc": np.nan,
                    "reconstruction_rgb_mae": np.nan,
                    "search_boundary": True,
                    "target_fov_bounds_pass": False,
                    "native_fov_bounds_pass": False,
                    "geometry_pass": False,
                    "failure_reason": "local_search_out_of_bounds",
                }
            )
            rows.append(record)
            continue

        registered_patch = extract_rgb(registered_image, match_x, match_y, size)
        reconstructed_ext = extract_rgb(
            reconstructed, match_x - margin, match_y - margin, size + 2 * margin
        )
        residual_dy, residual_dx, ncc = residual_integer_offset(
            reconstructed_ext, registered_patch, margin
        )
        reconstruction = reconstructed_ext[
            margin + residual_dy : margin + residual_dy + size,
            margin + residual_dx : margin + residual_dx + size,
        ]
        mae = float(np.mean(np.abs(reconstruction.astype(float) - registered_patch)))
        total_dx = legacy_dx + residual_dx
        total_dy = legacy_dy + residual_dy
        target_top_left_x = int(row.center_x - args.max_model_fov // 2 + total_dx)
        target_top_left_y = int(row.center_y - args.max_model_fov // 2 + total_dy)
        target_bounds = bool(
            target_top_left_x >= 0
            and target_top_left_y >= 0
            and target_top_left_x + args.max_model_fov <= registered_image.width
            and target_top_left_y + args.max_model_fov <= registered_image.height
        )
        corners = target_square_native_corners(
            affine, target_top_left_x, target_top_left_y, args.max_model_fov
        )
        native_bounds = corners_within_native(
            corners, native_image.width, native_image.height
        )
        boundary = bool(abs(residual_dx) >= margin or abs(residual_dy) >= margin)
        location_pass = bool(
            gate["global_transform_pass"]
            and ncc >= args.minimum_location_ncc
            and not boundary
            and target_bounds
            and native_bounds
        )
        failure = []
        if not gate["global_transform_pass"]:
            failure.append("global_transform_gate")
        if ncc < args.minimum_location_ncc:
            failure.append("low_ncc")
        if boundary:
            failure.append("search_boundary")
        if not target_bounds:
            failure.append("target_fov_bounds")
        if not native_bounds:
            failure.append("native_fov_bounds")
        record.update(
            {
                "native_recovery_dx": residual_dx,
                "native_recovery_dy": residual_dy,
                "total_target_dx": total_dx,
                "total_target_dy": total_dy,
                "native_recovery_ncc": ncc,
                "reconstruction_rgb_mae": mae,
                "search_boundary": boundary,
                "target_fov_top_left_x": target_top_left_x,
                "target_fov_top_left_y": target_top_left_y,
                **{
                    f"native_corner_{axis}{index}": float(corners[index, axis_index])
                    for index in range(4)
                    for axis_index, axis in enumerate(("x", "y"))
                },
                "target_fov_bounds_pass": target_bounds,
                "native_fov_bounds_pass": native_bounds,
                "geometry_pass": location_pass,
                "failure_reason": ";".join(failure) if failure else "pass",
            }
        )
        rows.append(record)

    frame = pd.DataFrame(rows)
    finite_ncc = pd.to_numeric(frame["native_recovery_ncc"], errors="coerce").dropna()
    failures = frame[~frame["geometry_pass"].astype(bool)]
    failure_reasons = sorted(
        {
            reason
            for value in failures["failure_reason"].astype(str)
            for reason in value.split(";")
            if reason and reason != "pass"
        }
    )
    cell_pass = bool(len(frame) == 100 and len(failures) == 0)
    cell = {
        "slide_id": str(locations.iloc[0]["slide_id"]),
        "scanner": scanner,
        "status": "ok",
        "native_path": str(native_path),
        "historical_registered_path": str(registered_path),
        "transform_cache_hit": cache_hit,
        **sift_metrics,
        "native_width": int(native_image.width),
        "native_height": int(native_image.height),
        "registered_width": int(registered_image.width),
        "registered_height": int(registered_image.height),
        **{key: value for key, value in gate.items() if key != "global_transform_checks"},
        **{
            f"global_check_{key}": value
            for key, value in gate["global_transform_checks"].items()
        },
        "locations_expected": 100,
        "locations_measured": int(finite_ncc.size),
        "location_ncc_min": float(finite_ncc.min()) if len(finite_ncc) else np.nan,
        "location_ncc_q05": float(finite_ncc.quantile(0.05))
        if len(finite_ncc)
        else np.nan,
        "location_ncc_median": float(finite_ncc.median()) if len(finite_ncc) else np.nan,
        "reconstruction_rgb_mae_median": float(
            pd.to_numeric(frame["reconstruction_rgb_mae"], errors="coerce").median()
        ),
        "residual_abs_shift_median": float(
            np.hypot(frame["native_recovery_dx"], frame["native_recovery_dy"]).median()
        ),
        "residual_abs_shift_q95": float(
            np.hypot(frame["native_recovery_dx"], frame["native_recovery_dy"]).quantile(0.95)
        ),
        "search_boundary_failures": int(frame["search_boundary"].astype(bool).sum()),
        "target_fov_bounds_failures": int((~frame["target_fov_bounds_pass"].astype(bool)).sum()),
        "native_fov_bounds_failures": int((~frame["native_fov_bounds_pass"].astype(bool)).sum()),
        "location_geometry_failures": int(len(failures)),
        "cell_pass": cell_pass,
        "failure_reason": ";".join(failure_reasons) if failure_reasons else "pass",
    }
    return rows, cell, affine


def main():
    import pyvips

    args = parse_args()
    manifest = pd.read_csv(args.manifest, dtype={"slide_id": str})
    slide_id, slide_order = resolve_slide_id(manifest, args.slide_id, args.slide_index)
    locations = (
        manifest[manifest["slide_id"].eq(slide_id)]
        .sort_values("location_id")
        .reset_index(drop=True)
    )
    if len(locations) != 100 or locations["location_id"].nunique() != 100:
        raise ValueError(f"{slide_id}: frozen manifest does not contain 100 unique locations")
    registry = pd.read_csv(args.registry, dtype={"slide_id": str}).set_index("slide_id")
    if slide_id not in registry.index:
        raise KeyError(f"{slide_id}: absent from registered-pair registry")

    output = Path(args.output) / "shards" / slide_id
    output.mkdir(parents=True, exist_ok=True)
    raw_root = Path(args.raw_root)
    location_rows = []
    cell_rows = []

    for scanner in SCANNERS:
        native_path = find_raw_path(raw_root, scanner, slide_id)
        registered_path = Path(str(registry.loc[slide_id, f"{scanner}_path"])).resolve()
        try:
            native_image = pyvips.Image.new_from_file(str(native_path), access="random")
            registered_image = pyvips.Image.new_from_file(
                str(registered_path), access="random"
            )
            if scanner == "at2":
                rows, cell, affine = process_reference(
                    locations,
                    native_path,
                    registered_path,
                    native_image,
                    registered_image,
                    args,
                )
                np.savez_compressed(output / "native_to_target_at2.npz", native_to_target=affine)
            else:
                rows, cell, _ = process_moving(
                    locations,
                    scanner,
                    native_path,
                    registered_path,
                    native_image,
                    registered_image,
                    output,
                    args,
                )
            matrix_path = (output / f"native_to_target_{scanner}.npz").resolve()
            singular = np.linalg.svd(affine[:2, :2], compute_uv=False)
            matrix_fields = {
                "native_to_target_m00": float(affine[0, 0]),
                "native_to_target_m01": float(affine[0, 1]),
                "native_to_target_m02": float(affine[0, 2]),
                "native_to_target_m10": float(affine[1, 0]),
                "native_to_target_m11": float(affine[1, 1]),
                "native_to_target_m12": float(affine[1, 2]),
                "explicit_aa_pre_scale": float(min(1.0, singular.min())),
                "native_to_target_path": str(matrix_path),
            }
            for row in rows:
                row.update(matrix_fields)
            cell.update(matrix_fields)
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            rows = failure_location_records(
                locations, scanner, native_path, registered_path, message
            )
            cell = {
                "slide_id": slide_id,
                "scanner": scanner,
                "status": "error",
                "native_path": str(native_path),
                "historical_registered_path": str(registered_path),
                "locations_expected": 100,
                "locations_measured": 0,
                "location_geometry_failures": 100,
                "cell_pass": False,
                "failure_reason": message,
                "traceback": traceback.format_exc(),
            }
        location_rows.extend(rows)
        cell_rows.append(cell)

    location_frame = pd.DataFrame(location_rows)
    cell_frame = pd.DataFrame(cell_rows)
    location_frame.to_csv(output / "native_geometry_locations.csv", index=False)
    cell_frame.to_csv(output / "native_geometry_cells.csv", index=False)
    summary = {
        "analysis": "e0_native_geometry_cohort_shard",
        "algorithm_version": ALGORITHM_VERSION,
        "slide_id": slide_id,
        "slide_index": slide_order.index(slide_id),
        "slide_count": len(slide_order),
        "thresholds": {
            "thumbnail_size": args.thumbnail_size,
            "minimum_sift_inliers": args.minimum_sift_inliers,
            "maximum_thumbnail_q95_px": args.maximum_thumbnail_q95_px,
            "maximum_affine_anisotropy": args.maximum_affine_anisotropy,
            "maximum_scale_relative_error": args.maximum_scale_relative_error,
            "match_patch_size": args.match_patch_size,
            "minimum_location_ncc": args.minimum_location_ncc,
            "maximum_at2_identity_mae": args.maximum_at2_identity_mae,
            "search_margin": args.search_margin,
            "max_model_fov": args.max_model_fov,
        },
        "locations_expected": 600,
        "locations_written": int(len(location_frame)),
        "cells_expected": 6,
        "cells_written": int(len(cell_frame)),
        "cells_passing": int(cell_frame["cell_pass"].astype(bool).sum()),
        "slide_pass": bool(cell_frame["cell_pass"].astype(bool).all()),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(cell_frame.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

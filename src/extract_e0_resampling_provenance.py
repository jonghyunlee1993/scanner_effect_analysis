"""Recover E0 native-grid and VALIS rigid-transform provenance.

This script is intentionally run with the historical VALIS 1.2.0 environment so
that its registrar pickles can be loaded.  It does not register or rewrite any
slide.  The resulting table describes the exact rigid transforms retained under
``rigid_all`` and separately audits metadata on the primary historical outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile


SCANNERS = ("at2", "gt450", "versa", "akoya", "s60", "s360")
PHYSICAL_UNITS = {"um", "µm", "micron", "microns", "micrometer", "micrometers"}
MPP_PATTERN = re.compile(r"(?:^|\|)\s*MPP\s*=\s*([0-9]+(?:\.[0-9]+)?)", re.I)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug-root",
        default=(
            "/mnt/isilon/oldridge_lab/batch_effects/pan_normal/"
            "registered_ref_at2_all/registered_images/rigid_all/debug_valis"
        ),
    )
    parser.add_argument(
        "--raw-root",
        default="/mnt/isilon/oldridge_lab/batch_effects/pan_normal/raw/raw_images",
    )
    parser.add_argument(
        "--registry",
        default=(
            "/mnt/isilon/oldridge_lab/batch_effects/pan_normal/"
            "registered_ref_at2_all/registered_images/qc_registered_eval/"
            "registered_pair_summary.csv"
        ),
    )
    parser.add_argument(
        "--pipeline-script",
        default="/mnt/isilon/oldridge_lab/batch_effects/scripts/02e_valis_registration.py",
    )
    parser.add_argument(
        "--valis-root",
        default=(
            "/mnt/isilon/oldridge_lab/irinaz/anaconda3/envs/valis310/"
            "lib/python3.10/site-packages/valis"
        ),
    )
    parser.add_argument("--output", default="outputs/e0_resampling_provenance")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scanner_from_name(name: str) -> str | None:
    lower = name.lower()
    for scanner in SCANNERS:
        if lower.endswith(f"_{scanner}"):
            return scanner
    return None


def find_raw_path(root: Path, scanner: str, slide_id: str) -> Path:
    matches = sorted(
        path
        for path in (root / scanner).glob(f"{slide_id}.*")
        if path.suffix.lower() in {".svs", ".ndpi", ".qptiff", ".tif", ".tiff"}
    )
    if len(matches) != 1:
        raise ValueError(f"expected one native slide for {scanner}/{slide_id}, got {matches}")
    return matches[0].resolve()


def native_mpp(path: Path, fallback_value=None, fallback_units=None):
    import pyvips

    image = pyvips.Image.new_from_file(str(path))
    for field in ("openslide.mpp-x", "aperio.MPP"):
        if image.get_typeof(field):
            value = float(image.get(field))
            if 0.05 < value < 2.0:
                return value, field
    for field in ("openslide.comment", "image-description"):
        if not image.get_typeof(field):
            continue
        match = MPP_PATTERN.search(str(image.get(field)))
        if match and 0.05 < float(match.group(1)) < 2.0:
            return float(match.group(1)), f"{field}:MPP"
    # libvips xres is always pixels/mm.  Values near 1 are the generic-TIFF
    # fallback and are deliberately rejected.
    if image.xres > 500:
        value = 1000.0 / float(image.xres)
        if 0.05 < value < 2.0:
            return value, "libvips.xres_px_per_mm"
    units = str(fallback_units or "").strip().lower()
    if fallback_value is not None and units in PHYSICAL_UNITS:
        value = float(fallback_value)
        if 0.05 < value < 2.0:
            return value, "valis.slide.resolution"
    raise ValueError(f"could not recover physical MPP from {path}")


def ome_physical_size_x(path: Path) -> float:
    try:
        with tifffile.TiffFile(path) as tif:
            xml = tif.ome_metadata or tif.pages[0].description or ""
        match = re.search(r'PhysicalSizeX="([0-9.eE+-]+)"', xml)
        if match:
            return float(match.group(1))
    except (OSError, ValueError, tifffile.TiffFileError):
        pass
    return math.nan


def primary_output_metadata(path: Path):
    import pyvips

    if not path.exists():
        return {
            "primary_output_exists": False,
            "primary_output_width": np.nan,
            "primary_output_height": np.nan,
            "primary_output_openslide_mpp": np.nan,
            "primary_output_ome_physical_size_x": np.nan,
            "primary_output_vendor": "",
        }
    image = pyvips.Image.new_from_file(str(path))
    if image.get_typeof("openslide.mpp-x"):
        mpp = float(image.get("openslide.mpp-x"))
    elif image.xres > 0:
        mpp = 1000.0 / float(image.xres)
    else:
        mpp = math.nan
    vendor = str(image.get("openslide.vendor")) if image.get_typeof("openslide.vendor") else ""
    return {
        "primary_output_exists": True,
        "primary_output_width": int(image.width),
        "primary_output_height": int(image.height),
        "primary_output_openslide_mpp": float(mpp),
        "primary_output_ome_physical_size_x": ome_physical_size_x(path),
        "primary_output_vendor": vendor,
    }


def effective_forward_matrix(slide) -> np.ndarray:
    """Return the full-resolution source-pixel -> aligned-pixel linear map.

    VALIS stores ``M`` as the inverse transformation on its processed images.
    ``warp_tools.warp_img`` therefore applies inv(M), bracketed by source and
    destination shape scaling.  Reference cropping changes only translation.
    """

    source_wh = np.asarray(slide.slide_dimensions_wh[0], dtype=float)
    source_processed_wh = np.asarray(slide.processed_img_shape_rc, dtype=float)[::-1]
    destination_wh = np.asarray(slide.aligned_slide_shape_rc, dtype=float)[::-1]
    destination_processed_wh = np.asarray(slide.reg_img_shape_rc, dtype=float)[::-1]
    source_scale = source_wh / source_processed_wh
    destination_scale = destination_wh / destination_processed_wh
    return (
        np.diag(destination_scale)
        @ np.linalg.inv(np.asarray(slide.M, dtype=float)[:2, :2])
        @ np.diag(1.0 / source_scale)
    )


def matrix_metrics(matrix: np.ndarray, native_um_per_px: float, target_um_per_px: float):
    u, singular, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    singular = np.sort(singular)
    geometric_scale = float(np.sqrt(np.prod(singular)))
    effective_mpp = native_um_per_px / geometric_scale
    return {
        "forward_xx": float(matrix[0, 0]),
        "forward_xy": float(matrix[0, 1]),
        "forward_yx": float(matrix[1, 0]),
        "forward_yy": float(matrix[1, 1]),
        "forward_det": float(np.linalg.det(matrix)),
        "output_px_per_native_px_min": float(singular[0]),
        "output_px_per_native_px_max": float(singular[1]),
        "output_px_per_native_px_geom": geometric_scale,
        "native_px_per_output_px": float(1.0 / geometric_scale),
        "affine_anisotropy_ratio": float(singular[1] / singular[0]),
        "affine_rotation_deg": float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))),
        "effective_target_mpp": float(effective_mpp),
        "target_mpp_error_percent": float(100.0 * (effective_mpp / target_um_per_px - 1.0)),
    }


def main():
    import pyvips

    args = parse_args()
    debug_root = Path(args.debug_root)
    raw_root = Path(args.raw_root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    registry = pd.read_csv(args.registry, dtype={"slide_id": str}).set_index("slide_id")
    pickles = sorted(debug_root.glob("*/valis_results/staging_input/data/staging_input_registrar.pickle"))
    if not pickles:
        raise FileNotFoundError(f"no registrar pickles under {debug_root}")

    rows = []
    errors = []
    for pickle_path in pickles:
        slide_id = pickle_path.relative_to(debug_root).parts[0]
        try:
            with pickle_path.open("rb") as handle:
                registrar = pickle.load(handle)
            native_at2 = find_raw_path(raw_root, "at2", slide_id)
            target_mpp, target_mpp_source = native_mpp(native_at2)
            for name, slide in registrar.slide_dict.items():
                scanner = scanner_from_name(str(name))
                if scanner is None:
                    continue
                path = find_raw_path(raw_root, scanner, slide_id)
                mpp, mpp_source = native_mpp(
                    path,
                    fallback_value=getattr(slide, "resolution", None),
                    fallback_units=getattr(slide, "units", None),
                )
                matrix = effective_forward_matrix(slide)
                record = {
                    "slide_id": slide_id,
                    "scanner": scanner,
                    "transform_route": "existing_valis_rigid",
                    "registrar_pickle": str(pickle_path),
                    "native_path": str(path),
                    "native_width": int(slide.slide_dimensions_wh[0][0]),
                    "native_height": int(slide.slide_dimensions_wh[0][1]),
                    "native_mpp": float(mpp),
                    "native_mpp_source": mpp_source,
                    "target_mpp": float(target_mpp),
                    "target_mpp_source": target_mpp_source,
                    "nominal_native_px_per_target_px": float(target_mpp / mpp),
                    "interpolation": "libvips bicubic (VALIS default)",
                    "explicit_antialias_prefilter": False,
                    "compression": "LZW",
                    **matrix_metrics(matrix, mpp, target_mpp),
                }
                if slide_id in registry.index:
                    primary_path = Path(str(registry.loc[slide_id, f"{scanner}_path"]))
                    record["primary_output_path"] = str(primary_path)
                    record.update(primary_output_metadata(primary_path))
                    output_mpp = record["primary_output_openslide_mpp"]
                    record["primary_output_mpp_matches_target"] = bool(
                        np.isfinite(output_mpp) and abs(output_mpp - target_mpp) <= 0.01 * target_mpp
                    )
                rows.append(record)
        except Exception as exc:  # retain a complete audit trail across historical artifacts
            errors.append({"slide_id": slide_id, "pickle": str(pickle_path), "error": repr(exc)})

    frame = pd.DataFrame(rows).sort_values(["scanner", "slide_id"]).reset_index(drop=True)
    if frame.empty:
        raise RuntimeError(f"all registrar loads failed: {errors[:3]}")
    frame.to_csv(output / "transform_provenance.csv", index=False)
    pd.DataFrame(errors, columns=["slide_id", "pickle", "error"]).to_csv(
        output / "provenance_errors.csv", index=False
    )

    scanner_rows = []
    for scanner, group in frame.groupby("scanner", sort=False):
        scanner_rows.append(
            {
                "scanner": scanner,
                "slides_with_exact_transform": int(group["slide_id"].nunique()),
                "native_mpp_median": float(group["native_mpp"].median()),
                "native_mpp_min": float(group["native_mpp"].min()),
                "native_mpp_max": float(group["native_mpp"].max()),
                "native_px_per_target_px_median": float(
                    group["nominal_native_px_per_target_px"].median()
                ),
                "effective_native_px_per_output_px_median": float(
                    group["native_px_per_output_px"].median()
                ),
                "effective_native_px_per_output_px_q05": float(
                    group["native_px_per_output_px"].quantile(0.05)
                ),
                "effective_native_px_per_output_px_q95": float(
                    group["native_px_per_output_px"].quantile(0.95)
                ),
                "anisotropy_ratio_q95": float(group["affine_anisotropy_ratio"].quantile(0.95)),
                "abs_target_mpp_error_percent_q95": float(
                    group["target_mpp_error_percent"].abs().quantile(0.95)
                ),
                "primary_output_metadata_match_fraction": float(
                    group["primary_output_mpp_matches_target"].mean()
                ),
            }
        )
    scanner_summary = pd.DataFrame(scanner_rows)
    scanner_summary.to_csv(output / "scanner_summary.csv", index=False)

    pipeline_files = [
        Path(args.pipeline_script),
        Path(args.valis_root) / "registration.py",
        Path(args.valis_root) / "warp_tools.py",
        Path(args.valis_root) / "slide_io.py",
    ]
    software = {
        "valis_version": "1.2.0",
        "pyvips_version": pyvips.__version__,
        "libvips_version": ".".join(str(pyvips.version(i)) for i in range(3)),
        "interpolation": "bicubic",
        "antialias_prefilter": None,
        "compression": "lzw",
        "frequency_axis_contract": "native AT2 openslide.mpp-x per physical slide",
        "source_sha256": {str(path): sha256(path) for path in pipeline_files},
    }
    (output / "software_contract.json").write_text(json.dumps(software, indent=2) + "\n")
    summary = {
        "analysis": "e0_resampling_provenance",
        "registrar_pickles_found": len(pickles),
        "registrar_pickles_loaded": int(frame["slide_id"].nunique()),
        "registrar_pickles_failed": len(errors),
        "transform_rows": len(frame),
        "scanners": sorted(frame["scanner"].unique().tolist()),
        "exact_transform_route": "existing_valis_rigid",
        "primary_route_exact_transform_artifacts_available": False,
        "primary_route_metadata_audited": True,
        "metadata_rewrite_bug": (
            "historical export rewrote scanner-native MPP onto AT2-grid pixels; "
            "output TIFF MPP must not define the frequency axis"
        ),
        "software_contract": software,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(scanner_summary.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Render one slide of frozen E0 native, explicitly anti-aliased RGB patches.

The native-geometry manifest supplies a global native-to-target affine plus a
per-location integer target-grid residual.  Historical registered RGB is never read.
For genuine reduction, the native WSI is first reduced with libvips Lanczos3 at the
smallest affine singular value.  The remaining affine is then evaluated with bicubic
interpolation directly on the requested target-grid patch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


SCANNERS = ("at2", "gt450", "versa", "akoya", "s60", "s360")
RENDER_VERSION = "e0_native_explicit_aa_lanczos3_bicubic_v1"
MODEL_FOV = {
    "resnet50": 256,
    "uni_v1": 256,
    "conch_v1": 512,
    "virchow2": 224,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide-id")
    parser.add_argument("--slide-index", type=int)
    parser.add_argument(
        "--geometry",
        default="outputs/e0_native_geometry_final/native_geometry_manifest.csv",
    )
    parser.add_argument("--output", default="outputs/e0_native_aa_grid/shards")
    parser.add_argument("--fov", type=int, default=512)
    parser.add_argument("--compression", choices=("lzf", "gzip", "none"), default="lzf")
    parser.add_argument("--gzip-level", type=int, default=1)
    return parser.parse_args()


def truth(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() == "true"


def resolve_slide_id(frame: pd.DataFrame, slide_id: str | None, slide_index: int | None):
    slides = sorted(frame["slide_id"].astype(str).unique())
    if (slide_id is None) == (slide_index is None):
        raise ValueError("provide exactly one of --slide-id or --slide-index")
    if slide_index is not None:
        if slide_index < 0 or slide_index >= len(slides):
            raise IndexError(f"slide index {slide_index} outside 0..{len(slides) - 1}")
        return slides[slide_index], slides
    if str(slide_id) not in slides:
        raise KeyError(f"slide {slide_id} absent from frozen native geometry")
    return str(slide_id), slides


def affine_from_row(row) -> np.ndarray:
    matrix = np.array(
        [
            [row.native_to_target_m00, row.native_to_target_m01, row.native_to_target_m02],
            [row.native_to_target_m10, row.native_to_target_m11, row.native_to_target_m12],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix[:2, :2])) < 1e-12:
        raise ValueError("native-to-target matrix is non-finite or singular")
    return matrix


def explicit_aa_pre_scale(native_to_target: np.ndarray) -> float:
    singular = np.linalg.svd(np.asarray(native_to_target)[:2, :2], compute_uv=False)
    return float(min(1.0, singular.min()))


def residual_after_prescale(native_to_target: np.ndarray, pre_scale: float) -> np.ndarray:
    """Map pre-scaled native coordinates to the unchanged global target grid."""

    if not np.isfinite(pre_scale) or pre_scale <= 0 or pre_scale > 1:
        raise ValueError(f"invalid explicit-AA pre-scale: {pre_scale}")
    residual = np.asarray(native_to_target, dtype=np.float64).copy()
    residual[:2, :2] /= pre_scale
    return residual


def exact_integer(value, label: str) -> int:
    numeric = float(value)
    rounded = int(round(numeric))
    if not np.isfinite(numeric) or not np.isclose(numeric, rounded, atol=1e-6):
        raise ValueError(f"{label} must be an integer target-grid displacement: {value}")
    return rounded


def target_top_left(row, fov: int) -> tuple[int, int]:
    if fov <= 0 or fov % 2:
        raise ValueError("target FOV must be a positive even integer")
    center_x = exact_integer(row.canonical_center_x, "canonical_center_x")
    center_y = exact_integer(row.canonical_center_y, "canonical_center_y")
    dx = exact_integer(row.total_target_dx, "total_target_dx")
    dy = exact_integer(row.total_target_dy, "total_target_dy")
    return center_x + dx - fov // 2, center_y + dy - fov // 2


def ensure_rgb(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim == 2:
        value = np.repeat(value[..., None], 3, axis=2)
    elif value.ndim == 3 and value.shape[2] == 1:
        value = np.repeat(value, 3, axis=2)
    elif value.ndim == 3 and value.shape[2] >= 3:
        value = value[..., :3]
    else:
        raise ValueError(f"unsupported rendered patch shape: {value.shape}")
    return np.asarray(np.clip(value, 0, 255), dtype=np.uint8)


def prepare_native_renderer(native_image, native_to_target):
    """Build one lazy libvips graph reused by all locations from a scanner WSI."""

    pre_scale = explicit_aa_pre_scale(native_to_target)
    source = (
        native_image.resize(pre_scale, kernel="lanczos3")
        if pre_scale < 1.0
        else native_image
    )
    residual = residual_after_prescale(native_to_target, pre_scale)
    return source, residual, pre_scale


def render_native_target_patch(
    native_image, native_to_target, target_x, target_y, fov, prepared=None
):
    """Render one global target-grid patch lazily from native WSI pixels."""

    import pyvips

    target_x = exact_integer(target_x, "target_x")
    target_y = exact_integer(target_y, "target_y")
    fov = int(fov)
    source, residual, pre_scale = (
        prepare_native_renderer(native_image, native_to_target)
        if prepared is None
        else prepared
    )
    interpolator = pyvips.Interpolate.new("bicubic")
    patch = source.affine(
        residual[:2, :2].reshape(-1).tolist(),
        interpolate=interpolator,
        oarea=[target_x, target_y, fov, fov],
        odx=float(residual[0, 2]),
        ody=float(residual[1, 2]),
        extend="black",
        premultiplied=bool(source.hasalpha()),
    )
    value = ensure_rgb(np.asarray(patch))
    if value.shape != (fov, fov, 3):
        raise ValueError(f"rendered {value.shape}, expected {(fov, fov, 3)}")
    return value, pre_scale


def validate_slide_geometry(frame: pd.DataFrame, slide_id: str) -> pd.DataFrame:
    slide = frame[frame["slide_id"].astype(str).eq(slide_id)].copy()
    keys = ["slide_id", "location_id", "scanner"]
    if len(slide) != 600 or slide.duplicated(keys).any():
        raise ValueError(f"{slide_id}: expected 600 unique scanner-location rows")
    if set(slide["scanner"].astype(str)) != set(SCANNERS):
        raise ValueError(f"{slide_id}: scanner set is not the frozen six-scanner panel")
    if not slide["geometry_pass"].map(truth).all():
        raise ValueError(f"{slide_id}: geometry failure reached native-AA rendering")
    counts = slide.groupby("scanner")["location_id"].nunique()
    if not counts.reindex(SCANNERS).eq(100).all():
        raise ValueError(f"{slide_id}: each scanner must contain 100 locations")
    return slide


def sha256(path: Path, block_size=1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def string_array(values):
    return np.asarray([str(value) for value in values], dtype=h5py.string_dtype("utf-8"))


def h5_compression(args):
    if args.compression == "none":
        return {}
    if args.compression == "gzip":
        if not 0 <= args.gzip_level <= 9:
            raise ValueError("--gzip-level must be in 0..9")
        return {"compression": "gzip", "compression_opts": args.gzip_level, "shuffle": True}
    return {"compression": "lzf", "shuffle": True}


def render_slide(frame: pd.DataFrame, slide_id: str, output_path: Path, args):
    import pyvips

    slide = validate_slide_geometry(frame, slide_id)
    locations = sorted(slide["location_id"].astype(int).unique())
    if locations != list(range(100)):
        raise ValueError(f"{slide_id}: location IDs must be exactly 0..99")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    compression = h5_compression(args)

    with h5py.File(temporary, "w") as store:
        rgb = store.create_dataset(
            "rgb",
            shape=(len(SCANNERS), 100, args.fov, args.fov, 3),
            dtype=np.uint8,
            chunks=(1, 1, args.fov, args.fov, 3),
            **compression,
        )
        store.create_dataset("scanner", data=string_array(SCANNERS))
        store.create_dataset("location_id", data=np.asarray(locations, dtype=np.int16))
        first = slide[slide["scanner"].eq("at2")].sort_values("location_id")
        store.create_dataset("replicate_id", data=first["replicate_id"].to_numpy(np.int8))
        store.create_dataset(
            "canonical_center_x", data=first["canonical_center_x"].to_numpy(np.int32)
        )
        store.create_dataset(
            "canonical_center_y", data=first["canonical_center_y"].to_numpy(np.int32)
        )
        target_x = store.create_dataset("target_top_left_x", shape=(6, 100), dtype=np.int32)
        target_y = store.create_dataset("target_top_left_y", shape=(6, 100), dtype=np.int32)
        total_dx = store.create_dataset("total_target_dx", shape=(6, 100), dtype=np.int16)
        total_dy = store.create_dataset("total_target_dy", shape=(6, 100), dtype=np.int16)
        matrices = store.create_dataset("native_to_target", shape=(6, 3, 3), dtype=np.float64)
        pre_scales = store.create_dataset("explicit_aa_pre_scale", shape=(6,), dtype=np.float64)
        native_paths = []
        alignment_versions = []

        for scanner_index, scanner in enumerate(SCANNERS):
            rows = slide[slide["scanner"].eq(scanner)].sort_values("location_id")
            paths = rows["native_path"].astype(str).unique()
            if len(paths) != 1:
                raise ValueError(f"{slide_id}/{scanner}: expected one native WSI path")
            matrices_in_rows = np.stack([affine_from_row(row) for row in rows.itertuples()])
            if not np.allclose(matrices_in_rows, matrices_in_rows[0], atol=1e-10):
                raise ValueError(f"{slide_id}/{scanner}: affine changes across locations")
            matrix = matrices_in_rows[0]
            native = pyvips.Image.new_from_file(paths[0], access="random")
            prepared = prepare_native_renderer(native, matrix)
            native_paths.append(paths[0])
            alignment_versions.append(";".join(sorted(rows["alignment_version"].astype(str).unique())))
            matrices[scanner_index] = matrix
            observed_pre_scale = None
            for location_index, row in enumerate(rows.itertuples(index=False)):
                x, y = target_top_left(row, args.fov)
                patch, pre_scale = render_native_target_patch(
                    native, matrix, x, y, args.fov, prepared=prepared
                )
                if observed_pre_scale is None:
                    observed_pre_scale = pre_scale
                elif not np.isclose(observed_pre_scale, pre_scale):
                    raise AssertionError("pre-scale changed within one scanner WSI")
                rgb[scanner_index, location_index] = patch
                target_x[scanner_index, location_index] = x
                target_y[scanner_index, location_index] = y
                total_dx[scanner_index, location_index] = exact_integer(
                    row.total_target_dx, "total_target_dx"
                )
                total_dy[scanner_index, location_index] = exact_integer(
                    row.total_target_dy, "total_target_dy"
                )
            pre_scales[scanner_index] = observed_pre_scale

        store.create_dataset("native_path", data=string_array(native_paths))
        store.create_dataset("alignment_version", data=string_array(alignment_versions))
        store.attrs["analysis"] = "e0_native_aa_grid_shard"
        store.attrs["render_version"] = RENDER_VERSION
        store.attrs["slide_id"] = slide_id
        store.attrs["target_mpp"] = 0.5052
        store.attrs["fov"] = int(args.fov)
        store.attrs["pixel_source"] = "native_wsi_only"
        store.attrs["historical_registered_rgb_used"] = False
        store.attrs["interpolation_contract"] = (
            "libvips Lanczos3 reduction at min affine singular value then bicubic residual affine"
        )
        store.attrs["model_fov_json"] = json.dumps(MODEL_FOV, sort_keys=True)
        store.attrs["geometry_manifest_path"] = str(Path(args.geometry).resolve())
        store.attrs["geometry_manifest_sha256"] = sha256(Path(args.geometry))
        store.flush()

    os.replace(temporary, output_path)


def main():
    args = parse_args()
    if args.fov != max(MODEL_FOV.values()):
        raise ValueError(
            f"the frozen shared grid FOV is {max(MODEL_FOV.values())}, got {args.fov}"
        )
    geometry_path = Path(args.geometry)
    geometry = pd.read_csv(geometry_path, dtype={"slide_id": str})
    slide_id, slide_order = resolve_slide_id(
        geometry, args.slide_id, args.slide_index
    )
    output_path = Path(args.output) / f"{slide_id}.h5"
    render_slide(geometry, slide_id, output_path, args)
    summary = {
        "analysis": "e0_native_aa_grid_shard",
        "render_version": RENDER_VERSION,
        "slide_id": slide_id,
        "slide_index": slide_order.index(slide_id),
        "slide_count": len(slide_order),
        "scanners": len(SCANNERS),
        "locations_per_scanner": 100,
        "shared_grid_fov": args.fov,
        "model_fov": MODEL_FOV,
        "pixel_source": "native_wsi_only",
        "output": str(output_path.resolve()),
        "output_sha256": sha256(output_path),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Render one slide's frozen glass coordinates through the final native-AA chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from e0d_same_chain_helpers import (
    batch_total_od_fft,
    frequency_geometry,
    glass_rejection_reasons,
    patch_statistics,
    radial_mean,
)
from render_e0_native_aa_shard import (
    RENDER_VERSION,
    SCANNERS,
    affine_from_row,
    prepare_native_renderer,
    render_native_target_patch,
)


ANALYSIS_VERSION = "e0d_same_chain_background_v1"
PATCH_SIZE = 256
TARGET_MPP = 0.5052
TABLES = {
    "spectra": "background_spectra.csv",
    "replicates": "background_replicate_spectra.csv",
    "coordinates": "rendered_coordinates.csv",
    "qc": "post_render_glass_qc.csv",
}


def parse_args():
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--slide-id")
    selection.add_argument("--slide-index", type=int)
    parser.add_argument(
        "--geometry",
        default="outputs/e0_native_geometry_final/native_geometry_manifest.csv",
    )
    parser.add_argument(
        "--background",
        default="outputs/exp07_raw_background_109x6",
    )
    parser.add_argument(
        "--output",
        default="outputs/e0d_same_chain_background/shards",
    )
    parser.add_argument("--bins", type=int, default=72)
    return parser.parse_args()


def sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_slide(frame: pd.DataFrame, slide_id: str | None, slide_index: int | None):
    slides = sorted(frame["slide_id"].astype(str).unique())
    if len(slides) != 109:
        raise ValueError(f"expected 109 slides in background metadata, got {len(slides)}")
    if slide_index is not None:
        if slide_index < 0 or slide_index >= len(slides):
            raise IndexError(f"slide index {slide_index} outside 0..{len(slides) - 1}")
        return slides[slide_index], slides
    if str(slide_id) not in slides:
        raise KeyError(f"slide absent from frozen background metadata: {slide_id}")
    return str(slide_id), slides


def mapped_target_top_left(
    native_to_target: np.ndarray,
    x_native: float,
    y_native: float,
    native_patch_size: int,
    target_size: int = PATCH_SIZE,
) -> tuple[int, int, float, float]:
    """Map the centre of a frozen native glass patch to an integer target crop."""

    center = np.array(
        [
            float(x_native) + float(native_patch_size) / 2.0,
            float(y_native) + float(native_patch_size) / 2.0,
            1.0,
        ],
        dtype=np.float64,
    )
    mapped = np.asarray(native_to_target, dtype=np.float64) @ center
    target_center_x = int(np.rint(mapped[0]))
    target_center_y = int(np.rint(mapped[1]))
    return (
        target_center_x - target_size // 2,
        target_center_y - target_size // 2,
        float(mapped[0]),
        float(mapped[1]),
    )


def spectrum_rows(images: np.ndarray, scanner: str, slide_id: str, bins: int):
    if images.shape != (100, PATCH_SIZE, PATCH_SIZE, 3):
        raise ValueError(f"{slide_id}/{scanner}: unexpected rendered RGB shape {images.shape}")
    index, valid, counts, frequency, _ = frequency_geometry(
        PATCH_SIZE, TARGET_MPP, bins
    )
    window_1d = np.hanning(PATCH_SIZE).astype(np.float32)
    window = window_1d[:, None] * window_1d[None, :]
    fourier = batch_total_od_fft(images, window)
    per_patch_power = np.abs(fourier) ** 2
    mean_radial = radial_mean(per_patch_power.mean(axis=0), index, valid, counts)
    spectra = [
        {
            "slide_id": slide_id,
            "scanner": scanner,
            "frequency_cyc_per_um": float(freq),
            "background_radial_power": float(power),
            "patches": len(images),
        }
        for freq, power in zip(frequency, mean_radial)
    ]
    replicate_rows = []
    replicate_ids = np.arange(len(images)) % 5
    for replicate in range(5):
        selected = replicate_ids == replicate
        if int(selected.sum()) != 20:
            raise AssertionError("same-chain background replicate is not 20 patches")
        radial = radial_mean(
            per_patch_power[selected].mean(axis=0), index, valid, counts
        )
        for freq, power in zip(frequency, radial):
            replicate_rows.append(
                {
                    "slide_id": slide_id,
                    "scanner": scanner,
                    "replicate": replicate,
                    "patches_in_replicate": 20,
                    "frequency_cyc_per_um": float(freq),
                    "background_radial_power": float(power),
                }
            )
    return spectra, replicate_rows


def cell_path(background_root: Path, row) -> Path:
    path = background_root / "cells" / (
        f"{int(row.task_id):04d}_{row.slide_id}_{row.scanner}"
    )
    if not (path / "_SUCCESS").exists():
        raise FileNotFoundError(f"frozen complete Exp07 cell absent: {path}")
    return path


def render_slide(
    slide_id: str,
    geometry: pd.DataFrame,
    metadata: pd.DataFrame,
    background_root: Path,
    bins: int,
):
    import pyvips

    spectra_rows = []
    replicate_rows = []
    coordinate_rows = []
    qc_rows = []
    slide_geometry = geometry[geometry["slide_id"].astype(str).eq(slide_id)]
    slide_metadata = metadata[metadata["slide_id"].astype(str).eq(slide_id)]
    if len(slide_geometry) != 600 or len(slide_metadata) != 6:
        raise ValueError(f"{slide_id}: expected 600 geometry rows and 6 background cells")

    for scanner in SCANNERS:
        rows = slide_geometry[slide_geometry["scanner"].eq(scanner)].sort_values(
            "location_id"
        )
        cell_meta = slide_metadata[slide_metadata["scanner"].eq(scanner)]
        if len(rows) != 100 or len(cell_meta) != 1:
            raise ValueError(f"{slide_id}/{scanner}: incomplete frozen inputs")
        matrices = np.stack([affine_from_row(row) for row in rows.itertuples()])
        if not np.allclose(matrices, matrices[0], atol=1e-10):
            raise ValueError(f"{slide_id}/{scanner}: affine varies across locations")
        matrix = matrices[0]
        meta = next(cell_meta.itertuples(index=False))
        accepted_path = cell_path(background_root, meta) / "accepted_glass_patches.csv"
        accepted = pd.read_csv(accepted_path).sort_values("patch_index")
        if len(accepted) != 100 or accepted["patch_index"].tolist() != list(range(100)):
            raise ValueError(f"{slide_id}/{scanner}: frozen glass identity is incomplete")
        native_paths = rows["native_path"].astype(str).unique()
        if len(native_paths) != 1 or not os.path.samefile(
            native_paths[0], str(meta.raw_path)
        ):
            raise ValueError(f"{slide_id}/{scanner}: geometry/background native WSI differs")

        native = pyvips.Image.new_from_file(native_paths[0], access="random")
        prepared = prepare_native_renderer(native, matrix)
        images = np.empty((100, PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
        for patch_index, item in enumerate(accepted.itertuples(index=False)):
            target_x, target_y, mapped_x, mapped_y = mapped_target_top_left(
                matrix,
                item.x_native,
                item.y_native,
                int(item.native_patch_size),
            )
            image, pre_scale = render_native_target_patch(
                native,
                matrix,
                target_x,
                target_y,
                PATCH_SIZE,
                prepared=prepared,
            )
            images[patch_index] = image
            stats = patch_statistics(image)
            reasons = glass_rejection_reasons(stats)
            coordinate_rows.append(
                {
                    "slide_id": slide_id,
                    "tissue_type": str(meta.tissue_type),
                    "scanner": scanner,
                    "patch_index": patch_index,
                    "x_native": int(item.x_native),
                    "y_native": int(item.y_native),
                    "native_patch_size": int(item.native_patch_size),
                    "mapped_target_center_x": mapped_x,
                    "mapped_target_center_y": mapped_y,
                    "target_top_left_x": target_x,
                    "target_top_left_y": target_y,
                    "explicit_aa_pre_scale": float(pre_scale),
                    "source_coordinate_qc_pass": True,
                    "post_render_qc_pass": not reasons,
                }
            )
            qc_rows.append(
                {
                    "slide_id": slide_id,
                    "tissue_type": str(meta.tissue_type),
                    "scanner": scanner,
                    "patch_index": patch_index,
                    "post_render_qc_pass": not reasons,
                    "post_render_rejection_reasons": ";".join(reasons),
                    "black_pixel_fraction": float(np.all(image == 0, axis=2).mean()),
                    **stats,
                }
            )
        scanner_spectra, scanner_replicates = spectrum_rows(
            images, scanner, slide_id, bins
        )
        for row in scanner_spectra:
            row["tissue_type"] = str(meta.tissue_type)
        for row in scanner_replicates:
            row["tissue_type"] = str(meta.tissue_type)
        spectra_rows.extend(scanner_spectra)
        replicate_rows.extend(scanner_replicates)

    return {
        "spectra": pd.DataFrame(spectra_rows),
        "replicates": pd.DataFrame(replicate_rows),
        "coordinates": pd.DataFrame(coordinate_rows),
        "qc": pd.DataFrame(qc_rows),
    }


def validate_tables(tables: dict[str, pd.DataFrame], slide_id: str, bins: int):
    expected = {
        "spectra": len(SCANNERS) * bins,
        "replicates": len(SCANNERS) * 5 * bins,
        "coordinates": len(SCANNERS) * 100,
        "qc": len(SCANNERS) * 100,
    }
    for name, frame in tables.items():
        if len(frame) != expected[name]:
            raise ValueError(f"{slide_id}: {name} rows {len(frame)} != {expected[name]}")
        numeric = frame.select_dtypes(include=[np.number])
        if not np.isfinite(numeric.to_numpy()).all():
            raise ValueError(f"{slide_id}: {name} contains non-finite values")
    if (tables["spectra"]["background_radial_power"] < 0).any():
        raise ValueError(f"{slide_id}: negative same-chain background power")


def main():
    args = parse_args()
    if args.bins != 72:
        raise ValueError("the frozen E0d radial contract uses 72 bins")
    geometry_path = Path(args.geometry)
    background_root = Path(args.background)
    metadata_path = background_root / "aggregate" / "cell_metadata.csv"
    geometry = pd.read_csv(geometry_path, dtype={"slide_id": str})
    metadata = pd.read_csv(metadata_path, dtype={"slide_id": str})
    slide_id, slides = resolve_slide(metadata, args.slide_id, args.slide_index)
    output = Path(args.output) / slide_id
    summary_path = output / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if (
            summary.get("analysis_version") == ANALYSIS_VERSION
            and summary.get("slide_gate_pass") is True
            and all((output / name).exists() for name in TABLES.values())
        ):
            print(json.dumps(summary, indent=2))
            return

    tables = render_slide(slide_id, geometry, metadata, background_root, args.bins)
    validate_tables(tables, slide_id, args.bins)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.mkdir(parents=True, exist_ok=False)
    for name, frame in tables.items():
        frame.to_csv(temporary / TABLES[name], index=False)
    summary = {
        "analysis": "same_processing_chain_operational_background_power",
        "analysis_version": ANALYSIS_VERSION,
        "slide_id": slide_id,
        "slide_index": slides.index(slide_id),
        "scanners": list(SCANNERS),
        "patches_per_scanner": 100,
        "replicates": 5,
        "patches_per_replicate": 20,
        "target_mpp": TARGET_MPP,
        "patch_size": PATCH_SIZE,
        "radial_bins": args.bins,
        "source_coordinate_contract": "Exp07 accepted native glass coordinates; no outcome-based reselection",
        "render_contract": RENDER_VERSION,
        "spectral_estimator": "E1-identical natural-log mean OD; patch-mean removal; 2D Hann; radial power",
        "geometry_manifest_sha256": sha256(geometry_path),
        "background_metadata_sha256": sha256(metadata_path),
        "post_render_qc_pass_fraction": float(tables["qc"]["post_render_qc_pass"].mean()),
        "maximum_black_pixel_fraction": float(tables["qc"]["black_pixel_fraction"].max()),
        "slide_gate_pass": True,
        "claim_scope": "operational glass/background noise floor after the tissue processing chain; not detector NPS, DQE, or absolute MTF",
    }
    (temporary / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

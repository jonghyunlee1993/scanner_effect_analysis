"""Measure one slide's paired ERT directly from the audited native-AA grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from prenorm.exp01.frequency import FixedLaplacianPyramid
from run_exp05_spectral_pilot import (
    BANDS,
    HF_GAINS,
    SCANNERS,
    frequency_geometry,
    geometric_band,
    intervention_power,
    normalized_transfer,
    project_to_hf_gain,
    radial_mean,
)


ANALYSIS_VERSION = "e1_native_aa_ert_v1"
PATCH_SIZE = 256
TARGET_MPP = 0.5052
TABLE_FILES = {
    "spectra": "slide_spectra.csv",
    "bands": "slide_band_summary.csv",
    "replicate_bands": "slide_band_replicates.csv",
    "interventions": "exp02_hf_gain_spectra.csv",
    "projection": "scanner_hf_gain_projection.csv",
    "patches": "selected_patches.csv",
    "alignment": "patch_alignment.csv",
}


def parse_args():
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--slide-id")
    selection.add_argument("--slide-index", type=int)
    parser.add_argument("--grid", default="outputs/e0_native_aa_grid/shards")
    parser.add_argument("--output", default="outputs/e1_native_aa_spectral_109/shards")
    parser.add_argument("--bins", type=int, default=72)
    return parser.parse_args()


def resolve_slide(grid_root: Path, slide_id: str | None, slide_index: int | None):
    slides = sorted(path.stem for path in grid_root.glob("*.h5"))
    if len(slides) != 109:
        raise ValueError(f"expected 109 audited native-AA grids, got {len(slides)}")
    if slide_index is not None:
        if slide_index < 0 or slide_index >= len(slides):
            raise IndexError(f"slide index {slide_index} outside 0..{len(slides) - 1}")
        return slides[slide_index]
    if str(slide_id) not in slides:
        raise KeyError(f"slide absent from native-AA grid: {slide_id}")
    return str(slide_id)


def central_256(rgb):
    value = np.asarray(rgb)
    if value.shape[-3:] != (512, 512, 3):
        raise ValueError(f"expected 512 px shared grid, got {value.shape}")
    return value[..., 128:384, 128:384, :]


def batch_total_od_fft(rgb: np.ndarray, window: np.ndarray):
    value = -np.log((np.asarray(rgb, dtype=np.float32) + 1.0) / 256.0).mean(axis=3)
    value -= value.mean(axis=(1, 2), keepdims=True)
    return np.fft.fft2(value * window[None, :, :], axes=(-2, -1))


def spectrum_tables(images: np.ndarray, replicate_ids: np.ndarray, bins: int, slide_id: str):
    index, valid, counts, frequency, nyquist = frequency_geometry(PATCH_SIZE, TARGET_MPP, bins)
    window_1d = np.hanning(PATCH_SIZE).astype(np.float32)
    window = window_1d[:, None] * window_1d[None, :]
    reference_fft = batch_total_od_fft(images[0], window)
    reference_power = np.mean(np.abs(reference_fft) ** 2, axis=0)
    radial_power = {}
    coherence = {}
    replicate_radial = {replicate: {} for replicate in range(5)}
    for scanner_index, scanner in enumerate(SCANNERS):
        fourier = reference_fft if scanner_index == 0 else batch_total_od_fft(images[scanner_index], window)
        mean_power = np.mean(np.abs(fourier) ** 2, axis=0)
        radial_power[scanner] = radial_mean(mean_power, index, valid, counts)
        cross = np.mean(fourier * np.conj(reference_fft), axis=0)
        coherence_2d = np.divide(
            np.abs(cross) ** 2,
            mean_power * reference_power,
            out=np.zeros_like(mean_power),
            where=(mean_power * reference_power) > 1e-20,
        )
        coherence[scanner] = np.clip(radial_mean(coherence_2d, index, valid, counts), 0.0, 1.0)
        for replicate in range(5):
            selected = replicate_ids == replicate
            if int(selected.sum()) != 20:
                raise ValueError(
                    f"{slide_id}: replicate {replicate} contains {int(selected.sum())}, expected 20"
                )
            power = np.mean(np.abs(fourier[selected]) ** 2, axis=0)
            replicate_radial[replicate][scanner] = radial_mean(power, index, valid, counts)

    spectra = []
    bands = []
    for scanner in SCANNERS:
        transfer = normalized_transfer(radial_power[scanner], radial_power["at2"], frequency)
        for bin_index, value in enumerate(frequency):
            spectra.append(
                {
                    "slide_id": slide_id,
                    "scanner": scanner,
                    "frequency_cyc_per_um": float(value),
                    "relative_transfer": float(transfer[bin_index]),
                    "log2_relative_transfer": float(np.log2(max(transfer[bin_index], 1e-20))),
                    "coherence_to_at2": float(coherence[scanner][bin_index]),
                    "radial_power": float(radial_power[scanner][bin_index]),
                }
            )
        for name, bounds in BANDS.items():
            value = geometric_band(transfer, frequency, bounds)
            bands.append(
                {
                    "slide_id": slide_id,
                    "scanner": scanner,
                    "band": name,
                    "lower_frequency": bounds[0],
                    "upper_frequency": bounds[1],
                    "relative_transfer": value,
                    "log2_relative_transfer": float(np.log2(value)),
                }
            )

    replicate_bands = []
    for replicate in range(5):
        for scanner in SCANNERS:
            transfer = normalized_transfer(
                replicate_radial[replicate][scanner],
                replicate_radial[replicate]["at2"],
                frequency,
            )
            for name, bounds in BANDS.items():
                value = geometric_band(transfer, frequency, bounds)
                replicate_bands.append(
                    {
                        "slide_id": slide_id,
                        "scanner": scanner,
                        "replicate": replicate,
                        "patches_in_replicate": 20,
                        "band": name,
                        "lower_frequency": bounds[0],
                        "upper_frequency": bounds[1],
                        "relative_transfer": value,
                        "log2_relative_transfer": float(np.log2(value)),
                    }
                )

    pyramid = FixedLaplacianPyramid(levels=4)
    intervention_2d = intervention_power(images[0], window, pyramid, HF_GAINS)
    intervention_radial = {
        gain: radial_mean(value, index, valid, counts) for gain, value in intervention_2d.items()
    }
    interventions = []
    for gain in HF_GAINS:
        transfer = normalized_transfer(intervention_radial[gain], intervention_radial[1.0], frequency)
        family = "raw" if gain == 1.0 else ("blur" if gain < 1.0 else "sharpen")
        parameter = 0.0 if gain == 1.0 else (1.0 - gain if gain < 1.0 else gain - 1.0)
        for bin_index, value in enumerate(frequency):
            interventions.append(
                {
                    "slide_id": slide_id,
                    "family": family,
                    "parameter": float(parameter),
                    "effective_hf_gain": float(gain),
                    "frequency_cyc_per_um": float(value),
                    "relative_transfer": float(transfer[bin_index]),
                    "log2_relative_transfer": float(np.log2(max(transfer[bin_index], 1e-20))),
                }
            )
    return (
        pd.DataFrame(spectra),
        pd.DataFrame(bands),
        pd.DataFrame(replicate_bands),
        pd.DataFrame(interventions),
        float(nyquist),
    )


def validate_tables(tables: dict[str, pd.DataFrame], slide_id: str):
    expected_rows = {
        "spectra": 6 * 72,
        "bands": 6 * 3,
        "replicate_bands": 6 * 5 * 3,
        "interventions": len(HF_GAINS) * 72,
        "projection": 6,
        "patches": 100,
        "alignment": 600,
    }
    for name, frame in tables.items():
        if len(frame) != expected_rows[name]:
            raise ValueError(f"{slide_id}: {name} has {len(frame)}, expected {expected_rows[name]}")
        numeric = frame.select_dtypes(include=[np.number])
        if not np.isfinite(numeric.to_numpy()).all():
            raise ValueError(f"{slide_id}: {name} contains non-finite values")
    at2 = tables["spectra"].query("scanner == 'at2'")["relative_transfer"]
    if not np.allclose(at2, 1.0, atol=1e-10):
        raise ValueError(f"{slide_id}: AT2 self-transfer differs from one")


def main():
    args = parse_args()
    if args.bins != 72:
        raise ValueError("the frozen E1 radial contract uses 72 bins")
    grid_root = Path(args.grid)
    slide_id = resolve_slide(grid_root, args.slide_id, args.slide_index)
    grid_path = grid_root / f"{slide_id}.h5"
    source_summary = json.loads(grid_path.with_suffix(".summary.json").read_text())
    output = Path(args.output) / slide_id
    existing_summary = output / "summary.json"
    if existing_summary.exists():
        existing = json.loads(existing_summary.read_text())
        if (
            existing.get("analysis_version") == ANALYSIS_VERSION
            and existing.get("source_grid_sha256") == source_summary["output_sha256"]
            and existing.get("slide_gate_pass") is True
            and all((output / filename).exists() for filename in TABLE_FILES.values())
        ):
            print(json.dumps(existing, indent=2))
            return

    with h5py.File(grid_path, "r") as store:
        images = central_256(store["rgb"][:])
        scanners = [value.decode() if isinstance(value, bytes) else str(value) for value in store["scanner"][:]]
        if scanners != list(SCANNERS):
            raise ValueError(f"{slide_id}: scanner order differs from frozen panel")
        locations = store["location_id"][:].astype(int)
        replicates = store["replicate_id"][:].astype(int)
        centers_x = store["canonical_center_x"][:].astype(int)
        centers_y = store["canonical_center_y"][:].astype(int)
        alignment_version = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in store["alignment_version"][:]
        ]
    spectra, bands, replicate_bands, interventions, nyquist = spectrum_tables(
        images, replicates, args.bins, slide_id
    )
    projection = project_to_hf_gain(spectra, interventions)
    patches = pd.DataFrame(
        {
            "slide_id": slide_id,
            "patch_index": locations,
            "location_id": locations,
            "replicate_id": replicates,
            "canonical_center_x": centers_x,
            "canonical_center_y": centers_y,
            "pixel_source": "native_wsi_only",
        }
    )
    alignment = pd.DataFrame(
        [
            {
                "slide_id": slide_id,
                "patch_index": int(location_id),
                "location_id": int(location_id),
                "scanner": scanner,
                "post_render_search_applied": False,
                "post_render_dx": 0,
                "post_render_dy": 0,
                "geometry_pass": True,
                "alignment_version": alignment_version[scanner_index],
            }
            for scanner_index, scanner in enumerate(SCANNERS)
            for location_id in locations
        ]
    )
    tables = {
        "spectra": spectra,
        "bands": bands,
        "replicate_bands": replicate_bands,
        "interventions": interventions,
        "projection": projection,
        "patches": patches,
        "alignment": alignment,
    }
    validate_tables(tables, slide_id)
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(output / TABLE_FILES[name], index=False)
    summary = {
        "analysis": "paired_native_aa_effective_relative_transfer",
        "analysis_version": ANALYSIS_VERSION,
        "slide_id": slide_id,
        "patches": 100,
        "replicate_groups": 5,
        "scanners": list(SCANNERS),
        "pixel_source": "native_wsi_only",
        "post_render_alignment_search": False,
        "patch_size": PATCH_SIZE,
        "target_mpp": TARGET_MPP,
        "nyquist_cyc_per_um": nyquist,
        "source_grid_sha256": source_summary["output_sha256"],
        "estimator": "mean total OD; patch-mean removal; 2D Hann; radial power; anchor-normalized amplitude ratio",
        "slide_gate_pass": True,
    }
    existing_summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

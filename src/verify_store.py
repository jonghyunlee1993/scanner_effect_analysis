"""Audit a version 3 crop-aligned store without modifying it.

The report separates content presence, geometry eligibility, and continuous
registration confidence, then renders the exact RGB images training will read.
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import qc, store
from utils.config import load_config, resolve_config_path

SLIDE = os.environ.get("VERIFY_SLIDE", "12.5_4")
N = int(os.environ.get("VERIFY_N", "10"))
OUT = REPO / "outputs" / "verify_store_v3"


def overlay(a, b):
    """Compose a green/magenta luminance registration overlay."""
    gray_a = np.dot(a[..., :3], [0.299, 0.587, 0.114]) / 255.0
    gray_b = np.dot(b[..., :3], [0.299, 0.587, 0.114]) / 255.0
    return np.clip(np.stack([gray_a, gray_b, gray_a], axis=-1), 0.0, 1.0)


def _stack(sidecar, column):
    """Stack a parquet list column into a tuple-by-scanner array."""
    return np.stack(sidecar[column].to_numpy())


def main():
    """Print summary diagnostics and save a registered RGB panel."""
    default = "configs/main/phase_0_preprocessing_10slide_v3.yaml"
    cfg = load_config(resolve_config_path(default))
    scanners = list(cfg.scanners)
    ref = cfg.reference_scanner
    OUT.mkdir(parents=True, exist_ok=True)

    sidecar = pd.read_parquet(Path(cfg.paths.qc_dir) / "sidecar" / f"{SLIDE}.parquet")
    count = len(sidecar)
    present = _stack(sidecar, "present").astype(bool)
    geom_ok = _stack(sidecar, "geom_ok").astype(bool)
    q_reg = _stack(sidecar, "q_reg")
    ncc_lp = _stack(sidecar, "ncc_lp")
    phase_response = _stack(sidecar, "phase_response")
    residual_shift = _stack(sidecar, "residual_shift")
    pad_fraction = _stack(sidecar, "pad_frac")
    peak_boundary = _stack(sidecar, "peak_boundary").astype(bool)
    dy = _stack(sidecar, "off_dy")
    dx = _stack(sidecar, "off_dx")

    print(f"=== {SLIDE}: {count} v3 tuples ===\n")
    print(
        f"{'scanner':8s} {'present':>8s} {'geom_ok':>8s} {'q_reg':>8s} "
        f"{'ncc_lp':>8s} {'phase':>8s} {'shift p90':>10s} {'boundary':>9s}"
    )
    for j, scanner in enumerate(scanners):
        print(
            f"{scanner:8s} {present[:, j].mean() * 100:7.1f}% "
            f"{geom_ok[:, j].mean() * 100:7.1f}% {q_reg[:, j].mean():8.3f} "
            f"{ncc_lp[:, j].mean():8.3f} {phase_response[:, j].mean():8.3f} "
            f"{np.percentile(residual_shift[:, j], 90):10.2f} "
            f"{peak_boundary[:, j].mean() * 100:8.1f}%"
        )
    print(f"\ngeom_ok scanners per tuple: {dict(zip(*np.unique(geom_ok.sum(1), return_counts=True)))}")
    print(f"tuples with absent scanner: {int((~present).any(1).sum())}")
    print(f"slots with pad_frac > 0.01: {int((pad_fraction > 0.01).sum())}")

    path = Path(cfg.paths.output_store) / f"{SLIDE}.h5"
    with h5py.File(path, "r") as handle:
        if int(handle.attrs.get("schema_version", -1)) != store.SCHEMA_VERSION:
            raise ValueError(f"{path} is not a schema_version=3 store")
        for field, expected in (
            ("present", present), ("geom_ok", geom_ok), ("q_reg", q_reg)
        ):
            actual = handle[field][:]
            if not np.allclose(actual, expected):
                raise ValueError(f"HDF5 {field} does not match its sidecar")

        rng = np.random.default_rng(0)
        probe = rng.choice(count, size=min(60, count), replace=False)
        measured = {scanner: [] for scanner in scanners if scanner != ref}
        zero_confirmed = 0
        for i in probe:
            images = store.read_tuple(handle, scanners, int(sidecar.tuple_id.iloc[i]))
            reference = images[ref]["rgb"]
            for j, scanner in enumerate(scanners):
                if scanner == ref:
                    continue
                rgb = images[scanner]["rgb"]
                if not present[i, j]:
                    zero_confirmed += int(rgb.max() == 0)
                elif geom_ok[i, j]:
                    measured[scanner].append(qc.registration_quality(reference, rgb)["q_reg"])

        print("\nStored-tile q_reg recomputed from RGB:")
        for scanner, values in measured.items():
            print(f"  {scanner:8s} n={len(values):3d} mean={np.mean(values):.3f}")
        absent_probed = int((~present[probe]).sum())
        print(f"absent slots probed: {absent_probed}; zero RGB confirmed: {zero_confirmed}")

        chosen = list(probe[:N])
        fig, axes = plt.subplots(
            len(chosen),
            2 * len(scanners) - 1,
            figsize=((2 * len(scanners) - 1) * 2.3, len(chosen) * 2.5),
            squeeze=False,
        )
        columns = scanners + [f"{scanner} vs {ref}" for scanner in scanners if scanner != ref]
        for j, column in enumerate(columns):
            axes[0, j].set_title(column, fontsize=9)
        for row, i in enumerate(chosen):
            images = store.read_tuple(handle, scanners, int(sidecar.tuple_id.iloc[i]))
            rendered = [images[scanner]["rgb"] for scanner in scanners]
            rendered += [overlay(images[scanner]["rgb"], images[ref]["rgb"])
                         for scanner in scanners if scanner != ref]
            for j, image in enumerate(rendered):
                axes[row, j].imshow(image)
                axes[row, j].set_xticks([])
                axes[row, j].set_yticks([])
            labels = [f"({sidecar.x.iloc[i]},{sidecar.y.iloc[i]})"]
            for j, scanner in enumerate(scanners):
                if scanner != ref:
                    state = "OK" if geom_ok[i, j] else "GEOM_BAD"
                    if not present[i, j]:
                        state = "ABSENT"
                    labels.append(
                        f"{scanner}: d=({dy[i, j]},{dx[i, j]}) q={q_reg[i, j]:.2f} {state}"
                    )
            axes[row, 0].set_ylabel(
                "\n".join(labels), fontsize=6.5, rotation=0, ha="right", va="center", labelpad=54
            )
        fig.suptitle(
            f"{SLIDE} v3 RGB store. Overlay: green={ref}, magenta=scanner; gray=registered.",
            fontsize=11,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.985])
        output = OUT / f"{SLIDE.replace('.', '_')}_store.png"
        fig.savefig(output, dpi=105, bbox_inches="tight")
        plt.close(fig)
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()

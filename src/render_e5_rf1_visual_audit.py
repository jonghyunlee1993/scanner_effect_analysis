"""Render deterministic paired-patch and spectrum diagnostics for E5-RF1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from e5_comparator_population import SCANNERS, centered_crop, reinhard_lab, rgb8_to_rgb01
from e5_reinhard_residual_frequency import (
    RF1_VERSION,
    fitted_residual_gains,
    fold_assignments,
    shared_od_residual_frequency,
)
from fetch_e0_pfm_checkpoints import sha256


VISUAL_VERSION = "e5_rf1_visual_v1"
VISUAL_FOV = 256


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", default="outputs/e0_native_aa_grid/shards")
    parser.add_argument("--statistics", default="outputs/e5_rf1_fold_statistics")
    parser.add_argument("--stability", default="outputs/e5_rf1_input_stability")
    parser.add_argument("--output", default="outputs/e5_rf1_visual_audit")
    return parser.parse_args()


def selected_examples(paths: list[Path]):
    examples = []
    for scanner in SCANNERS[1:]:
        candidates = []
        for path in paths:
            for location_index in range(100):
                token = f"{VISUAL_VERSION}:{scanner}:{path.stem}:{location_index}"
                candidates.append((hashlib.sha256(token.encode()).hexdigest(), path, location_index))
        _, path, location_index = min(candidates)
        examples.append((scanner, path, location_index))
    return examples


def main():
    args = parse_args()
    stability_root = Path(args.stability)
    stability_path = stability_root / "summary.json"
    stability = json.loads(stability_path.read_text())
    if not (
        stability.get("analysis") == "e5_rf1_input_stability"
        and stability.get("rf1_version") == RF1_VERSION
        and stability.get("outcome_access") is False
        and stability.get("stability_gate_pass") is True
    ):
        raise RuntimeError("E5-RF1 input stability gate has not passed")
    cap = float(stability["selected_gain_cap"])
    paths = sorted(Path(args.grid).glob("*.h5"))
    if len(paths) != 109:
        raise ValueError("RF1 visual audit requires 109 grid shards")
    assignments = fold_assignments([path.stem for path in paths])

    figure, axes = plt.subplots(len(SCANNERS) - 1, 5, figsize=(16, 15))
    column_titles = ("Paired AT2", "Raw source", "Reinhard", "RF1", "|RF1−Reinhard| ×5")
    for axis, title in zip(axes[0], column_titles):
        axis.set_title(title, fontsize=11)
    rows = []
    for row_index, (scanner, path, location_index) in enumerate(selected_examples(paths)):
        scanner_index = SCANNERS.index(scanner)
        fold = assignments[path.stem]
        statistics_path = Path(args.statistics) / f"fov_{VISUAL_FOV}_fold_{fold}.npz"
        expected_sha = stability["fold_statistics_sha256"][f"fov_{VISUAL_FOV}_fold_{fold}"]
        if sha256(statistics_path) != expected_sha:
            raise ValueError(f"{statistics_path}: statistics hash mismatch")
        with np.load(statistics_path) as source:
            statistics = {name: source[name] for name in source.files}
        gains = fitted_residual_gains(
            statistics["post_reinhard_source_power"],
            statistics["raw_at2_target_power"],
            statistics["radial_frequency"],
            cap,
        )
        with h5py.File(path, "r") as source:
            target8 = centered_crop(source["rgb"][0, location_index : location_index + 1], VISUAL_FOV)
            source8 = centered_crop(
                source["rgb"][scanner_index, location_index : location_index + 1], VISUAL_FOV
            )
            location = source["location_id"][location_index]
        target = rgb8_to_rgb01(target8)
        raw = rgb8_to_rgb01(source8)
        lab_mean = torch.as_tensor(statistics["lab_mean"], dtype=torch.float32)
        lab_std = torch.as_tensor(statistics["lab_std"], dtype=torch.float32)
        base = reinhard_lab(
            raw,
            lab_mean[scanner_index].reshape(1, 1, 1, 3),
            lab_std[scanner_index].reshape(1, 1, 1, 3),
            lab_mean[0].reshape(1, 1, 1, 3),
            lab_std[0].reshape(1, 1, 1, 3),
        )
        report = shared_od_residual_frequency(
            base["output"], gains[scanner_index - 1], statistics["radial_frequency"]
        )
        images = (
            target[0].numpy(),
            raw[0].numpy(),
            base["output"][0].numpy(),
            report["output"][0].numpy(),
            np.clip(np.abs(report["output"][0].numpy() - base["output"][0].numpy()) * 5, 0, 1),
        )
        for axis, image in zip(axes[row_index], images):
            axis.imshow(image)
            axis.set_axis_off()
        axes[row_index, 0].text(
            -0.07,
            0.5,
            scanner.upper(),
            transform=axes[row_index, 0].transAxes,
            ha="right",
            va="center",
            rotation=90,
            fontsize=10,
            fontweight="bold",
            clip_on=False,
        )
        axes[row_index, 3].text(
            0.01,
            0.99,
            f"proj MAE {float(report['projection_rgb_mae'][0]):.4g}\nmaterial {float(report['material_range_fraction'][0]):.4g}",
            transform=axes[row_index, 3].transAxes,
            ha="left",
            va="top",
            fontsize=7,
            color="black",
            bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
        )
        rows.append(
            {
                "scanner": scanner,
                "slide_id": path.stem,
                "location_index": location_index,
                "location_id": location.decode() if isinstance(location, bytes) else str(location),
                "fold": fold,
                "fov": VISUAL_FOV,
                "selected_gain_cap": cap,
                "base_preclip_range_fraction": float(base["preclip_range_fraction"][0]),
                "base_clip_pixel_mae": float(base["clip_pixel_mae"][0]),
                "material_range_fraction": float(report["material_range_fraction"][0]),
                "projection_fraction": float(report["projection_fraction"][0]),
                "projection_rgb_mae": float(report["projection_rgb_mae"][0]),
                "final_clamp_mae": float(report["final_clamp_mae"][0]),
                "statistics_sha256": expected_sha,
            }
        )
    figure.suptitle(
        f"E5-RF1 deterministic paired-patch audit (FOV {VISUAL_FOV}, gain cap {cap:g})",
        fontsize=14,
    )
    figure.tight_layout()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    figures = {}
    for suffix in ("png", "pdf"):
        path = output / f"paired_patch_audit.{suffix}"
        figure.savefig(path, dpi=250 if suffix == "png" else None, bbox_inches="tight")
        figures[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    plt.close(figure)

    sample_path = output / "sample_manifest.csv"
    pd.DataFrame(rows).to_csv(sample_path, index=False)
    candidates = pd.read_csv(stability_root / "gain_cap_candidates.csv")
    selected = candidates[np.isclose(candidates["gain_cap"], cap)].copy()
    selected["rmse_ratio"] = (
        selected["output_log_spectrum_rmse"] / selected["base_log_spectrum_rmse"]
    )
    selected_path = output / "selected_cap_spectrum_cells.csv"
    selected.to_csv(selected_path, index=False)
    summary = {
        "analysis": "e5_rf1_visual_audit",
        "rf1_version": RF1_VERSION,
        "visual_version": VISUAL_VERSION,
        "selection": "minimum salted SHA256 independently for each source scanner",
        "fov": VISUAL_FOV,
        "examples": len(rows),
        "selected_gain_cap": cap,
        "mean_spectrum_rmse_ratio": float(selected["rmse_ratio"].mean()),
        "cells_with_spectrum_improvement": int((selected["rmse_ratio"] < 1).sum()),
        "cells": len(selected),
        "sample_manifest_sha256": sha256(sample_path),
        "spectrum_cells_sha256": sha256(selected_path),
        "input_stability_sha256": sha256(stability_path),
        "figures": figures,
        "visual_gate_pass": bool(
            len(rows) == 5
            and len(selected) == 15
            and np.isfinite(selected["rmse_ratio"]).all()
            and all(row["final_clamp_mae"] <= 1e-6 for row in rows)
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["visual_gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

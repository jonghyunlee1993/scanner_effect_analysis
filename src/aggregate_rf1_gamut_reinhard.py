"""Aggregate the outer-held-out gamut-safe Reinhard image-only pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from e5_comparator_population import FOVS, SCANNERS, radial_geometry
from e5_reinhard_residual_frequency import RF1_FOLDS, log_spectrum_rmse
from fetch_e0_pfm_checkpoints import sha256
from rf1_gamut_reinhard import GAMUT_REINHARD_VERSION


BOOTSTRAP_SEED = 20260803
BOOTSTRAP_REPLICATES = 5000


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default="outputs/rf1_improvement_pilot/gamut_reinhard/shards"
    )
    parser.add_argument(
        "--pilot-contract", default="docs/e5_rf1_improvement_pilot_contract.md"
    )
    parser.add_argument("--output", default="outputs/rf1_improvement_pilot/gamut_reinhard")
    return parser.parse_args()


def mean_ci(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    value = np.asarray(values, dtype=np.float64)
    if value.ndim != 1 or len(value) != 109 or not np.isfinite(value).all():
        raise ValueError("paired bootstrap requires 109 finite slide values")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(value), size=(BOOTSTRAP_REPLICATES, len(value)))
    means = value[indices].mean(axis=1)
    return float(value.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main():
    args = parse_args()
    input_root = Path(args.input)
    contract_sha = sha256(Path(args.pilot_contract))
    patch_frames = []
    shard_hashes = {}
    for fov in FOVS:
        for fold in range(RF1_FOLDS):
            stem = f"fov_{fov}_fold_{fold}"
            summary_path = input_root / f"{stem}.summary.json"
            summary = json.loads(summary_path.read_text())
            patch_path = input_root / f"{stem}.patches.csv"
            spectrum_path = input_root / f"{stem}.spectrum.csv"
            array_path = input_root / f"{stem}.npz"
            valid = bool(
                summary.get("analysis") == "rf1_gamut_reinhard_outer_fold_pilot"
                and summary.get("version") == GAMUT_REINHARD_VERSION
                and summary.get("outcome_access") is False
                and summary.get("fov") == fov
                and summary.get("fold") == fold
                and summary.get("pilot_contract_sha256") == contract_sha
                and summary.get("patch_csv_sha256") == sha256(patch_path)
                and summary.get("spectrum_csv_sha256") == sha256(spectrum_path)
                and summary.get("array_sha256") == sha256(array_path)
                and summary.get("pilot_gate_pass") is True
            )
            if not valid:
                raise RuntimeError(f"invalid gamut-Reinhard shard: {summary_path}")
            patch_frames.append(pd.read_csv(patch_path))
            shard_hashes[stem] = sha256(summary_path)

    patches = pd.concat(patch_frames, ignore_index=True)
    expected_rows = 109 * len(FOVS) * 100 * (len(SCANNERS) - 1)
    if len(patches) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} patch rows, observed {len(patches)}")
    numeric_columns = [
        name
        for name in patches.columns
        if name
        not in {"fov", "fold", "slide_id", "location_index", "location_id", "scanner"}
    ]
    slides = (
        patches.groupby(["fov", "scanner", "slide_id"], as_index=False)[numeric_columns]
        .mean()
        .sort_values(["fov", "scanner", "slide_id"])
    )
    if len(slides) != len(FOVS) * (len(SCANNERS) - 1) * 109:
        raise RuntimeError("slide aggregation has an invalid population")

    cell_rows = []
    scanner_order = {scanner: index for index, scanner in enumerate(SCANNERS[1:])}
    for (fov, scanner), cell in slides.groupby(["fov", "scanner"], sort=True):
        rgb = mean_ci(
            cell["ray_minus_clipped_rgb_mae"].to_numpy(),
            BOOTSTRAP_SEED + int(fov) * 10 + scanner_order[scanner],
        )
        gradient = mean_ci(
            cell["ray_minus_clipped_gradient_mae"].to_numpy(),
            BOOTSTRAP_SEED + int(fov) * 20 + scanner_order[scanner],
        )
        row = {
            "fov": int(fov),
            "scanner": scanner,
            "slides": len(cell),
            "clipped_rgb_mae_to_at2": float(cell["clipped_rgb_mae_to_at2"].mean()),
            "ray_rgb_mae_to_at2": float(cell["ray_rgb_mae_to_at2"].mean()),
            "ray_minus_clipped_rgb_mae": rgb[0],
            "ray_minus_clipped_rgb_mae_ci_lower": rgb[1],
            "ray_minus_clipped_rgb_mae_ci_upper": rgb[2],
            "clipped_mean_od_gradient_mae": float(
                cell["clipped_mean_od_gradient_mae"].mean()
            ),
            "ray_mean_od_gradient_mae": float(cell["ray_mean_od_gradient_mae"].mean()),
            "ray_minus_clipped_gradient_mae": gradient[0],
            "ray_minus_clipped_gradient_mae_ci_lower": gradient[1],
            "ray_minus_clipped_gradient_mae_ci_upper": gradient[2],
            "base_preclip_range_fraction": float(
                cell["base_preclip_range_fraction"].mean()
            ),
            "base_clip_pixel_mae": float(cell["base_clip_pixel_mae"].mean()),
            "ray_limited_pixel_fraction": float(
                cell["ray_limited_pixel_fraction"].mean()
            ),
            "ray_mean_scale_loss": float(cell["ray_mean_scale_loss"].mean()),
            "ray_projection_rgb_mae": float(cell["ray_projection_rgb_mae"].mean()),
            "ray_final_clamp_mae_max": float(cell["ray_final_clamp_mae"].max()),
            "ray_vs_clipped_rgb_mae": float(cell["ray_vs_clipped_rgb_mae"].mean()),
        }
        cell_rows.append(row)
    cells = pd.DataFrame(cell_rows)

    spectrum_rows = []
    for fov in FOVS:
        target_power = None
        clipped_power = None
        ray_power = None
        for fold in range(RF1_FOLDS):
            with np.load(input_root / f"fov_{fov}_fold_{fold}.npz") as source:
                if target_power is None:
                    target_power = source["target_power"].copy()
                    clipped_power = source["clipped_power"].copy()
                    ray_power = source["ray_power"].copy()
                else:
                    target_power += source["target_power"]
                    clipped_power += source["clipped_power"]
                    ray_power += source["ray_power"]
        frequency = radial_geometry(fov).frequency
        for scanner_index, scanner in enumerate(SCANNERS[1:]):
            clipped_rmse = log_spectrum_rmse(
                clipped_power[scanner_index], target_power[scanner_index], frequency
            )
            ray_rmse = log_spectrum_rmse(
                ray_power[scanner_index], target_power[scanner_index], frequency
            )
            spectrum_rows.append(
                {
                    "fov": fov,
                    "scanner": scanner,
                    "clipped_log_spectrum_rmse": clipped_rmse,
                    "ray_log_spectrum_rmse": ray_rmse,
                    "ray_minus_clipped_log_spectrum_rmse": ray_rmse - clipped_rmse,
                }
            )
    spectrum = pd.DataFrame(spectrum_rows)
    cells = cells.merge(spectrum, on=["fov", "scanner"], validate="one_to_one")

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    patch_path = output_root / "patch_metrics.csv"
    slide_path = output_root / "slide_metrics.csv"
    cell_path = output_root / "cell_summary.csv"
    patches.to_csv(patch_path, index=False)
    slides.to_csv(slide_path, index=False)
    cells.to_csv(cell_path, index=False)
    summary = {
        "analysis": "rf1_gamut_reinhard_image_only_pilot",
        "version": GAMUT_REINHARD_VERSION,
        "outcome_access": False,
        "slides": 109,
        "patch_rows": len(patches),
        "cell_rows": len(cells),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "rgb_cells_improved_point": int((cells["ray_minus_clipped_rgb_mae"] < 0).sum()),
        "rgb_cells_improved_ci": int(
            (cells["ray_minus_clipped_rgb_mae_ci_upper"] < 0).sum()
        ),
        "gradient_cells_improved_point": int(
            (cells["ray_minus_clipped_gradient_mae"] < 0).sum()
        ),
        "spectrum_cells_improved_point": int(
            (cells["ray_minus_clipped_log_spectrum_rmse"] < 0).sum()
        ),
        "max_final_clamp_mae": float(cells["ray_final_clamp_mae_max"].max()),
        "pilot_contract_sha256": contract_sha,
        "shard_summary_sha256": shard_hashes,
        "patch_metrics_sha256": sha256(patch_path),
        "slide_metrics_sha256": sha256(slide_path),
        "cell_summary_sha256": sha256(cell_path),
        "pilot_gate_pass": bool(
            len(cells) == 15 and cells["ray_final_clamp_mae_max"].max() <= 1e-6
        ),
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not summary["pilot_gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

"""Render deterministic paired source-to-AT2 patches across image corrections."""

from __future__ import annotations

import argparse
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
from e5_macenko_supplement import aggregate_loso_reference, macenko_normalize_batch
from e5_reinhard_residual_frequency import fitted_residual_gains, shared_od_residual_frequency
from extract_e5_image_features import (
    heldout_parameters,
    load_stability as load_e5_stability,
    load_statistics as load_e5_statistics,
    render as render_e5,
)
from extract_e5_macenko_features import load_reference as load_macenko_reference
from extract_e5_rf1_features import (
    load_fold_statistics as load_rf1_fold_statistics,
    load_stability as load_rf1_stability,
)
from fetch_e0_pfm_checkpoints import sha256


FOV = 256
METHODS = (
    "target_at2",
    "raw_source",
    "reinhard_lab",
    "paired_od_affine",
    "frequency_calibration",
    "macenko_supplement",
    "reinhard_residual_frequency",
)
LABELS = {
    "target_at2": "Paired target\nAT2",
    "raw_source": "Raw source",
    "reinhard_lab": "Reinhard\n(LOSO)",
    "paired_od_affine": "Paired OD\naffine",
    "frequency_calibration": "Frequency\ncalibration",
    "macenko_supplement": "Macenko\n(Supplement)",
    "reinhard_residual_frequency": "Ours: Reinhard +\nresidual frequency",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default="outputs/e5_rf1_visual_audit/sample_manifest.csv")
    parser.add_argument("--grid", default="outputs/e0_native_aa_grid/shards")
    parser.add_argument("--e5-statistics", default="outputs/e5_image_statistics")
    parser.add_argument("--e5-stability", default="outputs/e5_input_stability/summary.json")
    parser.add_argument("--e5-contract", default="docs/e5_comparator_execution_contract.md")
    parser.add_argument("--macenko-reference", default="outputs/e5_macenko_references")
    parser.add_argument("--rf1-statistics", default="outputs/e5_rf1_fold_statistics")
    parser.add_argument("--rf1-stability", default="outputs/e5_rf1_input_stability/summary.json")
    parser.add_argument("--rf1-contract", default="docs/e5_reinhard_residual_frequency_contract.md")
    parser.add_argument("--output", default="outputs/e5_rf1_visual_comparison")
    parser.add_argument("--dpi", type=int, default=250)
    return parser.parse_args()


def to_numpy(report):
    value = report["output"]
    if value.shape != (1, FOV, FOV, 3) or not torch.isfinite(value).all():
        raise ValueError(f"invalid rendered patch: {tuple(value.shape)}")
    return value[0].detach().cpu().numpy()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("source-to-AT2 image comparison requires a visible CUDA device")
    device = "cuda"
    sample_path = Path(args.samples)
    samples = pd.read_csv(sample_path, dtype={"slide_id": str, "location_id": str})
    if (
        len(samples) != 5
        or set(samples["scanner"]) != set(SCANNERS[1:])
        or not samples["fov"].eq(FOV).all()
    ):
        raise ValueError("visual comparison requires the five frozen FOV-256 scanner samples")

    e5_stability_path = Path(args.e5_stability)
    e5_contract_path = Path(args.e5_contract)
    e5_stability = load_e5_stability(e5_stability_path, e5_contract_path)
    _, e5_statistics_summary, e5_statistics = load_e5_statistics(
        Path(args.e5_statistics), FOV, e5_stability
    )
    macenko_path, macenko_summary, macenko = load_macenko_reference(
        Path(args.macenko_reference), FOV
    )
    macenko_slide_ids = [str(value) for value in macenko["slide_ids"]]

    rf1_stability_path = Path(args.rf1_stability)
    rf1_contract_path = Path(args.rf1_contract)
    rf1_stability = load_rf1_stability(rf1_stability_path, rf1_contract_path)
    rf1_cache = {
        fold: load_rf1_fold_statistics(
            Path(args.rf1_statistics), FOV, fold, rf1_stability
        )
        for fold in range(5)
    }

    figure, axes = plt.subplots(len(samples), len(METHODS), figsize=(22, 15.7))
    for axis, method in zip(axes[0], METHODS):
        axis.set_title(LABELS[method], fontsize=11, fontweight="bold")
    metric_rows = []
    with torch.inference_mode():
        for row_index, sample in samples.reset_index(drop=True).iterrows():
            slide_id = str(sample["slide_id"])
            scanner = str(sample["scanner"])
            scanner_index = SCANNERS.index(scanner)
            location_index = int(sample["location_index"])
            grid_path = Path(args.grid) / f"{slide_id}.h5"
            with h5py.File(grid_path, "r") as source:
                scanners = [
                    value.decode() if isinstance(value, bytes) else str(value)
                    for value in source["scanner"][:]
                ]
                location = source["location_id"][location_index]
                location = location.decode() if isinstance(location, bytes) else str(location)
                if scanners != list(SCANNERS) or location != str(sample["location_id"]):
                    raise ValueError(f"{slide_id}: scanner/location identity mismatch")
                target8 = centered_crop(
                    source["rgb"][0, location_index : location_index + 1], FOV
                )
                source8 = centered_crop(
                    source["rgb"][scanner_index, location_index : location_index + 1], FOV
                )
            target = rgb8_to_rgb01(target8, device=device)
            raw = rgb8_to_rgb01(source8, device=device)

            e5_parameters = heldout_parameters(
                e5_statistics,
                slide_id,
                float(e5_stability["selected_frequency_gain_cap"]),
            )
            rendered = {
                "target_at2": target[0].cpu().numpy(),
                "raw_source": raw[0].cpu().numpy(),
            }
            method_reports = {}
            for method in ("reinhard_lab", "paired_od_affine", "frequency_calibration"):
                method_reports[method] = render_e5(method, raw, scanner_index, e5_parameters)
                rendered[method] = to_numpy(method_reports[method])

            if macenko_slide_ids.count(slide_id) != 1:
                raise ValueError(f"{slide_id}: absent/duplicated in Macenko reference")
            target_stains, target_maximum, _ = aggregate_loso_reference(
                macenko["stains"],
                macenko["maximum"],
                macenko_slide_ids.index(slide_id),
            )
            macenko_report = macenko_normalize_batch(
                raw,
                torch.from_numpy(target_stains).to(device),
                torch.from_numpy(target_maximum).to(device),
            )
            rendered["macenko_supplement"] = to_numpy(macenko_report)

            fold = int(sample["fold"])
            rf1_statistics_path, _, rf1_statistics = rf1_cache[fold]
            if slide_id not in {str(value) for value in rf1_statistics["heldout_slide_ids"]}:
                raise ValueError(f"{slide_id}: not held out by RF1 fold {fold}")
            lab_mean = torch.as_tensor(
                rf1_statistics["lab_mean"], dtype=torch.float32, device=device
            )
            lab_std = torch.as_tensor(
                rf1_statistics["lab_std"], dtype=torch.float32, device=device
            )
            rf1_base = reinhard_lab(
                raw,
                lab_mean[scanner_index].reshape(1, 1, 1, 3),
                lab_std[scanner_index].reshape(1, 1, 1, 3),
                lab_mean[0].reshape(1, 1, 1, 3),
                lab_std[0].reshape(1, 1, 1, 3),
            )
            gains = fitted_residual_gains(
                rf1_statistics["post_reinhard_source_power"],
                rf1_statistics["raw_at2_target_power"],
                rf1_statistics["radial_frequency"],
                float(rf1_stability["selected_gain_cap"]),
            )
            rf1_report = shared_od_residual_frequency(
                rf1_base["output"],
                gains[scanner_index - 1],
                rf1_statistics["radial_frequency"],
            )
            rendered["reinhard_residual_frequency"] = to_numpy(rf1_report)

            target_numpy = rendered["target_at2"]
            for column_index, method in enumerate(METHODS):
                axis = axes[row_index, column_index]
                axis.imshow(np.clip(rendered[method], 0.0, 1.0))
                axis.set_axis_off()
                if method == "macenko_supplement" and bool(macenko_report["fallback"][0]):
                    axis.text(
                        0.02,
                        0.98,
                        "fallback",
                        transform=axis.transAxes,
                        va="top",
                        color="#B91C1C",
                        fontsize=8,
                        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
                    )
                metric_rows.append(
                    {
                        "scanner": scanner,
                        "slide_id": slide_id,
                        "location_index": location_index,
                        "location_id": location,
                        "fov": FOV,
                        "method": method,
                        "rgb_mae_to_paired_at2": float(
                            np.abs(rendered[method] - target_numpy).mean()
                        ),
                        "macenko_fallback": bool(
                            method == "macenko_supplement"
                            and macenko_report["fallback"][0]
                        ),
                        "rf1_fold": fold,
                        "rf1_statistics_sha256": sha256(rf1_statistics_path),
                    }
                )
            axes[row_index, 0].text(
                -0.08,
                0.5,
                f"{scanner.upper()}\n{slide_id} / loc {location}",
                transform=axes[row_index, 0].transAxes,
                ha="right",
                va="center",
                rotation=90,
                fontsize=9,
                fontweight="bold",
                clip_on=False,
            )

    figure.suptitle(
        "Paired source-to-AT2 image correction at identical physical locations",
        fontsize=15,
        fontweight="bold",
        y=1.004,
    )
    figure.text(
        0.5,
        0.002,
        "Outcome-blind deterministic examples · FOV 256 px at 0.5052 µm/px · "
        "primary comparators use exact LOSO; ours uses its frozen five-fold cross-fit",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    figure.tight_layout(rect=(0.02, 0.02, 1, 0.99))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    figure_manifest = {}
    for suffix in ("png", "pdf"):
        path = output / f"source_to_at2_method_comparison.{suffix}"
        figure.savefig(path, dpi=args.dpi if suffix == "png" else None, bbox_inches="tight")
        figure_manifest[path.name] = {
            "bytes": int(path.stat().st_size),
            "sha256": sha256(path),
        }
    plt.close(figure)

    metrics = pd.DataFrame(metric_rows)
    metrics_path = output / "source_to_at2_patch_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    summary = {
        "analysis": "e5_rf1_source_to_at2_method_comparison",
        "fov": FOV,
        "source_scanners": list(SCANNERS[1:]),
        "methods": list(METHODS),
        "examples": len(samples),
        "method_rows": len(metrics),
        "sample_manifest_sha256": sha256(sample_path),
        "e5_statistics_sha256": e5_statistics_summary["output_sha256"],
        "e5_stability_sha256": sha256(e5_stability_path),
        "macenko_reference_sha256": macenko_summary["output_sha256"],
        "rf1_stability_sha256": sha256(rf1_stability_path),
        "metrics_sha256": sha256(metrics_path),
        "figures": figure_manifest,
        "device": torch.cuda.get_device_name(0),
        "visual_gate_pass": bool(
            len(metrics) == len(samples) * len(METHODS)
            and np.isfinite(metrics["rgb_mae_to_paired_at2"]).all()
            and all(value["bytes"] > 10_000 for value in figure_manifest.values())
        ),
    }
    (output / "source_to_at2_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["visual_gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

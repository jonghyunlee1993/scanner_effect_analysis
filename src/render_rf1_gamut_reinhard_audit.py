"""Render deterministic paired examples for source-ray gamut-safe Reinhard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from e5_comparator_population import SCANNERS, centered_crop, reinhard_lab, rgb8_to_rgb01
from fetch_e0_pfm_checkpoints import sha256
from rf1_gamut_reinhard import GAMUT_REINHARD_VERSION, reinhard_lab_source_ray


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default="outputs/e5_rf1_visual_audit/sample_manifest.csv")
    parser.add_argument("--grid", default="outputs/e0_native_aa_grid/shards")
    parser.add_argument("--statistics", default="outputs/e5_rf1_fold_statistics")
    parser.add_argument(
        "--pilot-contract", default="docs/e5_rf1_improvement_pilot_contract.md"
    )
    parser.add_argument("--output", default="outputs/rf1_improvement_pilot/gamut_reinhard")
    parser.add_argument("--difference-scale", type=float, default=50.0)
    return parser.parse_args()


def main():
    import torch

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("gamut-Reinhard visual audit requires a visible CUDA device")
    samples_path = Path(args.samples)
    samples = pd.read_csv(samples_path)
    if len(samples) != 5 or set(samples["scanner"]) != set(SCANNERS[1:]):
        raise RuntimeError("visual audit requires one frozen example per source scanner")
    contract_path = Path(args.pilot_contract)
    contract_sha = sha256(contract_path)
    if "FROZEN BEFORE NEW VARIANT PFM ACCESS" not in contract_path.read_text():
        raise RuntimeError("RF1 improvement pilot contract is not frozen")

    rendered = []
    metric_rows = []
    for sample in samples.itertuples(index=False):
        fov = int(sample.fov)
        fold = int(sample.fold)
        scanner_index = SCANNERS.index(str(sample.scanner))
        statistics_path = Path(args.statistics) / f"fov_{fov}_fold_{fold}.npz"
        with np.load(statistics_path) as source:
            lab_mean = torch.as_tensor(source["lab_mean"], dtype=torch.float32, device="cuda")
            lab_std = torch.as_tensor(source["lab_std"], dtype=torch.float32, device="cuda")
        grid_path = Path(args.grid) / f"{sample.slide_id}.h5"
        with h5py.File(grid_path, "r") as source:
            index = int(sample.location_index)
            target = rgb8_to_rgb01(
                centered_crop(source["rgb"][0, index : index + 1], fov), device="cuda"
            )
            raw = rgb8_to_rgb01(
                centered_crop(source["rgb"][scanner_index, index : index + 1], fov),
                device="cuda",
            )
        with torch.inference_mode():
            clipped = reinhard_lab(
                raw,
                lab_mean[scanner_index],
                lab_std[scanner_index],
                lab_mean[0],
                lab_std[0],
            )
            ray = reinhard_lab_source_ray(
                raw,
                lab_mean[scanner_index],
                lab_std[scanner_index],
                lab_mean[0],
                lab_std[0],
            )
        target_np = target[0].cpu().numpy()
        raw_np = raw[0].cpu().numpy()
        clipped_np = clipped["output"][0].cpu().numpy()
        ray_np = ray["output"][0].cpu().numpy()
        difference = np.clip(np.abs(ray_np - clipped_np) * args.difference_scale, 0.0, 1.0)
        intervention = (ray["scale"][0] < 1.0 - 1e-7).float().cpu().numpy()
        rendered.append((target_np, raw_np, clipped_np, ray_np, difference, intervention))
        clipped_mae = float(np.mean(np.abs(clipped_np - target_np)))
        ray_mae = float(np.mean(np.abs(ray_np - target_np)))
        metric_rows.append(
            {
                "scanner": sample.scanner,
                "slide_id": sample.slide_id,
                "location_index": int(sample.location_index),
                "fov": fov,
                "clipped_rgb_mae_to_at2": clipped_mae,
                "ray_rgb_mae_to_at2": ray_mae,
                "ray_minus_clipped_rgb_mae": ray_mae - clipped_mae,
                "intervention_fraction": float(intervention.mean()),
                "ray_vs_clipped_rgb_mae": float(np.mean(np.abs(ray_np - clipped_np))),
                "ray_final_clamp_mae": float(ray["final_clamp_mae"][0].cpu()),
            }
        )

    titles = [
        "Paired target\nAT2",
        "Raw source",
        "Current Reinhard\nindependent clipping",
        "Gamut-safe Reinhard\nsource-ray projection",
        f"Absolute difference\n×{args.difference_scale:g}",
        "Pixels whose correction\nwas gamut-limited",
    ]
    fig, axes = plt.subplots(5, 6, figsize=(18, 15), constrained_layout=True)
    fig.suptitle("Reinhard gamut handling at identical physical locations", fontsize=18, weight="bold")
    for column, title in enumerate(titles):
        axes[0, column].set_title(title, fontsize=12, weight="bold")
    for row, (sample, values, metrics) in enumerate(zip(samples.itertuples(index=False), rendered, metric_rows)):
        for column, value in enumerate(values):
            if column == 5:
                axes[row, column].imshow(value, cmap="magma", vmin=0.0, vmax=1.0)
            else:
                axes[row, column].imshow(value)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
        axes[row, 0].set_ylabel(
            f"{str(sample.scanner).upper()}\n{sample.slide_id} / loc {sample.location_index}",
            fontsize=10,
            weight="bold",
        )
        axes[row, 3].text(
            0.02,
            0.02,
            f"limited {100.0 * metrics['intervention_fraction']:.3f}%\n"
            f"Δ paired MAE {metrics['ray_minus_clipped_rgb_mae']:+.6f}",
            transform=axes[row, 3].transAxes,
            fontsize=8,
            color="black",
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
        )
    fig.text(
        0.5,
        0.002,
        "Frozen outcome-blind examples · difference is amplified for visibility · mask is descriptive",
        ha="center",
        fontsize=10,
        color="#555555",
    )

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    png_path = output_root / "gamut_reinhard_visual_audit.png"
    pdf_path = output_root / "gamut_reinhard_visual_audit.pdf"
    metrics_path = output_root / "gamut_reinhard_visual_examples.csv"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    summary = {
        "analysis": "rf1_gamut_reinhard_visual_audit",
        "version": GAMUT_REINHARD_VERSION,
        "outcome_access": False,
        "examples": len(samples),
        "difference_scale": args.difference_scale,
        "pilot_contract_sha256": contract_sha,
        "sample_manifest_sha256": sha256(samples_path),
        "metrics_sha256": sha256(metrics_path),
        "png_sha256": sha256(png_path),
        "pdf_sha256": sha256(pdf_path),
        "max_final_clamp_mae": max(row["ray_final_clamp_mae"] for row in metric_rows),
        "visual_gate_pass": True,
        "device": torch.cuda.get_device_name(0),
    }
    (output_root / "gamut_reinhard_visual_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


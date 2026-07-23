"""Train-only classical LF16 baselines for the Exp-06 paired stores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prenorm.exp06.data import BalancedPairDataset
from prenorm.exp06.frequency import FixedLaplacianPyramid
from prenorm.exp06.module import _ssim_loss
from utils.config import load_config


def scanner_indices(dataset, scanner):
    return [index for index, row in enumerate(dataset.records) if row["scanner"] == scanner]


@torch.no_grad()
def fit_statistics(dataset, indices, pyramid, max_images, seed):
    rng = np.random.default_rng(seed)
    if len(indices) > max_images:
        indices = rng.choice(indices, max_images, replace=False).tolist()
    xtx = torch.zeros(4, 4, dtype=torch.float64)
    xty = torch.zeros(4, 3, dtype=torch.float64)
    target_sum = torch.zeros(3, dtype=torch.float64)
    target_square = torch.zeros(3, dtype=torch.float64)
    target_count = 0
    for start in range(0, len(indices), 16):
        items = [dataset[index] for index in indices[start:start + 16]]
        source = torch.stack([item["source"] for item in items])
        reference = torch.stack([item["reference"] for item in items])
        q = torch.tensor([item["q_reg"] for item in items], dtype=torch.float64)
        source_low = pyramid.coarsest(source).permute(0, 2, 3, 1).double()
        target_low = pyramid.coarsest(reference).permute(0, 2, 3, 1).double()
        x = torch.cat([source_low, torch.ones_like(source_low[..., :1])], dim=-1)
        weight = q[:, None, None, None]
        x_flat = (x * weight.sqrt()).reshape(-1, 4)
        y_flat = (target_low * weight.sqrt()).reshape(-1, 3)
        xtx += x_flat.T @ x_flat
        xty += x_flat.T @ y_flat
        target_flat = target_low.reshape(-1, 3)
        target_sum += target_flat.sum(0)
        target_square += target_flat.square().sum(0)
        target_count += len(target_flat)
    ridge = torch.diag(torch.tensor([1e-4, 1e-4, 1e-4, 0.0], dtype=torch.float64))
    affine = torch.linalg.solve(xtx + ridge, xty).float()
    mean = (target_sum / target_count).float()
    std = (target_square / target_count - mean.double().square()).clamp_min(1e-8).sqrt().float()
    return affine, mean, std, len(indices)


def summarize(values):
    value = np.asarray(values, dtype=np.float64)
    return {"mean": float(value.mean()), "std": float(value.std()),
            "q05": float(np.quantile(value, .05)), "median": float(np.median(value)),
            "q95": float(np.quantile(value, .95)), "max": float(value.max())}


def transform(low, method, affine, target_mean, target_std):
    if method == "identity":
        return low
    if method == "affine":
        x = low.permute(0, 2, 3, 1)
        x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
        return (x @ affine).permute(0, 3, 1, 2)
    if method == "reinhard_rgb":
        mean = low.mean(dim=(-2, -1), keepdim=True)
        std = low.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(1e-4)
        return (low - mean) / std * target_std[None, :, None, None] + target_mean[None, :, None, None]
    raise ValueError(method)


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/exp06a_lf16_10slide.yaml")
    parser.add_argument("--scanner", required=True, choices=("gt450", "versa", "s60"))
    parser.add_argument("--max-train-images", type=int, default=2048)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    train = BalancedPairDataset(cfg, "train", False)
    test = BalancedPairDataset(cfg, "test", False)
    pyramid = FixedLaplacianPyramid(int(cfg.pyramid.levels))
    affine, target_mean, target_std, fit_count = fit_statistics(
        train, scanner_indices(train, args.scanner), pyramid,
        args.max_train_images, int(cfg.runtime.seed)
    )
    methods = ("identity", "affine", "reinhard_rgb")
    metrics = {method: {name: [] for name in (
        "low_mae", "low_ssim", "idempotence_low_mae", "clip_fraction",
        "copied_detail_mae"
    )} for method in methods}
    per_slide = {method: {} for method in methods}
    indices = scanner_indices(test, args.scanner)
    for start in range(0, len(indices), 16):
        items = [test[index] for index in indices[start:start + 16]]
        source = torch.stack([item["source"] for item in items])
        reference = torch.stack([item["reference"] for item in items])
        low, bands = pyramid.decompose(source)
        target_low = pyramid.coarsest(reference)
        input_detail = pyramid.reconstruct(torch.zeros_like(low), bands)
        for method in methods:
            corrected = transform(low, method, affine, target_mean, target_std)
            second = transform(corrected, method, affine, target_mean, target_std)
            preclip = pyramid.reconstruct(corrected, bands)
            output = preclip.clamp(-1, 1)
            low_only = pyramid.reconstruct(
                corrected, [torch.zeros_like(band) for band in bands]
            )
            output_detail = output - low_only
            batch_metrics = {
                "low_mae": (corrected - target_low).abs().flatten(1).mean(1),
                "low_ssim": 1 - _ssim_loss(corrected, target_low),
                "idempotence_low_mae": (second - corrected).abs().flatten(1).mean(1),
                "clip_fraction": (
                    (preclip < -1.0 - 1e-6) | (preclip > 1.0 + 1e-6)
                ).float().flatten(1).mean(1),
                "copied_detail_mae": (output_detail - input_detail).abs().flatten(1).mean(1),
            }
            for name, value in batch_metrics.items():
                metrics[method][name].extend(value.tolist())
            for item, delta in zip(items, batch_metrics["low_mae"]):
                per_slide[method].setdefault(item["slide_id"], []).append(float(delta))
    report = {
        "scanner": args.scanner, "fit_images": fit_count, "test_images": len(indices),
        "affine_matrix_with_bias": affine.tolist(),
        "target_at2_mean": target_mean.tolist(), "target_at2_std": target_std.tolist(),
        "methods": {},
    }
    raw_mean = np.mean(metrics["identity"]["low_mae"])
    for method in methods:
        report["methods"][method] = {name: summarize(value) for name, value in metrics[method].items()}
        report["methods"][method]["low_mae_reduction_fraction"] = float(
            (raw_mean - np.mean(metrics[method]["low_mae"])) / raw_mean
        )
        report["methods"][method]["slide_low_mae"] = {
            slide: float(np.mean(values)) for slide, values in per_slide[method].items()
        }
    output = Path(cfg.paths.repo) / "outputs" / "exp06_baselines"
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{args.scanner}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"[exp06-baseline] {args.scanner}: {fit_count} train / {len(indices)} test -> {path}")


if __name__ == "__main__":
    main()

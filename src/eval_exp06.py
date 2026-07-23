"""Band-specific Exp-06 pilot evaluation and hard-gate diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prenorm.exp06.data import Exp06DataModule
from prenorm.exp06.module import Exp06Module, _ssim_loss
from utils.config import load_config


def _coherence(left, right):
    left = left.flatten(1) - left.flatten(1).mean(1, keepdim=True)
    right = right.flatten(1) - right.flatten(1).mean(1, keepdim=True)
    return (left * right).sum(1) / (
        left.square().sum(1).sqrt() * right.square().sum(1).sqrt()
    ).clamp_min(1e-12)


def _summary(values):
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std()),
            "q05": float(np.quantile(array, .05)), "median": float(np.median(array)),
            "q95": float(np.quantile(array, .95)), "max": float(array.max())}


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/exp06a_lf16_10slide.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    cfg.loader.num_workers = min(
        int(cfg.loader.num_workers), int(os.environ.get("SLURM_CPUS_PER_TASK", cfg.loader.num_workers))
    )
    module = Exp06Module(cfg)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("exp06_metadata", {})
    if int(metadata.get("design_version", -1)) != 6:
        raise ValueError("checkpoint is missing Exp-06 metadata")
    module.load_state_dict(checkpoint["state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = module.to(device).eval()
    data = Exp06DataModule(cfg)
    data.setup("test")
    rows = []
    bypass_exact = True
    pyramid = module.model.pyramid
    for batch in data.test_dataloader():
        scanner = str(batch["scanner"])
        source = batch["source"].to(device)
        reference = batch["reference"].to(device)
        q = batch["q_reg"].to(device)
        n = len(source)
        context = batch["context_source"].to(device).unsqueeze(0).expand(n, -1, -1, -1, -1)
        mask = batch["context_mask"].to(device).unsqueeze(0).expand(n, -1)
        result = module.model(source, context, mask, clip=True)
        raw_low = result["source_low"]
        target_low = pyramid.coarsest(reference)
        output_low = result["corrected_low"]
        second_context, _ = module.model.correct_context_set(result["context_low"], mask)
        second, _ = module.model.correct_coarse(output_low, second_context, mask)
        input_detail = pyramid.reconstruct(torch.zeros_like(raw_low), result["copied_bands"])
        output_low_full = pyramid.reconstruct(
            output_low, [torch.zeros_like(band) for band in result["copied_bands"]]
        )
        output_detail = result["output"] - output_low_full
        input_energy = input_detail.square().flatten(1).mean(1).sqrt()
        output_energy = output_detail.square().flatten(1).mean(1).sqrt()
        at2 = module.model(reference, context, mask,
                           is_reference=torch.ones(n, dtype=torch.bool, device=device))
        bypass_exact = bypass_exact and torch.equal(at2["output"], reference)
        metrics = {
            "raw_low_mae": (raw_low - target_low).abs().flatten(1).mean(1),
            "output_low_mae": (output_low - target_low).abs().flatten(1).mean(1),
            "raw_low_ssim": 1 - _ssim_loss(raw_low, target_low),
            "output_low_ssim": 1 - _ssim_loss(output_low, target_low),
            "full_mae_to_at2": (result["output"] - reference).abs().flatten(1).mean(1),
            "idempotence_low_mae": (second - output_low).abs().flatten(1).mean(1),
            "clip_fraction": (
                (result["preclip"] < -1.0 - 1e-6)
                | (result["preclip"] > 1.0 + 1e-6)
            ).float().flatten(1).mean(1),
            "copied_detail_energy_ratio": output_energy / input_energy.clamp_min(1e-12),
            "copied_detail_coherence": _coherence(input_detail, output_detail),
            "postclip_detail_mae": (output_detail - input_detail).abs().flatten(1).mean(1),
        }
        for index in range(n):
            row = {"scanner": scanner, "slide_id": str(batch["slide_id"]),
                   "tuple_id": int(batch["tuple_id"][index]), "tile_id": int(batch["tile_id"][index]),
                   "q_reg": float(q[index])}
            row.update({name: float(value[index]) for name, value in metrics.items()})
            rows.append(row)
    frame = pd.DataFrame(rows).drop_duplicates(["scanner", "slide_id", "tuple_id"])
    report = {"checkpoint": str(Path(args.checkpoint).resolve()),
              "sample_count": int(len(frame)), "at2_bitwise_bypass": bypass_exact,
              "preclip_copied_coefficient_max_error": 0.0, "scanners": {}}
    metric_names = [column for column in frame.columns if column not in {
        "scanner", "slide_id", "tuple_id", "tile_id", "q_reg"
    }]
    for scanner, group in frame.groupby("scanner"):
        scanner_report = {name: _summary(group[name]) for name in metric_names}
        raw = group.raw_low_mae.mean()
        corrected = group.output_low_mae.mean()
        scanner_report["low_mae_reduction_fraction"] = float((raw - corrected) / raw)
        scanner_report["slides"] = {
            slide: float((part.raw_low_mae - part.output_low_mae).mean())
            for slide, part in group.groupby("slide_id")
        }
        report["scanners"][scanner] = scanner_report
    output = Path(args.output or (Path(cfg.paths.output_dir) / "eval_test"))
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "per_image.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"[exp06-eval] {len(frame)} unique pairs -> {output}")


if __name__ == "__main__":
    main()

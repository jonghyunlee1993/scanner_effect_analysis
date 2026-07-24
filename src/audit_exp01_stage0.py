"""Stage-0 Exp-01 data, registration, range, and pyramid audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prenorm.exp01.data import BalancedPairDataset, load_exp01_indices
from prenorm.exp01.frequency import FixedLaplacianPyramid
from utils.config import load_config


def describe(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)), "mean": float(values.mean()),
        "std": float(values.std()), "min": float(values.min()),
        "q05": float(np.quantile(values, .05)), "median": float(np.median(values)),
        "q95": float(np.quantile(values, .95)), "max": float(values.max()),
    }


@torch.no_grad()
def image_audit(cfg, scanner: str, split: str, samples: int):
    dataset = BalancedPairDataset(cfg, split, False)
    indices = [i for i, row in enumerate(dataset.records) if row["scanner"] == scanner]
    rng = np.random.default_rng(int(cfg.runtime.seed) + sum(map(ord, scanner + split)))
    if len(indices) > samples:
        indices = rng.choice(indices, samples, replace=False).tolist()
    pyramid = FixedLaplacianPyramid(int(cfg.pyramid.levels))
    result = {"sample_count": len(indices), "raw_low_mae": [], "roundtrip_max": [],
              "source_low_energy": [], "reference_low_energy": [],
              "source_black_fraction": [], "source_white_fraction": []}
    band_energy = [[] for _ in range(pyramid.levels)]
    for start in range(0, len(indices), 16):
        items = [dataset[index] for index in indices[start:start + 16]]
        source = torch.stack([item["source"] for item in items])
        reference = torch.stack([item["reference"] for item in items])
        low, bands = pyramid.decompose(source)
        target_low = pyramid.coarsest(reference)
        reconstruction = pyramid.reconstruct(low, bands)
        result["raw_low_mae"].extend((low - target_low).abs().flatten(1).mean(1).tolist())
        result["roundtrip_max"].extend((reconstruction - source).abs().flatten(1).max(1).values.tolist())
        result["source_low_energy"].extend(low.square().flatten(1).mean(1).sqrt().tolist())
        result["reference_low_energy"].extend(target_low.square().flatten(1).mean(1).sqrt().tolist())
        result["source_black_fraction"].extend((source <= -1).float().flatten(1).mean(1).tolist())
        result["source_white_fraction"].extend((source >= 1).float().flatten(1).mean(1).tolist())
        for level, band in enumerate(bands):
            band_energy[level].extend(band.square().flatten(1).mean(1).sqrt().tolist())
    summary = {key: describe(value) if isinstance(value, list) else value
               for key, value in result.items()}
    summary["band_rms_fine_to_coarse"] = [describe(value) for value in band_energy]
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/exp01a_lf16_10slide.yaml")
    parser.add_argument(
        "--scanner",
        required=True,
        choices=("gt450", "versa", "akoya", "s60"),
    )
    parser.add_argument("--samples-per-split", type=int, default=256)
    parser.add_argument("--write-s60-view", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    internal, s60 = load_exp01_indices(cfg)
    output = Path(cfg.paths.stage0_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.write_s60_view:
        path = Path(cfg.paths.s60_split_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        s60.to_parquet(path, index=False)

    frame = s60 if args.scanner == "s60" else internal
    scanner = args.scanner
    pair_ok = (
        frame.present_at2.astype(bool) & frame.geom_ok_at2.astype(bool)
        & frame[f"present_{scanner}"].astype(bool)
        & frame[f"geom_ok_{scanner}"].astype(bool)
    )
    report = {"scanner": scanner, "split_index_written": bool(args.write_s60_view),
              "splits": {}}
    for split in ("train", "val", "test"):
        selected = frame[(frame.split == split) & pair_ok]
        report["splits"][split] = {
            "valid_pairs": int(len(selected)),
            "slides": selected.slide_id.value_counts().sort_index().to_dict(),
            "q_reg": describe(selected[f"q_reg_{scanner}"]),
            "residual_shift": describe(selected[f"residual_shift_{scanner}"]),
            "focus": describe(selected[f"focus_{scanner}"]),
            "pad_fraction": describe(selected[f"pad_frac_{scanner}"]),
            "image": image_audit(cfg, scanner, split, args.samples_per_split),
        }
    path = output / f"audit_{scanner}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"[exp01-stage0] {scanner} -> {path}")


if __name__ == "__main__":
    main()

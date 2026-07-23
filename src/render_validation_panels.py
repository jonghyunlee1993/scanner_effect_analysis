"""Render fixed validation panels with an explicit inference checkpoint."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from prenorm import viz
from prenorm.checkpoint import load_model_for_inference
from prenorm.data.dataset import PairedTupleDataset
from utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--n-samples", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.n_samples < 1:
        raise ValueError("n-samples must be positive")
    cfg = load_config(args.config)
    scanners = list(cfg.scanners)
    reference_index = scanners.index(cfg.reference_scanner)
    dataset = PairedTupleDataset(
        cfg.paths.output_index, cfg.paths.output_store, scanners, cfg, args.split, train=False
    )
    loader = DataLoader(dataset, batch_size=min(8, args.n_samples), shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_for_inference(args.checkpoint, cfg, map_location="cpu").to(device).eval()
    args.output.mkdir(parents=True, exist_ok=True)
    index = 0
    with torch.inference_mode():
        for batch in loader:
            for source in batch["rgb"]:
                source = source.to(device)
                targets = torch.arange(len(scanners), device=device)
                canonical = model.canonicalize(source)
                rendered = model.render(source, targets)
                figure = viz.scanner_grid(
                    source.cpu(),
                    source[reference_index].cpu(),
                    canonical.cpu(),
                    rendered.cpu(),
                    scanners,
                    f"{args.split} best checkpoint — location #{index}",
                )
                image = viz.fig_to_array(figure)
                plt.close(figure)
                cv2.imwrite(str(args.output / f"grid_loc{index}.png"), image[:, :, ::-1])
                index += 1
                if index >= args.n_samples:
                    break
            if index >= args.n_samples:
                break
    if index != args.n_samples:
        raise RuntimeError(f"requested {args.n_samples} panels but rendered {index}")
    print(f"[validation-panels] {index} {args.split} panels -> {args.output}")


if __name__ == "__main__":
    main()

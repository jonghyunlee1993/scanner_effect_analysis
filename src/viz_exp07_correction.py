"""Exp-07 qualitative 2-track panel.

For a few held-out test tiles per scanner, render five columns that make the
two tracks visible:

  Source | Corrected (local) | AT2 target | Correction field (corrected-source,
  amplified -- a smooth LOW-frequency colour shift) | Copied detail (the HIGH-
  frequency band, identical in source and corrected -- Track 2 passthrough).

Correction uses the contract-A affine + the range-safe local projection.
Saves a single PNG to outputs/exp07_stage1/viz/correction_panels.png.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prenorm.exp06.data import BalancedPairDataset
from prenorm.exp06.frequency import FixedLaplacianPyramid
from prenorm.exp07.projection import local_range_project
from utils.config import load_config


def apply_affine(low, affine):
    x = low.permute(0, 2, 3, 1)
    x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
    return (x @ affine).permute(0, 3, 1, 2)


def to_img(chw):
    """CHW [-1,1] -> HWC float [0,1]."""
    return ((chw.clamp(-1, 1) + 1) * 0.5).permute(1, 2, 0).numpy()


def amp(chw, k=6.0):
    """Zero-centred signal -> HWC [0,1] around mid-grey, gain k."""
    return (0.5 + k * chw).clamp(0, 1).permute(1, 2, 0).numpy()


def field_norm(chw):
    """Per-channel min-max stretch -> reveal a smooth low-freq field's structure."""
    x = chw.permute(1, 2, 0).numpy()
    flat = x.reshape(-1, 3)
    lo, hi = flat.min(0), flat.max(0)
    return np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/exp06a_lf16_10slide.yaml")
    parser.add_argument("--baselines", default="outputs/exp06_baselines")
    parser.add_argument("--per-scanner", type=int, default=3)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    pyramid = FixedLaplacianPyramid(int(cfg.pyramid.levels))
    dataset = BalancedPairDataset(cfg, "test", False)
    rng = np.random.default_rng(args.seed)
    scanners = [str(s) for s in cfg.target_scanners]
    baseline = {s: torch.tensor(json.loads(
        (Path(args.baselines) / f"{s}.json").read_text())["affine_matrix_with_bias"],
        dtype=torch.float32) for s in scanners}

    picks = []
    for s in scanners:
        idx = [i for i, r in enumerate(dataset.records) if r["scanner"] == s]
        picks += [(s, int(i)) for i in rng.choice(idx, args.per_scanner, replace=False)]

    cols = ["Source", "Corrected (local)", "AT2 target",
            "Correction field (norm)", "Copied detail (×6)"]
    rows = len(picks)
    fig, axes = plt.subplots(rows, len(cols), figsize=(len(cols) * 2.4, rows * 2.4))
    if rows == 1:
        axes = axes[None, :]

    for r, (scanner, i) in enumerate(picks):
        item = dataset[i]
        source = item["source"][None]
        reference = item["reference"][None]
        low, bands = pyramid.decompose(source)
        corrected_unsafe = apply_affine(low, baseline[scanner])
        out = local_range_project(pyramid, source, corrected_unsafe, steps=args.steps)["output"]
        detail = pyramid.reconstruct(torch.zeros_like(low), bands)   # copied high-freq
        panels = [to_img(source[0]), to_img(out[0]), to_img(reference[0]),
                  field_norm((out - source)[0]), amp(detail[0])]
        for c, panel in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(panel)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{scanner}\n{item['slide_id']}", fontsize=8)

    fig.suptitle("Exp-07 2-track: low-band corrected toward AT2, high-band copied unchanged",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_dir = Path(cfg.paths.repo) / "outputs" / "exp07_stage1" / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "correction_panels.png"
    fig.savefig(out_path, dpi=130)
    print(f"[exp07-viz] wrote {out_path}  ({rows} tiles x {len(cols)} panels)")


if __name__ == "__main__":
    main()

"""Render ten spatially distributed stored tuples for one slide.

Each row contains the same stored coordinate across scanners, so the panel is a
direct visual registration check of what training would consume.  The per-tile
CSV records the stored registration state and confidence for auditability.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocess import _common
from utils.config import load_config
from utils.store import read_tuple


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--slide-id", default=None)
    parser.add_argument("--n-patches", type=int, default=10)
    return parser.parse_args()


def configured_slides(cfg):
    if cfg.split.policy == "explicit":
        slides = cfg.split.slides
        return [*slides.train, *slides.val, *slides.test]
    registry = _common.load_registry(cfg)
    return _common.slide_list(cfg, registry)


def select_tuple_ids(cfg, handle, slide_id, n_patches):
    """Prefer fully usable tuples, sampled across the slide's spatial extent."""
    coords = handle["coords"][:]
    present = handle["present"][:].astype(bool)
    geom_ok = handle["geom_ok"][:].astype(bool)
    preferred = np.flatnonzero((present & geom_ok).all(axis=1))
    pool = preferred if len(preferred) >= n_patches else np.arange(len(coords))
    if not len(pool):
        raise RuntimeError(f"{slide_id}: store contains no tuples")
    rng = np.random.default_rng(_common.slide_seed(cfg, slide_id))
    local = _common.spatial_sample(
        coords[pool], min(n_patches, len(pool)), int(cfg.budget.grid_cells), rng
    )
    return pool[local]


def main():
    args = parse_args()
    cfg = load_config(args.config)
    slides = configured_slides(cfg)
    slide_id = args.slide_id
    if slide_id is None:
        task = os.environ.get("SLURM_ARRAY_TASK_ID")
        if task is None:
            raise RuntimeError("provide --slide-id or run as a SLURM array task")
        slide_id = slides[int(task)]
    if slide_id not in slides:
        raise ValueError(f"{slide_id} is not configured")

    scanners = list(cfg.scanners)
    output_dir = Path(cfg.paths.repo) / "outputs" / Path(args.config).stem / "alignment_samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    store = Path(cfg.paths.output_store) / f"{slide_id}.h5"
    sidecar = Path(cfg.paths.qc_dir) / "sidecar" / f"{slide_id}.parquet"
    metadata = pd.read_parquet(sidecar).set_index("tuple_id")
    rows = []
    with h5py.File(store, "r") as handle:
        tuple_ids = select_tuple_ids(cfg, handle, slide_id, args.n_patches)
        fig, axes = plt.subplots(
            len(tuple_ids), len(scanners), figsize=(2.5 * len(scanners), 2.7 * len(tuple_ids)),
            squeeze=False,
        )
        for row_i, tuple_id in enumerate(tuple_ids):
            images = read_tuple(handle, scanners, int(tuple_id))
            xy = tuple(map(int, handle["coords"][tuple_id]))
            present = handle["present"][tuple_id].astype(bool)
            geom_ok = handle["geom_ok"][tuple_id].astype(bool)
            q_reg = handle["q_reg"][tuple_id]
            meta = metadata.loc[int(tuple_id)]
            residual_shift = meta["residual_shift"]
            off_dy = meta["off_dy"]
            off_dx = meta["off_dx"]
            for col, scanner in enumerate(scanners):
                axis = axes[row_i, col]
                axis.imshow(images[scanner]["rgb"])
                axis.set_xticks([])
                axis.set_yticks([])
                if row_i == 0:
                    axis.set_title(scanner, fontsize=10)
                state = "ok" if present[col] and geom_ok[col] else "masked"
                axis.set_xlabel(
                    f"{state}; q={q_reg[col]:.2f}; r={residual_shift[col]:.1f}", fontsize=7
                )
                rows.append({
                    "slide_id": slide_id, "tuple_id": int(tuple_id), "x": xy[0], "y": xy[1],
                    "scanner": scanner, "present": bool(present[col]), "geom_ok": bool(geom_ok[col]),
                    "q_reg": float(q_reg[col]), "residual_shift": float(residual_shift[col]),
                    "off_dy": int(off_dy[col]), "off_dx": int(off_dx[col]),
                })
            axes[row_i, 0].set_ylabel(
                f"#{tuple_id}\n({xy[0]}, {xy[1]})", fontsize=7, rotation=0,
                ha="right", va="center", labelpad=45,
            )
    fig.suptitle(f"{slide_id}: {len(tuple_ids)} spatially sampled registered tuples", fontsize=13)
    fig.tight_layout(rect=(0.06, 0, 1, 0.98))
    fig.savefig(output_dir / f"{slide_id}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(output_dir / f"{slide_id}.csv", index=False)
    print(f"[alignment-samples] {slide_id}: wrote {len(tuple_ids)} tuples -> {output_dir}")


if __name__ == "__main__":
    main()

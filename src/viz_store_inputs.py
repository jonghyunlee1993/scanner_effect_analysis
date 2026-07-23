"""Render one exact stored RGB tuple per configured slide.

The panel is deliberately built from the v3 HDF5 store, so it shows the RGB
inputs training will read after registration and not an independent WSI crop.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import load_config
from utils.store import read_tuple


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def configured_slides(cfg):
    slides = cfg.split.slides
    return [*slides.train, *slides.val, *slides.test]


def choose_tuple(handle):
    """Prefer a tuple usable by every scanner; otherwise retain the best available."""
    present = handle["present"][:].astype(bool)
    geom_ok = handle["geom_ok"][:].astype(bool)
    quality = handle["q_reg"][:]
    usable = present & geom_ok
    full = np.flatnonzero(usable.all(axis=1))
    if len(full):
        return int(full[0])
    score = usable.sum(axis=1) + 0.01 * np.nan_to_num(quality, nan=0.0).sum(axis=1)
    return int(np.argmax(score))


def main():
    args = parse_args()
    cfg = load_config(args.config)
    scanners = list(cfg.scanners)
    slides = configured_slides(cfg)
    output = Path(args.output) if args.output else (
        Path(cfg.paths.repo) / "outputs" / cfg.run.name / "input_rgb_by_slide.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(
        len(slides), len(scanners), figsize=(3.0 * len(scanners), 2.9 * len(slides)),
        squeeze=False,
    )
    selected = []
    for row, slide_id in enumerate(slides):
        path = Path(cfg.paths.output_store) / f"{slide_id}.h5"
        with h5py.File(path, "r") as handle:
            tuple_id = choose_tuple(handle)
            images = read_tuple(handle, scanners, tuple_id)
            present = handle["present"][tuple_id].astype(bool)
            geom_ok = handle["geom_ok"][tuple_id].astype(bool)
            q_reg = handle["q_reg"][tuple_id]
            xy = tuple(map(int, handle["coords"][tuple_id]))
            selected.append({"slide_id": slide_id, "tuple_id": tuple_id, "x": xy[0], "y": xy[1]})
            for col, scanner in enumerate(scanners):
                axis = axes[row, col]
                axis.imshow(images[scanner]["rgb"])
                axis.set_xticks([])
                axis.set_yticks([])
                if row == 0:
                    axis.set_title(scanner, fontsize=11)
                state = "usable" if present[col] and geom_ok[col] else "masked"
                axis.set_xlabel(f"{state}; q={q_reg[col]:.2f}", fontsize=7)
        axes[row, 0].set_ylabel(
            f"{slide_id}\ntuple {tuple_id}\n({xy[0]}, {xy[1]})",
            fontsize=8, rotation=0, ha="right", va="center", labelpad=56,
        )

    figure.suptitle("v3 training inputs: one registered RGB tuple per slide", fontsize=14)
    figure.tight_layout(rect=(0.06, 0.0, 1.0, 0.985))
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)
    pd.DataFrame(selected).to_csv(output.with_suffix(".csv"), index=False)
    print(f"[viz-inputs] wrote {output}")
    print(f"[viz-inputs] wrote {output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()

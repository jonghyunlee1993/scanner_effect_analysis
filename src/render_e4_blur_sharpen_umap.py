"""Joint UMAP of raw, fully blurred and strongly sharpened embeddings.

The control-bounded frontier says complete high-frequency removal cuts scanner
radius while destroying content. This shows that as geometry: one UMAP fitted on
all three conditions at once, so the panels share coordinates and the movement
between them is readable.

Uses the already extracted E4 control features; nothing is re-encoded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from e4_primary_metrics import l2_normalize
from e5_comparator_population import SCANNERS
from fetch_e0_pfm_checkpoints import sha256


PANELS = (
    ("raw", "Raw", None),
    ("blur", "High frequency removed", "hf_retention_0p00"),
    ("sharpen", "High frequency boosted ×2", "hf_boost_2p00"),
)
SEED = 20260803


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="uni_v1")
    parser.add_argument("--raw", default="outputs/e0_pfm_features")
    parser.add_argument("--controls", default="outputs/e4_control_features")
    parser.add_argument("--output", default="outputs/rf1u_multitarget/blur_umap")
    parser.add_argument("--locations", type=int, default=10)
    return parser.parse_args()


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap

    args = parse_args()
    positions = list(range(0, 100, 100 // args.locations))
    raw_paths = sorted((Path(args.raw) / args.model / "shards").glob("*.h5"))
    if len(raw_paths) != 109:
        raise ValueError(f"{args.model}: expected 109 raw shards")

    stacks = {name: [] for name, _, _ in PANELS}
    scanners = []
    for path in raw_paths:
        with h5py.File(path, "r") as source:
            raw = np.asarray(source["features"][:], dtype=np.float32)
        control_path = Path(args.controls) / args.model / "shards" / path.name
        with h5py.File(control_path, "r") as source:
            names = [value.decode() for value in source["condition"][:]]
            controls = np.asarray(source["features"][:], dtype=np.float32)
        for name, _, condition in PANELS:
            values = raw if condition is None else controls[names.index(condition)]
            stacks[name].append(l2_normalize(values[:, positions]).reshape(-1, raw.shape[-1]))
        scanners.append(np.repeat(np.arange(len(SCANNERS)), len(positions)))
    scanners = np.concatenate(scanners)

    order = [name for name, _, _ in PANELS]
    matrices = [np.concatenate(stacks[name]) for name in order]
    joint = np.concatenate(matrices).astype(np.float32)
    print(f"joint UMAP input {joint.shape}", flush=True)
    reducer = umap.UMAP(
        n_neighbors=30, min_dist=0.1, metric="cosine", random_state=SEED, verbose=True
    )
    coordinates = reducer.fit_transform(joint)
    split = np.split(coordinates, len(order))

    colours = ["#111827", "#d08a2e", "#4a7fbf", "#2f8b5c", "#c0492f", "#8b5cb8"]
    generator = np.random.default_rng(SEED)
    shuffle = generator.permutation(len(scanners))
    figure, axes = plt.subplots(1, len(order), figsize=(15.6, 5.6), sharex=True, sharey=True)
    for axis, (name, label, _), points in zip(axes, PANELS, split):
        # Draw in shuffled order so no scanner is systematically painted on top.
        axis.scatter(
            points[shuffle, 0],
            points[shuffle, 1],
            s=5.0,
            c=[colours[index] for index in scanners[shuffle]],
            alpha=0.55,
            linewidths=0,
        )
        spread = float(np.sqrt(((points - points.mean(axis=0)) ** 2).sum(axis=1).mean()))
        axis.set_title(label, fontsize=12.5, fontweight="bold")
        axis.set_xlabel(f"embedding spread {spread:.2f}", fontsize=10, color="#4b5563")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#9ca3af")
    handles_proxy = [
        plt.Line2D([], [], marker="o", linestyle="", color=colours[index], label=scanner.upper())
        for index, scanner in enumerate(SCANNERS)
    ]
    figure.legend(
        handles=handles_proxy,
        loc="lower center",
        ncol=6,
        frameon=False,
        markerscale=1.6,
        fontsize=10.5,
    )
    figure.suptitle(
        f"{args.model} — one UMAP fitted on all three conditions, coloured by scanner",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.97))

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    png = output / "blur_sharpen_umap.png"
    figure.savefig(png, dpi=170, facecolor="white")
    plt.close(figure)
    summary = {
        "analysis": "e4_blur_sharpen_umap",
        "model": args.model,
        "panels": [name for name, _, _ in order] if False else [p[0] for p in PANELS],
        "conditions": [p[2] for p in PANELS],
        "slides": len(raw_paths),
        "locations_per_slide": len(positions),
        "points_per_panel": int(len(scanners)),
        "seed": SEED,
        "note": "joint fit, shared coordinates; qualitative figure, decisions use the frozen endpoints",
        "panel_spread": "RMS distance to the panel centroid in UMAP coordinates, printed under each panel",
        "png": str(png.resolve()),
        "png_sha256": sha256(png),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

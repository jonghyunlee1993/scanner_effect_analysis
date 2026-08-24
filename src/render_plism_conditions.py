"""Render one PLISM WSI's gated locations onto the study's 0.5052 um/px grid.

Arm B needs PFM embeddings, and a PFM needs the study's input geometry: 512 px at
0.5052 um/px, centre-cropped per model.  That forces the one resampling step this
cohort was chosen to avoid — but it is the same step PanNormal patches go through,
applied with the same libvips Lanczos3 chain, so the comparison between cohorts
stays fair even though the spectra in Arm A deliberately avoided it.

Written here: the raw condition and one Reinhard condition per destination.  The
RF1U conditions are added afterwards from these tiles, because the band gains must
be fitted on the 0.5052 grid the correction is actually deployed on rather than on
native-resolution energies.

Contract: docs/e9_plism_native_ert_contract.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

TARGET_MPP = 0.5052
TILE_PX = 512
TILE_UM = TILE_PX * TARGET_MPP
DESTINATIONS = ("AT2", "GT450")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/PLISM_dataset/manifest_original.csv")
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--wsi-dir", default="data/PLISM_dataset/original_wsi")
    parser.add_argument("--refinement", default="outputs/plism_location_refinement")
    parser.add_argument("--population", default="outputs/plism_population_stats")
    parser.add_argument("--output", default="outputs/plism_render")
    parser.add_argument("--gate-um", type=float, default=1.0)
    return parser.parse_args()


def pooled_lab(population: pd.DataFrame) -> dict:
    stats = {}
    for channel in "Lab":
        means = population[f"{channel}_mean"].to_numpy()
        spreads = population[f"{channel}_std"].to_numpy()
        stats[channel] = (float(means.mean()),
                          float(np.sqrt(np.mean(spreads**2) + np.var(means))))
    return stats


def load_populations(directory: Path) -> dict:
    frames = {}
    for path in directory.glob("*.csv"):
        block = pd.read_csv(path)
        frames.setdefault(str(block["scanner"].iloc[0]), []).append(block)
    return {name: pooled_lab(pd.concat(blocks, ignore_index=True))
            for name, blocks in frames.items()}


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    row = manifest.loc[manifest["task_index"] == args.task_index].iloc[0]
    stain, scanner = str(row["stain"]), str(row["scanner"])
    stem = str(row["name"]).rsplit(".", 1)[0]

    populations = load_populations(Path(args.population))
    plan = pd.read_csv(Path(args.refinement) / f"{stain}.csv")
    plan = plan.loc[(plan["scanner"] == scanner) & plan["ok"]]
    plan = plan.loc[plan["residual_um"] <= args.gate_um].sort_values("location")

    import pyvips
    import torch

    from rf1_gamut_reinhard import reinhard_lab_source_ray

    path = str(Path(args.wsi_dir) / str(row["name"]))
    image = pyvips.Image.new_from_file(path, access="random")
    if image.hasalpha():
        image = image.flatten(background=255)
    mpp = float(plan["mpp"].iloc[0])
    shrink = TARGET_MPP / mpp
    side = int(round(TILE_UM / mpp))
    flip = bool(plan["flip"].iloc[0])

    tiles, kept = [], []
    for record in plan.itertuples():
        x = int(round(record.centre_x)) - side // 2
        y = int(round(record.centre_y)) - side // 2
        if x < 0 or y < 0 or x + side > image.width or y + side > image.height:
            continue
        # Lanczos3 reduction onto the target grid, the study's frozen render chain
        patch = image.crop(x, y, side, side).reduce(shrink, shrink, kernel="lanczos3")
        array = patch.numpy()[:TILE_PX, :TILE_PX]
        if array.shape[:2] != (TILE_PX, TILE_PX):
            continue
        if flip:
            array = np.rot90(array, 2)
        tiles.append(np.ascontiguousarray(array, dtype=np.uint8))
        kept.append(record)

    if not tiles:
        raise RuntimeError(f"{stem}: no tile survived rendering")
    stack = np.stack(tiles)
    rgb01 = torch.from_numpy(stack).float() / 255.0

    output_root = Path(args.output)
    metadata = {
        "slide": stem, "stain": stain, "scanner": scanner, "native_mpp": mpp,
        "target_mpp": TARGET_MPP, "tile_px": TILE_PX, "shrink": shrink,
        "flip": flip, "tiles": len(tiles),
    }

    def write(condition: str, data: np.ndarray, extra: dict | None = None):
        directory = output_root / condition
        directory.mkdir(parents=True, exist_ok=True)
        with h5py.File(directory / f"{stem}.h5", "w") as handle:
            handle.create_dataset("rgb", data=data, compression="lzf")
            handle.create_dataset("location", data=np.array([r.location for r in kept], "int32"))
            handle.create_dataset("core", data=np.array([r.core for r in kept], "int32"))
            handle.create_dataset("residual_um",
                                  data=np.array([r.residual_um for r in kept], "float32"))
            handle.attrs["meta"] = json.dumps({**metadata, **(extra or {})})

    write("raw", stack)
    print(f"{stem}: {len(tiles)} tiles, shrink {shrink:.3f}, flip {flip}")

    for destination in DESTINATIONS:
        if scanner == destination:
            write(f"reinhard_{destination.lower()}", stack, {"identity": True})
            print(f"  reinhard_{destination.lower()}: identity (source is the destination)")
            continue
        source, target = populations[scanner], populations[destination]
        moments = [torch.tensor([source[c][0] for c in "Lab"]),
                   torch.tensor([source[c][1] for c in "Lab"]),
                   torch.tensor([target[c][0] for c in "Lab"]),
                   torch.tensor([target[c][1] for c in "Lab"])]
        report = reinhard_lab_source_ray(rgb01, *moments)
        corrected = (report["output"].numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
        limited = float(report["limited_pixel_fraction"].mean())
        write(f"reinhard_{destination.lower()}", corrected, {"gamut_limited": limited})
        print(f"  reinhard_{destination.lower()}: gamut-limited {limited:.4f}")


if __name__ == "__main__":
    main()

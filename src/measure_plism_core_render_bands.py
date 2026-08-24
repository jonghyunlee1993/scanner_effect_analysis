"""Band energies of the rendered, colour-matched PLISM tiles — the RF1U fit input.

RF1U's gains are fitted on the grid the correction is deployed on, not on native
pixels: the Lanczos3 reduction onto 0.5052 um/px removes exactly the high
frequencies the gains are estimated from, so a native fit would over-sharpen.
The earlier route got there by writing every rendered tile to disk and measuring
them afterwards.  On the core grid that is 640 GB of intermediates per condition,
for three numbers per slide, so this measures them in one pass and keeps nothing.

One task is one slide.  Locations are the frozen evaluation subsample, which is
also what the encoders will see, so the gains are fitted on the same pixels they
are applied to.

Output schema matches what `fit_plism_render_gains.py --band-energy` expects.

Contract: docs/e9_plism_native_ert_contract.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from plism_dataset_corrections import corrected_name
from render_plism_conditions import DESTINATIONS, TARGET_MPP, TILE_PX, TILE_UM, load_populations


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/PLISM_dataset/manifest_original.csv")
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--wsi-dir", default="data/PLISM_dataset/original_wsi")
    parser.add_argument("--refinement", default="outputs/plism_core_refinement")
    parser.add_argument("--population", default="outputs/plism_core_population_stats")
    parser.add_argument("--eval-locations", default="outputs/plism_core_eval_locations")
    parser.add_argument("--eval-encoder", default="uni_v2",
                        help="whose location subsample to measure on; stride 1 is the widest")
    parser.add_argument("--output", default="outputs/plism_core_render_gains/band_energy")
    parser.add_argument("--gate-um", type=float, default=1.0)
    parser.add_argument("--batch", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    row = manifest.loc[manifest["task_index"] == args.task_index].iloc[0]
    stain, scanner = str(row["stain"]), str(row["scanner"])
    stem = str(row["name"]).rsplit(".", 1)[0]

    populations = load_populations(Path(args.population))
    wanted = json.loads(
        (Path(args.eval_locations) / args.eval_encoder / f"{stain}.json").read_text())
    keep = set(int(v) for v in wanted["locations"])

    plan = pd.read_csv(Path(args.refinement) / f"{stain}.csv")
    plan = plan.loc[(plan["scanner"] == scanner) & plan["ok"]]
    plan = plan.loc[plan["residual_um"] <= args.gate_um]
    plan = plan.loc[plan["location"].isin(keep)].sort_values("location")
    if plan.empty:
        raise RuntimeError(f"{stem}: no evaluation location survived the gate")

    import pyvips
    import torch

    from e5_comparator_population import rgb01_to_od
    from rf1_gamut_reinhard import reinhard_lab_source_ray
    from rf1m_combined import RF1M_SIGMAS, band_energy

    actual, swapped = corrected_name(stain, scanner, str(row["name"]))
    image = pyvips.Image.new_from_file(str(Path(args.wsi_dir) / actual), access="random")
    if image.hasalpha():
        image = image.flatten(background=255)
    mpp = float(plan["mpp"].iloc[0])
    shrink = TARGET_MPP / mpp
    side = int(round(TILE_UM / mpp))
    flip = bool(plan["flip"].iloc[0])

    tiles = []
    for record in plan.itertuples():
        x = int(round(record.centre_x)) - side // 2
        y = int(round(record.centre_y)) - side // 2
        if x < 0 or y < 0 or x + side > image.width or y + side > image.height:
            continue
        patch = image.crop(x, y, side, side).reduce(shrink, shrink, kernel="lanczos3")
        array = patch.numpy()[:TILE_PX, :TILE_PX]
        if array.shape[:2] != (TILE_PX, TILE_PX):
            continue
        if flip:
            array = np.rot90(array, 2)
        tiles.append(np.ascontiguousarray(array, dtype=np.uint8))
    if not tiles:
        raise RuntimeError(f"{stem}: no tile survived rendering")
    stack = np.stack(tiles)

    rows = []
    for destination in DESTINATIONS:
        source, target = populations[scanner], populations[destination]
        moments = [torch.tensor([source[c][0] for c in "Lab"]),
                   torch.tensor([source[c][1] for c in "Lab"]),
                   torch.tensor([target[c][0] for c in "Lab"]),
                   torch.tensor([target[c][1] for c in "Lab"])]
        energies, limited = [], []
        for start in range(0, len(stack), args.batch):
            rgb01 = torch.from_numpy(stack[start:start + args.batch]).float() / 255.0
            if scanner == destination:
                corrected = rgb01
                limited.append(0.0)
            else:
                report = reinhard_lab_source_ray(rgb01, *moments)
                corrected = report["output"]
                limited.append(float(report["limited_pixel_fraction"].mean()))
            energies.append(band_energy(rgb01_to_od(corrected).mean(dim=-1),
                                        RF1M_SIGMAS).numpy())
        energies = np.concatenate(energies)
        entry = {"stain": stain, "scanner": scanner, "tiles": len(energies),
                 "destination": destination.lower(),
                 "identity": bool(scanner == destination),
                 "gamut_limited": float(np.mean(limited)),
                 "file_read": actual, "file_swapped": bool(swapped)}
        for index in range(energies.shape[1]):
            entry[f"b{index + 1}"] = float(energies[:, index].mean())
        rows.append(entry)
        print(f"{stem} -> reinhard_{destination.lower()}: "
              + " ".join(f"b{i + 1}={entry[f'b{i + 1}']:.4e}" for i in range(energies.shape[1]))
              + f"  gamut_limited={entry['gamut_limited']:.4f}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / f"{stem}.csv", index=False)


if __name__ == "__main__":
    main()

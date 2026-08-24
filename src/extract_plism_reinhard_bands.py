"""Post-Reinhard band energies for one PLISM WSI — Arm A of the correction replication.

Section 9 of the report does not turn on raw band power.  It turns on *the band
power that survives Reinhard's contrast matching*: GT450 carries 4.14x AT2's
fine-band power on that measure, and it is that quantity, not raw sharpness,
which decides whether aiming a correction at a destination amplifies or
attenuates.  PLISM showed raw band power is dominated by how darkly a scanner
renders the stain, so this pass measures the quantity the finding actually uses.

Every scanner is mapped to one common colour reference (the AT2 population), so
contrast is equalised across the panel and what remains in the bands is the
frequency residual.  Gains toward any destination follow from ratios of these
numbers, because Reinhard toward a destination differs from Reinhard toward the
reference only by the destination's own Lab scale, which multiplies every source
alike.

Bands are the study's sigma 1, 2, 4 px at 0.5052 um/px, carried over as the
physical scales 0.5052, 1.0104 and 2.0208 um.

Contract: docs/e9_plism_native_ert_contract.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_plism_population_stats import BAND_NAMES, band_energies
from extract_plism_native_psd import PATCH_UM, open_level
from plism_dataset_corrections import corrected_name

REFERENCE = "AT2"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/PLISM_dataset/manifest_original.csv")
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--wsi-dir", default="data/PLISM_dataset/original_wsi")
    parser.add_argument("--refinement", default="outputs/plism_core_refinement")
    parser.add_argument("--population", default="outputs/plism_core_population_stats")
    parser.add_argument("--output", default="outputs/plism_core_reinhard_bands")
    parser.add_argument("--gate-um", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0,
                        help="smoke-test cap on locations; 0 reads every one")
    return parser.parse_args()


def pooled_lab(population: pd.DataFrame) -> dict:
    """Population Lab moments over all pixels, from per-patch means and spreads.

    The pooled variance is E[within-patch variance] + Var(patch means); using the
    mean of the per-patch standard deviations instead would understate the spread
    the correction has to match.
    """
    stats = {}
    for channel in "Lab":
        means = population[f"{channel}_mean"].to_numpy()
        spreads = population[f"{channel}_std"].to_numpy()
        stats[channel] = (
            float(means.mean()),
            float(np.sqrt(np.mean(spreads**2) + np.var(means))),
        )
    return stats


def reinhard(rgb: np.ndarray, source: dict, target: dict) -> tuple[np.ndarray, float]:
    """The study's Reinhard: CIE Lab moment matching closed by source-ray gamut
    projection.  Independent per-channel clipping would change hue and, on the
    scanners whose contrast is expanded most, would eat the band energy this pass
    exists to measure."""
    import torch

    from rf1_gamut_reinhard import reinhard_lab_source_ray

    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).float().unsqueeze(0) / 255.0
    moments = [
        torch.tensor([source[c][0] for c in "Lab"]),
        torch.tensor([source[c][1] for c in "Lab"]),
        torch.tensor([target[c][0] for c in "Lab"]),
        torch.tensor([target[c][1] for c in "Lab"]),
    ]
    report = reinhard_lab_source_ray(tensor, *moments)
    corrected = (report["output"][0].numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    return corrected, float(report["limited_pixel_fraction"][0])


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    row = manifest.loc[manifest["task_index"] == args.task_index].iloc[0]
    stain, scanner = str(row["stain"]), str(row["scanner"])

    population_dir = Path(args.population)
    frames = {}
    for path in population_dir.glob("*.csv"):
        block = pd.read_csv(path)
        frames.setdefault(str(block["scanner"].iloc[0]), []).append(block)
    pooled = {name: pooled_lab(pd.concat(blocks, ignore_index=True))
              for name, blocks in frames.items()}
    if REFERENCE not in pooled:
        raise RuntimeError(f"no {REFERENCE} population statistics")

    plan = pd.read_csv(Path(args.refinement) / f"{stain}.csv")
    plan = plan.loc[(plan["scanner"] == scanner) & plan["ok"]]
    plan = plan.loc[plan["residual_um"] <= args.gate_um].sort_values("location")
    if args.limit:
        plan = plan.head(args.limit)

    actual, swapped = corrected_name(stain, scanner, str(row["name"]))
    path = str(Path(args.wsi_dir) / actual)
    stem = str(row["name"]).rsplit(".", 1)[0]
    mpp = float(plan["mpp"].iloc[0])
    side = int(round(PATCH_UM / mpp))
    full = open_level(path, 0)

    records = []
    for record in plan.itertuples():
        x = int(round(record.centre_x)) - side // 2
        y = int(round(record.centre_y)) - side // 2
        if x < 0 or y < 0 or x + side > full.width or y + side > full.height:
            continue
        patch = full.crop(x, y, side, side).numpy()
        corrected, clipped = reinhard(patch, pooled[scanner], pooled[REFERENCE])
        entry = {"stain": stain, "scanner": scanner, "location": int(record.location),
                 "core": int(record.core), "tissue_type": str(record.tissue_type),
                 "residual_um": float(record.residual_um),
                 "response": float(record.response), "limited_frac": clipped}
        for name, value in band_energies(patch, mpp).items():
            entry[f"raw_{name}"] = value
        for name, value in band_energies(corrected, mpp).items():
            entry[f"rein_{name}"] = value
        records.append(entry)

    frame = pd.DataFrame(records)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / f"{stem}.csv", index=False)
    ratio = frame["rein_b1"].mean() / frame["raw_b1"].mean()
    print(f"{stem}: {len(frame)} patches  raw_b1={frame['raw_b1'].mean():.3e} "
          f"rein_b1={frame['rein_b1'].mean():.3e} (x{ratio:.2f}) "
          f"gamut_limited={frame['limited_frac'].mean():.4f}")


if __name__ == "__main__":
    main()

"""Population Lab moments and OD band energies for one PLISM WSI.

Arm A of the correction replication needs exactly what RF1U needs at fit time:
a source population and a target population, no paired acquisitions.  This pass
reads the gated, refined locations once and reports the two population summaries
Reinhard and the band-gain step are fitted from.

Band definition follows the study, translated into physical units.  RF1M/RF1U
split the OD mean with Gaussians of sigma 1, 2 and 4 pixels **on the 0.5052 um/px
target grid**, so the physical scales are 0.5052, 1.0104 and 2.0208 um.  Applying
the pixel sigmas directly on a native grid would make the bands mean different
things on different scanners, which is the mistake this cohort exists to avoid.

Contract: docs/e9_plism_native_ert_contract.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from extract_plism_native_psd import PATCH_UM, open_level
from plism_dataset_corrections import corrected_name

# sigma 1, 2, 4 px at the study's 0.5052 um/px target grid
BAND_SIGMA_UM = (0.5052, 1.0104, 2.0208)
BAND_NAMES = ("b1", "b2", "b3")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/PLISM_dataset/manifest_original.csv")
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--wsi-dir", default="data/PLISM_dataset/original_wsi")
    parser.add_argument("--refinement", default="outputs/plism_core_refinement")
    parser.add_argument("--output", default="outputs/plism_core_population_stats")
    parser.add_argument("--gate-um", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0,
                        help="smoke-test cap on locations; 0 reads every one")
    return parser.parse_args()


def to_lab(rgb: np.ndarray) -> np.ndarray:
    """CIE Lab through the study's own converter, so the moments Reinhard is
    fitted from are in the same space the frozen implementation uses."""
    import torch

    from e5_comparator_population import rgb01_to_lab

    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).float() / 255.0
    return rgb01_to_lab(tensor).numpy()


def band_energies(rgb: np.ndarray, mpp: float) -> dict:
    """Mean square of each Laplacian band of the mean OD, at fixed physical scale."""
    import cv2

    od = -np.log10(np.clip(rgb.astype(np.float32), 1.0, 255.0) / 255.0).mean(axis=2)
    blurred = []
    for sigma_um in BAND_SIGMA_UM:
        sigma_px = sigma_um / mpp
        radius = int(max(1, round(3 * sigma_px)))
        blurred.append(cv2.GaussianBlur(od, (2 * radius + 1, 2 * radius + 1), sigma_px))
    bands = [od - blurred[0], blurred[0] - blurred[1], blurred[1] - blurred[2]]
    out = {name: float(np.mean(band**2)) for name, band in zip(BAND_NAMES, bands)}
    out["base"] = float(np.mean(blurred[2]))
    return out


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    row = manifest.loc[manifest["task_index"] == args.task_index].iloc[0]
    stain, scanner = str(row["stain"]), str(row["scanner"])

    plan = pd.read_csv(Path(args.refinement) / f"{stain}.csv")
    plan = plan.loc[(plan["scanner"] == scanner) & plan["ok"]]
    plan = plan.loc[plan["residual_um"] <= args.gate_um].sort_values("location")
    if args.limit:
        plan = plan.head(args.limit)

    # The label names the section; two SQ files hold each other's.  The stem
    # stays the published label because that is the section actually read.
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
        lab = to_lab(patch)
        entry = {
            "stain": stain, "scanner": scanner, "location": int(record.location),
            "core": int(record.core), "tissue_type": str(record.tissue_type),
            "residual_um": float(record.residual_um),
            "response": float(record.response),
        }
        for index, channel in enumerate("Lab"):
            entry[f"{channel}_mean"] = float(lab[..., index].mean())
            entry[f"{channel}_std"] = float(lab[..., index].std())
        entry.update(band_energies(patch, mpp))
        records.append(entry)

    frame = pd.DataFrame(records)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / f"{stem}.csv", index=False)
    print(f"{stem}: {len(frame)} gated patches, mpp={mpp:.5f} "
          f"L={frame['L_mean'].mean():.1f} b1={frame['b1'].mean():.3e}"
          + (f"  [read {actual}]" if swapped else ""))


if __name__ == "__main__":
    main()

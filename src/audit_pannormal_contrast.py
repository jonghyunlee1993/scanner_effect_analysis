"""Is the locked ERT ordering a contrast ordering? — the audit PLISM forced.

On PLISM the raw band-power ratio between scanners is almost entirely explained by
how darkly each renders the stain: the power ratio against AT2 tracks the squared
optical-density ratio across all seven scanners.  Section 3.1 of the report uses
the same raw statistic on this cohort, and that relationship has never been checked
here.

This reads the audited native-AA render — the same pixels the locked features came
from — and asks the same question.  If the relationship holds, section 3.1's
"effective relative transfer" is a composite of contrast and frequency response and
must be restated; if it does not, the locked ordering is spectral after all and the
PLISM inversion is a property of that cohort's rendering.

No locked result is recomputed and nothing here enters the frozen ranking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

SCANNERS = ("at2", "gt450", "versa", "akoya", "s60", "s360")
LOCKED_ERT = {"gt450": 1.478, "s60": 1.044, "at2": 1.000,
              "versa": 0.938, "s360": 0.842, "akoya": 0.335}
# sigma 1, 2, 4 px on the 0.5052 um/px grid, as in RF1M/RF1U
SIGMAS = (1.0, 2.0, 4.0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", default="outputs/e0_native_aa_grid/shards")
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--output", default="outputs/pannormal_contrast_audit")
    parser.add_argument("--fov", type=int, default=256)
    return parser.parse_args()


def band_energies(rgb: np.ndarray) -> dict:
    import cv2

    od = -np.log10(np.clip(rgb.astype(np.float32), 1.0, 255.0) / 255.0).mean(axis=2)
    blurred = []
    for sigma in SIGMAS:
        radius = int(max(1, round(3 * sigma)))
        blurred.append(cv2.GaussianBlur(od, (2 * radius + 1, 2 * radius + 1), sigma))
    bands = [od - blurred[0], blurred[0] - blurred[1], blurred[1] - blurred[2]]
    out = {f"b{i + 1}": float(np.mean(b**2)) for i, b in enumerate(bands)}
    out["od_mean"] = float(od.mean())
    out["od_std"] = float(od.std())
    return out


def main() -> None:
    args = parse_args()
    shards = sorted(Path(args.grid).glob("*.h5"))
    shard = shards[args.task_index]

    rows = []
    with h5py.File(shard, "r") as handle:
        scanners = [s.decode() if isinstance(s, bytes) else str(s)
                    for s in handle["scanner"][:]]
        stack = handle["rgb"]
        _, locations, height, width, _ = stack.shape
        top = (height - args.fov) // 2
        left = (width - args.fov) // 2
        for index, scanner in enumerate(scanners):
            for location in range(locations):
                patch = stack[index, location,
                              top:top + args.fov, left:left + args.fov]
                rows.append({"slide": shard.stem, "scanner": scanner,
                             "location": location, **band_energies(patch)})

    frame = pd.DataFrame(rows)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / f"{shard.stem}.csv", index=False)
    summary = frame.groupby("scanner")[["b1", "od_std"]].mean()
    print(f"{shard.stem}: {len(frame)} patches")
    print(summary.round(5).to_string())


if __name__ == "__main__":
    main()

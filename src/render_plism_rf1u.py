"""Apply the fitted RF1U band gains to one rendered PLISM slide.

Reads the Reinhard tiles already on disk and adds the band correction through the
study's own operator, so the shared-OD residual and the exact gamut projection are
the frozen ones rather than a reimplementation.  A source that is its own
destination gets identity, and a band whose gain could not be estimated reliably
has alpha = 0 upstream, which is identity too — RF1U degrading to plain Reinhard
is a designed outcome, not a failure.

Contract: docs/e9_plism_native_ert_contract.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from rf1m_combined import RF1M_SIGMAS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/PLISM_dataset/manifest_original.csv")
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--render", default="outputs/plism_render")
    parser.add_argument("--gains", default="outputs/plism_render_gains/fitted_gains.csv")
    parser.add_argument("--destinations", default="at2,gt450")
    parser.add_argument("--batch", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    row = manifest.loc[manifest["task_index"] == args.task_index].iloc[0]
    stem = str(row["name"]).rsplit(".", 1)[0]
    scanner = str(row["scanner"])

    import torch

    from rf1m_combined import shared_od_multiscale_many

    gains_table = pd.read_csv(args.gains)
    render_root = Path(args.render)

    for destination in [d.strip() for d in args.destinations.split(",") if d.strip()]:
        source_path = render_root / f"reinhard_{destination}" / f"{stem}.h5"
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        block = gains_table.loc[(gains_table["destination"] == destination)
                                & (gains_table["source"] == scanner)]
        if block.empty:
            gains = np.ones((1, len(RF1M_SIGMAS)))
            note = "identity (source is the destination)"
        else:
            gains = block.sort_values("band")["gain"].to_numpy()[None, :]
            note = " ".join(f"{g:.3f}" for g in gains[0])

        destination_dir = render_root / f"rf1u_{destination}"
        destination_dir.mkdir(parents=True, exist_ok=True)
        with h5py.File(source_path, "r") as source, \
                h5py.File(destination_dir / f"{stem}.h5", "w") as target:
            meta = json.loads(source.attrs["meta"])
            total = source["rgb"].shape[0]
            output = target.create_dataset("rgb", shape=source["rgb"].shape,
                                           dtype="uint8", compression="lzf")
            projection = []
            for start in range(0, total, args.batch):
                chunk = source["rgb"][start:start + args.batch]
                rgb01 = torch.from_numpy(np.ascontiguousarray(chunk)).float() / 255.0
                report = shared_od_multiscale_many(rgb01, gains, RF1M_SIGMAS)
                corrected = report["output"][0].numpy()
                output[start:start + args.batch] = (
                    (corrected * 255.0).round().clip(0, 255).astype(np.uint8))
                projection.append(float(report["projection_fraction"].mean()))
            for key in ("location", "core", "residual_um"):
                target.create_dataset(key, data=source[key][:])
            target.attrs["meta"] = json.dumps({
                **meta, "condition": f"rf1u_{destination}",
                "gains": gains[0].tolist(), "sigmas_px": list(RF1M_SIGMAS),
                "projection_fraction": float(np.mean(projection))})
        print(f"{stem} -> rf1u_{destination}: gains {note}, "
              f"projection {np.mean(projection):.4f}")


if __name__ == "__main__":
    main()

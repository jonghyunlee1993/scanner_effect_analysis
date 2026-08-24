"""Freeze the location subsample the PLISM endpoints are evaluated on.

The core grid carries ~8,950 locations per section, and that density is what the
*fitting* problem needed: CORAL estimates a full covariance, and the report's
standing qualification against it was an estimation artefact of having only a few
hundred samples per section.  Fitting therefore uses every location.

Evaluation does not want them all.  Three of the four locked endpoints are
O(L^2) in the number of locations -- the unmatched-content quantile, the pairwise
distance and the collapse gram are all location-by-location matrices -- so the
cost grows as the square while the estimate stops moving after a few hundred.
Worse, encoding the four image conditions at full density would cost 4.5x the GPU
time for numbers that do not change.

So the endpoints are read on a fixed subsample, frozen here and written out, and
every arm reads the same file: the raw features, the Reinhard conditions and the
RF1U conditions are then paired on identical physical locations rather than on
whatever each pass happened to keep.

The subsample is drawn per (encoder, section) from the location ids that survive
in *every* scanner of that section, because an endpoint compares seven scanners
at one place.  Encoders differ here only because they differ in lattice stride:
CONCHv1.5's field of view is twice the lattice pitch, so it holds a quarter of
the locations to begin with.

Contract: docs/e9_plism_native_ert_contract.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

SEED = 20260822


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/PLISM_dataset/features")
    parser.add_argument("--output", default="outputs/plism_core_eval_locations")
    parser.add_argument("--locations", type=int, default=2000,
                        help="per section; fewer are kept if the intersection is smaller")
    return parser.parse_args()


def section_intersection(directory: Path) -> dict[str, dict]:
    """stain -> {'scanners': [...], 'shared': array of location ids}."""
    sections: dict[str, dict[str, np.ndarray]] = {}
    for path in sorted(directory.glob("*.h5")):
        with h5py.File(path, "r") as handle:
            meta = json.loads(handle.attrs["meta"])
            sections.setdefault(meta["stain"], {})[meta["scanner"]] = handle["location"][:]
    out = {}
    for stain, scanners in sections.items():
        shared = None
        for ids in scanners.values():
            shared = ids if shared is None else np.intersect1d(shared, ids)
        out[stain] = {"scanners": sorted(scanners), "shared": np.sort(shared)}
    return out


def main() -> None:
    args = parse_args()
    root = Path(args.features)
    output = Path(args.output)
    encoders = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name != "analysis")

    summary = []
    for encoder in encoders:
        sections = section_intersection(root / encoder)
        directory = output / encoder
        directory.mkdir(parents=True, exist_ok=True)
        kept, available = 0, 0
        for stain, entry in sorted(sections.items()):
            shared = entry["shared"]
            # Seeded per (encoder, section) so adding an encoder never disturbs
            # the sets already frozen for the others.
            rng = np.random.default_rng(abs(hash((SEED, encoder, stain))) % (2**32))
            take = min(args.locations, len(shared))
            chosen = np.sort(rng.choice(shared, size=take, replace=False))
            (directory / f"{stain}.json").write_text(json.dumps({
                "stain": stain, "encoder": encoder, "seed": SEED,
                "scanners": entry["scanners"],
                "available": int(len(shared)), "kept": int(take),
                "locations": [int(v) for v in chosen],
            }))
            kept += take
            available += len(shared)
        summary.append({"encoder": encoder, "sections": len(sections),
                        "available": available, "kept": kept})
        print(f"{encoder}: {len(sections)} sections, {available:,} shared locations, "
              f"{kept:,} kept ({kept / max(available, 1):.1%})")

    (output / "summary.json").write_text(json.dumps(
        {"seed": SEED, "per_section": args.locations, "encoders": summary}, indent=2))
    print(f"-> {output}")


if __name__ == "__main__":
    main()

"""How much of a representation is scanner, and is that scanner part a shift?

Scanner radius answers "how far apart do six acquisitions of one spot land" on a
scale whose maximum is 2, which is hard to hold in mind and impossible to compare
against anything outside this study. A variance decomposition answers the same
question in a unit anyone reads: what fraction of the embedding variance is the
instrument, and what fraction is the tissue.

The design makes the decomposition exact rather than approximate. Every physical
location is imaged by all six scanners, so slide, location-within-slide and
scanner are balanced and crossed, and the sums of squares are orthogonal:

    x[slide, location, scanner] = mean
                                + slide
                                + location within slide      <- content
                                + scanner                    <- instrument, global
                                + residual                   <- scanner x content

The last two lines are the point. A scanner main effect is a single offset per
instrument, identical everywhere. The residual holds everything about the
scanner that depends on what is being imaged. If the moment-ladder reading is
right -- that a per-scanner mean vector recovers most of what a full covariance
correction achieves -- then the scanner main effect should dominate the residual,
and correcting it should move the main effect to near zero while leaving content
untouched. That is a falsifiable version of "the scanner is a translation in
feature space", and it is checked here rather than asserted.

Runs over any condition in the registry layout, so a correction can be scored on
the same decomposition as raw.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np

from e4_primary_metrics import l2_normalize
from e5_comparator_population import SCANNERS
from fetch_e0_pfm_checkpoints import sha256


BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260803
LOCATIONS = 100


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", default="outputs/e0_pfm_contract/checkpoint_manifest.json")
    parser.add_argument("--encoder-index", type=int, required=True)
    parser.add_argument("--raw", default="outputs/e0_pfm_features")
    parser.add_argument("--source", action="append", default=[], metavar="LABEL=ROOT")
    parser.add_argument("--replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def slide_sums(features: np.ndarray):
    """Per-slide sufficient statistics for the crossed decomposition.

    The content sum of squares is a within-slide quantity, so it is reduced to a
    scalar here rather than carried as a 100-by-D array into the bootstrap; only
    the slide total, the per-scanner totals and two scalars survive, which is
    what keeps 2,000 slide resamples cheap.
    """
    value = l2_normalize(features)
    if value.shape[:2] != (len(SCANNERS), LOCATIONS):
        raise ValueError(f"expected 6x100xD, got {value.shape}")
    total = value.sum(axis=(0, 1))
    location_mean = value.sum(axis=0) / len(SCANNERS)
    slide_mean = total / (len(SCANNERS) * LOCATIONS)
    centered = location_mean - slide_mean[None]
    return {
        "total": total,
        "by_scanner": value.sum(axis=1),
        "sum_square": float(np.sum(value**2)),
        "ss_content": len(SCANNERS) * float(np.sum(centered**2)),
    }


def decompose(entries, index: np.ndarray):
    """Orthogonal sums of squares for one resample of slides.

    With every location imaged by every scanner the design is balanced, so the
    slide, content, scanner and residual sums of squares partition the total
    exactly and each can be written from the cached totals.
    """
    slides = len(index)
    n = slides * len(SCANNERS) * LOCATIONS

    totals = np.stack([entries[i]["total"] for i in index])
    grand = totals.sum(axis=0) / n
    sum_square = sum(entries[i]["sum_square"] for i in index)
    ss_total = sum_square - n * float(grand @ grand)

    # Slide: each slide mean over its 600 acquisitions.
    slide_means = totals / (len(SCANNERS) * LOCATIONS)
    ss_slide = len(SCANNERS) * LOCATIONS * float(
        np.sum((slide_means - grand[None]) ** 2)
    )

    # Scanner main effect: one offset per instrument, pooled over everything.
    scanner_total = np.stack([entries[i]["by_scanner"] for i in index]).sum(axis=0)
    scanner_mean = scanner_total / (slides * LOCATIONS)
    ss_scanner = LOCATIONS * slides * float(np.sum((scanner_mean - grand[None]) ** 2))

    # Content: location within slide, averaged over the six scanners. Within-slide
    # and therefore already reduced per slide.
    ss_content = float(sum(entries[i]["ss_content"] for i in index))

    ss_residual = ss_total - ss_slide - ss_scanner - ss_content
    return {
        "slide": ss_slide / ss_total,
        "content": ss_content / ss_total,
        "scanner": ss_scanner / ss_total,
        "residual": ss_residual / ss_total,
    }


def main():
    args = parse_args()
    contract = json.loads(Path(args.contract).read_text())
    model = contract["models"][args.encoder_index]
    model_id = model["encoder_id"]
    feature_dim = int(model["feature_dim"])

    sources = {}
    for value in args.source:
        label, root = value.split("=", 1)
        sources[label] = Path(root)

    raw_paths = sorted((Path(args.raw) / model_id / "shards").glob("*.h5"))
    if len(raw_paths) != 109:
        raise ValueError(f"{model_id}: expected 109 raw shards, got {len(raw_paths)}")

    cached: dict[str, list] = {}
    for path in raw_paths:
        slide_id = path.stem
        with h5py.File(path, "r") as handle:
            raw = np.asarray(handle["features"][:], dtype=np.float32)
        if raw.shape != (len(SCANNERS), LOCATIONS, feature_dim):
            raise ValueError(f"{model_id}/{slide_id}: unexpected raw shape")
        cached.setdefault("raw", []).append(slide_sums(raw))
        for label, root in sources.items():
            with h5py.File(root / model_id / "shards" / f"{slide_id}.h5", "r") as handle:
                names = [item.decode() for item in handle["condition"][:]]
                values = np.asarray(handle["features"][:], dtype=np.float32)
            for position, name in enumerate(names):
                cached.setdefault(f"{label}:{name}", []).append(
                    slide_sums(values[position])
                )
        print(f"{model_id} {slide_id}", flush=True)

    generator = np.random.default_rng(BOOTSTRAP_SEED)
    resamples = generator.integers(0, len(raw_paths), size=(args.replicates, len(raw_paths)))
    identity = np.arange(len(raw_paths))

    rows = []
    for condition, entries in cached.items():
        point = decompose(entries, identity)
        draws = {key: [] for key in point}
        for replicate in resamples:
            drawn = decompose(entries, replicate)
            for key, value in drawn.items():
                draws[key].append(value)
        for component, value in point.items():
            sample = np.asarray(draws[component])
            rows.append(
                {
                    "encoder_id": model_id,
                    "condition": condition,
                    "component": component,
                    "variance_fraction": value,
                    "ci95_low": float(np.quantile(sample, 0.025)),
                    "ci95_high": float(np.quantile(sample, 0.975)),
                }
            )
        print(
            f"{model_id} {condition:38s} "
            + " ".join(f"{key} {point[key]:6.3f}" for key in ("scanner", "content", "slide", "residual")),
            flush=True,
        )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    table = output / f"{model_id}.csv"
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "analysis": "e8_variance_components",
        "status": "POST_CORE_EXPLORATORY_EXTENSION",
        "encoder_id": model_id,
        "model": (
            "crossed, balanced decomposition of L2-normalised embeddings into "
            "slide, location-within-slide (content), scanner main effect and "
            "residual (scanner x content interaction)"
        ),
        "reading": (
            "The scanner main effect is a single offset per instrument. A large "
            "main effect relative to the residual is what makes a per-scanner "
            "mean shift an adequate correction."
        ),
        "conditions": list(cached),
        "bootstrap": {
            "replicates": args.replicates,
            "seed": BOOTSTRAP_SEED,
            "unit": "physical slide",
        },
        "artifacts": {table.name: sha256(table)},
    }
    (output / f"{model_id}.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

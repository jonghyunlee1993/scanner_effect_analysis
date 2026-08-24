"""Does a corrected patch still look like something a real scanner produced?

The destination sweep peaks at the one point on its axis that corresponds to a
physically realisable acquisition, and the explanation offered for that is
on-manifold: away from lambda = 1 the correction synthesises image statistics no
instrument produces, and an encoder trained on real images degrades off that
manifold. So far that is a story rather than a measurement.

The cheapest model-intrinsic proxy does not settle it. Feature norm is available
for free -- the stored features are pre-normalisation -- but the gap between
corrected sources and real GT450 shrinks as lambda grows past 1, the opposite of
the performance curve, and every gap is far inside a within-condition standard
deviation. Norm is not the quantity that matters here.

This measures manifold membership directly: for each corrected patch, the cosine
distance to its nearest neighbours among the raw acquisitions of *other physical
slides*. Excluding the query's own slide is what makes this a manifold question
rather than an accuracy question -- a patch cannot score well by resembling its
own uncorrected self, only by resembling a plausible real acquisition of some
other tissue.

The test is asymmetric in the right way, which is why it can discriminate. Both
lambda = 0.5 and lambda = 2 are one octave from the peak, but only one of them
moves toward statistics that real scanners actually produce: the reference bank
contains AKOYA and AT2, which are softer than GT450, and contains nothing sharper
than GT450. If distance from the bank tracks the performance curve, off-manifold
is the mechanism. If distance grows symmetrically with |log lambda|, the measure
is only reporting how much the image changed and says nothing.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from e4_primary_metrics import l2_normalize
from e5_comparator_population import SCANNERS
from fetch_e0_pfm_checkpoints import sha256


LOCATION_STRIDE = 4
NEIGHBOURS = 10
SEED = 20260803


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", default="outputs/e0_pfm_contract/checkpoint_manifest.json")
    parser.add_argument("--encoder-index", type=int, required=True)
    parser.add_argument("--raw", default="outputs/e0_pfm_features")
    parser.add_argument("--source", action="append", default=[], metavar="LABEL=ROOT")
    parser.add_argument("--destination", default="gt450", choices=SCANNERS)
    parser.add_argument("--neighbours", type=int, default=NEIGHBOURS)
    parser.add_argument("--location-stride", type=int, default=LOCATION_STRIDE)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_bank(root: Path, model_id: str, slide_ids, feature_dim: int):
    """All raw acquisitions, L2-normalised, with the slide each one came from."""
    blocks, owners = [], []
    for index, slide_id in enumerate(slide_ids):
        with h5py.File(root / model_id / "shards" / f"{slide_id}.h5", "r") as handle:
            values = np.asarray(handle["features"][:], dtype=np.float32)
        if values.shape != (len(SCANNERS), 100, feature_dim):
            raise ValueError(f"{slide_id}: unexpected raw shape {values.shape}")
        blocks.append(l2_normalize(values).reshape(-1, feature_dim).astype(np.float32))
        owners.append(np.full(len(SCANNERS) * 100, index, dtype=np.int32))
    return torch.from_numpy(np.concatenate(blocks)), torch.from_numpy(np.concatenate(owners))


def mean_neighbour_similarity(queries, bank, owners, slide_index, k):
    """Mean cosine to the k nearest bank entries that are not from this slide."""
    similarity = queries @ bank.T
    similarity[:, owners == slide_index] = -2.0
    top = similarity.topk(k, dim=1).values
    return top.mean(dim=1)


def main():
    args = parse_args()
    contract = json.loads(Path(args.contract).read_text())
    model = contract["models"][args.encoder_index]
    model_id = model["encoder_id"]
    feature_dim = int(model["feature_dim"])
    destination = SCANNERS.index(args.destination)
    positions = list(range(0, 100, args.location_stride))

    raw_root = Path(args.raw)
    slide_ids = sorted(path.stem for path in (raw_root / model_id / "shards").glob("*.h5"))
    if len(slide_ids) != 109:
        raise ValueError(f"{model_id}: expected 109 raw shards, got {len(slide_ids)}")

    sources = {}
    for value in args.source:
        label, root = value.split("=", 1)
        sources[label] = Path(root)

    bank, owners = load_bank(raw_root, model_id, slide_ids, feature_dim)
    print(f"{model_id}: bank {tuple(bank.shape)}", flush=True)

    rows = []
    per_condition: dict[str, list] = {}
    for index, slide_id in enumerate(slide_ids):
        with h5py.File(raw_root / model_id / "shards" / f"{slide_id}.h5", "r") as handle:
            raw = np.asarray(handle["features"][:], dtype=np.float32)
        queries = {"raw": raw}
        for label, root in sources.items():
            with h5py.File(root / model_id / "shards" / f"{slide_id}.h5", "r") as handle:
                names = [item.decode() for item in handle["condition"][:]]
                values = np.asarray(handle["features"][:], dtype=np.float32)
            for position, name in enumerate(names):
                queries[f"{label}:{name}"] = values[position]

        for condition, values in queries.items():
            # Source scanners only: the destination passes through uncorrected,
            # so including it would dilute every condition with the same patches.
            rows_of_interest = [i for i in range(len(SCANNERS)) if i != destination]
            block = l2_normalize(values[rows_of_interest][:, positions]).reshape(
                -1, feature_dim
            )
            similarity = mean_neighbour_similarity(
                torch.from_numpy(block.astype(np.float32)), bank, owners, index, args.neighbours
            )
            per_condition.setdefault(condition, []).append(float(similarity.mean()))
        print(f"[{index + 1}/109] {model_id} {slide_id}", flush=True)

    generator = np.random.default_rng(SEED)
    draws = generator.integers(0, len(slide_ids), size=(5000, len(slide_ids)))
    baseline = np.asarray(per_condition["raw"])
    for condition, values in per_condition.items():
        value = np.asarray(values)
        delta = value - baseline
        rows.append(
            {
                "encoder_id": model_id,
                "condition": condition,
                "mean_neighbour_cosine": float(value.mean()),
                "ci95_low": float(np.quantile(value[draws].mean(axis=1), 0.025)),
                "ci95_high": float(np.quantile(value[draws].mean(axis=1), 0.975)),
                "delta_vs_raw": float(delta.mean()),
                "delta_ci95_low": float(np.quantile(delta[draws].mean(axis=1), 0.025)),
                "delta_ci95_high": float(np.quantile(delta[draws].mean(axis=1), 0.975)),
            }
        )
        print(
            f"{model_id} {condition:44s} cos {value.mean():.4f} "
            f"delta {delta.mean():+.4f}",
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
        "analysis": "e8_manifold_distance",
        "status": "POST_CORE_EXPLORATORY_EXTENSION",
        "question": (
            "Does a corrected patch resemble a real acquisition of some other "
            "tissue? Higher mean neighbour cosine means closer to the manifold "
            "of real acquisitions."
        ),
        "encoder_id": model_id,
        "bank": "all raw acquisitions, six scanners, 109 slides",
        "exclusion": "the query slide's own raw acquisitions are removed from the bank",
        "destination_excluded_from_queries": args.destination,
        "neighbours": args.neighbours,
        "location_stride": args.location_stride,
        "bootstrap": {"replicates": 5000, "seed": SEED, "unit": "physical slide"},
        "conditions": list(per_condition),
        "artifacts": {table.name: sha256(table)},
    }
    (output / f"{model_id}.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

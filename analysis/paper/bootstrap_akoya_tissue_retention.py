#!/usr/bin/env python3
"""Paired uncertainty for the existing held-out AKOYA tissue-probe predictions."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


CONTRASTS = (
    ("target_trained", ("target:akoya", "combined:akoya"),
     ("target:akoya", "reinhard:akoya")),
    ("within_arm", ("combined:akoya", "combined:akoya"),
     ("reinhard:akoya", "reinhard:akoya")),
)


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (position - low) * (ordered[high] - ordered[low])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()

    predictions: dict[tuple[str, str], dict[str, tuple[str, int]]] = defaultdict(dict)
    with args.input.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["train_arm"], row["test_arm"])
            slide = row["slide_id"]
            if slide in predictions[key]:
                raise ValueError(f"duplicate prediction for {key}, {slide}")
            predictions[key][slide] = (row["tissue_type"], int(row["correct"]))

    rng = random.Random(args.seed)
    rows = []
    for name, treatment, comparator in CONTRASTS:
        left, right = predictions[treatment], predictions[comparator]
        if set(left) != set(right):
            raise ValueError(f"slide sets differ for {name}")
        slides = sorted(left)
        tissue_deltas: dict[str, list[int]] = defaultdict(list)
        slide_deltas = []
        for slide in slides:
            tissue, value = left[slide]
            other_tissue, other_value = right[slide]
            if tissue != other_tissue:
                raise ValueError(f"tissue mismatch for {slide}")
            delta = value - other_value
            tissue_deltas[tissue].append(delta)
            slide_deltas.append(delta)
        tissue_values = [sum(values) / len(values)
                         for _, values in sorted(tissue_deltas.items())]
        n_tissues, n_slides = len(tissue_values), len(slide_deltas)
        if n_tissues != 36 or n_slides != 102:
            raise ValueError((name, n_tissues, n_slides))

        for metric, values, unit in (
            ("macro_recall_difference", tissue_values, "tissue_type"),
            ("slide_accuracy_difference", slide_deltas, "physical_slide"),
        ):
            n = len(values)
            draws = [sum(values[rng.randrange(n)] for _ in range(n)) / n
                     for _ in range(args.replicates)]
            rows.append({
                "contrast": name,
                "metric": metric,
                "estimate": sum(values) / n,
                "ci_low": percentile(draws, 0.025),
                "ci_high": percentile(draws, 0.975),
                "n_slides": n_slides,
                "n_tissues": n_tissues,
                "bootstrap_unit": unit,
                "bootstrap_replicates": args.replicates,
                "bootstrap_seed": args.seed,
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()

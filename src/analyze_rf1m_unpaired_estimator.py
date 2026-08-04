"""Check whether the band-gain estimator needs paired acquisition.

RF1M fits its band gains from pooled population energy, never from per-pair
ratios, so it has the same structure as Reinhard: match a source population
statistic to a target population statistic.  This quantifies what pairing
actually buys, and whether an unpaired reliability guardrail is available.

Two measurements, both from the stored per-slide band energies:

* `disjoint`: fit the target energy on one half of the slides and the source
  energy on the disjoint other half, repeatedly, and compare with the fit that
  uses one half for both.  The gap is the tissue-composition confound that
  pairing removes.
* `bootstrap`: slide bootstrap of the population gain when source and target
  come from the same slides.  This is the sampling precision available when a
  paired calibration set exists, and it understates true unpaired uncertainty.

No PFM endpoint is read and no image is rendered.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_rf1m_slide_adaptive import load_energy
from e5_comparator_population import FOVS, SCANNERS
from fetch_e0_pfm_checkpoints import sha256
from rf1m_combined import RF1M_SIGMAS, RF1M_VERSION


REPLICATES = 2000
SEED = 20260803
SHRINKAGE_MULTIPLIERS = (2.0, 3.0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--energy", default="outputs/rf1m_slide_adaptive/energy")
    parser.add_argument("--output", default="outputs/rf1m_unpaired_estimator")
    return parser.parse_args()


def log_population_gain(target: np.ndarray, source: np.ndarray) -> float:
    """Half the log ratio of pooled band energies, the RF1M gain in log space."""
    if target.ndim != 1 or source.ndim != 1 or not len(target) or not len(source):
        raise ValueError("population gain needs non-empty energy vectors")
    if np.any(target <= 0) or np.any(source <= 0):
        raise ValueError("band energies must be positive")
    return 0.5 * float(np.log(target.mean()) - np.log(source.mean()))


def main():
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(SEED)

    rows = []
    hashes = {}
    for fov in FOVS:
        values, summary = load_energy(Path(args.energy), fov)
        hashes[f"fov_{fov}"] = summary["output_sha256"]
        target = values["target_band_energy"]
        # Fold 0 Reinhard statistics; the transform is a per-scanner constant, so
        # the choice of fold shifts every slide alike and cannot change a ratio.
        source = values["source_band_energy"][:, :, 0, :]
        slides = len(target)
        for scanner_index, scanner in enumerate(SCANNERS[1:]):
            for band_index, sigma in enumerate(RF1M_SIGMAS):
                t = target[:, band_index]
                s = source[:, scanner_index, band_index]
                paired = log_population_gain(t, s)

                disjoint = np.empty(REPLICATES)
                for replicate in range(REPLICATES):
                    order = generator.permutation(slides)
                    first, second = order[: slides // 2], order[slides // 2 :]
                    disjoint[replicate] = log_population_gain(
                        t[first], s[second]
                    ) - log_population_gain(t[first], s[first])

                index = generator.integers(0, slides, size=(REPLICATES, slides))
                replicates = 0.5 * (
                    np.log(t[index].mean(axis=1)) - np.log(s[index].mean(axis=1))
                )
                bootstrap_se = float(replicates.std(ddof=1))
                row = {
                    "fov": fov,
                    "scanner": scanner,
                    "sigma_px": sigma,
                    "slides": slides,
                    "population_gain": float(np.exp(paired)),
                    "abs_log_gain": abs(paired),
                    "bootstrap_se": bootstrap_se,
                    "bootstrap_noise_ratio": bootstrap_se / abs(paired),
                    "disjoint_abs_median": float(np.median(np.abs(disjoint))),
                    "disjoint_abs_q95": float(np.quantile(np.abs(disjoint), 0.95)),
                    "disjoint_noise_ratio": float(np.median(np.abs(disjoint))) / abs(paired),
                }
                for k in SHRINKAGE_MULTIPLIERS:
                    tag = f"k{k:g}"
                    row[f"alpha_calibrated_{tag}"] = max(
                        0.0, 1.0 - k * row["bootstrap_noise_ratio"]
                    )
                    row[f"alpha_unpaired_{tag}"] = max(
                        0.0, 1.0 - k * row["disjoint_noise_ratio"]
                    )
                rows.append(row)

    table = output / "unpaired_estimator.csv"
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_scanner = {}
    for scanner in SCANNERS[1:]:
        selected = [row for row in rows if row["scanner"] == scanner]
        by_scanner[scanner] = {
            "bootstrap_noise_ratio": float(
                np.mean([row["bootstrap_noise_ratio"] for row in selected])
            ),
            "disjoint_noise_ratio": float(
                np.mean([row["disjoint_noise_ratio"] for row in selected])
            ),
            "alpha_calibrated_k2": float(
                np.mean([row["alpha_calibrated_k2"] for row in selected])
            ),
            "alpha_unpaired_k2": float(
                np.mean([row["alpha_unpaired_k2"] for row in selected])
            ),
        }
    result = {
        "analysis": "rf1m_unpaired_estimator_check",
        "rf1m_version": RF1M_VERSION,
        "status": "DIAGNOSTIC_NOT_A_FROZEN_METHOD",
        "outcome_access": False,
        "pfm_feature_access": False,
        "finding": (
            "The band gain is a population moment ratio, so it is structurally unpaired "
            "like Reinhard. Pairing removes the tissue-composition confound between the "
            "source and target populations; without it the same absolute gain error "
            "appears for every scanner, which is tolerable where the gain is large and "
            "overwhelming where the scanner already matches the reference."
        ),
        "caveat": (
            "The bootstrap uses one slide set for both populations and therefore measures "
            "sampling precision with a paired calibration set, not true unpaired "
            "deployment uncertainty; the disjoint columns are the guide for that."
        ),
        "cells": len(rows),
        "replicates": REPLICATES,
        "seed": SEED,
        "by_scanner": by_scanner,
        "energy_sha256": hashes,
        "artifacts": {table.name: sha256(table)},
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

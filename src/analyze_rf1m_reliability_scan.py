"""Retrospective scan of a reliability-shrinkage selector on the stored RF1M cells.

RF1M chose a cap from 14 discrete candidates with a one- then two-standard-error
rule, and failed the per-fold no-harm gate on S360.  This scans an alternative
rule that needs no candidate list of its own:

    alpha = max(0, 1 - k * SE(best) / gain(best))

where gain and SE come only from the four inner folds.  The target strength
`alpha * strength(best)` is then matched against the candidates actually
rendered, using each candidate's mean absolute log band gain as its strength, so
the scan runs on stored data without re-rendering.

This is a diagnostic scan, not a frozen selector.  Choosing `k` from its output
would be tuning; a real contract must fix `k` before evaluation.  No PFM endpoint
is read.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_rf1_noharm_gain import SPECTRUM_TOLERANCE, gamut_gate, sha256
from analyze_rf1m_selection import candidate_metrics, candidate_power, load_cell
from e5_comparator_population import FOVS, SCANNERS
from e5_reinhard_residual_frequency import RF1_FOLDS, log_spectrum_rmse
from rf1m_combined import RF1M_CAP_CANDIDATES, RF1M_IDENTITY_CAP, RF1M_VERSION


SCAN_MULTIPLIERS = (1.0, 2.0, 3.0, 4.0, 5.0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", default="outputs/rf1m_candidate/cells")
    parser.add_argument("--output", default="outputs/rf1m_candidate/reliability_scan")
    return parser.parse_args()


def candidate_strength(gains: np.ndarray, scanner_index: int) -> np.ndarray:
    """Mean absolute log band gain of each candidate, a monotone strength scale."""
    values = np.abs(np.log(np.asarray(gains, dtype=np.float64)[:, scanner_index]))
    strength = values.mean(axis=1)
    if strength[0] > 1e-12 or np.any(np.diff(strength) < -1e-12):
        raise ValueError("candidate strength must start at identity and be non-decreasing")
    return strength


def shrunk_selection(
    strength: np.ndarray, eligible: np.ndarray, best: int, alpha: float
) -> int:
    """Strongest eligible candidate at or below the shrunk target strength."""
    target = float(alpha) * float(strength[best])
    allowed = np.flatnonzero(eligible & (strength <= target + 1e-12))
    return int(allowed[-1]) if len(allowed) else 0


def main():
    args = parse_args()
    cells_root = Path(args.cells)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    caps = np.asarray(RF1M_CAP_CANDIDATES, dtype=np.float64)

    rows = []
    for fov in FOVS:
        outer = {
            fold: load_cell(cells_root, fov, "outer", fold, fold)[0]
            for fold in range(RF1_FOLDS)
        }
        inner = {
            (h, j): load_cell(cells_root, fov, "inner", h, j)[0]
            for h in range(RF1_FOLDS)
            for j in range(RF1_FOLDS)
            if j != h
        }
        frequency = outer[0]["radial_frequency"]
        for scanner_index, scanner in enumerate(SCANNERS[1:]):
            for outer_fold in range(RF1_FOLDS):
                inner_folds = [f for f in range(RF1_FOLDS) if f != outer_fold]
                delta = np.empty((len(inner_folds), len(caps)))
                gamut = np.empty_like(delta, dtype=bool)
                strengths = []
                for row_index, inner_fold in enumerate(inner_folds):
                    cell = inner[outer_fold, inner_fold]
                    target = cell["validation_target_power"]
                    base = log_spectrum_rmse(
                        cell["validation_base_power"][scanner_index], target, frequency
                    )
                    for j in range(len(caps)):
                        delta[row_index, j] = (
                            log_spectrum_rmse(
                                candidate_power(cell, j, scanner_index), target, frequency
                            )
                            - base
                        )
                        m = candidate_metrics(cell, j, scanner_index)
                        gamut[row_index, j] = gamut_gate(
                            m["material_range_fraction"],
                            m["projection_rgb_mae"],
                            m["final_clamp_mae"],
                        )
                    strengths.append(candidate_strength(cell["gains"], scanner_index))
                strength = np.mean(strengths, axis=0)
                eligible = np.all(gamut, axis=0) & np.all(delta <= SPECTRUM_TOLERANCE, axis=0)
                mean_delta = delta.mean(axis=0)
                best = int(np.flatnonzero(eligible)[np.argmin(mean_delta[eligible])])
                gain = float(-mean_delta[best])
                se = float(delta[:, best].std(ddof=1) / np.sqrt(len(inner_folds)))

                cell = outer[outer_fold]
                target = cell["validation_target_power"]
                base = log_spectrum_rmse(
                    cell["validation_base_power"][scanner_index], target, frequency
                )
                row = {
                    "fov": fov,
                    "scanner": scanner,
                    "outer_fold": outer_fold,
                    "inner_gain": gain,
                    "inner_se": se,
                    "noise_ratio": se / gain if gain > 0 else np.inf,
                    "base_log_spectrum_rmse": base,
                }
                for k in SCAN_MULTIPLIERS:
                    alpha = max(0.0, 1.0 - k * se / gain) if gain > 0 else 0.0
                    selected = shrunk_selection(strength, eligible, best, alpha)
                    after = log_spectrum_rmse(
                        candidate_power(cell, selected, scanner_index), target, frequency
                    )
                    m = candidate_metrics(cell, selected, scanner_index)
                    tag = f"k{k:g}"
                    row[f"{tag}_alpha"] = alpha
                    row[f"{tag}_cap"] = float(caps[selected])
                    row[f"{tag}_rmse_reduction_percent"] = 100.0 * (base - after) / base
                    row[f"{tag}_nonworse"] = bool(after <= base + SPECTRUM_TOLERANCE)
                    row[f"{tag}_gamut_pass"] = gamut_gate(
                        m["material_range_fraction"],
                        m["projection_rgb_mae"],
                        m["final_clamp_mae"],
                    )
                rows.append(row)

    table = output / "reliability_scan.csv"
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    scan = {}
    for k in SCAN_MULTIPLIERS:
        tag = f"k{k:g}"
        nonworse = sum(row[f"{tag}_nonworse"] for row in rows)
        identity = sum(np.isclose(row[f"{tag}_cap"], RF1M_IDENTITY_CAP) for row in rows)
        s360 = [row for row in rows if row["scanner"] == "s360"]
        scan[tag] = {
            "multiplier": k,
            "outer_nonworse": int(nonworse),
            "outer_total": len(rows),
            "outer_gamut_pass": int(sum(row[f"{tag}_gamut_pass"] for row in rows)),
            "identity_selections": int(identity),
            "s360_identity_selections": int(
                sum(np.isclose(row[f"{tag}_cap"], RF1M_IDENTITY_CAP) for row in s360)
            ),
            "mean_rmse_reduction_percent": float(
                np.mean([row[f"{tag}_rmse_reduction_percent"] for row in rows])
            ),
            "worst_rmse_reduction_percent": float(
                np.min([row[f"{tag}_rmse_reduction_percent"] for row in rows])
            ),
            "would_pass_frozen_gate": bool(nonworse == len(rows)),
        }
    summary = {
        "analysis": "rf1m_reliability_shrinkage_scan",
        "rf1m_version": RF1M_VERSION,
        "status": "DIAGNOSTIC_SCAN_NOT_A_FROZEN_SELECTOR",
        "outcome_access": False,
        "pfm_feature_access": False,
        "rule": "alpha = max(0, 1 - k * SE(best) / gain(best)), inner folds only",
        "strength_scale": "mean absolute log band gain of each rendered candidate",
        "caveat": (
            "Selecting k from this scan would be tuning on the same data that failed. "
            "A contract must fix k before evaluation or choose it in an inner loop."
        ),
        "cells": len(rows),
        "scan": scan,
        "artifacts": {table.name: sha256(table)},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

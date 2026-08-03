"""Compare historical and native-AA ERT and audit anchor-band sensitivity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_exp05_spectral_pilot import BANDS, SCANNERS, geometric_band


ANCHORS = {
    "primary_0.03_0.10": (0.03, 0.10),
    "lower_0.02_0.08": (0.02, 0.08),
    "broad_0.02_0.12": (0.02, 0.12),
    "upper_0.05_0.12": (0.05, 0.12),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical", default="outputs/exp05_spectral_cohort_109")
    parser.add_argument("--native-aa", default="outputs/e1_native_aa_spectral_109")
    parser.add_argument("--output", default="outputs/e0_native_aa_ert_sensitivity")
    return parser.parse_args()


def anchor_normalized_transfer(power, reference, frequency, bounds):
    ratio = np.sqrt(
        np.maximum(np.asarray(power, dtype=float), 1e-20)
        / np.maximum(np.asarray(reference, dtype=float), 1e-20)
    )
    selected = (frequency >= bounds[0]) & (frequency <= bounds[1])
    if not selected.any():
        raise ValueError(f"anchor {bounds} contains no radial bins")
    scale = np.exp(np.mean(np.log(np.maximum(ratio[selected], 1e-20))))
    return ratio / scale


def compare_routes(historical: pd.DataFrame, native: pd.DataFrame):
    keys = ["slide_id", "scanner", "band"]
    merged = historical[keys + ["log2_relative_transfer"]].merge(
        native[keys + ["log2_relative_transfer"]],
        on=keys,
        how="outer",
        suffixes=("_historical", "_native_aa"),
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all() or len(merged) != 109 * 6 * 3:
        raise ValueError("historical/native-AA band tables do not share the full identity")
    merged["delta_native_aa_minus_historical_log2"] = (
        merged["log2_relative_transfer_native_aa"]
        - merged["log2_relative_transfer_historical"]
    )
    summary = (
        merged.groupby(["scanner", "band"], as_index=False)
        .agg(
            historical_mean_log2=("log2_relative_transfer_historical", "mean"),
            native_aa_mean_log2=("log2_relative_transfer_native_aa", "mean"),
            signed_delta_median_log2=("delta_native_aa_minus_historical_log2", "median"),
            absolute_delta_median_log2=(
                "delta_native_aa_minus_historical_log2",
                lambda values: float(np.median(np.abs(values))),
            ),
            absolute_delta_q95_log2=(
                "delta_native_aa_minus_historical_log2",
                lambda values: float(np.quantile(np.abs(values), 0.95)),
            ),
        )
    )
    return merged.drop(columns="_merge"), summary


def anchor_sensitivity(spectra: pd.DataFrame):
    rows = []
    for slide_id, slide in spectra.groupby("slide_id", sort=True):
        reference = slide[slide["scanner"].eq("at2")].sort_values("frequency_cyc_per_um")
        frequency = reference["frequency_cyc_per_um"].to_numpy(float)
        reference_power = reference["radial_power"].to_numpy(float)
        for scanner in SCANNERS:
            frame = slide[slide["scanner"].eq(scanner)].sort_values("frequency_cyc_per_um")
            if not np.array_equal(frame["frequency_cyc_per_um"].to_numpy(float), frequency):
                raise ValueError(f"{slide_id}/{scanner}: radial frequency grid differs")
            power = frame["radial_power"].to_numpy(float)
            raw_ratio = np.sqrt(np.maximum(power, 1e-20) / np.maximum(reference_power, 1e-20))
            curves = {
                anchor: anchor_normalized_transfer(power, reference_power, frequency, bounds)
                for anchor, bounds in ANCHORS.items()
            }
            for band, bounds in BANDS.items():
                raw_value = geometric_band(raw_ratio, frequency, bounds)
                primary = geometric_band(curves["primary_0.03_0.10"], frequency, bounds)
                for anchor, curve in curves.items():
                    value = geometric_band(curve, frequency, bounds)
                    rows.append(
                        {
                            "slide_id": str(slide_id),
                            "scanner": scanner,
                            "band": band,
                            "anchor": anchor,
                            "anchor_lower": ANCHORS[anchor][0],
                            "anchor_upper": ANCHORS[anchor][1],
                            "log2_relative_transfer": float(np.log2(value)),
                            "delta_from_primary_log2": float(np.log2(value) - np.log2(primary)),
                            "unnormalized_amplitude_ratio": raw_value,
                            "unnormalized_log2_amplitude_ratio": float(np.log2(raw_value)),
                        }
                    )
    result = pd.DataFrame(rows)
    expected = 109 * 6 * len(BANDS) * len(ANCHORS)
    if len(result) != expected or not np.isfinite(result.select_dtypes(include=[np.number])).all().all():
        raise ValueError("anchor sensitivity output is incomplete or non-finite")
    summary = (
        result.groupby(["scanner", "band", "anchor"], as_index=False)
        .agg(
            mean_log2_transfer=("log2_relative_transfer", "mean"),
            median_delta_from_primary_log2=("delta_from_primary_log2", "median"),
            median_absolute_delta_from_primary_log2=(
                "delta_from_primary_log2",
                lambda values: float(np.median(np.abs(values))),
            ),
            q95_absolute_delta_from_primary_log2=(
                "delta_from_primary_log2",
                lambda values: float(np.quantile(np.abs(values), 0.95)),
            ),
            mean_unnormalized_log2_amplitude_ratio=(
                "unnormalized_log2_amplitude_ratio",
                "mean",
            ),
        )
    )
    return result, summary


def main():
    args = parse_args()
    historical_root = Path(args.historical)
    native_root = Path(args.native_aa)
    historical = pd.read_csv(historical_root / "slide_band_summary.csv", dtype={"slide_id": str})
    native = pd.read_csv(native_root / "slide_band_summary.csv", dtype={"slide_id": str})
    route_rows, route_summary = compare_routes(historical, native)
    spectra = pd.read_csv(native_root / "slide_spectra.csv", dtype={"slide_id": str})
    anchor_rows, anchor_summary = anchor_sensitivity(spectra)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    route_rows.to_csv(output / "historical_vs_native_aa_band_deltas.csv", index=False)
    route_summary.to_csv(output / "historical_vs_native_aa_summary.csv", index=False)
    anchor_rows.to_csv(output / "anchor_sensitivity.csv", index=False)
    anchor_summary.to_csv(output / "anchor_sensitivity_summary.csv", index=False)
    run_summary = {
        "analysis": "e0_native_aa_ert_and_anchor_sensitivity",
        "slides": 109,
        "scanners": list(SCANNERS),
        "bands": BANDS,
        "anchors": ANCHORS,
        "historical_route": "registered WSI plus legacy integer refinement",
        "primary_route": "final native WSI geometry plus explicit AA",
        "absolute_mtf_claim": False,
        "noise_floor_sensitivity_complete": False,
        "noise_floor_note": "requires background rendered through the same native-AA transform chain",
    }
    (output / "summary.json").write_text(json.dumps(run_summary, indent=2) + "\n")
    print(route_summary.to_string(index=False))
    print(anchor_summary.query("band == 'high'").to_string(index=False))
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()

"""Audit E1 ERT against operational background rendered by the same AA chain."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from render_e0_native_aa_shard import SCANNERS
from render_e0d_same_chain_background_slide import ANALYSIS_VERSION, TABLES
from run_exp05_spectral_pilot import BANDS, geometric_band, normalized_transfer


ANCHOR = (0.03, 0.10)
SNR_THRESHOLDS = (1.0, 2.0, 5.0, 10.0)
EPSILON = 1e-20


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shards", default="outputs/e0d_same_chain_background/shards"
    )
    parser.add_argument(
        "--tissue-spectra",
        default="outputs/e1_native_aa_spectral_109/slide_spectra.csv",
    )
    parser.add_argument(
        "--output", default="outputs/e0d_same_chain_background"
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    return parser.parse_args()


def load_shards(root: Path):
    directories = sorted(path.parent for path in root.glob("*/summary.json"))
    if len(directories) != 109:
        raise ValueError(f"expected 109 same-chain background shards, got {len(directories)}")
    tables = {name: [] for name in TABLES}
    summaries = []
    for directory in directories:
        summary = json.loads((directory / "summary.json").read_text())
        if (
            summary.get("analysis_version") != ANALYSIS_VERSION
            or summary.get("slide_gate_pass") is not True
        ):
            raise ValueError(f"invalid same-chain background shard: {directory}")
        summaries.append(summary)
        for name, filename in TABLES.items():
            tables[name].append(
                pd.read_csv(directory / filename, dtype={"slide_id": str})
            )
    result = {name: pd.concat(parts, ignore_index=True) for name, parts in tables.items()}
    expected = {
        "spectra": 109 * 6 * 72,
        "replicates": 109 * 6 * 5 * 72,
        "coordinates": 109 * 6 * 100,
        "qc": 109 * 6 * 100,
    }
    for name, frame in result.items():
        if len(frame) != expected[name]:
            raise ValueError(f"aggregate {name} rows {len(frame)} != {expected[name]}")
        numeric = frame.select_dtypes(include=[np.number])
        if not np.isfinite(numeric.to_numpy()).all():
            raise ValueError(f"aggregate {name} contains non-finite numeric values")
        if set(frame["scanner"].astype(str)) != set(SCANNERS):
            raise ValueError(f"aggregate {name} scanner panel differs from E1")
    identities = {
        "spectra": ["slide_id", "scanner", "frequency_cyc_per_um"],
        "replicates": [
            "slide_id",
            "scanner",
            "replicate",
            "frequency_cyc_per_um",
        ],
        "coordinates": ["slide_id", "scanner", "patch_index"],
        "qc": ["slide_id", "scanner", "patch_index"],
    }
    for name, keys in identities.items():
        if result[name].duplicated(keys).any():
            raise ValueError(f"aggregate {name} contains duplicate frozen identity")
    summary_slides = [str(item["slide_id"]) for item in summaries]
    if len(set(summary_slides)) != 109:
        raise ValueError("same-chain shard summaries do not contain 109 unique slides")
    return result, summaries


def merge_tissue_background(tissue: pd.DataFrame, background: pd.DataFrame):
    keys = ["slide_id", "scanner", "frequency_cyc_per_um"]
    keep = keys + ["tissue_type", "background_radial_power"]
    merged = tissue.merge(
        background[keep], on=keys, how="outer", validate="one_to_one", indicator=True
    )
    if len(merged) != 109 * 6 * 72 or not merged["_merge"].eq("both").all():
        raise ValueError("E1 tissue and E0d background spectra do not share full identity")
    merged = merged.drop(columns="_merge")
    merged = merged.rename(columns={"radial_power": "tissue_radial_power"})
    tissue_power = merged["tissue_radial_power"].to_numpy(float)
    background_power = merged["background_radial_power"].to_numpy(float)
    if (tissue_power < 0).any() or (background_power < 0).any():
        raise ValueError("negative input periodogram power")
    corrected = tissue_power - background_power
    merged["noise_subtracted_radial_power"] = corrected
    merged["negative_after_subtraction"] = corrected <= 0
    merged["tissue_to_background_snr"] = tissue_power / np.maximum(
        background_power, EPSILON
    )
    merged["background_fraction_of_tissue_power"] = background_power / np.maximum(
        tissue_power, EPSILON
    )
    return merged


def noise_floor_band_sensitivity(merged: pd.DataFrame):
    rows = []
    for slide_id, slide in merged.groupby("slide_id", sort=True):
        reference = slide[slide["scanner"].eq("at2")].sort_values(
            "frequency_cyc_per_um"
        )
        frequency = reference["frequency_cyc_per_um"].to_numpy(float)
        reference_corrected = reference["noise_subtracted_radial_power"].to_numpy(float)
        reference_snr = reference["tissue_to_background_snr"].to_numpy(float)
        anchor_mask = (frequency >= ANCHOR[0]) & (frequency <= ANCHOR[1])
        if not anchor_mask.any():
            raise ValueError("primary anchor has no frequency bins")
        for scanner in SCANNERS:
            frame = slide[slide["scanner"].eq(scanner)].sort_values(
                "frequency_cyc_per_um"
            )
            scanner_frequency = frame["frequency_cyc_per_um"].to_numpy(float)
            if not np.array_equal(scanner_frequency, frequency):
                raise ValueError(f"{slide_id}/{scanner}: frequency grid differs")
            corrected = frame["noise_subtracted_radial_power"].to_numpy(float)
            scanner_snr = frame["tissue_to_background_snr"].to_numpy(float)
            raw_curve = frame["relative_transfer"].to_numpy(float)
            for band, bounds in BANDS.items():
                band_mask = (frequency >= bounds[0]) & (frequency < bounds[1])
                required = anchor_mask | band_mask
                raw_value = geometric_band(raw_curve, frequency, bounds)
                for threshold in SNR_THRESHOLDS:
                    eligible = bool(
                        np.all(corrected[required] > 0)
                        and np.all(reference_corrected[required] > 0)
                        and np.all(scanner_snr[required] >= threshold)
                        and np.all(reference_snr[required] >= threshold)
                    )
                    corrected_value = np.nan
                    corrected_log2 = np.nan
                    delta = np.nan
                    if eligible:
                        curve = normalized_transfer(
                            corrected, reference_corrected, frequency
                        )
                        corrected_value = geometric_band(curve, frequency, bounds)
                        corrected_log2 = float(np.log2(corrected_value))
                        delta = corrected_log2 - float(np.log2(raw_value))
                    rows.append(
                        {
                            "slide_id": str(slide_id),
                            "tissue_type": str(frame["tissue_type"].iloc[0]),
                            "scanner": scanner,
                            "band": band,
                            "lower_frequency": bounds[0],
                            "upper_frequency": bounds[1],
                            "anchor_lower": ANCHOR[0],
                            "anchor_upper": ANCHOR[1],
                            "minimum_snr_threshold": threshold,
                            "eligible": eligible,
                            "raw_relative_transfer": raw_value,
                            "raw_log2_relative_transfer": float(np.log2(raw_value)),
                            "noise_corrected_relative_transfer": corrected_value,
                            "noise_corrected_log2_relative_transfer": corrected_log2,
                            "delta_corrected_minus_raw_log2": delta,
                            "minimum_scanner_snr_required_bins": float(
                                np.min(scanner_snr[required])
                            ),
                            "minimum_at2_snr_required_bins": float(
                                np.min(reference_snr[required])
                            ),
                            "negative_scanner_bins_required": int(
                                np.sum(corrected[required] <= 0)
                            ),
                            "negative_at2_bins_required": int(
                                np.sum(reference_corrected[required] <= 0)
                            ),
                        }
                    )
    result = pd.DataFrame(rows)
    expected = 109 * 6 * len(BANDS) * len(SNR_THRESHOLDS)
    if len(result) != expected:
        raise ValueError(f"band sensitivity rows {len(result)} != {expected}")
    return result


def stable_seed(*values) -> int:
    digest = hashlib.sha256(":".join(map(str, values)).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def summarize_band_sensitivity(rows: pd.DataFrame, bootstrap: int):
    if bootstrap < 100:
        raise ValueError("at least 100 slide-bootstrap replicates are required")
    summaries = []
    draws = []
    groups = ["scanner", "band", "minimum_snr_threshold"]
    for keys, frame in rows.groupby(groups, sort=True):
        scanner, band, threshold = keys
        eligible = frame[frame["eligible"]].copy()
        values = eligible["delta_corrected_minus_raw_log2"].dropna().to_numpy(float)
        ci_low = np.nan
        ci_high = np.nan
        median_delta = np.nan
        mean_corrected = np.nan
        mean_raw = np.nan
        if len(values):
            median_delta = float(np.median(values))
            mean_corrected = float(
                eligible["noise_corrected_log2_relative_transfer"].mean()
            )
            mean_raw = float(eligible["raw_log2_relative_transfer"].mean())
            rng = np.random.default_rng(
                stable_seed("e0d", scanner, band, threshold)
            )
            indices = rng.integers(0, len(values), size=(bootstrap, len(values)))
            boot_values = np.median(values[indices], axis=1)
            ci_low, ci_high = np.quantile(boot_values, [0.025, 0.975])
            for replicate, value in enumerate(boot_values):
                draws.append(
                    {
                        "scanner": scanner,
                        "band": band,
                        "minimum_snr_threshold": threshold,
                        "bootstrap_replicate": replicate,
                        "median_delta_corrected_minus_raw_log2": float(value),
                    }
                )
        summaries.append(
            {
                "scanner": scanner,
                "band": band,
                "minimum_snr_threshold": threshold,
                "slides_total": len(frame),
                "slides_eligible": len(eligible),
                "eligible_fraction": float(len(eligible) / len(frame)),
                "mean_raw_log2_transfer_eligible": mean_raw,
                "mean_noise_corrected_log2_transfer": mean_corrected,
                "median_delta_corrected_minus_raw_log2": median_delta,
                "median_delta_bootstrap_ci95_low": float(ci_low),
                "median_delta_bootstrap_ci95_high": float(ci_high),
                "median_minimum_scanner_snr": float(
                    frame["minimum_scanner_snr_required_bins"].median()
                ),
                "median_minimum_at2_snr": float(
                    frame["minimum_at2_snr_required_bins"].median()
                ),
                "slides_with_negative_required_scanner_bins": int(
                    (frame["negative_scanner_bins_required"] > 0).sum()
                ),
                "slides_with_negative_required_at2_bins": int(
                    (frame["negative_at2_bins_required"] > 0).sum()
                ),
            }
        )
    return pd.DataFrame(summaries), pd.DataFrame(draws)


def summarize_spectra(merged: pd.DataFrame):
    rows = []
    for scanner in SCANNERS:
        frame = merged[merged["scanner"].eq(scanner)]
        frequency = frame["frequency_cyc_per_um"].to_numpy(float)
        for band, bounds in BANDS.items():
            selected = frame[
                (frame["frequency_cyc_per_um"] >= bounds[0])
                & (frame["frequency_cyc_per_um"] < bounds[1])
            ]
            rows.append(
                {
                    "scanner": scanner,
                    "band": band,
                    "bins": len(selected),
                    "negative_after_subtraction_fraction": float(
                        selected["negative_after_subtraction"].mean()
                    ),
                    "background_fraction_median": float(
                        selected["background_fraction_of_tissue_power"].median()
                    ),
                    "background_fraction_q95": float(
                        selected["background_fraction_of_tissue_power"].quantile(0.95)
                    ),
                    "tissue_to_background_snr_median": float(
                        selected["tissue_to_background_snr"].median()
                    ),
                    "tissue_to_background_snr_q05": float(
                        selected["tissue_to_background_snr"].quantile(0.05)
                    ),
                }
            )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    shard_tables, shard_summaries = load_shards(Path(args.shards))
    tissue = pd.read_csv(args.tissue_spectra, dtype={"slide_id": str})
    merged = merge_tissue_background(tissue, shard_tables["spectra"])
    band_rows = noise_floor_band_sensitivity(merged)
    band_summary, bootstrap_draws = summarize_band_sensitivity(
        band_rows, args.bootstrap
    )
    spectral_summary = summarize_spectra(merged)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    shard_tables["spectra"].to_csv(
        output / "same_chain_background_spectra.csv.gz",
        index=False,
        compression="gzip",
    )
    shard_tables["replicates"].to_csv(
        output / "same_chain_background_replicate_spectra.csv.gz",
        index=False,
        compression="gzip",
    )
    shard_tables["coordinates"].to_csv(
        output / "same_chain_rendered_coordinates.csv.gz",
        index=False,
        compression="gzip",
    )
    shard_tables["qc"].to_csv(
        output / "same_chain_post_render_glass_qc.csv.gz",
        index=False,
        compression="gzip",
    )
    merged.to_csv(
        output / "tissue_background_noise_subtracted_spectra.csv.gz",
        index=False,
        compression="gzip",
    )
    band_rows.to_csv(output / "noise_floor_band_sensitivity.csv", index=False)
    band_summary.to_csv(
        output / "noise_floor_band_sensitivity_summary.csv", index=False
    )
    bootstrap_draws.to_csv(
        output / "noise_floor_slide_bootstrap.csv.gz",
        index=False,
        compression="gzip",
    )
    spectral_summary.to_csv(output / "same_chain_snr_summary.csv", index=False)

    qc = shard_tables["qc"]
    summary = {
        "analysis": "e0d_same_chain_background_noise_floor_sensitivity",
        "analysis_version": ANALYSIS_VERSION,
        "slides": 109,
        "scanners": list(SCANNERS),
        "background_patches": len(qc),
        "background_patches_per_scanner_slide": 100,
        "snr_thresholds": list(SNR_THRESHOLDS),
        "primary_anchor": list(ANCHOR),
        "bands": BANDS,
        "bootstrap_unit": "physical slide among SNR-eligible slides",
        "bootstrap_replicates": args.bootstrap,
        "negative_subtractions_are_reported_not_clipped": True,
        "post_render_glass_qc_pass_fraction": float(qc["post_render_qc_pass"].mean()),
        "maximum_black_pixel_fraction": float(qc["black_pixel_fraction"].max()),
        "shard_post_render_qc_pass_fraction_range": [
            float(min(item["post_render_qc_pass_fraction"] for item in shard_summaries)),
            float(max(item["post_render_qc_pass_fraction"] for item in shard_summaries)),
        ],
        "claim_scope": "operational glass/background noise floor through the E1 processing chain; not detector NPS, DQE, or absolute MTF",
        "absolute_mtf_claim": False,
        "noise_floor_sensitivity_complete": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(spectral_summary.query("band == 'high'").to_string(index=False))
    print(band_summary.query("band == 'high'").to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

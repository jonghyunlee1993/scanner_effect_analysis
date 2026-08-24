"""Acquisition provenance for the manuscript: resampling ratio and scan schedule.

Two reviewer questions about the spectral characterization are answerable from
the native headers alone, and neither is currently answered in the report.

The first is whether the measured transfer is the scanner or the render chain.
Every scan is resampled onto one 0.5052 um/px grid, and resampling can only
remove high-frequency content, so a reviewer is right to ask whether the
ordering in section 3.1 is an artefact of which scanner was reduced hardest.
The header audit already holds the native pixel size, so the ratio is a join
away, and the check is direct: if transfer tracked the reduction, the most
heavily reduced scanner would be the softest.

The second is scan schedule. Six acquisitions of one physical slide are only a
clean scanner contrast if section fading over the interval between them is not
doing the work. Dates live in the TIFF DateTime tag for some vendors and inside
the ImageDescription for others, and the existing audit truncates the
description before reaching them, so this re-reads the headers.

Neither result is a new measurement -- no pixel is decoded, only headers -- and
the locked transfer estimates are read, not recomputed.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import openslide
import pandas as pd
import tifffile

from fetch_e0_pfm_checkpoints import sha256


TARGET_MPP = 0.5052
SCANNERS = ("at2", "gt450", "versa", "akoya", "s60", "s360")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="outputs/e0_native_geometry_final/native_geometry_manifest.csv",
    )
    parser.add_argument(
        "--locked", default="outputs/e0_e3_locked_results/locked_results.csv"
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", default="outputs/e8_acquisition_provenance")
    return parser.parse_args()


def first_match(pattern: str, text: str):
    found = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    return found.group(1).strip() if found else ""


def parse_timestamp(text: str, tiff_datetime: str):
    """Return an ISO-ish timestamp from whichever vendor field carries one.

    Aperio writes `Date = 12/09/25|Time = 11:55:19` into the ImageDescription,
    Versa writes an ISO instant, Hamamatsu and Akoya use the TIFF DateTime tag.
    The raw string is kept alongside so nothing is silently reinterpreted.
    """
    iso = first_match(r"(?:^|\|)Date\s*=\s*(\d{4}-\d{2}-\d{2}T[\d:]+Z?)", text)
    if iso:
        return iso.replace("T", " ").rstrip("Z"), "description_iso"
    date = first_match(r"(?:^|\|)Date\s*=\s*(\d{1,2}/\d{1,2}/\d{2,4})", text)
    if date:
        time = first_match(r"(?:^|\|)Time\s*=\s*([\d:]+)", text)
        month, day, year = date.split("/")
        year = f"20{year}" if len(year) == 2 else year
        return f"{year}-{int(month):02d}-{int(day):02d} {time or '00:00:00'}", "description_us"
    if tiff_datetime:
        return tiff_datetime.replace(":", "-", 2), "tiff_datetime"
    return "", "absent"


def read_one(row):
    path = Path(row["native_path"])
    record = {
        "scanner": row["scanner"],
        "slide_id": row["slide_id"],
        "native_path": str(path),
        "native_mpp": float("nan"),
        "timestamp": "",
        "timestamp_source": "unreadable",
    }
    try:
        with openslide.OpenSlide(str(path)) as slide:
            properties = dict(slide.properties)
        description = properties.get("openslide.comment", "") or ""
        mpp = properties.get("openslide.mpp-x")
        if mpp:
            record["native_mpp"] = float(mpp)
        tiff_datetime = ""
        with tifffile.TiffFile(str(path)) as handle:
            page = handle.pages[0]
            tag = page.tags.get("DateTime")
            if tag is not None:
                tiff_datetime = str(tag.value).strip()
            if not description:
                image_description = page.tags.get("ImageDescription")
                if image_description is not None:
                    description = str(image_description.value)
        if not np.isfinite(record["native_mpp"]):
            # OpenSlide exposes mpp-x only for vendors it recognises; VERSA is
            # read as generic TIFF, so its pixel size has to come from the
            # vendor's own ImageDescription field.
            described = first_match(r"(?:^|\|)MPP\s*=\s*([\d.]+)", description)
            if described:
                record["native_mpp"] = float(described)
        stamp, source = parse_timestamp(description, tiff_datetime)
        record["timestamp"] = stamp
        record["timestamp_source"] = source
    except Exception as error:  # header-only read; a failure is data, not a crash
        record["timestamp_source"] = f"error:{type(error).__name__}"
        print(f"{row['scanner']}/{row['slide_id']}: {error}", flush=True)
    return record


def locked_transfer(path: Path) -> dict:
    """High-band AT2-relative log2 transfer, read from the locked E1 result."""
    transfer = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if (
                row["phase"] == "E1"
                and row["endpoint"] == "AT2-relative fixed log2 transfer"
                and row["band"] == "high"
            ):
                transfer[row["scanner"]] = {
                    "log2_estimate": float(row["estimate"]),
                    "ci95_low": float(row["ci95_low"]),
                    "ci95_high": float(row["ci95_high"]),
                    "fold_vs_at2": 2.0 ** float(row["estimate"]),
                }
    transfer["at2"] = {
        "log2_estimate": 0.0,
        "ci95_low": 0.0,
        "ci95_high": 0.0,
        "fold_vs_at2": 1.0,
    }
    return transfer


def main():
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    required = {"scanner", "slide_id", "native_path"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"manifest needs {sorted(required)}, has {list(manifest.columns)}")
    # The geometry manifest is one row per scanner-slide-location; headers are a
    # per-file property, so collapse to the 654 distinct native files first.
    files = (
        manifest[["scanner", "slide_id", "native_path"]]
        .drop_duplicates()
        .sort_values(["scanner", "slide_id"])
    )
    rows = files.to_dict("records")
    if len(rows) != 654:
        raise ValueError(f"expected 654 scanner-slide files, got {len(rows)}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(read_one, rows))
    frame = pd.DataFrame(records)

    transfer = locked_transfer(Path(args.locked))
    provenance = []
    for scanner in SCANNERS:
        block = frame[frame["scanner"] == scanner]
        if len(block) != 109:
            raise ValueError(f"{scanner}: {len(block)} files, expected 109")
        native = float(block["native_mpp"].median())
        stamps = sorted(value for value in block["timestamp"] if value)
        provenance.append(
            {
                "scanner": scanner,
                "native_mpp": native,
                "reduction_to_target": TARGET_MPP / native,
                "high_band_fold_vs_at2": transfer[scanner]["fold_vs_at2"],
                "high_band_log2": transfer[scanner]["log2_estimate"],
                "ci95_low": transfer[scanner]["ci95_low"],
                "ci95_high": transfer[scanner]["ci95_high"],
                "timestamps_recovered": len(stamps),
                "first_scan": stamps[0] if stamps else "",
                "last_scan": stamps[-1] if stamps else "",
                "timestamp_source": (
                    block["timestamp_source"].mode().iat[0] if len(block) else ""
                ),
            }
        )
    provenance_frame = pd.DataFrame(provenance)

    # The direct rebuttal: resampling only removes high-frequency content, so if
    # the reduction drove the ordering the hardest-reduced scanner would be the
    # softest. Restricting to the barely-resampled scanners makes it sharper --
    # whatever spread survives there cannot be the render chain.
    light = provenance_frame[provenance_frame["reduction_to_target"] <= 1.15]
    spearman = float(
        provenance_frame["reduction_to_target"].corr(
            provenance_frame["high_band_fold_vs_at2"], method="spearman"
        )
    )
    resampling_check = {
        "spearman_reduction_vs_transfer": spearman,
        "hardest_reduced_scanner": str(
            provenance_frame.loc[provenance_frame["reduction_to_target"].idxmax(), "scanner"]
        ),
        "hardest_reduced_fold": float(provenance_frame["reduction_to_target"].max()),
        "hardest_reduced_transfer": float(
            provenance_frame.loc[
                provenance_frame["reduction_to_target"].idxmax(), "high_band_fold_vs_at2"
            ]
        ),
        "light_resample_scanners": light["scanner"].tolist(),
        "light_resample_transfer_min": float(light["high_band_fold_vs_at2"].min()),
        "light_resample_transfer_max": float(light["high_band_fold_vs_at2"].max()),
        "light_resample_transfer_span": float(
            light["high_band_fold_vs_at2"].max() / max(light["high_band_fold_vs_at2"].min(), 1e-12)
        ),
    }

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    provenance_path = output / "scanner_provenance.csv"
    schedule_path = output / "scan_schedule.csv"
    provenance_frame.to_csv(provenance_path, index=False)
    frame.sort_values(["scanner", "timestamp"]).to_csv(schedule_path, index=False)

    summary = {
        "analysis": "e8_acquisition_provenance",
        "status": "POST_CORE_EXPLORATORY_EXTENSION",
        "note": (
            "Header-only. No pixels decoded, no transfer estimate recomputed; "
            "the locked E1 high-band values are joined, not refitted."
        ),
        "target_mpp": TARGET_MPP,
        "files": len(frame),
        "resampling_check": resampling_check,
        "timestamps_recovered_by_scanner": {
            row["scanner"]: row["timestamps_recovered"] for row in provenance
        },
        "artifacts": {
            provenance_path.name: sha256(provenance_path),
            schedule_path.name: sha256(schedule_path),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(provenance_frame.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

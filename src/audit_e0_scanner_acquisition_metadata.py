"""Audit acquisition metadata from every native scanner-slide WSI header."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import openslide
import pandas as pd
import tifffile


SCANNERS = ("at2", "gt450", "versa", "akoya", "s60", "s360")
REPORTED_MODELS = {
    "at2": "Aperio AT2",
    "gt450": "Aperio Leica Biosystems GT450",
    "versa": "VERSA (exact manufacturer/model confirmation pending)",
    "akoya": "Akoya/PerkinElmer QPI instrument (exact model confirmation pending)",
    "s60": "Hamamatsu S60; TIFF model C13210",
    "s360": "Hamamatsu S360; TIFF model C13220",
}
MODEL_EVIDENCE = {
    "at2": "cohort label + Aperio SVS header/ICC profile",
    "gt450": "Aperio ScannerType and ImageDescription",
    "versa": "native ImageDescription label only",
    "akoya": "PerkinElmer-QPI header and cohort label only",
    "s60": "cohort label + TIFF Make/Model",
    "s360": "cohort label + TIFF Make/Model",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="outputs/e0_native_geometry_final/native_geometry_manifest.csv",
    )
    parser.add_argument("--output", default="outputs/e0_scanner_acquisition_metadata")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def first_match(pattern: str, text: str):
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def parse_description(text: str):
    jpeg_quality = first_match(r"(?:^|[|\s])Q\s*=\s*(\d+)", text)
    if not jpeg_quality:
        jpeg_quality = first_match(r"<JPEGQuality>(.*?)</JPEGQuality>", text)
    product = first_match(r"^(.*?)(?=\d+x\d+)", text)
    if "<" in product:
        product = ""
    product = re.sub(r"\s+", " ", product).strip().removesuffix("->").strip()
    return {
        "description_prefix": re.sub(r"\s+", " ", text[:160]).strip(),
        "description_product": product,
        "description_scanner_type": first_match(r"(?:^|\|)ScannerType\s*=\s*([^|<]+)", text),
        "description_app_mag": first_match(r"(?:^|\|)AppMag\s*=\s*([^|<]+)", text),
        "description_mpp": first_match(r"(?:^|\|)MPP\s*=\s*([^|<]+)", text),
        "description_jpeg_quality": jpeg_quality,
        "acquisition_software": first_match(r"<AcquisitionSoftware>(.*?)</AcquisitionSoftware>", text),
        "description_objective": first_match(r"<Objective>(.*?)</Objective>", text),
        "scan_profile_name": first_match(r"<Name>(.*?)</Name>", text),
        "description_pixel_size_um": first_match(r"<PixelSizeMicrons>(.*?)</PixelSizeMicrons>", text),
        "description_magnification": first_match(r"<Magnification>(.*?)</Magnification>", text),
        "description_objective_name": first_match(r"<ObjectiveName>(.*?)</ObjectiveName>", text),
        "description_compression": first_match(r"<Compression>(.*?)</Compression>", text),
        "description_instrument_type": first_match(r"<InstrumentType>(.*?)</InstrumentType>", text),
        "description_camera_type": first_match(r"<CameraType>(.*?)</CameraType>", text),
    }


def as_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def read_slide_header(record: dict):
    path = Path(record["native_path"])
    row = {
        "scanner": record["scanner"],
        "slide_id": str(record["slide_id"]),
        "native_path": str(path),
        "suffix": path.suffix.lower(),
        "file_bytes": path.stat().st_size if path.is_file() else 0,
        "header_status": "pass",
        "error": "",
    }
    try:
        with openslide.OpenSlide(str(path)) as slide:
            properties = slide.properties
            row.update(
                {
                    "width": slide.dimensions[0],
                    "height": slide.dimensions[1],
                    "levels": slide.level_count,
                    "openslide_vendor": properties.get("openslide.vendor", ""),
                    "openslide_mpp_x": as_float(properties.get("openslide.mpp-x")),
                    "openslide_mpp_y": as_float(properties.get("openslide.mpp-y")),
                    "openslide_objective_power": as_float(properties.get("openslide.objective-power")),
                }
            )
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            tags = page.tags
            description = str(tags["ImageDescription"].value) if "ImageDescription" in tags else ""
            row.update(parse_description(description))
            row.update(
                {
                    "tiff_compression": page.compression.name,
                    "tiff_photometric": page.photometric.name,
                    "tile_width": page.tilewidth or "",
                    "tile_height": page.tilelength or "",
                    "tiff_make": str(tags["Make"].value) if "Make" in tags else "",
                    "tiff_model": str(tags["Model"].value) if "Model" in tags else "",
                    "tiff_software": str(tags["Software"].value) if "Software" in tags else "",
                    "tiff_datetime": str(tags["DateTime"].value) if "DateTime" in tags else "",
                }
            )
        description_mpp = as_float(row.get("description_mpp"))
        row["effective_mpp_x"] = (
            row["openslide_mpp_x"] if np.isfinite(row["openslide_mpp_x"]) else description_mpp
        )
        row["effective_mpp_y"] = (
            row["openslide_mpp_y"] if np.isfinite(row["openslide_mpp_y"]) else description_mpp
        )
    except Exception as error:
        row["header_status"] = "fail"
        row["error"] = f"{type(error).__name__}: {error}"
    return row


def joined_unique(series: pd.Series):
    values = sorted({str(value).strip() for value in series.dropna() if str(value).strip()})
    return " | ".join(values)


def numeric_values(series: pd.Series):
    values = sorted({float(value) for value in series.dropna() if np.isfinite(float(value))})
    return " | ".join(f"{value:g}" for value in values)


def summarize_scanners(headers: pd.DataFrame):
    rows = []
    for scanner in SCANNERS:
        group = headers[headers["scanner"].eq(scanner)].copy()
        effective = group["effective_mpp_x"].dropna()
        row = {
            "scanner": scanner,
            "reported_scanner_model": REPORTED_MODELS[scanner],
            "model_evidence": MODEL_EVIDENCE[scanner],
            "scanner_slides": group["slide_id"].nunique(),
            "native_files": group["native_path"].nunique(),
            "header_failures": int(group["header_status"].ne("pass").sum()),
            "file_suffixes": joined_unique(group["suffix"]),
            "openslide_vendors": joined_unique(group["openslide_vendor"]),
            "native_mpp_x_median": float(effective.median()) if len(effective) else np.nan,
            "native_mpp_x_min": float(effective.min()) if len(effective) else np.nan,
            "native_mpp_x_max": float(effective.max()) if len(effective) else np.nan,
            "objective_power_values": numeric_values(group["openslide_objective_power"]),
            "description_app_mag_values": joined_unique(group["description_app_mag"]),
            "description_objective_values": joined_unique(group["description_objective"]),
            "description_magnification_values": joined_unique(group["description_magnification"]),
            "compression_values": joined_unique(group["tiff_compression"]),
            "jpeg_quality_values": joined_unique(group["description_jpeg_quality"]),
            "photometric_values": joined_unique(group["tiff_photometric"]),
            "tile_geometry_values": joined_unique(
                group["tile_width"].astype(str) + "x" + group["tile_height"].astype(str)
            ),
            "tiff_make_values": joined_unique(group["tiff_make"]),
            "tiff_model_values": joined_unique(group["tiff_model"]),
            "tiff_software_values": joined_unique(group["tiff_software"]),
            "acquisition_software_values": joined_unique(group["acquisition_software"]),
            "scan_profile_values": joined_unique(group["scan_profile_name"]),
            "description_product_values": joined_unique(group["description_product"]),
            "objective_na_status": "not present in native WSI headers",
            "firmware_status": "not separately identifiable from acquisition-software metadata",
        }
        row["mpp_range"] = row["native_mpp_x_max"] - row["native_mpp_x_min"]
        row["mpp_consistent_within_0_005"] = bool(row["mpp_range"] <= 0.005)
        rows.append(row)
    return pd.DataFrame(rows)


def build_missing_metadata(summary: pd.DataFrame):
    rows = []
    for scanner in SCANNERS:
        rows.extend(
            [
                {
                    "scanner": scanner,
                    "field": "objective_numerical_aperture",
                    "status": "requires acquisition record or operator confirmation",
                },
                {
                    "scanner": scanner,
                    "field": "firmware_version",
                    "status": "requires acquisition record or operator confirmation",
                },
            ]
        )
    for scanner in ("versa", "akoya"):
        rows.append(
            {
                "scanner": scanner,
                "field": "exact_manufacturer_and_model",
                "status": "cohort label/header insufficient; requires operator confirmation",
            }
        )
    for scanner in ("s60", "s360"):
        rows.append(
            {
                "scanner": scanner,
                "field": "jpeg_quality_setting",
                "status": "JPEG compression detected but quality value absent from header",
            }
        )
    return pd.DataFrame(rows)


def run(manifest_path: Path, output: Path, workers: int):
    manifest = pd.read_csv(manifest_path, dtype={"slide_id": str})
    cells = manifest[["scanner", "slide_id", "native_path"]].drop_duplicates()
    if len(cells) != 654 or cells.duplicated(["scanner", "slide_id"]).any():
        raise ValueError("native manifest does not contain 654 unique scanner-slide cells")
    if set(cells["scanner"]) != set(SCANNERS):
        raise ValueError("native manifest scanner set differs")

    records = cells.sort_values(["scanner", "slide_id"]).to_dict("records")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        headers = pd.DataFrame(executor.map(read_slide_header, records))
    summary = summarize_scanners(headers)
    missing = build_missing_metadata(summary)

    header_gate = bool(
        len(headers) == 654
        and headers["header_status"].eq("pass").all()
        and summary["scanner_slides"].eq(109).all()
        and summary["native_files"].eq(109).all()
        and summary["mpp_consistent_within_0_005"].all()
    )
    result = {
        "analysis": "e0_scanner_acquisition_metadata_audit",
        "native_manifest": str(manifest_path),
        "native_manifest_sha256": sha256_file(manifest_path),
        "scanner_slide_headers_expected": 654,
        "scanner_slide_headers_observed": len(headers),
        "scanner_slide_headers_passing": int(headers["header_status"].eq("pass").sum()),
        "scanners": list(SCANNERS),
        "slides_per_scanner": {
            row["scanner"]: int(row["scanner_slides"]) for _, row in summary.iterrows()
        },
        "native_mpp_consistency_gate": "within-scanner max-min <= 0.005 micrometres/pixel",
        "header_population_gate_pass": header_gate,
        "manuscript_acquisition_metadata_complete": False,
        "remaining_user_fields": sorted(missing["field"].unique()),
        "claim_scope": "native WSI header audit; software strings are not assumed to be firmware",
    }
    output.mkdir(parents=True, exist_ok=True)
    headers.to_csv(output / "scanner_slide_headers.csv", index=False)
    summary.to_csv(output / "scanner_summary.csv", index=False)
    missing.to_csv(output / "missing_metadata.csv", index=False)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not header_gate:
        raise SystemExit(2)
    return result


def main():
    args = parse_args()
    run(Path(args.manifest), Path(args.output), args.workers)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Paired slide/section audit of incremental frequency correction."""

from pathlib import Path
import json
import hashlib

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/scanner_batch_effect_analysis_2026-09-17"
HERE = Path(__file__).resolve().parent
OUT = HERE / "results/frequency_pfm_audit"
ARMS = ["raw", "reinhard", "combined"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_mean(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), (20000, len(values)))].mean(axis=1)
    return float(values.mean()), *map(float, np.quantile(samples, [0.025, 0.975]))


def summarize_delta(frame: pd.DataFrame, group: str, unit: str, left: str,
                    right: str, metric: str, seed: int) -> dict:
    wide = frame.pivot(index=unit, columns="arm", values=metric)
    if wide[[left, right]].isna().any().any():
        raise ValueError(f"Missing paired values in {group}: {metric}")
    delta = (wide[left] - wide[right]).to_numpy(float)
    estimate, lo, hi = bootstrap_mean(delta, seed)
    return {"group": group, "metric": metric, "contrast": f"{left}-{right}",
            "estimate": estimate, "ci_low": lo, "ci_high": hi,
            "positive_units": int((delta > 0).sum()), "n_units": len(delta)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    uni_path = BASE / "03_uni/location_metrics.csv"
    image_path = BASE / "02_correction/location_metrics.csv"
    plism_path = BASE / "12_manuscript_completion/05_plism_external_correction/plism_external_section_metrics.csv"
    uni = pd.read_csv(uni_path, dtype={"slide_id": str})
    uni = uni[uni.arm.isin(ARMS)]
    uni_slide = uni.groupby(["slide_id", "tissue_type", "scanner", "arm"], as_index=False).agg(
        target_gain=("target_gain", "mean"),
        uni_distance=("generated_target_cosine", lambda x: 1 - x.mean()),
    )
    if len(uni_slide) != 103 * 5 * 3:
        raise ValueError(f"UNI slide rows: {len(uni_slide)}")
    image = pd.read_csv(image_path, dtype={"slide_id": str})
    image = image[image.arm.isin(ARMS)]
    image_slide = image.groupby(["slide_id", "tissue_type", "scanner", "arm"], as_index=False)[
        ["distance", "fidelity_ncc", "target_gradient_ncc", "saturation_fraction",
         "generated_frequency_low_mid", "generated_frequency_mid", "generated_frequency_high",
         "target_frequency_low_mid", "target_frequency_mid", "target_frequency_high"]
    ].mean()
    if len(image_slide) != 103 * 5 * 3:
        raise ValueError(f"Image slide rows: {len(image_slide)}")
    means = uni_slide.groupby(["scanner", "arm"], as_index=False)[["target_gain", "uni_distance"]].mean()
    means = means.merge(image_slide.groupby(["scanner", "arm"], as_index=False)[
        ["distance", "fidelity_ncc", "target_gradient_ncc", "saturation_fraction"]
    ].mean(), on=["scanner", "arm"], validate="one_to_one")
    means.to_csv(OUT / "pannormal_scanner_arm_means.csv", index=False)
    for band in ["low_mid", "mid", "high"]:
        image_slide[f"frequency_residual_{band}"] = (
            image_slide[f"generated_frequency_{band}"] - image_slide[f"target_frequency_{band}"]
        )
    rows = []
    scanners = sorted(uni_slide.scanner.unique())
    for n, scanner in enumerate(scanners + ["all"]):
        u = uni_slide if scanner == "all" else uni_slide[uni_slide.scanner == scanner]
        i = image_slide if scanner == "all" else image_slide[image_slide.scanner == scanner]
        if scanner == "all":
            u = u.groupby(["slide_id", "arm"], as_index=False)[["target_gain", "uni_distance"]].mean()
            i = i.groupby(["slide_id", "arm"], as_index=False)[
                ["distance", "fidelity_ncc", "target_gradient_ncc", "saturation_fraction"]
            ].mean()
        rows.append(summarize_delta(u, scanner, "slide_id", "combined", "reinhard", "target_gain", 20+n))
        rows.append(summarize_delta(i, scanner, "slide_id", "reinhard", "combined", "distance", 30+n))
        rows.append(summarize_delta(i, scanner, "slide_id", "combined", "reinhard", "fidelity_ncc", 40+n))
        rows.append(summarize_delta(i, scanner, "slide_id", "combined", "reinhard", "target_gradient_ncc", 50+n))
        rows.append(summarize_delta(i, scanner, "slide_id", "reinhard", "combined", "saturation_fraction", 60+n))

    # PLISM has one physical section per observation and three shared scanners.
    plism = pd.read_csv(plism_path)
    plism = plism[(plism.analysis_set == "primary_all") & (plism.method.isin(ARMS))]
    plism = plism.rename(columns={"method": "arm", "section": "section_id", "uni_target_gain": "target_gain"})
    for n, scanner in enumerate(sorted(plism.scanner.unique()) + ["all"]):
        p = plism if scanner == "all" else plism[plism.scanner == scanner]
        if scanner == "all":
            p = p.groupby(["section_id", "arm"], as_index=False)[["target_gain", "distance", "source_gradient_ncc"]].mean()
        rows.append(summarize_delta(p, f"PLISM_{scanner}", "section_id", "combined", "reinhard", "target_gain", 70+n))
        rows.append(summarize_delta(p, f"PLISM_{scanner}", "section_id", "reinhard", "combined", "distance", 80+n))

    pd.DataFrame(rows).to_csv(OUT / "paired_incremental_summary.csv", index=False)
    band_rows = []
    for scanner in scanners:
        subset = image_slide[image_slide.scanner == scanner]
        for arm in ARMS:
            arm_rows = subset[subset.arm == arm]
            for band in ["low_mid", "mid", "high"]:
                residual = arm_rows[f"frequency_residual_{band}"].to_numpy(float)
                band_rows.append({"scanner": scanner, "arm": arm, "band": band,
                                  "mean_signed_residual": float(residual.mean()),
                                  "mean_abs_residual": float(np.abs(residual).mean())})
    pd.DataFrame(band_rows).to_csv(OUT / "frequency_residual_by_scanner.csv", index=False)
    tissue = uni_slide[uni_slide.scanner == "akoya"].pivot(index=["tissue_type", "slide_id"], columns="arm", values="target_gain")
    tissue["incremental_gain"] = tissue["combined"] - tissue["reinhard"]
    tissue = tissue.reset_index().groupby("tissue_type", as_index=False).agg(
        n_slides=("slide_id", "size"), mean_incremental_gain=("incremental_gain", "mean"),
        positive_slides=("incremental_gain", lambda x: int((x > 0).sum())),
    )
    tissue.to_csv(OUT / "akoya_tissue_consistency.csv", index=False)
    output_names = ["paired_incremental_summary.csv", "frequency_residual_by_scanner.csv",
                    "akoya_tissue_consistency.csv", "pannormal_scanner_arm_means.csv"]
    manifest = {"unit_pan": "physical_slide", "unit_plism": "physical_section",
                "bootstrap_replicates": 20000, "contrast": "combined minus Reinhard for UNI gain",
                "n_pan_slides": int(uni_slide.slide_id.nunique()),
                "n_plism_sections": int(plism.section_id.nunique()),
                "input_sha256": {str(path): sha256(path) for path in [uni_path, image_path, plism_path]},
                "code_sha256": sha256(Path(__file__)),
                "output_sha256": {name: sha256(OUT / name) for name in output_names},
                "note": "Exploratory audit of existing frozen outputs; no new image transformations."}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Aggregate paired augmentation SSIM/LPIPS with slide-level uncertainty."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
AUGMENTATION = PROJECT / "outputs/augmentation_ood_v1"
OUTPUT = Path(__file__).resolve().parent / "results/augmentation_paired_metrics"
ARMS = ("identity", "default_oracle", "strong_oracle", "strong_unconstrained_oracle")
METRICS = ("ssim", "lpips_vgg")
REPLICATES = 5000


def summarize(frame: pd.DataFrame, sample: str) -> pd.DataFrame:
    rows = []
    for scanner in ("all", "versa", "akoya", "gt450", "s360", "s60"):
        sub = frame if scanner == "all" else frame[frame.scanner.eq(scanner)]
        slide_means = sub.groupby(["slide_id", "arm"], as_index=False)[list(METRICS)].mean()
        for metric in METRICS:
            pivot = slide_means.pivot(index="slide_id", columns="arm", values=metric)[list(ARMS)]
            if pivot.isna().any().any() or len(pivot) != 103:
                raise ValueError(f"incomplete slide means: {sample}/{scanner}/{metric}")
            values = pivot.to_numpy(float)
            raw = values[:, 0]
            rng = np.random.default_rng(20260924 + len(rows))
            draws = rng.integers(0, len(values), size=(REPLICATES, len(values)))
            sampled = values[draws].mean(axis=1)
            for index, arm in enumerate(ARMS):
                average = values[:, index].mean()
                raw_average = raw.mean()
                sign = 1 if metric == "ssim" else -1
                improvement = 100 * sign * (average - raw_average) / raw_average
                boot = 100 * sign * (sampled[:, index] - sampled[:, 0]) / sampled[:, 0]
                rows.append({
                    "sample": sample,
                    "scanner": scanner,
                    "arm": arm,
                    "metric": metric,
                    "n_slides": len(values),
                    "mean": average,
                    "raw_mean": raw_average,
                    "improvement_pct": improvement,
                    "improvement_ci_low": np.quantile(boot, 0.025),
                    "improvement_ci_high": np.quantile(boot, 0.975),
                })
    return pd.DataFrame(rows)


def condition_contrasts(frame: pd.DataFrame, sample: str) -> pd.DataFrame:
    slides = frame.groupby(["slide_id", "arm"], as_index=False)[list(METRICS)].mean()
    comparisons = (
        ("default_oracle", "identity"),
        ("strong_oracle", "default_oracle"),
        ("strong_unconstrained_oracle", "strong_oracle"),
    )
    rows = []
    for metric in METRICS:
        pivot = slides.pivot(index="slide_id", columns="arm", values=metric)[list(ARMS)]
        values = pivot.to_numpy(float)
        raw = values[:, 0].mean()
        rng = np.random.default_rng(20261024 + len(rows))
        draws = rng.integers(0, len(values), size=(REPLICATES, len(values)))
        sampled = values[draws].mean(axis=1)
        for left, right in comparisons:
            i, j = ARMS.index(left), ARMS.index(right)
            sign = 1 if metric == "ssim" else -1
            change = 100 * sign * (values[:, i].mean() - values[:, j].mean()) / raw
            boot = 100 * sign * (sampled[:, i] - sampled[:, j]) / sampled[:, 0]
            rows.append({
                "sample": sample,
                "metric": metric,
                "arm": left,
                "comparator": right,
                "additional_raw_relative_improvement_pp": change,
                "ci_low": np.quantile(boot, 0.025),
                "ci_high": np.quantile(boot, 0.975),
            })
    return pd.DataFrame(rows)


def main() -> None:
    cohort = pd.read_csv(AUGMENTATION / "00_contract/cohort.csv", dtype={"slide_id": str})
    parts = []
    common_parts = []
    for slide_id in cohort.slide_id:
        path = OUTPUT / "shards" / f"{slide_id}.csv.gz"
        if not path.exists():
            raise FileNotFoundError(path)
        part = pd.read_csv(path, dtype={"slide_id": str})
        if len(part) != 40 * 5 * len(ARMS):
            raise ValueError(f"bad image metric shard: {slide_id}")
        if not np.isfinite(part[list(METRICS)].to_numpy()).all():
            raise ValueError(f"nonfinite metric in {slide_id}")
        parts.append(part)
        with h5py.File(AUGMENTATION / "03_pfm_uni/shards" / f"{slide_id}.h5", "r") as store:
            common_locations = set(np.asarray(store["location_index"], dtype=int).tolist())
        common = part[part.location_index.isin(common_locations)]
        if len(common) != 20 * 5 * len(ARMS):
            raise ValueError(f"bad common UNI subset: {slide_id}")
        common_parts.append(common)
    all_image = pd.concat(parts, ignore_index=True)
    common_image = pd.concat(common_parts, ignore_index=True)
    summary = pd.concat([
        summarize(common_image, "same_20_as_uni"),
        summarize(all_image, "all_40_image_locations"),
    ], ignore_index=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    condition_contrasts(common_image, "same_20_as_uni").to_csv(
        OUTPUT / "condition_contrasts.csv", index=False
    )
    image_wide = summary[
        summary["sample"].eq("same_20_as_uni")
    ].pivot(index=["scanner", "arm"], columns="metric", values="improvement_pct").reset_index()
    image_wide = image_wide.rename(columns={
        "ssim": "ssim_improvement_pct",
        "lpips_vgg": "lpips_reduction_pct",
    })
    uni = pd.read_csv(AUGMENTATION / "03_pfm_uni/summary.csv")
    uni = uni[uni.arm.isin(ARMS)].copy()
    raw_uni = uni[uni.arm.eq("identity")].set_index("scanner")["generated_target_cosine"]
    uni["uni_distance_change_pct"] = [
        -100 * row.target_gain / (1 - raw_uni.loc[row.scanner])
        for row in uni.itertuples(index=False)
    ]
    joint = image_wide.merge(
        uni[["scanner", "arm", "target_gain", "target_gain_ci_low", "target_gain_ci_high", "uni_distance_change_pct"]],
        on=["scanner", "arm"], validate="one_to_one",
    )
    if len(joint) != 6 * len(ARMS):
        raise ValueError("image and UNI summaries did not fully match")
    joint.to_csv(OUTPUT / "joint_summary.csv", index=False)

    raw = summary[
        summary["sample"].eq("same_20_as_uni")
        & summary["scanner"].eq("all")
        & summary["arm"].eq("identity")
    ].set_index("metric")
    benchmark = pd.read_csv(Path(__file__).resolve().parent / "table2_main_summary.csv")
    baseline = benchmark[benchmark.dataset.eq("PanNormal") & benchmark.method.eq("raw")].iloc[0]
    checks = {
        "same_20_raw_ssim": float(raw.loc["ssim", "mean"]),
        "benchmark_raw_ssim": float(baseline.target_ssim_mean),
        "same_20_raw_lpips": float(raw.loc["lpips_vgg", "mean"]),
        "benchmark_raw_lpips": float(baseline.lpips_vgg_mean),
    }
    # Augmentation uses the registered image cache; the general benchmark
    # reads its aligned-valid prediction export. This is a gross consistency
    # check, not an assertion that the two pixel arrays are identical.
    if abs(checks["same_20_raw_ssim"] - checks["benchmark_raw_ssim"]) > 0.01:
        raise ValueError(f"SSIM baseline disagrees with benchmark: {checks}")
    if abs(checks["same_20_raw_lpips"] - checks["benchmark_raw_lpips"]) > 0.01:
        raise ValueError(f"LPIPS baseline disagrees with benchmark: {checks}")
    (OUTPUT / "validation.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(summary[summary["sample"].eq("same_20_as_uni") & summary["scanner"].eq("all")].to_string(index=False))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()

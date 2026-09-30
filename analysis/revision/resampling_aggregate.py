#!/usr/bin/env python3
"""RV10 Stages B and C: aggregate the native re-renders and write the summary.

Stage B (which resampler does the registered patch match?): per scanner and variant,
the OD pixel correlation after alignment, the patch-level high-band (0.60-0.90
cycles/um) log2 amplitude ratio variant/registered, the high-band residual fraction
|F(variant - registered)|^2 / |F(registered)|^2, and the share of locations where each
variant is the closest.  Slide-level values are location means; scanner values are
means over slides with 2,000-resample slide-bootstrap CIs.

Stage C: the paper's slide-level spectral transfer (sum of Hann-windowed mean-OD power
over the slide's locations, radial mean, amplitude ratio to AT2 normalised over
0.03-0.10 cycles/um, geometric mean over the band) for the registered pipeline patches
and for each variant, on the same set20 locations; UNI v1 cosine distance to AT2.
Primary endpoint: sign and ordering of the five scanners' high-band log2 ratios under
each variant.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import stats

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_image_study import (  # noqa: E402
    BANDS_CYC_PER_UM, geometric_band, normalized_transfer)
from resampling_geometry import COHORT, GEOMETRY, OUTPUT, SCANNERS, TARGETS  # noqa: E402
from resampling_render import RENDER, VARIANTS  # noqa: E402
from resampling_uni import UNI  # noqa: E402

PAPER_BANDS = PROJECT / "outputs/final_image_study_v1/03_frequency/band_summary.csv"
CONDITIONS = ("pipeline", *VARIANTS)
BOOTSTRAP = 2000
SEED = 20260929
PARITY_MIN = 0.999


def bootstrap_indices(n: int) -> np.ndarray:
    return np.random.default_rng(SEED).integers(0, n, size=(BOOTSTRAP, n))


def mean_ci(values: np.ndarray, draws: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    resampled = np.nanmean(values[draws], axis=1)
    return float(np.nanmean(values)), float(np.nanpercentile(resampled, 2.5)), float(
        np.nanpercentile(resampled, 97.5))


def stage_b(cohort: pd.DataFrame, draws: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = [pd.read_csv(RENDER / "shards" / f"{slide}.csv", dtype={"slide_id": str})
              for slide in cohort.slide_id]
    table = pd.concat(frames, ignore_index=True)
    rendered = table[table.variant.isin(VARIANTS)].copy()
    rendered["abs_high_log2"] = rendered.high_log2_amplitude_ratio_to_registered.abs()
    key = ["slide_id", "location_index", "scanner"]
    best = []
    for metric, ascending in (("od_correlation", False), ("abs_high_log2", True),
                              ("high_residual_fraction", True), ("rgb_mae", True)):
        winner = rendered.sort_values(metric, ascending=ascending).groupby(key).head(1)
        best.append(winner.groupby("scanner").variant.value_counts(normalize=True)
                    .rename(f"share_best_{metric}"))
    shares = pd.concat(best, axis=1).fillna(0.0).reset_index()

    metrics = ("od_correlation", "rgb_mae", "high_log2_amplitude_ratio_to_registered",
               "abs_high_log2", "high_residual_fraction", "all_residual_fraction")
    slide_means = rendered.groupby(["slide_id", "scanner", "variant"])[list(metrics)].mean()
    rows = []
    for (scanner, variant), part in slide_means.groupby(level=["scanner", "variant"]):
        part = part.droplevel(["scanner", "variant"]).reindex(cohort.slide_id)
        record = {"scanner": scanner, "variant": variant, "slides": int(part.notna().all(axis=1).sum()),
                  "locations": int(((rendered.scanner == scanner) & (rendered.variant == variant)).sum())}
        for metric in metrics:
            mean, low, high = mean_ci(part[metric].to_numpy(), draws)
            record.update({metric: mean, f"{metric}_ci_low": low, f"{metric}_ci_high": high})
        loc = rendered[(rendered.scanner == scanner) & (rendered.variant == variant)]
        record["od_correlation_location_median"] = float(loc.od_correlation.median())
        record["high_log2_location_median"] = float(loc.high_log2_amplitude_ratio_to_registered.median())
        record["high_log2_location_p10"] = float(loc.high_log2_amplitude_ratio_to_registered.quantile(0.1))
        record["high_log2_location_p90"] = float(loc.high_log2_amplitude_ratio_to_registered.quantile(0.9))
        rows.append(record)
    summary = pd.DataFrame(rows).merge(shares, on=["scanner", "variant"], how="left")
    share_columns = [column for column in summary.columns if column.startswith("share_best_")]
    summary[share_columns] = summary[share_columns].fillna(0.0)
    summary["scanner"] = pd.Categorical(summary.scanner, TARGETS)
    summary = summary.sort_values(["scanner", "variant"]).reset_index(drop=True)
    return table, summary, rendered


def slide_transfer(power_target: np.ndarray, power_reference: np.ndarray,
                   frequency_um: np.ndarray) -> dict[str, float]:
    transfer = normalized_transfer(power_target, power_reference, frequency_um)
    return {band: float(np.log2(geometric_band(transfer, frequency_um, bounds)))
            for band, bounds in BANDS_CYC_PER_UM.items()}


def stage_c_bands(cohort: pd.DataFrame, rendered_ok: dict) -> pd.DataFrame:
    rows = []
    for slide in cohort.itertuples(index=False):
        with h5py.File(RENDER / "shards" / f"{slide.slide_id}.h5", "r") as store:
            frequency_um = np.asarray(store["frequency_cyc_per_um"])
            ok = np.asarray(store["rendered_ok"], dtype=bool)
            power = {name: np.asarray(store[f"radial_power/{name}"]) for name in
                     ("pipeline", "native_at2", *VARIANTS)}
        rendered_ok[slide.slide_id] = ok
        for target_offset, scanner in enumerate(TARGETS):
            selected = ok[:, target_offset]
            scanner_index = SCANNERS.index(scanner)
            for condition in CONDITIONS:
                if condition == "pipeline":
                    target = power["pipeline"][selected, scanner_index].sum(axis=0)
                    reference = power["pipeline"][selected, 0].sum(axis=0)
                else:
                    target = power[condition][selected, scanner_index].sum(axis=0)
                    reference = power["native_at2"][selected, 0].sum(axis=0)
                values = slide_transfer(target, reference, frequency_um)
                for band, value in values.items():
                    rows.append({"slide_id": slide.slide_id, "tissue_type": slide.tissue_type,
                                 "scanner": scanner, "condition": condition, "band": band,
                                 "locations": int(selected.sum()), "log2_ratio_to_at2": value})
    return pd.DataFrame(rows)


def scanner_band_summary(bands: pd.DataFrame, cohort: pd.DataFrame, draws: np.ndarray) -> pd.DataFrame:
    rows = []
    for (band, condition, scanner), part in bands.groupby(["band", "condition", "scanner"]):
        values = part.set_index("slide_id").log2_ratio_to_at2.reindex(cohort.slide_id).to_numpy()
        pipeline = bands[(bands.band == band) & (bands.condition == "pipeline") & (bands.scanner == scanner)]
        base = pipeline.set_index("slide_id").log2_ratio_to_at2.reindex(cohort.slide_id).to_numpy()
        mean, low, high = mean_ci(values, draws)
        dmean, dlow, dhigh = mean_ci(values - base, draws)
        rows.append({"band": band, "condition": condition, "scanner": scanner, "slides": int(np.isfinite(values).sum()),
                     "mean_log2": mean, "ci_low": low, "ci_high": high,
                     "diff_vs_pipeline": dmean, "diff_ci_low": dlow, "diff_ci_high": dhigh})
    return pd.DataFrame(rows)


def ordering_endpoint(bands: pd.DataFrame, cohort: pd.DataFrame, draws: np.ndarray) -> tuple[pd.DataFrame, dict]:
    high = bands[bands.band == "high"].pivot_table(index=["slide_id", "condition"], columns="scanner",
                                                   values="log2_ratio_to_at2")
    rows, detail = [], {}
    pipeline_means = high.xs("pipeline", level="condition").reindex(cohort.slide_id)[list(TARGETS)].mean()
    pipeline_order = list(pipeline_means.sort_values(ascending=False).index)
    for condition in CONDITIONS:
        matrix = high.xs(condition, level="condition").reindex(cohort.slide_id)[list(TARGETS)].to_numpy()
        means = np.nanmean(matrix, axis=0)
        order = [TARGETS[i] for i in np.argsort(-means)]
        boot = np.nanmean(matrix[draws], axis=1)  # (B, scanners)
        boot_order_same = np.mean([list(np.array(TARGETS)[np.argsort(-b)]) == order for b in boot])
        gt450_top = float(np.mean(np.argmax(boot, axis=1) == TARGETS.index("gt450")))
        akoya_bottom = float(np.mean(np.argmin(boot, axis=1) == TARGETS.index("akoya")))
        pairwise = {}
        for a, b in combinations(range(len(TARGETS)), 2):
            diff = boot[:, a] - boot[:, b]
            pairwise[f"{TARGETS[a]}>{TARGETS[b]}"] = float(np.mean(diff > 0))
        tau = stats.kendalltau([pipeline_order.index(s) for s in TARGETS],
                               [order.index(s) for s in TARGETS]).statistic
        signs = {}
        for i, scanner in enumerate(TARGETS):
            low, high_ = np.percentile(boot[:, i], [2.5, 97.5])
            signs[scanner] = "+" if low > 0 else "-" if high_ < 0 else "0 (CI spans 0)"
        spread = boot[:, TARGETS.index("gt450")] - boot[:, TARGETS.index("akoya")]
        rows.append({"condition": condition, "order_high_to_low": " > ".join(order),
                     "gt450_minus_akoya": float(means[TARGETS.index("gt450")] - means[TARGETS.index("akoya")]),
                     "gt450_minus_akoya_ci_low": float(np.percentile(spread, 2.5)),
                     "gt450_minus_akoya_ci_high": float(np.percentile(spread, 97.5)),
                     "same_order_as_pipeline_set20": order == pipeline_order,
                     "kendall_tau_vs_pipeline": float(tau),
                     "bootstrap_share_same_order": float(boot_order_same),
                     "bootstrap_share_gt450_highest": gt450_top,
                     "bootstrap_share_akoya_lowest": akoya_bottom,
                     **{f"sign_{s}": signs[s] for s in TARGETS},
                     **{f"mean_{s}": float(means[i]) for i, s in enumerate(TARGETS)}})
        detail[condition] = pairwise
    return pd.DataFrame(rows), detail


def uni_distances(cohort: pd.DataFrame, draws: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows, parity = [], {}
    for slide in cohort.itertuples(index=False):
        path = UNI / "shards" / f"{slide.slide_id}.h5"
        with h5py.File(path, "r") as store:
            location_index = np.asarray(store["location_index"])
            pipeline = np.asarray(store["pipeline"])
            at2 = np.asarray(store["native_at2"])
            variant = np.asarray(store["variant"])
        parity[slide.slide_id] = json.loads(path.with_suffix(".json").read_text())["min_cosine_to_stored"]
        for target_offset, scanner in enumerate(TARGETS):
            scanner_index = SCANNERS.index(scanner)
            ok = np.isfinite(variant[:, target_offset]).all(axis=(1, 2))
            for k in np.flatnonzero(ok):
                record = {"slide_id": slide.slide_id, "tissue_type": slide.tissue_type,
                          "location_index": int(location_index[k]), "scanner": scanner,
                          "pipeline": float(1 - pipeline[k, scanner_index] @ pipeline[k, 0]),
                          "native_at2_vs_pipeline_at2": float(1 - at2[k] @ pipeline[k, 0])}
                for v, name in enumerate(VARIANTS):
                    record[name] = float(1 - variant[k, target_offset, v] @ at2[k])
                    record[f"{name}_vs_pipeline_target"] = float(
                        1 - variant[k, target_offset, v] @ pipeline[k, scanner_index])
                rows.append(record)
    table = pd.DataFrame(rows)
    slide = table.groupby(["slide_id", "scanner"]).mean(numeric_only=True).drop(columns="location_index")
    summary = []
    for scanner in TARGETS:
        part = slide.xs(scanner, level="scanner").reindex(cohort.slide_id)
        for condition in CONDITIONS:
            mean, low, high = mean_ci(part[condition].to_numpy(), draws)
            dmean, dlow, dhigh = mean_ci((part[condition] - part["pipeline"]).to_numpy(), draws)
            record = {"scanner": scanner, "condition": condition, "uni_distance_to_at2": mean,
                      "ci_low": low, "ci_high": high, "diff_vs_pipeline": dmean,
                      "diff_ci_low": dlow, "diff_ci_high": dhigh}
            if condition != "pipeline":
                record["distance_to_pipeline_target"] = float(part[f"{condition}_vs_pipeline_target"].mean())
            summary.append(record)
    summary = pd.DataFrame(summary)
    parity_frame = pd.DataFrame(parity).T
    parity_summary = {"min_cosine_to_stored_overall": float(parity_frame.min().min()),
                      "min_cosine_to_stored_by_scanner": parity_frame.min().to_dict(),
                      "slides_below_threshold": int((parity_frame.min(axis=1) < PARITY_MIN).sum()),
                      "native_at2_vs_pipeline_at2_max_distance": float(table.native_at2_vs_pipeline_at2.max())}
    return table, summary, parity_summary


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}"


def write_summary(stage_a: dict, b_summary: pd.DataFrame, band_summary: pd.DataFrame,
                  ordering: pd.DataFrame, uni_summary: pd.DataFrame, parity: dict,
                  render_qc: dict, paper: pd.Series) -> None:
    lines = ["# RV10 Robustness of the high-frequency phenotype to resampling", "",
             "Generated by `analysis/revision/resampling_aggregate.py`. Protocol: "
             "`analysis/revision/README.md` (RV10). Outputs in this directory.", "",
             "## What ran", "",
             "- **Stage A** (`resampling_geometry.py`): E0 native-to-registered similarity per slide "
             "and scanner as the start; wide native render, coarse-to-fine NCC and ECC affine on "
             "blurred mean OD against the cached registered patch; per-slide affine refit; second "
             "pass. Per-location affine (patch pixel to native level-0 pixel) stored for Stage B.",
             "- **Stage B** (`resampling_render.py`): for every confirmed set20 location, native "
             "re-render with (i) Catmull-Rom bicubic point sampling without pre-filter, (ii) "
             "Lanczos-3 widened by the scale (anti-aliased), (iii) box average over the AT2 pixel "
             "footprint (16x16 supersampling); identical geometry for all three.",
             "- **Stage C**: the paper's slide-level spectral transfer (`src/final_image_study.py` "
             "functions) on the same set20 locations for the pipeline patches and each variant; UNI v1 "
             "(`resampling_uni.py`, same preprocessing as the stored `03_uni` shards).", "",
             "## QC", ""]
    for record in stage_a["per_scanner"]:
        lines.append(f"- Stage A {record['scanner']}: {record['confirmed']}/{record['locations']} confirmed "
                     f"(ECC median {record['ecc_median']:.4f}, p05 {record['ecc_p05']:.4f}); "
                     f"refit mean scale {record['scale_mean_median']:.4f} vs MPP ratio "
                     f"{record['expected_scale']:.4f}.")
    lines += [f"- Stage A verdict: **{'PASS' if stage_a['stage_a_pass'] else 'FAIL'}** "
              f"(gate: >= {stage_a['gate']['pass_fraction_min']:.0%} confirmed per target scanner, "
              f"ECC >= {stage_a['gate']['ecc_min']}).",
              f"- Native AT2 read at the registered coordinates identical to the cached AT2 patch: "
              f"{render_qc['at2_identical_slides']}/{render_qc['slides']} slides "
              f"(max abs difference {render_qc['at2_max_abs_difference']}).",
              f"- Rendered locations per scanner: {render_qc['rendered']}.",
              f"- UNI parity (re-embedded pipeline patches vs stored `03_uni`): minimum cosine "
              f"{parity['min_cosine_to_stored_overall']:.5f}; slides below {PARITY_MIN}: "
              f"{parity['slides_below_threshold']}.", "",
              "## Stage B: which resampler does the registered patch match?", "",
              "Slide means over set20 (95% slide-bootstrap CI). `high log2` is log2 of the "
              "variant/registered high-band amplitude (0 = same high-band energy). Shares are the "
              "fraction of locations where the variant is closest.", "",
              "| Scanner | Variant | OD corr | high log2 variant/registered | high residual fraction | "
              "share best (corr) | share best (abs high log2) |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in b_summary.itertuples(index=False):
        lines.append(
            f"| {row.scanner} | {row.variant} | {row.od_correlation:.4f} | "
            f"{fmt(row.high_log2_amplitude_ratio_to_registered)} "
            f"[{fmt(row.high_log2_amplitude_ratio_to_registered_ci_low)}, "
            f"{fmt(row.high_log2_amplitude_ratio_to_registered_ci_high)}] | "
            f"{row.high_residual_fraction:.3f} | {row.share_best_od_correlation:.2f} | "
            f"{row.share_best_abs_high_log2:.2f} |")
    best_corr = b_summary.loc[b_summary.groupby("scanner", observed=True).od_correlation.idxmax()]
    best_amp = b_summary.loc[b_summary.groupby("scanner", observed=True).abs_high_log2.idxmin()]
    best_res = b_summary.loc[b_summary.groupby("scanner", observed=True).high_residual_fraction.idxmin()]
    by = b_summary.set_index(["scanner", "variant"]).high_log2_amplitude_ratio_to_registered
    lines += ["", "Closest variant by mean OD correlation: " + ", ".join(
        f"{r.scanner} {r.variant}" for r in best_corr.itertuples()) + ".",
        "Closest variant by high-band amplitude (smallest mean |log2|): " + ", ".join(
        f"{r.scanner} {r.variant}" for r in best_amp.itertuples()) + ".",
        "Closest variant by high-band residual fraction: " + ", ".join(
        f"{r.scanner} {r.variant} ({r.high_residual_fraction:.3f})" for r in best_res.itertuples())
        + " (0 would be a pixel-exact reproduction).", "",
        "Mean high-band log2 variant/registered, bicubic | lanczos_aa | box: " + "; ".join(
            f"{sc} {fmt(by[(sc, 'bicubic')], 2)} | {fmt(by[(sc, 'lanczos_aa')], 2)} | "
            f"{fmt(by[(sc, 'box')], 2)}" for sc in TARGETS) + ". AT2 is read natively for every "
        "condition (scale 1, integer shift), as in the pipeline, so only the target side changes.", "",
        "## Stage C: high-band log2 ratio to AT2 (primary endpoint)", "",
        "Slide means (95% CI); `pipeline` = cached registered patches on the same set20 locations; "
        "paper = published whole-slide value (all matched locations).", "",
        "| Scanner | paper | pipeline (set20) | bicubic | lanczos_aa | box |",
        "| --- | --- | --- | --- | --- | --- |"]
    high = band_summary[band_summary.band == "high"].set_index(["scanner", "condition"])
    for scanner in TARGETS:
        cells = [f"{fmt(paper[scanner])}"]
        for condition in CONDITIONS:
            r = high.loc[(scanner, condition)]
            cells.append(f"{fmt(r.mean_log2)} [{fmt(r.ci_low)}, {fmt(r.ci_high)}]")
        lines.append(f"| {scanner} | " + " | ".join(cells) + " |")
    lines += ["", "Difference variant minus pipeline (same slides and locations):", "",
              "| Scanner | bicubic | lanczos_aa | box |", "| --- | --- | --- | --- |"]
    for scanner in TARGETS:
        cells = []
        for condition in VARIANTS:
            r = high.loc[(scanner, condition)]
            cells.append(f"{fmt(r.diff_vs_pipeline)} [{fmt(r.diff_ci_low)}, {fmt(r.diff_ci_high)}]")
        lines.append(f"| {scanner} | " + " | ".join(cells) + " |")
    lines += ["", "Sign and ordering:", "",
              "| Condition | order (high to low) | same as pipeline | P(GT450 highest) | "
              "P(AKOYA lowest) | signs versa/akoya/gt450/s360/s60 | GT450 minus AKOYA |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in ordering.itertuples(index=False):
        signs = "/".join(getattr(row, f"sign_{s}") for s in TARGETS)
        lines.append(f"| {row.condition} | {row.order_high_to_low} | {row.same_order_as_pipeline_set20} | "
                     f"{row.bootstrap_share_gt450_highest:.3f} | {row.bootstrap_share_akoya_lowest:.3f} | "
                     f"{signs} | {row.gt450_minus_akoya:.3f} [{row.gt450_minus_akoya_ci_low:.3f}, "
                     f"{row.gt450_minus_akoya_ci_high:.3f}] |")
    lines += ["", "## Stage C: UNI v1 cosine distance to AT2", "",
              "| Scanner | pipeline | bicubic | lanczos_aa | box |", "| --- | --- | --- | --- | --- |"]
    uni = uni_summary.set_index(["scanner", "condition"])
    for scanner in TARGETS:
        cells = []
        for condition in CONDITIONS:
            r = uni.loc[(scanner, condition)]
            text = f"{r.uni_distance_to_at2:.4f}"
            if condition != "pipeline":
                text += f" ({fmt(r.diff_vs_pipeline, 4)})"
            cells.append(text)
        lines.append(f"| {scanner} | " + " | ".join(cells) + " |")
    lines += ["", "Parentheses: difference from the pipeline distance (slide means).", "",
              "## Implementation choices not fixed by the protocol (proposed deviations)", "",
              "- Stage A gate (ECC >= 0.95 on blurred mean OD, integer peak off the search boundary, "
              ">= 95% of set20 per target scanner) was set after a one-slide pilot, before the "
              "cohort run.",
              "- The E0 manifest transforms refer to the historical registered frame; current AKOYA "
              "(and some other) registrations differ, including non-rigid components (per-slide "
              "affine refit RMS up to ~22 px). E0 was therefore only the starting transform; every "
              "location uses its own ECC-refined affine (a patch-wise affine approximation of the "
              "VALIS warp).",
              "- bicubic = Catmull-Rom (a = -0.5, the libvips/VALIS default kernel); box = 16 x 16 "
              "supersampled footprint average; lanczos_aa = Lanczos-3 in output-pixel units.",
              "- Locations not confirmed in Stage A (1-2 per scanner) are excluded from Stages B and C; "
              "the pipeline values in Stage C use the same locations.",
              "- UNI v1 ran in fp32 on CPU because the GPU per-user cap was reached; parity with the "
              "stored fp16 GPU embeddings is reported above.", ""]
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    cohort = pd.read_csv(COHORT, dtype={"slide_id": str})[["slide_id", "tissue_type"]]
    draws = bootstrap_indices(len(cohort))
    stage_a = json.loads((GEOMETRY / "summary.json").read_text())

    render_json = [json.loads((RENDER / "shards" / f"{s}.json").read_text()) for s in cohort.slide_id]
    render_qc = {"slides": len(render_json),
                 "at2_identical_slides": int(sum(r["at2_native_identical_to_cache"] for r in render_json)),
                 "at2_max_abs_difference": int(max(r["at2_native_max_abs_difference"] for r in render_json)),
                 "rendered": {s: int(sum(r["rendered"][s] for r in render_json)) for s in TARGETS}}

    table, b_summary, _ = stage_b(cohort, draws)
    table.to_csv(OUTPUT / "stage_b_location_similarity.csv.gz", index=False)
    b_summary.to_csv(OUTPUT / "stage_b_variant_match.csv", index=False)

    rendered_ok: dict = {}
    bands = stage_c_bands(cohort, rendered_ok)
    bands.to_csv(OUTPUT / "stage_c_slide_bands.csv", index=False)
    band_summary = scanner_band_summary(bands, cohort, draws)
    band_summary.to_csv(OUTPUT / "stage_c_scanner_bands.csv", index=False)
    ordering, pairwise = ordering_endpoint(bands, cohort, draws)
    ordering.to_csv(OUTPUT / "stage_c_primary_ordering.csv", index=False)

    uni_table, uni_summary, parity = uni_distances(cohort, draws)
    uni_table.to_csv(OUTPUT / "stage_c_uni_location_distances.csv.gz", index=False)
    uni_summary.to_csv(OUTPUT / "stage_c_uni_summary.csv", index=False)

    paper_bands = pd.read_csv(PAPER_BANDS, dtype={"slide_id": str})
    paper = paper_bands[paper_bands.band == "high"].groupby("scanner").log2_relative_transfer.mean()

    payload = {
        "analysis": "RV10 resampling robustness",
        "stage_a_pass": stage_a["stage_a_pass"],
        "render_qc": render_qc,
        "uni_parity": parity,
        "paper_high_band_slide_means": paper.reindex(TARGETS).to_dict(),
        "primary_endpoint": ordering.to_dict(orient="records"),
        "pairwise_order_bootstrap": pairwise,
        "bootstrap": {"resamples": BOOTSTRAP, "seed": SEED, "unit": "slide"},
    }
    (OUTPUT / "summary.json").write_text(json.dumps(payload, indent=2, default=float) + "\n")
    write_summary(stage_a, b_summary, band_summary, ordering, uni_summary, parity, render_qc, paper)
    print((OUTPUT / "summary.md").read_text())


if __name__ == "__main__":
    main()

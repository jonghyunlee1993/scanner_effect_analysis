#!/usr/bin/env python3
"""RV02: band sensitivity and target-direction gain on `set20` in four PFMs.

Aggregates `band_manipulation_set20_features.py` shards. Statistics follow the paper
(`scripts/frequency/bandwise_uni_experiment.py`, `plot_bandwise_uni.py`,
`review_bandwise_crosspfm_aggregate.py`): slide means with the five target scanners equally
weighted, 95% CIs from 2,000 bootstrap resamples of the 103 slides (seed 20260924, the seed of
the UNI v1 script; one resample matrix shared by every statistic so paired differences use the
same resamples). Each statistic is computed on (i) the paper's stored outputs (checks this
code), (ii) the re-run restricted to the paper's three locations, and (iii) all 20 `set20`
locations (the RV02 result).

Primary endpoints per PFM: high-minus-low-mid representation shift at dose 0.25; mean
additional target gain from high-band manipulation in the target direction (dose 0.25).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


PROJECT = Path(__file__).resolve().parents[2]
COHORT = PROJECT / "outputs/scanner_batch_effect_analysis_2026-09-17/00_contract/cohort.csv"
SET20 = PROJECT / "analysis/revision/results/location_sets/set20.csv"
PAPER_UNI = PROJECT / "outputs/bandwise_uni_experiment"
PAPER_CROSS = PROJECT / "outputs/bandwise_crosspfm_2026-09-25/shards"
OUTPUT = Path(__file__).resolve().parent / "results/band_manipulation_set20"
MODELS = ("uni_v1", "uni2", "virchow2", "hoptimus1")
LABELS = {"uni_v1": "UNI v1", "uni2": "UNI2-h", "virchow2": "Virchow2", "hoptimus1": "H-optimus-1"}
BANDS = ("low_mid", "mid", "high")
BAND_LABELS = {"low_mid": "Low-mid", "mid": "Mid", "high": "High"}
SCANNERS = ("versa", "akoya", "gt450", "s360", "s60")
DOSES = (0.25, 0.5)
KEY = ["model", "slide_id", "location_index", "scanner", "band", "dose_fraction", "sign"]
N_BOOT = 2000
SEED = 20260924
EXPECTED_SIGN = {"uni_v1": 1, "uni2": 1, "virchow2": -1, "hoptimus1": -1}
PUBLISHED = {
    # (model, statistic, dose): (value, ci_low, ci_high); None where the paper gives no CI
    ("uni_v1", "shift_high", 0.25): (0.0056, None, None),
    ("uni_v1", "shift_low_mid", 0.25): (0.0043, None, None),
    ("uni_v1", "shift_mid", 0.25): (0.0043, None, None),
    ("uni_v1", "high_minus_low_mid", 0.25): (0.0013, 0.0009, 0.0017),
    ("uni_v1", "target_gain_high", 0.25): (-0.0007, -0.0011, -0.0002),
    ("uni_v1", "shift_high", 0.5): (0.0210, None, None),
    ("uni_v1", "shift_low_mid", 0.5): (0.0170, None, None),
    ("uni_v1", "shift_mid", 0.5): (0.0169, None, None),
    ("uni_v1", "target_gain_high", 0.5): (-0.0082, -0.0094, -0.0070),
    ("uni2", "high_minus_low_mid", 0.25): (0.0035, 0.0028, 0.0042),
    ("virchow2", "high_minus_low_mid", 0.25): (-0.0006, -0.0008, -0.0004),
    ("hoptimus1", "high_minus_low_mid", 0.25): (-0.0016, -0.0020, -0.0012),
}
COLORS = {"low_mid": "#5A7DA5", "mid": "#50A698", "high": "#D46B43"}


# ----------------------------------------------------------------------------- loading
def load_rerun(cohort: pd.DataFrame, set20: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for model in MODELS:
        for slide_id in cohort.slide_id:
            path = OUTPUT / "shards" / model / f"{slide_id}.csv.gz"
            if not path.exists():
                raise FileNotFoundError(path)
            frames.append(pd.read_csv(path, dtype={"slide_id": str}))
    frame = pd.concat(frames, ignore_index=True)
    per_model = len(cohort) * 20 * len(SCANNERS) * len(BANDS) * len(DOSES) * 2
    for model in MODELS:
        part = frame[frame.model.eq(model)]
        if len(part) != per_model or part.slide_id.nunique() != len(cohort):
            raise ValueError(f"{model}: incomplete rows ({len(part)} of {per_model})")
        locations = part.groupby("slide_id").location_index.apply(lambda v: tuple(sorted(set(v))))
        expected = set20.groupby("slide_id").location_index.apply(lambda v: tuple(sorted(v)))
        if not (locations.sort_index() == expected.loc[locations.sort_index().index]).all():
            raise ValueError(f"{model}: locations differ from set20")
    if frame.duplicated(KEY).any():
        raise ValueError("duplicate perturbation rows")
    values = ["embedding_displacement", "target_gain", "target_rms_od", "achieved_rms_od", "coefficient"]
    if not np.isfinite(frame[values].to_numpy()).all():
        raise ValueError("nonfinite values")
    probe = set20.set_index(["slide_id", "location_index"]).paper_band_probe
    flags = probe.loc[list(zip(frame.slide_id, frame.location_index))].to_numpy(bool)
    if not np.array_equal(flags, frame.paper_band_probe.to_numpy(bool)):
        raise ValueError("paper probe flags inconsistent with set20")
    return frame


def load_paper() -> pd.DataFrame:
    frames = []
    for fold in range(5):
        part = pd.read_csv(PAPER_UNI / f"fold_{fold}.csv.gz", dtype={"slide_id": str})
        frames.append(part.assign(model="uni_v1"))
    for model in MODELS[1:]:
        for fold in range(5):
            frames.append(pd.read_csv(PAPER_CROSS / model / f"fold_{fold}.csv.gz", dtype={"slide_id": str}))
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------------- statistics
def slide_vector(rows: pd.DataFrame, value: str, slides: np.ndarray) -> np.ndarray:
    """Mean within slide x scanner, then equal-weight mean of the five scanners per slide."""
    by_scanner = rows.groupby(["slide_id", "scanner"])[value].mean()
    counts = by_scanner.groupby(level="slide_id").size()
    vector = by_scanner.groupby(level="slide_id").mean().reindex(slides)
    if vector.isna().any():
        raise ValueError(f"missing slides for {value}")
    return vector.to_numpy(float), counts.reindex(slides).to_numpy()


def summarize(frame: pd.DataFrame, label: str, slides: np.ndarray, draws: np.ndarray) -> pd.DataFrame:
    records = []

    def add(model, statistic, dose, subset, scanner, vector, scanners_per_slide):
        boot = vector[draws].mean(axis=1)
        records.append({"set": label, "model": model, "statistic": statistic, "dose_fraction": dose,
                        "subset": subset, "scanner": scanner, "mean": float(vector.mean()),
                        "ci_low": float(np.quantile(boot, 0.025)), "ci_high": float(np.quantile(boot, 0.975)),
                        "slides": len(vector), "min_scanners_per_slide": int(np.min(scanners_per_slide))})

    for model in MODELS:
        part = frame[frame.model.eq(model)]
        wide = part.pivot_table(index=["slide_id", "location_index", "scanner", "dose_fraction", "sign"],
                                columns="band", values="embedding_displacement").reset_index()
        for dose in DOSES:
            at_dose = part[part.dose_fraction.eq(dose)]
            for band in BANDS:
                vector, n = slide_vector(at_dose[at_dose.band.eq(band)], "embedding_displacement", slides)
                add(model, f"shift_{band}", dose, "both_signs", "all", vector, n)
                target = at_dose[at_dose.band.eq(band) & at_dose.target_like]
                vector, n = slide_vector(target, "target_gain", slides)
                add(model, f"target_gain_{band}", dose, "target_direction", "all", vector, n)
                opposite = at_dose[at_dose.band.eq(band) & ~at_dose.target_like]
                vector, n = slide_vector(opposite, "target_gain", slides)
                add(model, f"target_gain_{band}", dose, "opposite_direction", "all", vector, n)
                for scanner in SCANNERS:
                    rows = target[target.scanner.eq(scanner)]
                    vector, n = slide_vector(rows, "target_gain", slides)
                    add(model, f"target_gain_{band}", dose, "target_direction", scanner, vector, n)
            wide_dose = wide[wide.dose_fraction.eq(dose)].copy()
            for comparator in ("low_mid", "mid"):
                wide_dose["contrast"] = wide_dose["high"] - wide_dose[comparator]
                for subset, sign in (("both_signs", None), ("increase", 1), ("decrease", -1)):
                    rows = wide_dose if sign is None else wide_dose[wide_dose.sign.eq(sign)]
                    vector, n = slide_vector(rows, "contrast", slides)
                    add(model, f"high_minus_{comparator}", dose, subset, "all", vector, n)
    return pd.DataFrame(records)


def parity(rerun: pd.DataFrame, paper: pd.DataFrame) -> pd.DataFrame:
    old = rerun[rerun.paper_band_probe].merge(paper, on=KEY, suffixes=("", "_paper"), how="outer", indicator=True)
    rows = []
    for model in MODELS:
        part = old[old.model.eq(model)]
        both = part[part._merge.eq("both")]
        row = {"model": model, "rerun_rows": int((part._merge != "right_only").sum()),
               "paper_rows": int((part._merge != "left_only").sum()), "matched_rows": len(both),
               "target_like_agree": bool((both.target_like == both.target_like_paper).all())}
        for column in ("target_rms_od", "achieved_rms_od", "coefficient", "embedding_displacement", "target_gain"):
            difference = (both[column] - both[f"{column}_paper"]).abs()
            row[f"{column}_max_abs_diff"] = float(difference.max())
            row[f"{column}_fraction_identical"] = float((difference == 0).mean())
        row["embedding_displacement_corr"] = float(np.corrcoef(both.embedding_displacement, both.embedding_displacement_paper)[0, 1])
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- figures
def plot_models(summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(len(MODELS), 2, figsize=(10, 3.4 * len(MODELS)), constrained_layout=True)
    for row, model in enumerate(MODELS):
        ax = axes[row, 0]
        for band in BANDS:
            for label, style in (("set20", dict(linestyle="-", marker="o")), ("probe3_rerun", dict(linestyle="--", marker="s", alpha=0.5))):
                part = summary[summary.set.eq(label) & summary.model.eq(model) & summary.statistic.eq(f"shift_{band}")].sort_values("dose_fraction")
                ax.errorbar(part.dose_fraction + (0.004 if label == "probe3_rerun" else 0), part["mean"],
                            yerr=(part["mean"] - part.ci_low, part.ci_high - part["mean"]), color=COLORS[band], capsize=3,
                            label=f"{BAND_LABELS[band]} ({'20 loc' if label == 'set20' else '3 loc'})", **style)
        ax.set(xlabel="Dose (fraction of weakest band RMS)", ylabel=f"{LABELS[model]} cosine displacement", xticks=DOSES)
        ax.grid(axis="y", alpha=0.2)
        if row == 0:
            ax.legend(frameon=False, fontsize=7, ncol=2)
        ax = axes[row, 1]
        for index, scanner in enumerate(SCANNERS):
            for band_index, band in enumerate(BANDS):
                item = summary[summary.set.eq("set20") & summary.model.eq(model) & summary.statistic.eq(f"target_gain_{band}")
                               & summary.dose_fraction.eq(0.25) & summary.subset.eq("target_direction") & summary.scanner.eq(scanner)].iloc[0]
                x = index + (band_index - 1) * 0.21
                ax.errorbar(x, item["mean"], yerr=[[item["mean"] - item.ci_low], [item.ci_high - item["mean"]]],
                            fmt="o", color=COLORS[band], capsize=2, markersize=4)
        for boundary in np.arange(len(SCANNERS) - 1) + 0.5:
            ax.axvline(boundary, color="#C8CDD3", linestyle="--", linewidth=0.8, zorder=0)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(range(len(SCANNERS)), [s.upper() for s in SCANNERS])
        ax.set(ylabel=f"{LABELS[model]} target gain over Reinhard", title="set20, dose 0.25, target direction")
        ax.title.set_fontsize(9)
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_endpoints(endpoints: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for ax, statistic, title in ((axes[0], "high_minus_low_mid", "High minus low-mid shift, dose 0.25"),
                                 (axes[1], "target_gain_high", "High-band target-direction gain, dose 0.25")):
        part = endpoints[endpoints.statistic.eq(statistic)].set_index("model").loc[list(MODELS)]
        y = np.arange(len(MODELS))
        for offset, (label, color) in enumerate((("paper_outputs", "#9a9a9a"), ("set20", "#2a5d9f"))):
            mean, low, high = part[f"{label}_mean"], part[f"{label}_ci_low"], part[f"{label}_ci_high"]
            ax.errorbar(mean, y + (0.15 if offset == 0 else -0.15), xerr=(mean - low, high - mean), fmt="o",
                        color=color, capsize=3, label="paper, 3 loc" if offset == 0 else "set20, 20 loc")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(y, [LABELS[m] for m in MODELS])
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        ax.grid(axis="x", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main() -> None:
    cohort = pd.read_csv(COHORT, dtype={"slide_id": str})
    set20 = pd.read_csv(SET20, dtype={"slide_id": str})
    slides = np.sort(cohort.slide_id.to_numpy())
    draws = np.random.default_rng(SEED).integers(0, len(slides), size=(N_BOOT, len(slides)))
    rerun = load_rerun(cohort, set20)
    paper = load_paper()

    rerun["dose_relative_error"] = (rerun.achieved_rms_od / rerun.target_rms_od - 1).abs()
    dose = rerun.groupby(["model", "band", "dose_fraction", "sign"], as_index=False).dose_relative_error.agg(
        median="median", maximum="max")
    dose.to_csv(OUTPUT / "dose_audit.csv", index=False)
    if (dose["median"] > 0.02).any():
        raise ValueError("median achieved dose differs from target by more than 2%")
    qc_frames = [pd.read_csv(path, dtype={"slide_id": str}) for path in sorted((OUTPUT / "shards/qc").glob("*.csv"))
                 if not path.name.endswith(".failed.csv")]
    baseline = pd.concat(qc_frames, ignore_index=True).groupby(["model", "check", "fatal"]).cosine.agg(
        rows="size", min="min", median="median", below_0995=lambda v: int((v < 0.995).sum())).reset_index()
    baseline.to_csv(OUTPUT / "baseline_agreement.csv", index=False)
    parity_frame = parity(rerun, paper)
    parity_frame.to_csv(OUTPUT / "parity_paper_locations.csv", index=False)

    summary = pd.concat([
        summarize(paper, "paper_outputs", slides, draws),
        summarize(rerun[rerun.paper_band_probe], "probe3_rerun", slides, draws),
        summarize(rerun, "set20", slides, draws),
    ], ignore_index=True)
    summary.to_csv(OUTPUT / "summary_statistics.csv", index=False)

    lookup = summary.set_index(["set", "model", "statistic", "dose_fraction", "subset", "scanner"])
    rows = []
    for model in MODELS:
        for statistic, subset in (("high_minus_low_mid", "both_signs"), ("target_gain_high", "target_direction")):
            for dose_fraction in DOSES:
                row = {"model": model, "statistic": statistic, "dose_fraction": dose_fraction,
                       "primary": dose_fraction == 0.25}
                published = PUBLISHED.get((model, statistic, dose_fraction))
                row.update({"published": published[0] if published else np.nan,
                            "published_ci_low": published[1] if published else np.nan,
                            "published_ci_high": published[2] if published else np.nan})
                for label in ("paper_outputs", "probe3_rerun", "set20"):
                    item = lookup.loc[(label, model, statistic, dose_fraction, subset, "all")]
                    row.update({f"{label}_mean": item["mean"], f"{label}_ci_low": item.ci_low, f"{label}_ci_high": item.ci_high})
                row["set20_ci_excludes_zero"] = bool(row["set20_ci_low"] > 0 or row["set20_ci_high"] < 0)
                if statistic == "high_minus_low_mid":
                    row["expected_sign"] = EXPECTED_SIGN[model]
                    row["sign_as_expected"] = bool(np.sign(row["set20_mean"]) == EXPECTED_SIGN[model])
                    row["sign_changed_vs_paper"] = bool(np.sign(row["set20_mean"]) != np.sign(row["paper_outputs_mean"]))
                rows.append(row)
    endpoints = pd.DataFrame(rows)
    endpoints.to_csv(OUTPUT / "endpoints.csv", index=False)
    published_check = []
    for (model, statistic, dose_fraction), (value, low, high) in PUBLISHED.items():
        subset = "target_direction" if statistic.startswith("target_gain") else "both_signs"
        item = lookup.loc[("paper_outputs", model, statistic, dose_fraction, subset, "all")]
        published_check.append({"model": model, "statistic": statistic, "dose_fraction": dose_fraction,
                                "published": value, "published_ci_low": low, "published_ci_high": high,
                                "recomputed": item["mean"], "recomputed_ci_low": item.ci_low, "recomputed_ci_high": item.ci_high,
                                "within_rounding": bool(abs(round(item["mean"], 4) - value) <= 0.0001 + 1e-12)})
    published_check = pd.DataFrame(published_check)
    published_check.to_csv(OUTPUT / "published_value_check.csv", index=False)

    plot_models(summary, OUTPUT / "band_sensitivity_set20.png")
    plot_endpoints(endpoints[endpoints.primary], OUTPUT / "primary_endpoints_set20_vs_paper.png")
    write_summary(endpoints, summary, parity_frame, baseline, dose, published_check)
    print(endpoints.to_string(index=False), flush=True)


def write_summary(endpoints, summary, parity_frame, baseline, dose, published_check) -> None:
    def ci(row, label):
        return f"{row[f'{label}_mean']:+.4f} ({row[f'{label}_ci_low']:+.4f} to {row[f'{label}_ci_high']:+.4f})"

    s20 = summary[summary.set.eq("set20")].set_index(["model", "statistic", "dose_fraction", "subset", "scanner"])
    lines = [
        "# RV02 frequency-band manipulation on set20 in four PFMs",
        "",
        "103 PanNormal slides x 20 `set20` locations x 5 target scanners; Reinhard (fold parameters) then",
        "low-mid / mid / high OD-band increase and decrease at doses 0.25 and 0.50 of the smallest band OD RMS;",
        "target direction = sign of the train-fold band-mean scanner gain. PFMs: UNI v1, UNI2-h, Virchow2,",
        "H-optimus-1, each with its paper embedding path. Slide means with five scanners equally weighted; 95% CIs",
        "from 2,000 slide bootstrap resamples (one shared resample matrix, seed 20260924).",
        "Code: `analysis/revision/band_manipulation_set20_features.py` (GPU), `analysis/revision/band_manipulation_set20.py`.",
        "",
        "## QC",
        "",
        f"- Rows: {len(MODELS)} PFMs x 123,600; complete, no duplicates, finite; locations equal `set20`.",
        f"- Dose audit: median |achieved/target - 1| max over cells {dose['median'].max():.2e} (limit 0.02); max {dose['maximum'].max():.2e}.",
    ]
    for item in baseline.itertuples(index=False):
        lines.append(f"- Frozen baseline `{item.check}` ({item.model}{', fatal' if item.fatal else ', recorded only'}): "
                     f"{item.rows} checks, min cosine {item.min:.5f}, below 0.995: {item.below_0995}.")
    lines += ["", "Parity with the paper's stored per-location outputs at the paper's 3 locations:", "",
              "| PFM | matched rows | image side (dose, coefficient) max abs diff | displacement max abs diff (identical rows) | target gain max abs diff |",
              "| --- | --- | --- | --- | --- |"]
    for row in parity_frame.itertuples(index=False):
        image = max(row.target_rms_od_max_abs_diff, row.achieved_rms_od_max_abs_diff, row.coefficient_max_abs_diff)
        lines.append(f"| {LABELS[row.model]} | {row.matched_rows} / {row.paper_rows} | {image:.1e} | "
                     f"{row.embedding_displacement_max_abs_diff:.1e} ({100 * row.embedding_displacement_fraction_identical:.1f}%) | "
                     f"{row.target_gain_max_abs_diff:.1e} |")
    lines += ["", "Published values recomputed from the paper's stored outputs by this code:", "",
              "| PFM | statistic | dose | published | recomputed | within rounding |", "| --- | --- | --- | --- | --- | --- |"]
    for row in published_check.itertuples(index=False):
        pub_ci = "" if pd.isna(row.published_ci_low) else f" ({row.published_ci_low:+.4f} to {row.published_ci_high:+.4f})"
        lines.append(f"| {LABELS[row.model]} | {row.statistic} | {row.dose_fraction:g} | {row.published:+.4f}{pub_ci} | "
                     f"{row.recomputed:+.4f} ({row.recomputed_ci_low:+.4f} to {row.recomputed_ci_high:+.4f}) | {'yes' if row.within_rounding else 'NO'} |")
    lines += ["", "## Primary endpoints (dose 0.25)", "",
              "| PFM | Endpoint | Paper outputs (3 loc) | Re-run, 3 loc | **set20** | CI excludes 0 | Expected sign | Sign changed vs paper |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    names = {"high_minus_low_mid": "High minus low-mid shift", "target_gain_high": "High-band target-direction gain"}
    for row in endpoints[endpoints.primary].to_dict("records"):
        expected = "" if row["statistic"] != "high_minus_low_mid" else ("positive" if row["expected_sign"] > 0 else "negative") + (" (met)" if row["sign_as_expected"] else " (NOT met)")
        changed = "" if row["statistic"] != "high_minus_low_mid" else ("YES" if row["sign_changed_vs_paper"] else "no")
        lines.append(f"| {LABELS[row['model']]} | {names[row['statistic']]} | {ci(row, 'paper_outputs')} | {ci(row, 'probe3_rerun')} | "
                     f"**{ci(row, 'set20')}** | {'yes' if row['set20_ci_excludes_zero'] else 'no'} | {expected} | {changed} |")
    lines += ["", "## Secondary (dose 0.50 and band means)", "",
              "| PFM | Endpoint | Paper outputs (3 loc) | set20 |", "| --- | --- | --- | --- |"]
    for row in endpoints[~endpoints.primary].to_dict("records"):
        lines.append(f"| {LABELS[row['model']]} | {names[row['statistic']]} (dose 0.50) | {ci(row, 'paper_outputs')} | {ci(row, 'set20')} |")
    lines += ["", "set20 representation shift by band (both signs):", "", "| PFM | dose | low-mid | mid | high |", "| --- | --- | --- | --- | --- |"]
    for model in MODELS:
        for dose_fraction in DOSES:
            values = [s20.loc[(model, f"shift_{band}", dose_fraction, "both_signs", "all")] for band in BANDS]
            lines.append(f"| {LABELS[model]} | {dose_fraction:g} | " + " | ".join(
                f"{v['mean']:.4f} ({v.ci_low:.4f}-{v.ci_high:.4f})" for v in values) + " |")
    lines += ["", "set20 high-minus-low-mid shift by perturbation sign:", "", "| PFM | dose | increase | decrease |", "| --- | --- | --- | --- |"]
    for model in MODELS:
        for dose_fraction in DOSES:
            values = [s20.loc[(model, "high_minus_low_mid", dose_fraction, subset, "all")] for subset in ("increase", "decrease")]
            lines.append(f"| {LABELS[model]} | {dose_fraction:g} | " + " | ".join(
                f"{v['mean']:+.4f} ({v.ci_low:+.4f} to {v.ci_high:+.4f})" for v in values) + " |")
    lines += ["", "set20 target-direction gain by band (dose 0.25, all scanners):", "", "| PFM | low-mid | mid | high |", "| --- | --- | --- | --- |"]
    for model in MODELS:
        values = [s20.loc[(model, f"target_gain_{band}", 0.25, "target_direction", "all")] for band in BANDS]
        lines.append(f"| {LABELS[model]} | " + " | ".join(f"{v['mean']:+.4f} ({v.ci_low:+.4f} to {v.ci_high:+.4f})" for v in values) + " |")
    lines += ["", "set20 high-band target-direction gain by scanner (dose 0.25):", "", "| PFM | " + " | ".join(s.upper() for s in SCANNERS) + " |",
              "| --- |" + " --- |" * len(SCANNERS)]
    for model in MODELS:
        values = [s20.loc[(model, "target_gain_high", 0.25, "target_direction", scanner)] for scanner in SCANNERS]
        lines.append(f"| {LABELS[model]} | " + " | ".join(f"{v['mean']:+.4f} ({v.ci_low:+.4f} to {v.ci_high:+.4f})" for v in values) + " |")
    lines += ["", "## Reading against the protocol", ""]
    for row in endpoints[endpoints.primary & endpoints.statistic.eq("high_minus_low_mid")].to_dict("records"):
        flip = row["sign_changed_vs_paper"] and row["set20_ci_excludes_zero"]
        lines.append(f"- {LABELS[row['model']]}: high minus low-mid {row['set20_mean']:+.4f}; expected "
                     f"{'positive' if row['expected_sign'] > 0 else 'negative'} -> {'met' if row['sign_as_expected'] else 'NOT met'}; "
                     f"sign change vs paper with CI excluding zero: {'YES' if flip else 'no'}.")
    for row in endpoints[endpoints.primary & endpoints.statistic.eq("target_gain_high")].to_dict("records"):
        lines.append(f"- {LABELS[row['model']]}: high-band target-direction gain {row['set20_mean']:+.4f} "
                     f"({row['set20_ci_low']:+.4f} to {row['set20_ci_high']:+.4f}); "
                     f"{'positive with CI excluding zero' if row['set20_ci_low'] > 0 else 'near zero or negative'}.")
    lines += ["", "## Implementation notes (proposed deviations, for the protocol log)", "",
              "- Images are rendered once per slide and embedded by all four PFMs in the same job, each with its paper",
              "  embedding function, precision and batch size (UNI v1: 32, fp16; UNI2-h/Virchow2: 8, bf16; H-optimus-1: 8, fp16).",
              "- For UNI2-h, Virchow2 and H-optimus-1 the stored Reinhard embeddings exist only at the paper's 3 locations,",
              "  so the fatal frozen-baseline check covers those; at all 20 locations the raw target embedding is compared",
              "  with the 40-location cross-encoder extraction (recorded, non-fatal).",
              "- CIs for all PFMs use 2,000 resamples (seed 20260924) as the protocol specifies; the paper's cross-PFM CIs used",
              "  3,000 resamples (seeds 20260925-7), so recomputed CIs for those PFMs differ slightly from the published ones.",
              "- Embedding differences against the paper outputs occur only for images that sat in the original run's final",
              "  partial batch (fp16/bf16 kernel dependence); image-side quantities are identical.",
              "", "## Files", "",
              "`endpoints.csv`, `summary_statistics.csv` (all statistics x set = paper_outputs / probe3_rerun / set20),",
              "`published_value_check.csv`, `parity_paper_locations.csv`, `baseline_agreement.csv`, `dose_audit.csv`,",
              "`band_sensitivity_set20.png`, `primary_endpoints_set20_vs_paper.png` (diagnostic only), `shards/<model>/<slide>.csv.gz`."]
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

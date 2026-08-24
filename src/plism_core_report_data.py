"""Every number the PLISM part of the report quotes, loaded from its own output.

The previous report builder carried its figures as literals in the prose.  That
was survivable while the cohort was frozen; it is not survivable now that the
cohort has been rebuilt, because a stale literal looks exactly like a fresh one.
So the sections read from here, and a number that has not been recomputed raises
rather than renders.

Sources are named per loader.  Anything derived — a ratio, a rank correlation, a
count of scanners on one side of a threshold — is derived here rather than in the
prose, so the same quantity cannot be computed two ways in two places.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

REFERENCE = "AT2"
# Display order, roughly by post-Reinhard detail power; also the row order of
# every scanner table in the report.
SCANNER_ORDER = ("GT450", "S210", "S60", "P", "S360", "SQ", "AT2")
SCANNER_LABEL = {"P": "Philips"}
ENCODER_ORDER = ("uni_v2", "conch_v15", "hoptimus1")
ENCODER_LABEL = {"uni_v2": "UNI2-h", "conch_v15": "CONCHv1.5", "hoptimus1": "H-optimus-1"}


def require(path: Path | str) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} — the report cannot quote a number that "
                                "has not been recomputed on the core grid")
    return path


def concat_csv(pattern: str) -> pd.DataFrame:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no files match {pattern}")
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def spearman(a, b) -> float:
    """Rank correlation, implemented here so the report has no scipy dependency."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra = pd.Series(a[ok]).rank().to_numpy()
    rb = pd.Series(b[ok]).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


# ------------------------------------------------------------------- section 15
def alignment(dataset="data/PLISM_dataset",
              registration="outputs/plism_core_registration",
              refinement="outputs/plism_core_refinement",
              sparse="outputs/plism_location_refinement") -> dict:
    """Coverage, residual and the three published defects.

    The sparse cohort is read too, so the report can state the size of what it
    replaced without carrying a literal that nobody would notice going stale.
    """
    root = Path(dataset)
    per_scanner = pd.read_csv(require(root / "alignment/per_scanner.csv"))
    per_section = pd.read_csv(require(root / "alignment/per_section.csv"))
    fit = pd.read_csv(require(root / "alignment/core_map_fit.csv"))
    corrections = pd.read_csv(require(root / "alignment/file_corrections.csv"))

    coverage = []
    for path in sorted(glob.glob(f"{registration}/*.json")):
        entry = json.loads(Path(path).read_text())
        coverage.append({"stain": entry["stain"], "cores": entry["cores"],
                         "locations": len(entry["locations_reference_thumb_xy"]),
                         "tissue_mm2": entry["tissue_mm2"],
                         "reached_mm2": entry["tissue_reached_mm2"],
                         "coverage": entry["tissue_reached_mm2"] / entry["tissue_mm2"],
                         "patch_um": entry["patch_um"]})
    coverage = pd.DataFrame(coverage)

    plan = concat_csv(f"{refinement}/*.csv")
    gate = plan["ok"] & (plan["residual_um"] <= 1.0)
    # The reference aligns to itself, so quoting the pooled pass rate would
    # flatter it by a seventh; every alignment figure below is non-reference.
    nonref = plan.loc[plan["scanner"] != REFERENCE]
    nonref_gate = nonref["ok"] & (nonref["residual_um"] <= 1.0)
    leak = (nonref["response"] < 0.3) & (nonref["residual_um"] <= 1.0)
    sparse_locations, sparse_rows, sparse_coverage = None, None, None
    if Path(sparse).exists():
        previous = concat_csv(f"{sparse}/*.csv")
        per_section = previous.groupby("stain")["location"].nunique()
        sparse_locations = int(per_section.sum())
        sparse_rows = int(len(previous))
        # Same measure as the lattice: patch footprint area over tissue area, per
        # section.  Generous to the sparse set, whose random placement can overlap.
        patch_mm2 = (coverage["patch_um"].iloc[0] / 1000.0) ** 2
        tissue = coverage.set_index("stain")["tissue_mm2"]
        shared_sections = per_section.index.intersection(tissue.index)
        sparse_coverage = float(
            (per_section[shared_sections] * patch_mm2 / tissue[shared_sections]).mean())

    return {
        "sparse_locations": sparse_locations,
        "sparse_rows": sparse_rows,
        "sparse_coverage": sparse_coverage,
        "gate_pass_nonref": float(nonref_gate.mean()),
        "residual_median_nonref": float(nonref.loc[nonref_gate, "residual_um"].median()),
        "leak_fraction": float(leak.mean()),
        "measurements_nonref": int(len(nonref)),
        "per_scanner": per_scanner, "per_section": per_section,
        "canvas_fit": fit, "corrections": corrections, "coverage": coverage,
        "sections": int(coverage["stain"].nunique()),
        "cores_max": int(coverage["cores"].max()),
        "locations_total": int(coverage["locations"].sum()),
        "locations_per_section": float(coverage["locations"].mean()),
        "coverage_mean": float(coverage["coverage"].mean()),
        "coverage_low": float(coverage["coverage"].min()),
        "coverage_high": float(coverage["coverage"].max()),
        "patch_um": float(coverage["patch_um"].iloc[0]),
        "residual_median_low": float(per_scanner["median_um"].min()),
        "residual_median_high": float(per_scanner["median_um"].max()),
        "gate_pass": float(gate.mean()),
        "measurements": int(len(plan)),
        "canvas_um_low": float(fit["canvas_um_per_unit"].min()),
        "canvas_um_high": float(fit["canvas_um_per_unit"].max()),
        "rotation_max": float(fit["rotation_deg"].abs().max()),
        "dice_low": float(fit["dice"].min()), "dice_high": float(fit["dice"].max()),
    }


# ------------------------------------------------------------------- section 16
def band_power(bands="outputs/plism_core_reinhard_bands",
               contrast="outputs/pannormal_contrast_audit") -> dict:
    """Raw and post-Reinhard fine-band power per scanner, against AT2."""
    frame = concat_csv(f"{require(bands)}/*.csv")
    per_section = frame.groupby(["stain", "scanner"])[["raw_b1", "rein_b1"]].mean()
    pooled = per_section.groupby("scanner").mean()
    ratio = pooled / pooled.loc[REFERENCE]
    ratio.columns = ["raw", "reinhard"]

    focus_ok = frame[~((frame["stain"] == "HRH") & (frame["scanner"] == "S60"))]
    pooled_ok = focus_ok.groupby(["stain", "scanner"])[["raw_b1", "rein_b1"]].mean() \
        .groupby("scanner").mean()
    ratio_ok = pooled_ok / pooled_ok.loc[REFERENCE]
    ratio_ok.columns = ["raw_focus_ok", "reinhard_focus_ok"]

    optical = frame.groupby("scanner")["limited_frac"].mean()
    return {"ratio": ratio.join(ratio_ok), "gamut_limited": optical,
            "per_section": per_section.reset_index(),
            "patches": int(len(frame)),
            "pannormal": contrast_decomposition(contrast),
            "plism": plism_contrast(focus_ok)}


def plism_contrast(bands: pd.DataFrame,
                   spectra="outputs/plism_core_native_psd") -> pd.DataFrame:
    """PLISM's own contrast decomposition, on the measure PanNormal's audit uses.

    Raw fine-band power comes from the Reinhard pass and the optical-density
    spread from the spectral pass; they run on the same gated locations, so the
    two are joined per location rather than pooled separately.  Without this the
    report would state PanNormal's excess as a number and PLISM's as a phrase.
    """
    rows = []
    for path in sorted(Path(require(spectra)).glob("*.npz")):
        with np.load(path, allow_pickle=False) as blob:
            meta = json.loads(str(blob["meta"]))
            columns = [str(c) for c in blob["band_names"]]
            block = pd.DataFrame(blob["band_per_patch"], columns=columns)
        block["stain"], block["scanner"] = meta["stain"], meta["scanner"]
        rows.append(block[["stain", "scanner", "location", "od_std"]])
    spread = pd.concat(rows, ignore_index=True)
    spread["location"] = spread["location"].astype(int)

    merged = bands[["stain", "scanner", "location", "raw_b1"]].merge(
        spread, on=["stain", "scanner", "location"], how="inner")
    per_section = merged.groupby(["stain", "scanner"])[["raw_b1", "od_std"]].mean()
    pooled = per_section.groupby("scanner").mean()
    reference = pooled.loc[REFERENCE]
    out = pd.DataFrame({
        "b1_vs_at2": pooled["raw_b1"] / reference["raw_b1"],
        "od_std": pooled["od_std"],
        "contrast_ratio": pooled["od_std"] / reference["od_std"]})
    out["predicted"] = out["contrast_ratio"] ** 2
    out["excess"] = out["b1_vs_at2"] / out["predicted"]
    out.attrs["locations"] = int(len(merged))
    return out


# The locked transfer of section 3.1, for the four scanners both cohorts share.
LOCKED_TRANSFER = {"gt450": 1.478, "s60": 1.044, "s360": 0.842, "akoya": 0.335, "at2": 1.0}


def contrast_decomposition(directory="outputs/pannormal_contrast_audit") -> pd.DataFrame:
    """Raw fine-band power against what contrast alone predicts, on PanNormal.

    Band power in a fixed band scales with the square of the optical-density
    spread, so dividing a scanner's raw ratio by the square of its contrast ratio
    leaves whatever is *not* contrast.  Contrast is the OD standard deviation, not
    the OD mean: the mean says how dark the render is, the spread says how much
    signal there is to resolve, and it is the spread the band power is made of.
    """
    frame = concat_csv(f"{require(directory)}/*.csv")
    per_slide = frame.groupby(["slide", "scanner"])[["b1", "od_std"]].mean()
    pooled = per_slide.groupby("scanner").mean()
    reference = pooled.loc["at2"]
    out = pd.DataFrame({
        "b1_vs_at2": pooled["b1"] / reference["b1"],
        "od_std": pooled["od_std"],
        "contrast_ratio": pooled["od_std"] / reference["od_std"]})
    out["predicted"] = out["contrast_ratio"] ** 2
    out["excess"] = out["b1_vs_at2"] / out["predicted"]
    out["locked"] = [LOCKED_TRANSFER.get(s, np.nan) for s in out.index]
    shared = out.dropna(subset=["locked"])
    out.attrs["rho_locked"] = spearman(shared["excess"], shared["locked"])
    out.attrs["shared"] = int(len(shared))
    return out


# ------------------------------------------------------------------- section 17
def destinations(root="outputs/plism_core_rf1u_destinations", variant="focus_ok") -> dict:
    directory = require(Path(root) / variant)
    summary = pd.read_csv(directory / "destination_summary.csv")
    detail = pd.read_csv(directory / "post_reinhard_detail.csv")
    merged = summary.merge(detail, left_on="target", right_index=True, how="left") \
        if "scanner" not in detail else summary.merge(
            detail, left_on="target", right_on="scanner", how="left")
    return {"summary": summary, "detail": detail, "merged": merged,
            "rho": spearman(summary["detail_b1_vs_AT2"], summary["mean_log_gain"]),
            "gains": pd.read_csv(directory / "fitted_gains.csv")}


def image_frontier(root="outputs/plism_core_pfm_frontier") -> dict:
    directory = require(root)
    return {"frontier": pd.read_csv(directory / "frontier.csv"),
            "per_section": pd.read_csv(directory / "per_section.csv"),
            "summary": json.loads((directory / "summary.json").read_text())}


# ------------------------------------------------------------------- section 18
def feature_frontier(root="outputs/plism_core_feature_correction") -> dict:
    directory = require(root)
    out = {"frontier": pd.read_csv(directory / "frontier.csv"),
           "per_section": pd.read_csv(directory / "per_section.csv"),
           "summary": json.loads((directory / "summary.json").read_text())}
    sweep_path = directory / "coral_sample_size.csv"
    if sweep_path.exists():
        sweep = pd.read_csv(sweep_path)
        # Group by the *requested* size, not the realised one.  A section supplies
        # whatever it has, so the "all locations" step lands on a different
        # fit_samples for each held-out section; grouping on that would split one
        # sweep point into thirteen single-section estimates.
        curve = (sweep.groupby(["encoder", "dim", "requested"])
                 .agg(fit_samples=("fit_samples", "mean"),
                      samples_per_dimension=("samples_per_dimension", "mean"),
                      rr=("rr", "mean"), sections=("rr", "size"))
                 .reset_index().sort_values(["encoder", "fit_samples"]))
        # A requested size larger than the data saturates at the same point as
        # "all"; keep one of them.
        keep = []
        for encoder_id, block in curve.groupby("encoder"):
            block = block.sort_values("fit_samples")
            previous = None
            for _, entry in block.iterrows():
                if previous is not None and entry["fit_samples"] <= 1.01 * previous:
                    continue
                keep.append(entry)
                previous = entry["fit_samples"]
        curve = pd.DataFrame(keep)
        out["sweep"] = sweep
        out["sweep_curve"] = curve
        # Where each encoder's CORAL first stops hurting, in samples per dimension.
        crossings = {}
        for encoder, block in curve.groupby("encoder"):
            block = block.sort_values("fit_samples")
            positive = block.loc[block["rr"] > 0]
            crossings[encoder] = (float(positive["samples_per_dimension"].iloc[0])
                                  if len(positive) else float("nan"))
        out["sweep_crossing"] = crossings
    return out


def retrieval(root="data/PLISM_dataset/features/analysis") -> dict:
    """Top-1 retrieval against AT2, the one scale that compares across encoders."""
    directory = require(root)
    frames = {}
    for path in sorted(p for p in directory.iterdir() if p.is_dir()):
        target = path / "cosine_vs_reference.csv"
        if target.exists():
            frames[path.name] = pd.read_csv(target).set_index("scanner")
    if not frames:
        raise FileNotFoundError(f"no cosine_vs_reference.csv under {directory}")
    top1 = pd.DataFrame({name: block["top1"] for name, block in frames.items()})
    agreement = {}
    names = list(top1.columns)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            agreement[(a, b)] = spearman(top1[a], top1[b])
    return {"top1": top1, "frames": frames, "agreement": agreement,
            "by_tissue": {name: pd.read_csv(directory / name / "cosine_by_tissue.csv")
                          for name in frames
                          if (directory / name / "cosine_by_tissue.csv").exists()}}


# ------------------------------------------------------------------- section 19
def nesting(slopes="outputs/plism_core_random_slopes",
            covariate="outputs/plism_core_stain_covariate",
            variant="focus_ok") -> dict:
    out = {}
    for level in ("section", "tissue"):
        directory = require(Path(slopes) / f"{level}_{variant}")
        out[level] = {"table": pd.read_csv(directory / "random_slopes.csv"),
                      "summary": json.loads((directory / "summary.json").read_text())}
    directory = require(Path(covariate) / variant)
    table = pd.read_csv(directory / "stain_covariate_high.csv")
    entry = {"table": table,
             "summary": json.loads((directory / "summary.json").read_text())}
    fold_path = directory / "stain_covariate_high_loo.csv"
    if fold_path.exists():
        entry["folds"] = pd.read_csv(fold_path)
        # Which single section carries a coefficient: the fold whose removal
        # weakens it most.  With thirteen sections this is the whole question.
        weakest = (entry["folds"].sort_values("p").groupby(["scanner", "covariate"])
                   .last()[["held_out", "beta", "p"]]
                   .rename(columns={"held_out": "weakest_fold",
                                    "beta": "weakest_beta", "p": "weakest_p"}))
        entry["weakest"] = weakest.reset_index()
    out["covariate"] = entry
    return out


def native_ert(root="outputs/plism_core_native_ert") -> dict:
    directory = require(root)
    summary = pd.read_csv(directory / "ert_summary.csv")
    locked = summary.dropna(subset=["locked_log2"])
    return {"summary": summary,
            "rho_locked": spearman(locked["locked_log2"], locked["estimate"]),
            "shared": len(locked)}


def load_all(variant: str = "focus_ok") -> dict:
    return {"alignment": alignment(), "band_power": band_power(),
            "destinations": destinations(variant=variant),
            "image_frontier": image_frontier(),
            "feature_frontier": feature_frontier(),
            "retrieval": retrieval(),
            "nesting": nesting(variant=variant),
            "native_ert": native_ert()}


if __name__ == "__main__":
    import sys

    which = sys.argv[1] if len(sys.argv) > 1 else "alignment"
    data = globals()[which]()
    for key, value in data.items():
        print(f"--- {key}")
        print(value if not isinstance(value, pd.DataFrame) else value.round(4).to_string())

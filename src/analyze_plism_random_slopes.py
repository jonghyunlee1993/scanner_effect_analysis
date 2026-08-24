"""Nested random-slope analysis of the PLISM scanner effect, with an alignment gate.

Runs the study's existing REML machinery on the external cohort.  For every
scanner (against AT2) and frequency band:

    y_section,core,location = fixed scanner mean
                              + section-specific scanner slope
                              + core-within-section scanner slope
                              + location sampling error

which is the diagonal-covariance form of ``0 + scanner + (0 + scanner | section)
+ (0 + scanner | section:core)``, fit by profiled REML with a boundary-mixture
LRT — the same estimator and the same code as the PanNormal tissue random-slope
analysis, so the two cohorts are compared on one method rather than two.

The level mapping differs from PanNormal's, and deliberately.  There the top
level is tissue type and the question is how much of a scanner's effect is
tissue-driven.  Here the top level is the **staining condition**, because that is
the axis PanNormal does not have: the same physical block is cut into 13 serial
sections and stained 13 ways, so a section-level slope variance answers whether
the scanner signature is a property of the instrument or of the chemistry.

Every patch carries the residual misalignment measured for it, so the gate is
applied here rather than baked into extraction, and gated and ungated fits are
reported side by side.

Contract: docs/e9_plism_native_ert_contract.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_exp05_tissue_random_slopes import benjamini_hochberg, fit_nested_reml
from plism_dataset_corrections import drop_blocks, parse_exclusions

REFERENCE = "AT2"
BANDS = ["high", "low", "extended"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/plism_native_psd")
    parser.add_argument("--output", default="outputs/plism_random_slopes")
    parser.add_argument("--gate-um", type=float, default=1.0,
                        help="drop a location for a scanner if its residual exceeds this")
    parser.add_argument("--level", default="section", choices=("section", "tissue"),
                        help="what sits at the top of the nesting; see build_replicates")
    parser.add_argument("--exclude", default="",
                        help="stain:scanner blocks to drop, or 'default' for the\n                              known-defective table in plism_dataset_corrections")
    parser.add_argument("--per-core", type=int, default=8,
                        help="locations kept per core after gating; blocks must be balanced")
    return parser.parse_args()


def load_patches(input_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(input_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            columns = [str(c) for c in data["band_names"]]
            frame = pd.DataFrame(data["band_per_patch"], columns=columns)
            frame["stain"] = meta["stain"]
            frame["scanner"] = meta["scanner"]
            if "tissue_names" in data and "tissue" in frame:
                names = [str(v) for v in data["tissue_names"]]
                frame["tissue_name"] = [names[int(i)] for i in frame["tissue"]]
            frames.append(frame)
    if not frames:
        raise RuntimeError(f"no spectra in {input_dir}")
    return pd.concat(frames, ignore_index=True)


def build_replicates(patches: pd.DataFrame, band: str, gate_um: float | None,
                     per_core: int, level: str = "section") -> pd.DataFrame:
    """Paired log2 band-power ratio against AT2 at the identical location.

    `level` chooses what sits at the top of the three-level nesting, and the two
    choices read the *same rows* differently rather than using different data --
    each core carries one tissue type and appears once per section, so a
    (section, core) cell and a (tissue, section) cell are the same set of
    locations.  Only the grouping changes.

    ``section``  top = staining condition, middle = core within it.  Asks whether
                 the scanner signature is a property of the instrument or of the
                 chemistry -- the axis PanNormal does not have.
    ``tissue``   top = tissue type, middle = the section that tissue was cut on.
                 This is PanNormal section 3.2's own nesting, and it could not be
                 run on PLISM before the 46 cores were named, so the two cohorts
                 can now be compared on one model rather than two.
    """
    keys = ["stain", "core", "location"]
    reference = patches.loc[patches["scanner"] == REFERENCE, keys + [band]]
    reference = reference.rename(columns={band: "reference"})
    merged = patches.loc[patches["scanner"] != REFERENCE].merge(reference, on=keys, how="inner")
    merged["log2_relative_transfer"] = np.log2(merged[band] / merged["reference"])

    if gate_um is not None:
        # a location is kept only where every scanner aligned, so the surviving set
        # is identical across scanners and the comparison stays paired
        ok = merged["residual_um"] <= gate_um
        good = merged.loc[ok].groupby(keys).size()
        wanted = merged["scanner"].nunique()
        usable = set(good[good == wanted].index)
        merged = merged[[tuple(row) in usable for row in merged[keys].to_numpy()]]

    core_label = merged["core"].astype(int).astype(str)
    if level == "tissue":
        if "tissue_name" not in merged:
            raise RuntimeError("tissue nesting needs spectra carrying tissue names; "
                               "re-run extract_plism_native_psd.py on the core grid")
        merged["tissue_type"] = merged["tissue_name"]
        merged["slide_id"] = merged["tissue_name"] + ":" + merged["stain"]
    else:
        merged["tissue_type"] = merged["stain"]
        merged["slide_id"] = merged["stain"] + ":core" + core_label

    balanced = []
    for (scanner, slide_id), block in merged.groupby(["scanner", "slide_id"]):
        block = block.sort_values("location")
        if len(block) < per_core:
            continue
        # Evenly spaced across the core, not the first `per_core` by location id.
        # On the sparse random sampling the head was as good as any subset; on the
        # core-grid lattice the ids run in raster order, so taking the head would
        # confine every block to one corner of its core and read a spatial
        # gradient as sampling error.  The surviving location set is identical
        # across scanners, so each scanner gets the same positions.
        pick = np.linspace(0, len(block) - 1, per_core).round().astype(int)
        block = block.iloc[pick].copy()
        block["replicate"] = np.arange(per_core)
        balanced.append(block)
    if not balanced:
        raise RuntimeError("no balanced blocks survived the gate")
    frame = pd.concat(balanced, ignore_index=True)

    # every scanner must see the same cores, or the fits are not comparable
    counts = frame.groupby("slide_id")["scanner"].nunique()
    shared = set(counts[counts == frame["scanner"].nunique()].index)
    return frame[frame["slide_id"].isin(shared)]


def fit_all(frame: pd.DataFrame, band: str, label: str) -> pd.DataFrame:
    rows = []
    for scanner, block in frame.groupby("scanner"):
        try:
            fit = fit_nested_reml(block[["tissue_type", "slide_id", "replicate",
                                         "log2_relative_transfer"]])
        except Exception as exc:  # a degenerate block should not kill the sweep
            rows.append({"condition": label, "band": band, "scanner": scanner,
                         "error": type(exc).__name__})
            continue
        section_var, core_var, residual_var = fit["variances"]
        total = section_var + core_var
        rows.append({
            "condition": label,
            "band": band,
            "scanner": scanner,
            "log2_ert": fit["full"]["fixed_mean"],
            "se": fit["full"]["fixed_se"],
            "ci_low": fit["full"]["fixed_mean"] - 1.96 * fit["full"]["fixed_se"],
            "ci_high": fit["full"]["fixed_mean"] + 1.96 * fit["full"]["fixed_se"],
            "var_section": section_var,
            "var_core": core_var,
            "var_location": residual_var,
            "section_share": section_var / total if total > 0 else np.nan,
            "lrt": fit["likelihood_ratio"],
            "p_section": fit["tissue_variance_p"],
            "n_sections": fit["n_tissues"],
            "n_obs": fit["full"]["total_n"],
        })
    out = pd.DataFrame(rows)
    if "p_section" in out and out["p_section"].notna().any():
        mask = out["p_section"].notna()
        out.loc[mask, "q_section"] = benjamini_hochberg(out.loc[mask, "p_section"].to_numpy())
    return out


def main() -> None:
    args = parse_args()
    patches = load_patches(Path(args.input))
    patches, removed = drop_blocks(patches, parse_exclusions(args.exclude))
    for entry in removed:
        print(f"excluded {entry['stain']}/{entry['scanner']}: "
              f"{entry['rows']:,} patches — {entry['reason']}")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(patches[patches["scanner"] != REFERENCE])
    gated = (patches.loc[patches["scanner"] != REFERENCE, "residual_um"] <= args.gate_um).sum()
    print(f"{len(patches)} patches, {patches['stain'].nunique()} sections, "
          f"{patches['scanner'].nunique()} scanners")
    print(f"alignment gate at {args.gate_um} um: {gated}/{total} "
          f"({100 * gated / total:.2f}%) of non-reference patches pass\n")

    results = []
    for band in BANDS:
        for label, gate in (("gated", args.gate_um), ("ungated", None)):
            frame = build_replicates(patches, band, gate, args.per_core, args.level)
            fits = fit_all(frame, band, label)
            fits["cores"] = frame["slide_id"].nunique()
            results.append(fits)
    table = pd.concat(results, ignore_index=True)
    table.to_csv(output_dir / "random_slopes.csv", index=False)

    pd.set_option("display.width", 220)
    for band in BANDS:
        block = table[table["band"] == band]
        if block.empty:
            continue
        print(f"=== band {band} ===")
        wide = block.pivot_table(index="scanner", columns="condition",
                                 values=["log2_ert", "section_share", "p_section"])
        print(wide.round(4).to_string())
        shift = (block[block.condition == "gated"].set_index("scanner")["log2_ert"]
                 - block[block.condition == "ungated"].set_index("scanner")["log2_ert"])
        print(f"gated - ungated log2 ERT: max |shift| = {shift.abs().max():.4f}, "
              f"median = {shift.abs().median():.4f}")
        print(f"cores per fit: {block['cores'].iloc[0]}, "
              f"observations: {block['n_obs'].iloc[0]}\n")

    summary = {
        "gate_um": args.gate_um,
        "level": args.level,
        "excluded_blocks": [f"{e['stain']}:{e['scanner']}" for e in removed],
        "per_core": args.per_core,
        "gate_pass_fraction": float(gated / total),
        "bands": BANDS,
        "levels": ({"top": "tissue type", "middle": "section within tissue",
                    "replicate": "location within core"} if args.level == "tissue"
                   else {"top": "stain section", "middle": "core within section",
                         "replicate": "location within core"}),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()

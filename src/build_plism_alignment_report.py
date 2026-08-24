"""Alignment report for the native PLISM cohort, written beside the data.

Everything here is aggregation.  The numbers come from
build_plism_location_refinement.py, which measured them at full resolution by
phase correlation, and from audit_plism_registration_models.py, which measured
the global fit before any per-location refinement.  Nothing is re-estimated.

The unit of report is the TMA core.  PLISM's own tissue names are attached by
build_plism_core_map.py, so a core is `18_liver` rather than a component rank,
and the same core is the same tissue in all thirteen sections.

Written to the dataset folder rather than the repository: the report describes
the data, and anyone who copies `PLISM_dataset` should get it with them.

Contract: docs/e9_plism_native_ert_contract.md
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd

REFERENCE = "AT2"
SCANNER_ORDER = ["AT2", "GT450", "P", "S210", "S360", "S60", "SQ"]
GATE_UM = 1.0
# Phase-correlation response below this means the two patches share no content,
# and then the micrometre figure beside it is a peak in noise, not a distance.
RESPONSE_FLOOR = 0.3

SCANNER_MODEL = {
    "AT2": "Leica Aperio AT2",
    "GT450": "Leica Aperio GT450",
    "P": "Philips Ultrafast Scanner",
    "S210": "Hamamatsu NanoZoomer-S210 C13239-01",
    "S360": "Hamamatsu NanoZoomer-S360 C13220-01",
    "S60": "Hamamatsu NanoZoomer-S60 C13210-01",
    "SQ": "Hamamatsu NanoZoomer-SQ C13140-D03",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-registration", default="outputs/plism_core_registration")
    parser.add_argument("--refinement", default="outputs/plism_core_refinement")
    parser.add_argument("--models", default="outputs/plism_registration_models")
    parser.add_argument("--qc-dir", default="outputs/plism_core_map_qc")
    parser.add_argument("--gallery", default="outputs/plism_core_gallery/core_gallery.json")
    parser.add_argument("--dest", default="data/PLISM_dataset")
    parser.add_argument("--gate-um", type=float, default=GATE_UM)
    parser.add_argument("--response-floor", type=float, default=RESPONSE_FLOOR)
    return parser.parse_args()


def load_locations(refinement: Path) -> pd.DataFrame:
    files = sorted(refinement.glob("*.csv"))
    if not files:
        raise RuntimeError(f"no refinement CSVs under {refinement}")
    frame = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "tissue_type" not in frame.columns:
        raise RuntimeError("refinement lacks tissue_type; run build_plism_core_map.py first")
    return frame


def summarise(frame: pd.DataFrame, keys: list[str], gate: float,
              floor: float = RESPONSE_FLOOR) -> pd.DataFrame:
    def block(group: pd.DataFrame) -> pd.Series:
        values = group["residual_um"].to_numpy(dtype=float)
        response = group["response"].to_numpy(dtype=float)
        keep = np.isfinite(values)
        finite, answered = values[keep], response[keep]
        if finite.size == 0:
            return pd.Series({"n": len(group), "n_measured": 0, "median_um": np.nan,
                              "p90_um": np.nan, "max_um": np.nan, "response_median": np.nan,
                              "locked_frac": np.nan, "pass_frac": np.nan,
                              "usable_frac": np.nan})
        locked = answered >= floor
        # Median, p90 and max describe only the locations that locked on.  Over a
        # block where nothing locked they are left empty rather than filled with a
        # peak in noise, which would read as a distance and is not one.
        described = finite[locked]
        return pd.Series({
            "n": len(group),
            "n_measured": int(finite.size),
            "median_um": float(np.median(described)) if described.size else np.nan,
            "p90_um": float(np.percentile(described, 90)) if described.size else np.nan,
            "max_um": float(described.max()) if described.size else np.nan,
            "response_median": float(np.median(answered)),
            "locked_frac": float(locked.mean()),
            "pass_frac": float((finite <= gate).mean()),
            "usable_frac": float((locked & (finite <= gate)).mean()),
        })

    out = frame.groupby(keys, dropna=False).apply(block, include_groups=False).reset_index()
    out["n"] = out["n"].astype(int)
    out["n_measured"] = out["n_measured"].astype(int)
    return out


def core_table(registrations: list[dict]) -> pd.DataFrame:
    rows = []
    for section in registrations:
        for entry in section["core_labels"]:
            rows.append({"stain": section["stain"], **entry})
    frame = pd.DataFrame(rows)
    # Reference contrast arrived with a later core map; tolerate its absence so an
    # older outputs/plism_core_registration still renders.
    for column in ("od_median", "od_std"):
        if column not in frame.columns:
            frame[column] = np.nan
    return frame


def corrections_table(registrations: list[dict]) -> pd.DataFrame:
    rows = []
    for section in registrations:
        for entry in section.get("file_corrections", []):
            rows.append({"stain": section["stain"], **entry})
    return pd.DataFrame(rows)


def canvas_table(registrations: list[dict]) -> pd.DataFrame:
    rows = []
    for section in registrations:
        fit = section["canvas_fit"]
        rows.append({
            "stain": section["stain"],
            "dice": fit["dice"],
            "cell_px": fit["cell_px"],
            "canvas_um_per_unit": fit["canvas_um_per_unit"],
            "rotation_deg": fit["rotation_deg"],
            "shift_x_px": fit["shift_px"][0],
            "shift_y_px": fit["shift_px"][1],
            "cores_covered": int(sum(1 for c in section["core_labels"] if c["sampled"] > 0)),
            "locations": int(len(section["location_core"])),
        })
    return pd.DataFrame(rows).sort_values("stain").reset_index(drop=True)


def section_transform_table(registrations: list[dict]) -> pd.DataFrame:
    rows = []
    for section in registrations:
        guides = section.get("mask_guides", {})
        for scanner, transform in section["transforms"].items():
            guide = guides.get(scanner, {})
            rows.append({
                "stain": section["stain"], "scanner": scanner,
                "ok": bool(transform["ok"]),
                "mask_dice": guide.get("dice"),
                "mask_rotation_deg": guide.get("rotation_deg"),
                "mask_ok": guide.get("ok"),
                "inliers": transform.get("inliers"),
                "matches": transform.get("matches"),
                "scale": transform.get("scale"),
                "rotation_deg": transform.get("rotation_deg"),
                "mpp": section["scanners"][scanner]["mpp"],
                "width": section["scanners"][scanner]["width"],
                "height": section["scanners"][scanner]["height"],
            })
    return pd.DataFrame(rows).sort_values(["stain", "scanner"]).reset_index(drop=True)


def pivot(frame: pd.DataFrame, index: str, column: str, value: str) -> pd.DataFrame:
    table = frame.pivot_table(index=index, columns=column, values=value, aggfunc="first")
    order = [s for s in SCANNER_ORDER if s in table.columns]
    return table[order]


def main() -> None:
    args = parse_args()
    core_dir = Path(args.core_registration)
    registrations = [json.loads(p.read_text()) for p in sorted(core_dir.glob("*.json"))]
    if not registrations:
        raise RuntimeError(f"no core registrations under {core_dir}")

    locations = load_locations(Path(args.refinement))
    moving = locations.loc[locations["scanner"] != REFERENCE].copy()

    floor = args.response_floor
    per_scanner = summarise(moving, ["scanner"], args.gate_um, floor)
    per_section = summarise(moving, ["stain", "scanner"], args.gate_um, floor)
    per_core_stain = summarise(moving, ["core", "tissue_type", "stain", "scanner"],
                               args.gate_um, floor)
    per_core = summarise(moving, ["core", "tissue_type", "scanner"], args.gate_um, floor)

    # Reference contrast travels with the per-core numbers, so a low usable
    # fraction can be read as "too little signal to check" rather than
    # "misaligned" without opening a second file.
    contrast = (core_table(registrations)
                .groupby(["core", "tissue_type"])[["od_median", "od_std"]]
                .median().reset_index())
    per_core = per_core.merge(contrast, on=["core", "tissue_type"], how="left")
    per_core_stain = per_core_stain.merge(
        core_table(registrations)[["stain", "core", "tissue_type", "od_median", "od_std"]],
        on=["stain", "core", "tissue_type"], how="left")

    unlocked = moving.loc[moving["response"] < floor, "residual_um"]
    leak = {
        "floor": floor,
        "unlocked": int(unlocked.notna().sum()),
        "unlocked_under_gate": int((unlocked <= args.gate_um).sum()),
        "unlocked_under_gate_frac": float((unlocked <= args.gate_um).mean())
                                    if unlocked.notna().any() else float("nan"),
        "of_all_frac": float((unlocked <= args.gate_um).sum() / max(len(moving), 1)),
        "total": int(moving["residual_um"].notna().sum()),
    }

    models = pd.concat(
        [pd.read_csv(p) for p in sorted(Path(args.models).glob("*.csv"))], ignore_index=True
    ) if any(Path(args.models).glob("*.csv")) else pd.DataFrame()

    dest = Path(args.dest)
    tables = dest / "alignment"
    tables.mkdir(parents=True, exist_ok=True)

    locations.to_csv(tables / "per_location.csv.gz", index=False, compression="gzip")
    per_scanner.to_csv(tables / "per_scanner.csv", index=False)
    per_section.to_csv(tables / "per_section.csv", index=False)
    per_core.to_csv(tables / "per_core.csv", index=False)
    per_core_stain.to_csv(tables / "per_core_stain.csv", index=False)
    core_table(registrations).to_csv(tables / "core_map.csv", index=False)
    canvas_table(registrations).to_csv(tables / "core_map_fit.csv", index=False)
    section_transform_table(registrations).to_csv(tables / "section_transform.csv", index=False)
    corrections = corrections_table(registrations)
    if not corrections.empty:
        corrections.to_csv(tables / "file_corrections.csv", index=False)
    if not models.empty:
        models.to_csv(tables / "global_model_audit.csv", index=False)

    payload = {
        "gate_um": args.gate_um,
        "floor": floor,
        "leak": leak,
        "locations": locations,
        "moving": moving,
        "per_scanner": per_scanner,
        "per_section": per_section,
        "per_core": per_core,
        "per_core_stain": per_core_stain,
        "canvas": canvas_table(registrations),
        "cores": core_table(registrations),
        "models": models,
        "registrations": registrations,
        "corrections": corrections,
        "qc_dir": Path(args.qc_dir),
        "gallery": json.loads(Path(args.gallery).read_text())
                   if Path(args.gallery).exists() else None,
    }

    markdown = render_markdown(payload)
    (dest / "ALIGNMENT.md").write_text(markdown)
    html = render_html(payload)
    (dest / "alignment_report.html").write_text(html)

    print(f"-> {dest / 'ALIGNMENT.md'}")
    print(f"-> {dest / 'alignment_report.html'} ({len(html) / 1e6:.1f} MB)")
    for name in sorted(p.name for p in tables.iterdir()):
        print(f"-> {tables / name}")


# --------------------------------------------------------------------------- markdown


def markdown_table(frame: pd.DataFrame, floats: dict[str, str] | None = None,
                   default_float: str | None = None) -> str:
    floats = floats or {}
    header = list(frame.columns)
    lines = ["| " + " | ".join(str(h) for h in header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for _, row in frame.iterrows():
        cells = []
        for column in header:
            value = row[column]
            if column in floats and pd.notna(value):
                cells.append(format(value, floats[column]))
            elif isinstance(value, float) and pd.isna(value):
                cells.append("—")
            elif default_float and isinstance(value, (float, np.floating)):
                cells.append(format(value, default_float))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_markdown(payload: dict) -> str:
    gate = payload["gate_um"]
    floor = payload["floor"]
    leak = payload["leak"]
    moving = payload["moving"]
    per_scanner = payload["per_scanner"].set_index("scanner")
    canvas = payload["canvas"]
    cores = payload["cores"]

    scanner_rows = []
    for scanner in SCANNER_ORDER:
        if scanner == REFERENCE or scanner not in per_scanner.index:
            continue
        row = per_scanner.loc[scanner]
        scanner_rows.append({
            "Scanner": scanner,
            "Model": SCANNER_MODEL[scanner],
            "Transform": MODEL_POLICY_TEXT.get(scanner, ""),
            "n": int(row["n_measured"]),
            "median µm": row["median_um"],
            "p90 µm": row["p90_um"],
            "max µm": row["max_um"],
            "response": row["response_median"],
            "usable %": row["usable_frac"] * 100,
        })
    scanner_frame = pd.DataFrame(scanner_rows)

    section_median = pivot(payload["per_section"], "stain", "scanner", "median_um")
    section_pass = pivot(payload["per_section"], "stain", "scanner", "usable_frac") * 100

    failures = payload["per_section"].loc[
        (payload["per_section"]["scanner"] != REFERENCE)
        & (payload["per_section"]["usable_frac"] < 0.95)
    ].sort_values("usable_frac")

    worst_cores = (
        payload["per_core"].loc[payload["per_core"]["usable_frac"] < 1.0]
        .sort_values("usable_frac")
        .head(15)
    )

    covered = cores.loc[cores["sampled"] > 0]
    uncovered = cores.loc[cores["sampled"] == 0]

    parts = [
        "# PLISM native WSIs — cross-scanner alignment report",
        "",
        f"Generated by `src/build_plism_alignment_report.py` from measurements made on the "
        f"91 native WSIs in `original_wsi/`. Companion to `PROVENANCE.md`.",
        "",
        "## 1. What was aligned, and what alignment means here",
        "",
        "Within one staining condition the seven scanners image the **same physical "
        "section**, so a location on one scanner has a true counterpart on the other six. "
        "Alignment is the problem of finding it. Each WSI has its own origin, its own "
        "pixel size (0.220–0.262 µm/px), and the Philips unit scans the slide rotated by "
        "180°.",
        "",
        f"`{REFERENCE}` is the reference in every section. Alignment runs in two stages:",
        "",
        "1. **Global transform**, fitted once per (section, scanner) on 16 µm/px "
        "thumbnails by AKAZE keypoints and RANSAC. The model is chosen per scanner from "
        "measured residual over all 13 sections, not by assumption — a blanket affine "
        "rescues Philips and S60 but is *worse* than a similarity for S210 and S360.",
        "2. **Per-location refinement** by phase correlation against the reference patch, "
        "applied as an **integer pixel shift**. A different crop, never an interpolation.",
        "",
        "The number reported as *residual* is measured **after** refinement, at full "
        "resolution, on the patch that was actually read. It is the distance between the "
        "reference patch and the scanner patch, in micrometres, from phase correlation of "
        "their optical-density images.",
        "",
        "> No analysed pixel is ever resampled. The transforms decide *where to read*; "
        "every patch is then taken by integer crop at level 0 on its own scanner's pixel "
        "grid. This is the property that makes the native cohort worth using over the "
        "authors' registered subset, and the alignment procedure preserves it.",
        "",
        "## 2. Headline: residual by scanner",
        "",
        f"Pooled over all 13 sections and all TMA cores. Reference `{REFERENCE}` is "
        "excluded — it is the fixed frame, residual 0 by construction.",
        "",
        markdown_table(scanner_frame, {"median µm": ".3f", "p90 µm": ".3f",
                                       "max µm": ".1f", "response": ".2f", "usable %": ".1f"}),
        "",
        f"A patch is {int(round(PATCH_UM_TEXT))} µm across, so a 0.1 µm residual is about "
        "one part in 1300 of the field of view — below the pixel pitch of every scanner in "
        "the panel.",
        "",
        "*Median*, *p90* and *max* are taken over the locations that locked on (see §2.1). "
        f"*usable* is the fraction passing both tests: residual ≤ {gate:g} µm **and** "
        f"response ≥ {floor:g}.",
        "",
        "### 2.1 The residual alone is not enough",
        "",
        "Phase correlation always returns a peak. When the two patches share no content — "
        "because the global transform landed on the wrong tissue — it returns a peak in "
        "noise, and the micrometre figure beside it is not a distance. The *response* is "
        "what distinguishes the two cases, and it is strongly bimodal here: locations that "
        "locked on sit near 1.0, locations that did not sit near 0.02.",
        "",
        f"That matters because **a peak in noise is not uniformly distributed** — the Hann "
        f"window biases it toward zero shift. Of the {leak['unlocked']:,} location–scanner "
        f"pairs with response < {floor:g}, {leak['unlocked_under_gate_frac'] * 100:.0f}% land "
        f"under the {gate:g} µm residual gate by chance. A residual-only gate therefore "
        f"admits {leak['unlocked_under_gate']:,} pairs "
        f"({leak['of_all_frac'] * 100:.2f}% of all measurements) that are not aligned at all.",
        "",
        f"Every table below gates on both. `alignment/per_location.csv.gz` carries "
        "`residual_um` and `response` for every location, so a different threshold can be "
        "applied without re-measuring.",
        "",
        "## 3. Per stain section",
        "",
        "Median residual, µm:",
        "",
        markdown_table(section_median.rename_axis("Section").reset_index(),
                       default_float=".3f"),
        "",
        f"Usable locations (residual ≤ {gate:g} µm and response ≥ {floor:g}), %:",
        "",
        markdown_table(section_pass.rename_axis("Section").reset_index(),
                       default_float=".1f"),
        "",
    ]

    if failures.empty:
        parts += ["Every (section, scanner) block is ≥95% usable.", ""]
    else:
        parts += [
            "### Blocks below 95% usable",
            "",
            "These are whole-block failures, not scattered bad locations. **None of them "
            "is a registration failure** — §5 works through the evidence for each. What "
            "the response column shows here is that the patches did not correlate, and "
            "that has more than one cause: the position can be right while the pixels "
            "carry nothing to correlate with.",
            "",
            markdown_table(
                failures.drop(columns=["n"])
                .rename(columns={"stain": "Section", "scanner": "Scanner",
                                         "n_measured": "n", "response_median": "response",
                                         "pass_frac": "residual-only %",
                                         "usable_frac": "usable %"})
                .assign(**{"residual-only %": lambda d: d["residual-only %"] * 100,
                           "usable %": lambda d: d["usable %"] * 100})[
                    ["Section", "Scanner", "n", "response", "residual-only %", "usable %"]],
                {"response": ".3f", "residual-only %": ".1f", "usable %": ".1f"}),
            "",
            "The gap between the two percentage columns is the leak of §2.1 — what a "
            "residual-only gate would have kept. It is widest exactly where the block is "
            "worst, which is why the residual alone cannot be trusted to police itself.",
            "",
        ]

    parts += [
        "## 4. Per TMA core",
        "",
        "The block is one TMA of 46 human tissue types, cut into 13 serial sections. Core "
        "identity comes from PLISM's own `PLISM_wsi_en.csv` tissue labels, carried onto "
        "each native section by a single global similarity fitted between the authors' "
        "registered canvas and that section's AT2 tissue mask. That fit is used **only to "
        "name a core**; a core is 2.5 mm across on a 3.6 mm pitch, so a naming that is "
        "right to a millimetre is right. It never places an analysed pixel.",
        "",
        markdown_table(
            canvas.rename(columns={
                "stain": "Section", "dice": "canvas fit Dice", "cell_px": "cell px",
                "canvas_um_per_unit": "canvas µm/unit", "rotation_deg": "rotation °",
                "cores_covered": "cores covered", "locations": "locations"})[
                ["Section", "canvas fit Dice", "canvas µm/unit", "rotation °",
                 "cores covered", "locations"]],
            {"canvas fit Dice": ".3f", "canvas µm/unit": ".4f", "rotation °": "+.2f"}),
        "",
        f"{covered['tissue_type'].nunique()} of 46 cores received at least one location. "
        + (f"Never covered: {', '.join(sorted(set(uncovered['tissue_type']) - set(covered['tissue_type'])))}."
           if len(set(uncovered['tissue_type']) - set(covered['tissue_type'])) else
           "Every core is covered in at least one section."),
        "",
        "Full per-core numbers are in `alignment/per_core.csv` (core × scanner, pooled over "
        "sections) and `alignment/per_core_stain.csv` (core × scanner × section).",
        "",
    ]

    if not worst_cores.empty:
        parts += [
            "### Cores with any failing location",
            "",
            markdown_table(
                worst_cores.drop(columns=["n"])
                .rename(columns={"core": "Core", "tissue_type": "Tissue",
                                            "scanner": "Scanner", "n_measured": "n",
                                            "median_um": "median µm", "max_um": "max µm",
                                            "usable_frac": "usable %"})
                .assign(**{"usable %": lambda d: d["usable %"] * 100})[
                    ["Core", "Tissue", "Scanner", "n", "median µm", "max µm", "usable %"]],
                {"median µm": ".3f", "max µm": ".1f", "usable %": ".1f"}),
            "",
            "Read this against §3: where a core fails it is almost always because the whole "
            "section failed for that scanner, not because the core is hard.",
            "",
        ]

    corrections = payload["corrections"]
    parts += [
        "## 5. Dataset integrity",
        "",
        "Three (section, scanner) blocks resisted alignment. None of them turned out "
        "to be an alignment problem, and the distinction matters: a registration "
        "defect is something to fix, while these are properties of the files that a "
        "user of this dataset needs to know about.",
        "",
        "The diagnosis rests on separating *where the transform landed* from *what was "
        "there when it landed*. Four explanations were excluded in turn.",
        "",
        "| Explanation | Test | Result |",
        "|---|---|---|",
        "| Transform sat on the wrong core | Dice of the whole section outline; core pitch is 3600 µm | Dice 0.93, transforms agree to 40–52 µm — right core |",
        "| Transform merely far off | normalised cross-correlation over ±800 µm | best peak 0.10–0.27, runner-up 0.88–0.99 of it — a noise field, not a displaced match |",
        "| Scale error | correlation swept over 0.4–2.5× | best 0.17 at 1.90×, against 0.92 at 1.00× on a healthy block |",
        "| Out of focus | high-band power, and NCC which tolerates blur | separates the cases — see below |",
        "",
        "### Two SQ files hold each other's section",
        "",
        "`GIVH_SQ.ndpi` contains the HRH section and `HRH_SQ.ndpi` contains the GIVH "
        "section. Matching each against the AT2 reference of all thirteen sections is "
        "unambiguous:",
        "",
        "| File | vs its own label | vs the other section | other 11 sections |",
        "|---|---|---|---|",
        "| `GIVH_SQ.ndpi` | 0.111 | **0.747** (HRH) | 0.10–0.32 |",
        "| `HRH_SQ.ndpi` | 0.130 | **0.831** (GIVH) | 0.09–0.29 |",
        "",
        "Repeated over eight TMA cores this is unanimous — 16 of 16 — while S360 put "
        "through the identical comparison is correct 16 of 16, at 0.824–0.983 against "
        "its own section. Six of the seven scanners agree with each other under the "
        "published labels and only SQ disagrees, so it is the two SQ files that are "
        "mislabelled, not the other six.",
        "",
        "Refitting SQ's transform against the corrected file confirms it independently: "
        "RANSAC inliers rise from 405 to 1,324 on GIVH and from 288 to 2,354 on HRH, "
        "back into the 1,000–6,000 range every healthy block occupies.",
        "",
        "**This report applies the swap.** Without it SQ contributes nothing to two of "
        "the thirteen sections and the 13 × 7 crossed design is broken. The correction "
        "is in `src/plism_dataset_corrections.py` and can be switched off with "
        "`--no-file-corrections` so the data as published stays reproducible.",
        "",
    ]
    if not corrections.empty:
        parts += [markdown_table(corrections.rename(columns={
            "stain": "Section", "scanner": "Scanner", "published": "Published as",
            "actual_file": "Actually holds", "refitted_ok": "Transform refitted"})[
            ["Section", "Scanner", "Published as", "Actually holds", "Transform refitted"]]), ""]

    parts += [
        "### A caveat on the outline-Dice column",
        "",
        "`alignment/section_transform.csv` carries a `mask_dice` per (section, scanner) "
        "— the overlap of the two tissue silhouettes at 16 µm/px. It is what separates "
        "*the transform is in the wrong place* from *the transform is right and the "
        "pixels still disagree*, and that separation is what §5 is built on. **It is not "
        "reliable for Philips.** The Philips scans carry saturated pixels, so a "
        "percentile white level puts the glass itself above the tissue threshold; taking "
        "the white level from the mode of the bright half fixes some sections (HRH, 0.42 "
        "→ 0.86) and not others (GIVH, 0.44 → 0.44). Philips alignment itself is "
        "unaffected and normal — median residual 0.10 µm, 99.4% usable — so read a low "
        "Philips Dice as a failure of this QC statistic, not of the registration.",
        "",
        "### One scan is out of focus",
        "",
        "S60's HRH scan is correctly labelled and correctly registered — its patches sit "
        "4–11 µm from the reference and normalised cross-correlation, which tolerates "
        "blur, confirms the tissue matches at 0.46–0.82. What has gone is the high "
        "frequency content: band power is down five- to tenfold against the reference. "
        "Phase correlation whitens the spectrum and is therefore dominated by high "
        "frequencies, which is why its response falls to 0.13–0.35 while the position is "
        "fine.",
        "",
        "This one cannot be repaired. It is recorded so that anyone measuring sharpness "
        "on this cohort — which is what the parent study does — knows the scan is soft "
        "before it enters an estimate.",
        "",
        "## 6. Global transform, before refinement",
        "",
        "How much the global model alone leaves behind, and why the per-scanner model "
        "policy is what it is. Median over sections of each section's median residual, µm:",
        "",
    ]
    if not payload["models"].empty:
        table = payload["models"].pivot_table(index="scanner", columns="model",
                                              values="median_um", aggfunc="median")
        order = [c for c in ("similarity", "affine", "homography") if c in table.columns]
        table = table[order].rename_axis("Scanner").reset_index()
        table["policy"] = table["Scanner"].map(MODEL_POLICY_TEXT)
        parts += [markdown_table(table, default_float=".2f"), ""]
    parts += [
        "Philips is the clear case: 28.2 µm under a similarity, 9.4 under an affine. The "
        "refinement in stage 2 then takes every scanner from these 5–9 µm figures down to "
        "the ~0.1 µm of §2.",
        "",
        "## 7. Files",
        "",
        "| File | Contents |",
        "|---|---|",
        "| `ALIGNMENT.md` | this report |",
        "| `alignment_report.html` | same report with the core-map QC overlays |",
        "| `alignment/per_location.csv.gz` | every measured location: section, scanner, core, tissue, crop centre, integer shift, residual |",
        "| `alignment/per_core.csv` | core × scanner, pooled over sections |",
        "| `alignment/per_core_stain.csv` | core × scanner × section |",
        "| `alignment/per_section.csv` | section × scanner |",
        "| `alignment/per_scanner.csv` | scanner, pooled |",
        "| `alignment/core_map.csv` | core ↔ tissue name ↔ position on each section's AT2 thumbnail |",
        "| `alignment/core_map_fit.csv` | canvas→section similarity fit and its Dice |",
        "| `alignment/section_transform.csv` | global transform per (section, scanner): inliers, scale, rotation |",
        "| `alignment/global_model_audit.csv` | similarity vs affine vs homography residual |",
        "",
        "### Using the alignment",
        "",
        "`per_location.csv.gz` carries `centre_x`, `centre_y` in **that scanner's own level-0 "
        "pixels**, already including the integer refinement shift. To read the patch that "
        f"was measured, crop `round(PATCH_UM / mpp)` pixels centred there — and rotate by "
        "180° if `flip` is true, which it is for Philips throughout.",
        "",
        f"Gate on `residual_um <= {gate:g}` **and** `response >= {floor:g}`. The parent "
        f"study's downstream analyses gate on the residual alone; §2.1 shows what that "
        f"lets through, and `alignment/per_location.csv.gz` carries both columns.",
        "",
        "## 8. Provenance",
        "",
        "- Data: `original_wsi/`, 91 native WSIs, md5-verified, CC BY 4.0. See `PROVENANCE.md`.",
        "- Tissue labels: `PLISM_wsi_en.csv`, shipped with the dataset.",
        "- The registered 512 px patch subset and the smartphone subset are **not** used here "
        "and are not downloaded.",
        "- Code: `/mnt/isilon/oldridge_lab/leej/prenorm`, `src/build_plism_section_registration.py`, "
        "`src/audit_plism_registration_models.py`, `src/build_plism_core_map.py`, "
        "`src/build_plism_location_refinement.py`, `src/build_plism_alignment_report.py`.",
        "- Cite PLISM: Ochi, M., Komura, D., Onoyama, T. et al. *Sci Data* **11**, 330 (2024).",
        "",
    ]
    return "\n".join(parts) + "\n"


PATCH_UM_TEXT = 256 * 0.5052
MODEL_POLICY_TEXT = {"GT450": "similarity", "P": "affine + 180° flip", "S210": "similarity",
                     "S360": "similarity", "S60": "affine", "SQ": "similarity"}


# --------------------------------------------------------------------------- html


def downscaled_jpeg(path: Path, width: int = 1000, quality: int = 78) -> str:
    """QC overlays are 1.5 MB PNGs; thirteen of them would not fit in one page."""
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        return ""
    if image.shape[1] > width:
        scale = width / image.shape[1]
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer.tobytes()).decode() if ok else ""


def heat_style(value: float, gate: float) -> str:
    """Green at zero residual through amber at the gate to red well past it."""
    if not np.isfinite(value):
        return "background:var(--absent)"
    ratio = float(np.clip(np.log10(max(value, 1e-3) / 0.05) / np.log10(gate * 20 / 0.05), 0, 1))
    return f"background:color-mix(in oklab, var(--good) {(1 - ratio) * 100:.0f}%, var(--bad))"


def pass_style(fraction: float) -> str:
    if not np.isfinite(fraction):
        return "background:var(--absent)"
    ratio = float(np.clip((1.0 - fraction) / 0.2, 0, 1))
    return f"background:color-mix(in oklab, var(--good) {(1 - ratio) * 100:.0f}%, var(--bad))"


def html_matrix(table: pd.DataFrame, styler, formatter, index_name: str) -> str:
    head = "".join(f"<th>{c}</th>" for c in table.columns)
    body = []
    for label, row in table.iterrows():
        cells = "".join(
            f'<td style="{styler(v)}">{formatter(v)}</td>' for v in row
        )
        body.append(f"<tr><th scope=row>{label}</th>{cells}</tr>")
    return (f'<div class="scroll"><table class="matrix"><thead><tr><th>{index_name}</th>{head}'
            f"</tr></thead><tbody>{''.join(body)}</tbody></table></div>")


def plain_table(frame: pd.DataFrame, formats: dict[str, str] | None = None) -> str:
    formats = formats or {}
    head = "".join(f"<th>{c}</th>" for c in frame.columns)
    body = []
    for _, row in frame.iterrows():
        cells = []
        for column in frame.columns:
            value = row[column]
            if column in formats and pd.notna(value):
                cells.append(f"<td>{format(value, formats[column])}</td>")
            elif isinstance(value, float) and pd.isna(value):
                cells.append("<td>—</td>")
            else:
                cells.append(f"<td>{value}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (f'<div class="scroll"><table><thead><tr>{head}</tr></thead>'
            f"<tbody>{''.join(body)}</tbody></table></div>")


def render_html(payload: dict) -> str:
    gate = payload["gate_um"]
    floor = payload["floor"]
    leak = payload["leak"]
    per_scanner = payload["per_scanner"].set_index("scanner")
    section_median = pivot(payload["per_section"], "stain", "scanner", "median_um")
    section_pass = pivot(payload["per_section"], "stain", "scanner", "usable_frac")

    core_index = (payload["per_core"][["core", "tissue_type"]].drop_duplicates()
                  .sort_values("core"))
    core_median = payload["per_core"].pivot_table(index="tissue_type", columns="scanner",
                                                  values="median_um", aggfunc="first")
    core_pass = payload["per_core"].pivot_table(index="tissue_type", columns="scanner",
                                                values="usable_frac", aggfunc="first")
    order = [t for t in core_index["tissue_type"] if t in core_median.index]
    columns = [s for s in SCANNER_ORDER if s in core_median.columns]
    core_median = core_median.loc[order, columns]
    core_pass = core_pass.loc[order, columns]

    cards = []
    for scanner in SCANNER_ORDER:
        if scanner == REFERENCE or scanner not in per_scanner.index:
            continue
        row = per_scanner.loc[scanner]
        cards.append(
            f'<div class="card"><div class="name">{scanner}</div>'
            f'<div class="value">{row["median_um"]:.3f}<span class="unit"> µm</span></div>'
            f'<div class="sub">median residual · {row["usable_frac"] * 100:.1f}% usable '
            f'· n={int(row["n_measured"]):,}</div></div>'
        )

    qc = []
    for section in payload["registrations"]:
        image = payload["qc_dir"] / f"{section['stain']}.png"
        if not image.exists():
            continue
        encoded = downscaled_jpeg(image)
        fit = section["canvas_fit"]
        qc.append(
            f'<figure><img alt="core map for {section["stain"]}" '
            f'src="data:image/jpeg;base64,{encoded}">'
            f'<figcaption><b>{section["stain"]}</b> — canvas fit Dice {fit["dice"]:.3f}, '
            f'rotation {fit["rotation_deg"]:+.2f}°, '
            f'{sum(1 for c in section["core_labels"] if c["sampled"] > 0)}/46 cores covered'
            f"</figcaption></figure>"
        )

    strips = ""
    if payload["gallery"]:
        blocks = []
        for strip in payload["gallery"]["strips"]:
            tiles = []
            for scanner in payload["gallery"]["scanner_order"]:
                if scanner not in strip["tiles"]:
                    continue
                residual = strip["residual"].get(scanner)
                caption = ("reference" if scanner == REFERENCE
                           else f"{residual:.2f} µm" if residual is not None else "—")
                tiles.append(
                    f'<div class="tile"><img alt="{strip["tissue_type"]} on {scanner}" '
                    f'src="data:image/jpeg;base64,{strip["tiles"][scanner]}">'
                    f'<div class="tag">{scanner}</div><div class="res">{caption}</div></div>')
            blocks.append(
                f'<div class="strip"><div class="striphead"><b>{strip["tissue_type"]}</b>'
                f'<span> · section {strip["stain"]} · location {strip["location"]}</span></div>'
                f'<div class="tiles">{"".join(tiles)}</div></div>')
        strips = "".join(blocks)

    failures = payload["per_section"].loc[payload["per_section"]["usable_frac"] < 0.95].copy()
    failures["pass_frac"] = failures["pass_frac"] * 100
    failures["usable_frac"] = failures["usable_frac"] * 100
    failure_html = plain_table(
        failures.drop(columns=["n"])
        .rename(columns={"stain": "Section", "scanner": "Scanner", "n_measured": "n",
                                 "response_median": "response",
                                 "pass_frac": "residual-only %", "usable_frac": "usable %"})[
            ["Section", "Scanner", "n", "response", "residual-only %", "usable %"]],
        {"response": ".3f", "residual-only %": ".1f", "usable %": ".1f"},
    ) if not failures.empty else "<p>Every (section, scanner) block is ≥95% usable.</p>"

    corrections = payload["corrections"]
    corrections_html = plain_table(
        corrections.rename(columns={"stain": "Section", "scanner": "Scanner",
                                    "published": "Published as",
                                    "actual_file": "Actually holds"})[
            ["Section", "Scanner", "Published as", "Actually holds"]]
    ) if not corrections.empty else "<p>No file corrections were applied.</p>"

    model_html = ""
    if not payload["models"].empty:
        table = payload["models"].pivot_table(index="scanner", columns="model",
                                              values="median_um", aggfunc="median")
        cols = [c for c in ("similarity", "affine", "homography") if c in table.columns]
        table = table[cols].reset_index()
        table["policy"] = table["scanner"].map(MODEL_POLICY_TEXT)
        model_html = plain_table(table, {c: ".2f" for c in cols})

    total = int(payload["moving"]["residual_um"].notna().sum())
    overall_pass = float(((payload["moving"]["residual_um"] <= gate)
                          & (payload["moving"]["response"] >= floor)).mean()) * 100

    return HTML_TEMPLATE.format(
        cards="".join(cards),
        total=f"{total:,}",
        overall_pass=f"{overall_pass:.1f}",
        gate=f"{gate:g}",
        floor=f"{floor:g}",
        leak_n=f"{leak['unlocked']:,}",
        leak_pass=f"{leak['unlocked_under_gate']:,}",
        leak_pct=f"{leak['unlocked_under_gate_frac'] * 100:.0f}",
        leak_of_all=f"{leak['of_all_frac'] * 100:.2f}",
        sections=len(payload["registrations"]),
        cores=core_median.shape[0],
        section_median=html_matrix(section_median, lambda v: heat_style(v, gate),
                                   lambda v: "—" if not np.isfinite(v) else f"{v:.3f}",
                                   "Section"),
        section_pass=html_matrix(section_pass, pass_style,
                                 lambda v: "—" if not np.isfinite(v) else f"{v * 100:.0f}",
                                 "Section"),
        core_median=html_matrix(core_median, lambda v: heat_style(v, gate),
                                lambda v: "—" if not np.isfinite(v) else f"{v:.3f}",
                                "TMA core"),
        core_pass=html_matrix(core_pass, pass_style,
                              lambda v: "—" if not np.isfinite(v) else f"{v * 100:.0f}",
                              "TMA core"),
        failures=failure_html,
        models=model_html,
        canvas=plain_table(
            payload["canvas"].rename(columns={
                "stain": "Section", "dice": "Dice", "canvas_um_per_unit": "canvas µm/unit",
                "rotation_deg": "rotation °", "cores_covered": "cores", "locations": "locations"})[
                ["Section", "Dice", "canvas µm/unit", "rotation °", "cores", "locations"]],
            {"Dice": ".3f", "canvas µm/unit": ".4f", "rotation °": "+.2f"}),
        qc="".join(qc),
        corrections=corrections_html,
        strips=strips or "<p>No crop strips were built.</p>",
    )


HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PLISM native alignment</title>
<style>
:root {{
  --bg: #fbfaf8; --panel: #ffffff; --ink: #1a1a1c; --muted: #6b6b73;
  --line: #e4e2dd; --accent: #2f6f5e;
  --good: #cfe8d8; --bad: #e8b4ac; --absent: #eeece8;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg: #17171a; --panel: #1f1f23; --ink: #ecebe8; --muted: #9b9aa2;
    --line: #33333a; --accent: #7fc4ae;
    --good: #2c4b3d; --bad: #6b3a34; --absent: #2a2a2f;
  }}
}}
:root[data-theme="dark"] {{
  --bg: #17171a; --panel: #1f1f23; --ink: #ecebe8; --muted: #9b9aa2;
  --line: #33333a; --accent: #7fc4ae;
  --good: #2c4b3d; --bad: #6b3a34; --absent: #2a2a2f;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--bg); color: var(--ink);
  font: 16px/1.6 ui-sans-serif, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}}
.wrap {{ max-width: 1180px; margin: 0 auto; padding: 3rem 1.25rem 6rem; }}
h1 {{ font-size: clamp(1.6rem, 3.4vw, 2.3rem); line-height: 1.2; margin: 0 0 .4rem; letter-spacing: -.01em; }}
h2 {{ font-size: 1.25rem; margin: 3.2rem 0 .5rem; letter-spacing: -.005em; }}
h3 {{ font-size: 1.02rem; margin: 2rem 0 .4rem; color: var(--muted); font-weight: 600; }}
p {{ margin: .7rem 0; max-width: 74ch; }}
.lede {{ color: var(--muted); max-width: 74ch; }}
.rule {{ height: 1px; background: var(--line); border: 0; margin: 2.4rem 0 0; }}
.cards {{ display: grid; gap: .7rem; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); margin: 1.6rem 0; }}
.card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: .85rem .95rem; }}
.card .name {{ font-size: .78rem; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }}
.card .value {{ font-size: 1.55rem; font-variant-numeric: tabular-nums; margin-top: .15rem; }}
.card .unit {{ font-size: .9rem; color: var(--muted); }}
.card .sub {{ font-size: .76rem; color: var(--muted); margin-top: .25rem; }}
.scroll {{ overflow-x: auto; margin: 1rem 0; border: 1px solid var(--line); border-radius: 10px; background: var(--panel); }}
table {{ border-collapse: collapse; width: 100%; font-size: .86rem; }}
th, td {{ padding: .42rem .6rem; text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }}
thead th {{ position: sticky; top: 0; background: var(--panel); border-bottom: 1px solid var(--line); color: var(--muted); font-weight: 600; }}
tbody th {{ text-align: left; font-weight: 500; }}
tbody tr + tr td, tbody tr + tr th {{ border-top: 1px solid var(--line); }}
.matrix td {{ color: var(--ink); }}
.note {{ border-left: 3px solid var(--accent); padding: .1rem 0 .1rem 1rem; margin: 1.3rem 0; color: var(--muted); }}
.note strong {{ color: var(--ink); }}
figure {{ margin: 0 0 1.6rem; background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: .7rem; }}
figure img {{ width: 100%; height: auto; border-radius: 6px; display: block; }}
figcaption {{ font-size: .8rem; color: var(--muted); margin-top: .5rem; }}
.gallery {{ display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); margin-top: 1.2rem; }}
code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .88em; background: var(--absent); padding: .1em .35em; border-radius: 4px; }}
.legend {{ display: flex; gap: .8rem; align-items: center; font-size: .78rem; color: var(--muted); margin-top: -.4rem; }}
.strip {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: .8rem .9rem; margin: 1rem 0; }}
.striphead {{ font-size: .9rem; margin-bottom: .6rem; }}
.striphead span {{ color: var(--muted); font-size: .82rem; }}
.tiles {{ display: flex; gap: .55rem; overflow-x: auto; padding-bottom: .2rem; }}
.tile {{ flex: 0 0 auto; width: 132px; }}
.tile img {{ width: 132px; height: 132px; border-radius: 5px; display: block; }}
.tile .tag {{ font-size: .76rem; margin-top: .3rem; }}
.tile .res {{ font-size: .72rem; color: var(--muted); font-variant-numeric: tabular-nums; }}
.swatch {{ display: inline-block; width: 2.4rem; height: .7rem; border-radius: 3px; border: 1px solid var(--line); }}
</style></head><body><div class="wrap">

<h1>PLISM native WSIs — cross-scanner alignment</h1>
<p class="lede">Seven scanners, thirteen H&amp;E staining conditions, one TMA block of 46 tissue
types. Within a condition all seven image the same physical section, so every location has a true
counterpart on the other six. This is how well they were found, and where they were not.</p>

<div class="cards">{cards}</div>

<div class="note"><strong>No analysed pixel is ever resampled.</strong> The transforms decide
<em>where to read</em>; every patch is then taken by integer crop at level 0 on its own scanner's
pixel grid. That is the property that makes the native cohort worth using over the authors'
registered subset, and the alignment procedure preserves it.</div>

<p>{total} location–scanner pairs measured across {sections} sections and {cores} TMA cores,
{overall_pass}% usable. Residual is measured <em>after</em> refinement, at full resolution, by phase
correlation of the two optical-density patches — the distance between the reference patch and the
patch that was actually read. <b>Usable</b> means it passes both tests: residual ≤ {gate} µm
<em>and</em> phase-correlation response ≥ {floor}.</p>

<div class="note"><strong>The residual alone is not enough.</strong> Phase correlation always
returns a peak. When two patches share no content — because the global transform landed on the wrong
tissue — it returns a peak in <em>noise</em>, and the micrometre figure beside it is not a distance.
The <em>response</em> separates the two cases, and it is sharply bimodal here: locations that locked
on sit near 1.0, locations that did not sit near 0.02. And a peak in noise is not uniformly
distributed — the Hann window biases it toward zero shift, so {leak_pct}% of the {leak_n} pairs with
response &lt; {floor} land under the {gate} µm residual gate by chance. A residual-only gate admits
{leak_pass} pairs ({leak_of_all}% of all measurements) that are not aligned at all. Every table here
gates on both; <code>per_location.csv.gz</code> carries both columns so a different threshold needs
no re-measuring.</div>

<hr class="rule">
<h2>Method</h2>
<p><code>AT2</code> is the reference in every section. Each WSI has its own origin, its own pixel
size (0.220–0.262 µm/px), and Philips scans the slide rotated 180°. Alignment runs in two stages:</p>
<p><b>1 · Global transform</b> — fitted once per (section, scanner) on 16 µm/px thumbnails by AKAZE
keypoints and RANSAC. The model is chosen per scanner from measured residual over all 13 sections,
not by assumption.</p>
<p><b>2 · Per-location refinement</b> — phase correlation against the reference patch, applied as an
<b>integer pixel shift</b>. A different crop, never an interpolation.</p>

<hr class="rule">
<h2>Per stain section</h2>
<h3>Median residual, µm</h3>
<div class="legend"><span class="swatch" style="background:var(--good)"></span> sub-pixel
<span class="swatch" style="background:var(--bad)"></span> past the gate</div>
{section_median}
<h3>Usable locations, %</h3>
{section_pass}

<h3>Blocks below 95% usable</h3>
<p>These are whole-section failures, not scattered bad locations: the global transform landed on the
wrong tissue and the local refinement could not recover it. The response column is the diagnosis —
at 0.01–0.25 these patch pairs share no content at all, so the fit landed on the <em>wrong tissue</em>
rather than landing imprecisely on the right tissue. The gap between the two percentage columns is
the leak described above: what a residual-only gate would have kept.</p>
{failures}

<hr class="rule">
<h2>Per TMA core</h2>
<p>Core identity comes from PLISM's own <code>PLISM_wsi_en.csv</code> tissue labels, carried onto each
native section by a single global similarity fitted between the authors' registered canvas and that
section's AT2 tissue mask. That fit names a core; it never places an analysed pixel. A core is 2.5 mm
across on a 3.6 mm pitch, so a naming right to a millimetre is right.</p>
{canvas}
<h3>Median residual by core, µm</h3>
{core_median}
<h3>Usable locations by core, %</h3>
<p>Read a low row against §<i>per stain section</i>. A core that is low on <em>one</em> scanner is a
registration failure for that scanner. A core that is low on <em>all six</em> carries too little
optical density to correlate — it is unverifiable rather than misaligned, and
<code>alignment/core_map.csv</code> gives its <code>od_median</code> and <code>od_std</code> so the
two can be told apart.</p>
{core_pass}

<hr class="rule">
<h2>Dataset integrity</h2>
<p>Three (section, scanner) blocks resisted alignment. <b>None of them turned out to be an
alignment problem</b>, and the distinction matters: a registration defect is something to fix,
while these are properties of the files that a user of this dataset needs to know about. The
diagnosis rests on separating <em>where the transform landed</em> from <em>what was there when it
landed</em> — four explanations were excluded in turn.</p>
<div class="scroll"><table><thead><tr><th>Explanation</th><th>Test</th><th>Result</th></tr></thead>
<tbody>
<tr><td>Wrong core</td><td>Dice of the whole section outline; core pitch 3600 µm</td><td>Dice 0.93, transforms agree to 40–52 µm — right core</td></tr>
<tr><td>Merely far off</td><td>NCC over ±800 µm</td><td>best peak 0.10–0.27, runner-up 0.88–0.99 of it — a noise field</td></tr>
<tr><td>Scale error</td><td>correlation swept 0.4–2.5×</td><td>best 0.17 at 1.90×, vs 0.92 at 1.00× on a healthy block</td></tr>
<tr><td>Out of focus</td><td>high-band power, plus blur-tolerant NCC</td><td>separates the two remaining cases</td></tr>
</tbody></table></div>

<h3>Two SQ files hold each other's section</h3>
<p><code>GIVH_SQ.ndpi</code> contains the HRH section and <code>HRH_SQ.ndpi</code> contains the GIVH
section. Matched against the AT2 reference of all thirteen sections, <code>GIVH_SQ</code> scores
0.747 on HRH and 0.111 on its own label; <code>HRH_SQ</code> scores 0.831 on GIVH and 0.130 on its
own; the other eleven sections score 0.09–0.32. Over eight cores this is unanimous, 16 of 16, while
S360 through the identical comparison is correct 16 of 16 at 0.824–0.983. Six of seven scanners
agree with each other under the published labels and only SQ disagrees.</p>
<p>Refitting SQ's transform against the corrected file confirms it independently: RANSAC inliers
rise from 405 to 1,324 on GIVH and from 288 to 2,354 on HRH, back into the 1,000–6,000 range every
healthy block occupies.</p>
<div class="note"><strong>This report applies the swap.</strong> Without it SQ contributes nothing
to two of the thirteen sections and the 13 × 7 crossed design is broken. The correction lives in
<code>src/plism_dataset_corrections.py</code> and can be switched off with
<code>--no-file-corrections</code>, so the data as published stays reproducible.</div>
{corrections}

<h3>A caveat on the outline-Dice column</h3>
<p><code>alignment/section_transform.csv</code> carries a <code>mask_dice</code> per (section,
scanner) — the overlap of the two tissue silhouettes at 16 µm/px. It is what separates <em>the
transform is in the wrong place</em> from <em>the transform is right and the pixels still
disagree</em>, and that separation is what this whole section is built on. <b>It is not reliable for
Philips.</b> Those scans carry saturated pixels, so a percentile white level puts the glass itself
above the tissue threshold; taking the white level from the mode of the bright half fixes some
sections (HRH, 0.42 → 0.86) and not others (GIVH, 0.44 → 0.44). Philips alignment itself is normal —
median residual 0.10 µm, 99.4% usable — so read a low Philips Dice as a failure of this QC
statistic, not of the registration.</p>

<h3>One scan is out of focus</h3>
<p>S60's HRH scan is correctly labelled and correctly registered — its patches sit 4–11 µm from the
reference, and blur-tolerant NCC confirms the tissue matches at 0.46–0.82. What has gone is the high
frequency content: band power is down five- to tenfold. Phase correlation whitens the spectrum and is
dominated by high frequencies, which is why its response falls to 0.13–0.35 while the position is
fine. This cannot be repaired; it is recorded so that anyone measuring sharpness on this cohort knows
the scan is soft before it enters an estimate.</p>

<hr class="rule">
<h2>Global transform, before refinement</h2>
<p>How much the global model alone leaves behind, and why the per-scanner policy is what it is.
Median over sections of each section's median residual, µm. Philips is the clear case: 28 µm under a
similarity, 9 under an affine.</p>
{models}

<hr class="rule">
<h2>What alignment looks like</h2>
<p>One core, one tissue, the same field of view on seven scanners. Every tile is the crop the
refinement actually used — <code>centre_x</code>, <code>centre_y</code> and <code>flip</code> taken
straight from the table, not recomputed — so the picture and the number below it describe the same
patch. The reticle marks the centre; correspondence is judged against it. Tiles are resized for
display only.</p>
{strips}

<hr class="rule">
<h2>Core map QC</h2>
<p>Each section's AT2 thumbnail with the tissue mask tinted, the fitted core centroids circled and
numbered with PLISM's tissue index, and the sampled locations marked. Dice is the overlap between the
warped canvas occupancy lattice and the section's own tissue mask.</p>
<div class="gallery">{qc}</div>

<hr class="rule">
<h2>Files</h2>
<p>Tables live in <code>alignment/</code> beside this report.
<code>per_location.csv.gz</code> carries <code>centre_x</code>, <code>centre_y</code> in that
scanner's own level-0 pixels, already including the integer refinement shift; crop
<code>round(129.33 / mpp)</code> pixels centred there, and rotate 180° where <code>flip</code> is
true. Gate on <code>residual_um &lt;= {gate}</code> <b>and</b>
<code>response &gt;= {floor}</code>.</p>
<p class="lede">Data: <code>original_wsi/</code>, 91 native WSIs, CC BY 4.0 — see
<code>PROVENANCE.md</code>. Cite Ochi, M., Komura, D., Onoyama, T. et al. <i>Sci Data</i>
<b>11</b>, 330 (2024).</p>

</div></body></html>
"""


if __name__ == "__main__":
    main()

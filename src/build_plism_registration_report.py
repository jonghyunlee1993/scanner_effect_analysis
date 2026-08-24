"""Standalone HTML report for the PLISM section registration.

Reads the registration JSONs, the crop assets and the native-ERT summary, and
writes one self-contained page: every image is inlined as a data URI so the file
can be moved or mailed on its own.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

STYLE = """
*, *::before, *::after { box-sizing: border-box; }
body, h1, h2, h3, p, figure, table, ul, ol { margin: 0; padding: 0; }
img { max-width: 100%; display: block; }
:root {
  --ground:#f6f7f9; --surface:#fff; --surface-sunk:#eef0f4;
  --ink:#161b22; --ink-soft:#525c6b; --ink-faint:#7a8593;
  --rule:#d9dee6; --rule-soft:#e7eaf0;
  --accent:#0e7c86; --accent-soft:#e2f0f1;
  --pass:#2b7049; --fail:#a1372a; --warn:#a75c14;
  --f-display:"Helvetica Neue",Helvetica,Arial,system-ui,sans-serif;
  --f-body:Charter,"Bitstream Charter","Iowan Old Style",Georgia,serif;
  --f-data:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground:#11151b; --surface:#181d25; --surface-sunk:#0d1116;
    --ink:#e2e7ee; --ink-soft:#9ba6b5; --ink-faint:#77828f;
    --rule:#2a3240; --rule-soft:#212833;
    --accent:#45b3bd; --accent-soft:#17323a;
    --pass:#5eb88c; --fail:#e57866; --warn:#dc9550;
  }
}
:root[data-theme="dark"] {
  --ground:#11151b; --surface:#181d25; --surface-sunk:#0d1116;
  --ink:#e2e7ee; --ink-soft:#9ba6b5; --ink-faint:#77828f;
  --rule:#2a3240; --rule-soft:#212833;
  --accent:#45b3bd; --accent-soft:#17323a;
  --pass:#5eb88c; --fail:#e57866; --warn:#dc9550;
}
body { background:var(--ground); color:var(--ink); font-family:var(--f-body);
       font-size:16.5px; line-height:1.65; -webkit-font-smoothing:antialiased; }
.page { max-width:64rem; margin:0 auto; padding:2.5rem 1.25rem 5rem; }
header.top { border-bottom:2px solid var(--ink); padding-bottom:1.5rem; margin-bottom:2rem; }
.kicker { font-family:var(--f-display); font-size:.72rem; letter-spacing:.14em;
          text-transform:uppercase; color:var(--ink-faint); margin-bottom:.6rem; }
h1 { font-family:var(--f-display); font-size:2rem; line-height:1.2; letter-spacing:-.01em; }
.lede { color:var(--ink-soft); margin-top:.9rem; max-width:62ch; }
h2 { font-family:var(--f-display); font-size:1.28rem; margin:2.6rem 0 .5rem;
     padding-top:1.4rem; border-top:1px solid var(--rule); }
h2 .num { color:var(--accent); font-variant-numeric:tabular-nums; margin-right:.5rem; }
h3 { font-family:var(--f-display); font-size:1rem; margin:1.6rem 0 .5rem; color:var(--ink-soft); }
p { margin:.7rem 0; max-width:64ch; }
p.wide { max-width:none; }
.stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(9rem,1fr)); gap:1px;
         background:var(--rule); border:1px solid var(--rule); margin:1.5rem 0; }
.stat { background:var(--surface); padding:.85rem 1rem; }
.stat .v { font-family:var(--f-display); font-size:1.45rem; font-weight:600;
           font-variant-numeric:tabular-nums; }
.stat .l { font-size:.74rem; color:var(--ink-faint); line-height:1.35; margin-top:.15rem; }
.scroll { overflow-x:auto; margin:1.1rem 0; border:1px solid var(--rule); background:var(--surface); }
table { border-collapse:collapse; width:100%; font-family:var(--f-data); font-size:.78rem; }
caption { caption-side:top; text-align:left; font-family:var(--f-display); font-size:.75rem;
          letter-spacing:.08em; text-transform:uppercase; color:var(--ink-faint);
          padding:.7rem 1rem .3rem; }
th, td { padding:.42rem .7rem; text-align:right; border-bottom:1px solid var(--rule-soft);
         white-space:nowrap; }
th:first-child, td:first-child { text-align:left; }
thead th { font-family:var(--f-display); font-size:.7rem; letter-spacing:.05em;
           text-transform:uppercase; color:var(--ink-faint); border-bottom:1px solid var(--rule); }
tbody tr:last-child td { border-bottom:none; }
tr.flag td { background:var(--accent-soft); }
.pass { color:var(--pass); font-weight:600; }
.fail { color:var(--fail); font-weight:600; }
.warn { color:var(--warn); font-weight:600; }
.gal { overflow-x:auto; margin:1.2rem 0; }
.gal table { font-size:.72rem; width:auto; }
.gal td, .gal th { padding:3px; border:none; white-space:nowrap; }
.gal img { width:132px; height:132px; border-radius:2px; }
.gal .res { font-family:var(--f-data); font-size:.62rem; text-align:center; color:var(--ink-faint); margin-top:1px; }
.gal .res.warn { color:var(--warn); font-weight:600; }
.gal thead th { text-align:center; padding-bottom:.4rem; }
.gal td.rowlab { text-align:left; font-family:var(--f-data); font-size:.7rem;
                 color:var(--ink-soft); padding-right:.6rem; }
.strip { display:flex; flex-wrap:wrap; gap:6px; margin:1.2rem 0; }
.strip figure { width:112px; }
.strip img { width:112px; height:112px; border-radius:2px; }
.strip figcaption { font-family:var(--f-data); font-size:.68rem; color:var(--ink-faint);
                    text-align:center; margin-top:.2rem; }
.note { background:var(--surface-sunk); border-left:3px solid var(--accent);
        padding:.85rem 1rem; margin:1.3rem 0; font-size:.92rem; }
.note strong { font-family:var(--f-display); }
footer { margin-top:3.5rem; padding-top:1.2rem; border-top:1px solid var(--rule);
         font-family:var(--f-data); font-size:.72rem; color:var(--ink-faint); }
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration", default="outputs/plism_section_registration")
    parser.add_argument("--assets", default="outputs/plism_report_assets/report_assets.json")
    parser.add_argument("--alignment", default="outputs/plism_report_assets/alignment_gallery.json")
    parser.add_argument("--ert", default="outputs/plism_native_ert/ert_summary.csv")
    parser.add_argument("--output", default="presentations/plism_registration_2026-08-08/index.html")
    return parser.parse_args()


def table(caption: str, headers: list[str], rows: list[list[str]], flag_rows=()) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = []
    for index, row in enumerate(rows):
        cells = "".join(f"<td>{c}</td>" for c in row)
        body.append(f'<tr class="flag">{cells}</tr>' if index in flag_rows else f"<tr>{cells}</tr>")
    return (
        f'<div class="scroll"><table><caption>{html.escape(caption)}</caption>'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
    )


def load_registration(directory: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text())
        for scanner, transform in data["transforms"].items():
            rows.append(
                {
                    "stain": data["stain"],
                    "scanner": scanner,
                    "ok": transform["ok"],
                    "inliers": transform.get("inliers"),
                    "matches": transform.get("matches"),
                    "scale": transform.get("scale"),
                    "rotation": transform.get("rotation_deg"),
                    "mpp": data["scanners"][scanner]["mpp"],
                    "candidates": data["shared_candidates"],
                    "locations": len(data["locations_reference_thumb_xy"]),
                }
            )
    return pd.DataFrame(rows)


def fold_rotation(value: float) -> float:
    return value if abs(value) < 90 else (abs(value) - 180) * (1 if value > 0 else 1)


def main() -> None:
    args = parse_args()
    frame = load_registration(Path(args.registration))
    assets = json.loads(Path(args.assets).read_text())
    order = assets["scanner_order"]

    moving = frame[frame["scanner"] != "AT2"]
    parts: list[str] = []

    n_sections = frame["stain"].nunique()
    parts.append(
        f'<div class="stats">'
        f'<div class="stat"><div class="v">{len(frame)}</div>'
        f'<div class="l">registration cells<br>{n_sections} sections × {frame["scanner"].nunique()} scanners</div></div>'
        f'<div class="stat"><div class="v pass">{int(frame["ok"].sum())} / {len(frame)}</div>'
        f'<div class="l">passed QC<br>inliers ≥ 100, scale 0.98–1.02</div></div>'
        f'<div class="stat"><div class="v">{int(moving["inliers"].median()):,}</div>'
        f'<div class="l">median RANSAC inliers<br>AKAZE at 16 µm/px</div></div>'
        f'<div class="stat"><div class="v">0</div>'
        f'<div class="l">pixels resampled<br>registration selects coordinates only</div></div>'
        f"</div>"
    )

    # 1. method
    parts.append('<h2><span class="num">1</span>What was aligned, and how</h2>')
    parts.append(
        "<p>Within one PLISM staining condition the seven scanners image the "
        "<strong>same physical section</strong>, but each whole-slide image has its own origin and "
        "the Philips unit scans the slide upside down. For each of the "
        f"{n_sections} sections a similarity transform was fitted per scanner onto that section's "
        "AT2 reference, from AKAZE keypoints matched at 16 µm/px and filtered by RANSAC.</p>"
    )
    parts.append(
        '<div class="note"><strong>The transform is used only to decide where to read.</strong> '
        "Patches are then taken by integer crop at level 0 on each scanner's own pixel grid. No "
        "analysed pixel is interpolated, warped or resampled — which is the entire reason this "
        "cohort was chosen, since its seven scanners share one native sampling density and ours "
        "do not.</div>"
    )

    # 2. metrics by scanner
    parts.append('<h2><span class="num">2</span>Registration quality by scanner</h2>')
    rows, flags = [], set()
    for index, scanner in enumerate([s for s in order if s != "AT2"]):
        block = moving[moving["scanner"] == scanner]
        folded = block["rotation"].apply(fold_rotation)
        deviation = (block["scale"].mean() - 1) * 1000
        if abs(deviation) > 5:
            flags.add(index)
        rows.append([
            scanner,
            f"{block['mpp'].iloc[0]:.5f}",
            f"{int(block['ok'].sum())}/{len(block)}",
            f"{int(block['inliers'].min()):,}",
            f"{int(block['inliers'].median()):,}",
            f"{int(block['inliers'].max()):,}",
            f"{block['scale'].mean():.4f}",
            f'<span class="{"fail" if abs(deviation) > 5 else "pass"}">{deviation:+.2f}</span>',
            f"{block['rotation'].median():+.2f}",
            f"{folded.median():+.3f}",
        ])
    parts.append(table(
        "Per scanner, over all sections, against that section's AT2",
        ["scanner", "native MPP", "passed", "inliers min", "median", "max",
         "fitted scale", "scale dev ‰", "rotation °", "folded ° from 0/180"],
        rows, flags))
    parts.append(
        "<p>Inlier counts split the panel in two. The Hamamatsu units return 1,800–2,600 matched "
        "keypoints; <strong>GT450 returns 722 and Philips 413</strong>. AKAZE keys on local "
        "contrast, so this is the same low-contrast rendering that dominates the spectral result — "
        "seen here through an entirely independent instrument.</p>"
    )

    # 3. MPP validation
    parts.append('<h2><span class="num">3</span>The fitted scale is an independent MPP audit</h2>')
    parts.append(
        "<p>Each thumbnail was built at exactly 16 µm/px <em>using that scanner's own MPP header</em>. "
        "If the headers are mutually consistent the fitted scale must come back at 1.0000, so any "
        "departure measures header error rather than registration error.</p>"
    )
    parts.append(
        "<p>Five of six agree with AT2 to within <strong>0.11 ‰</strong> across all "
        f"{n_sections} sections, which is the empirical licence for expressing every spectrum in "
        "cycles per micrometre. <strong>Philips is the exception at −9.84 ‰</strong>, consistently "
        "in every section (0.9887–0.9931). Its declared 0.250 µm/px is the only round number in a "
        "panel of measured ones — 0.2208, 0.2211, 0.2297, 0.2532, 0.2621 — and the fit puts its true "
        "value near <strong>0.2476 µm/px</strong>. A 1 % error moves the frequency axis by 1 %, small "
        "beside the contrast effect but systematic, and it should be corrected before Philips enters "
        "any physical-units comparison.</p>"
    )

    # 4. alignment gallery, refined
    align = json.loads(Path(args.alignment).read_text())
    order = align["scanner_order"]
    residuals = {s: [r["residual"][s] for r in align["gallery"] if s in r["residual"]] for s in order}
    pooled = np.array([v for s in order if s != "AT2" for v in residuals[s]])

    parts.append('<h2><span class="num">4</span>The same tissue on seven scanners</h2>')
    parts.append(
        "<p>Sections <strong>" + ", ".join(html.escape(s) for s in align["sections"]) +
        "</strong>, ten TMA cores each. Every row is one core; every column is that core read on one "
        "scanner at the position the registration maps it to. Tiles cover "
        f"{align['patch_um']:.1f} µm and carry a faint centre reticle, so correspondence can be judged "
        "against a fixed mark rather than by impression. The number under each tile is the residual "
        "offset measured on that tile by phase correlation against the reference.</p>")
    parts.append(
        '<div class="stats">'
        f'<div class="stat"><div class="v">{len(align["gallery"]) * (len(order) - 1)}</div>'
        f'<div class="l">tile pairs shown<br>{len(align["gallery"])} cores × 6 scanners</div></div>'
        f'<div class="stat"><div class="v pass">{np.median(pooled):.2f} µm</div>'
        '<div class="l">median residual<br>0.1 % of the tile width</div></div>'
        f'<div class="stat"><div class="v">{100 * (1 - np.median(pooled) / align["patch_um"]) ** 2:.2f} %</div>'
        '<div class="l">estimated content overlap</div></div>'
        f'<div class="stat"><div class="v warn">{100 * (pooled > 1).mean():.1f} %</div>'
        '<div class="l">tiles still above 1 µm<br>all of them SQ</div></div>'
        "</div>")

    rows = []
    for scanner in order:
        if scanner == "AT2":
            continue
        values = np.array(residuals[scanner])
        above = int((values > 1).sum())
        rows.append([
            scanner,
            align["model_policy"].get(scanner, "—"),
            "180°" if scanner in align["flipped"] else "—",
            f"{len(values)}",
            f"{np.median(values):.2f}",
            f"{np.percentile(values, 90):.2f}",
            f"{values.max():.2f}",
            f'<span class="{"warn" if above else "pass"}">{above}</span>',
        ])
    parts.append(table(
        "Residual offset per scanner on the tiles below, after model choice and refinement",
        ["scanner", "global model", "display rotation", "tiles", "median µm", "p90 µm", "max µm", "> 1 µm"],
        rows))

    for section in align["sections"]:
        block = [r for r in align["gallery"] if r["stain"] == section]
        if not block:
            continue
        parts.append(f"<h3>Section {html.escape(section)}</h3>")
        head = "".join(f"<th>{html.escape(s)}</th>" for s in order)
        body = []
        for row in block:
            cells = ""
            for scanner in order:
                tile = row["tiles"].get(scanner)
                if tile is None:
                    cells += "<td></td>"
                    continue
                offset = row["residual"].get(scanner, float("nan"))
                cls = "res warn" if offset > 1 else "res"
                cells += (
                    f'<td><img loading="lazy" alt="{html.escape(section)} core {row["core"]} {scanner}" '
                    f'src="data:image/jpeg;base64,{tile}">'
                    f'<div class="{cls}">{offset:.2f} µm</div></td>')
            body.append(f'<tr><td class="rowlab">core {row["core"]}</td>{cells}</tr>')
        parts.append(
            f'<div class="gal"><table><thead><tr><th></th>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')

    parts.append(
        '<div class="note"><strong>Read these for correspondence, not for sharpness.</strong> '
        "Tiles are resized to a common display size, so the resampling this cohort exists to avoid is "
        "present in the pictures. Sharpness is measured on unresampled pixels by radial spectrum. What "
        "the gallery shows is that the seven scanners land on the same cells, and how differently they "
        "render colour and contrast — GT450 and Philips pale, the Hamamatsu units and AT2 saturated. "
        "Across all 13 sections GT450's mean patch optical density is <strong>0.133 against AT2's "
        "0.301</strong>, and band power tracks the square of that ratio, which is why the raw spectral "
        "ordering is a contrast ordering.</div>")

    comparison = align.get("comparison", [])
    if comparison:
        parts.append('<h2><span class="num">5</span>What the refinement bought</h2>')
        parts.append(
            "<p>Two changes, both decided on measured residual rather than on argument. The global "
            "transform model is now chosen per scanner — a blanket affine rescues Philips and S60 but is "
            "<em>worse</em> than the similarity for S210 and S360, because two extra degrees of freedom "
            "fit keypoint noise when there is no real anisotropy to capture. Then each location is "
            "refined against the reference by phase correlation, applied as an <strong>integer pixel "
            "shift</strong>, so no analysed pixel is interpolated.</p>")
        before, after = {}, {}
        for row in comparison:
            for key, value in row["residual"].items():
                scanner, _, tag = key.rpartition("_")
                (before if tag == "before" else after).setdefault(scanner, []).append(value)
        rows = []
        for scanner in order:
            if scanner not in before:
                continue
            b, a = float(np.median(before[scanner])), float(np.median(after[scanner]))
            rows.append([scanner, align["model_policy"].get(scanner, "—"), f"{b:.2f}", f"{a:.2f}",
                         f"{b / max(a, 1e-6):.0f}×",
                         f"{100 * (1 - b / align['patch_um']) ** 2:.1f} %",
                         f"{100 * (1 - a / align['patch_um']) ** 2:.1f} %"])
        parts.append(table(
            "Global similarity alone against model choice plus per-location refinement",
            ["scanner", "model used", "before µm", "after µm", "gain", "overlap before", "overlap after"],
            rows))
        parts.append(
            "<p>Philips is the case that changes a decision. On the global similarity it sat at 28 µm "
            "median over all 13 sections — 21 % of a tile, 62 % content overlap, phase-correlation "
            "response 0.46 — which would not pass an honest gate. An affine takes it to 9.4 µm and "
            "refinement to 0.11 µm. Its problem was never a failed registration; it was a model too "
            "rigid for a scanner with anisotropic distortion, on top of a slide mounted 180° round.</p>")
        parts.append(
            '<div class="note"><strong>The gate that was missing.</strong> The frozen QC accepted a '
            "scanner on inlier count and fitted scale, and Philips passed both while sitting 28 µm out. "
            "A residual-offset gate measured at full resolution belongs in the contract, and 5 % of SQ "
            "tiles still exceed 1 µm after refinement, so it would do real work.</div>")

    # 5. stain strip
    parts.append('<h2><span class="num">6</span>The thirteen staining conditions</h2>')
    parts.append(
        "<p>One core per section on its own AT2 reference. This is the axis our own cohort does not "
        "have: colour varies here independently of the scanner, which is what makes it possible to "
        "ask whether the scanner signature is optics or chemistry.</p>"
    )
    strip = "".join(
        f'<figure><img loading="lazy" alt="{html.escape(k)}" '
        f'src="data:image/jpeg;base64,{v}"><figcaption>{html.escape(k)}</figcaption></figure>'
        for k, v in sorted(assets["stain_strip"].items())
    )
    parts.append(f'<div class="strip">{strip}</div>')

    # 6. per-section table
    parts.append('<h2><span class="num">7</span>Per section</h2>')
    rows = []
    for stain_name, block in frame.groupby("stain"):
        movers = block[block["scanner"] != "AT2"]
        rows.append([
            stain_name,
            f"{int(block['ok'].sum())}/{len(block)}",
            f"{int(movers['inliers'].min()):,}",
            f"{int(movers['inliers'].median()):,}",
            f"{movers['scale'].min():.4f}",
            f"{movers['scale'].max():.4f}",
            f"{int(block['candidates'].iloc[0]):,}",
            f"{int(block['locations'].iloc[0]):,}",
        ])
    parts.append(table(
        "Per staining condition, six moving scanners against AT2",
        ["section", "passed", "inliers min", "inliers median", "scale min", "scale max",
         "shared candidate positions", "locations sampled"],
        rows))

    # 7. differences from PanNormal
    parts.append('<h2><span class="num">8</span>What differs from the PanNormal cohort</h2>')
    parts.append(
        "<p class=\"wide\">PanNormal stays the primary cohort. PLISM is reported alongside it, and "
        "the reason is in the first row of this table.</p>"
    )
    comparison = [
        ["<strong>Sampling density</strong>",
         "mixed: 20× for AT2, AKOYA, S360, S60; 40× for VERSA, GT450. Native MPP 0.2624–0.5052, "
         "so Nyquist spans 0.99–1.91 cyc/µm",
         "uniform 40×, native MPP 0.2208–0.2621, Nyquist 1.91–2.27 cyc/µm",
         "In PanNormal the high band's upper edge, 0.99 cyc/µm, <em>is</em> AT2's Nyquist, while "
         "GT450 sits at half of its own. Sampling density and measured transfer are rank-correlated "
         "at 0.90, so optics and sampling are not separable there. They are here."],
        ["<strong>MPP header accuracy</strong>",
         "not independently checked; the native-to-target affines contain the same information and "
         "have not been read back",
         "five of six consistent to 0.11 ‰; Philips off by 9.84 ‰",
         "The physical-frequency argument rests on these numbers. PLISM's now carry a measurement; "
         "PanNormal's should get the same audit."],
        ["<strong>Contrast rendering</strong>",
         "unmeasured across scanners",
         "GT450 and Philips render tissue at roughly half the optical density of AT2; band power "
         "tracks od_std² across all seven scanners",
         "This is the finding that inverted the external replication. Raw band power measures how "
         "darkly a scanner draws the stain as much as how sharply. §3.1 uses the same raw statistic "
         "and needs auditing; §9 works after Reinhard and is insulated."],
        ["<strong>Replicate unit</strong>",
         "109 physical slides, 37 tissue types, 100 locations each — 65,400 paired acquisitions",
         "13 serial sections of one TMA block, 46 cores, one institution",
         "An eight-fold difference in replicate count. Every comparator gate, non-inferiority margin "
         "and collapse threshold in the locked analysis depends on the larger base."],
        ["<strong>Independent colour axis</strong>",
         "none — one staining protocol",
         "13 H&amp;E conditions fully crossed with the 7 scanners",
         "Only PLISM can ask whether the signature is instrument or chemistry. It answers: scanner "
         "66 %, stain 30 %, interaction 4 % of log band-power variance."],
        ["<strong>Geometry</strong>",
         "alias-audited render chain, 65,400/65,400 patches passing integer alignment, residual "
         "phase, padding and bounds checks",
         "coarse similarity per section, used for coordinate selection only; residual jitter of "
         "order 16 µm on a 129 µm patch",
         "PanNormal supports pixel-paired work — paired OD affine, the paired oracle, same-location "
         "content margin. PLISM as processed here does not, and registering it properly would "
         "reintroduce the interpolation the cohort was chosen to avoid."],
        ["<strong>Slide placement</strong>",
         "handled inside the frozen geometry",
         "Philips scans rotated 180° in 13 of 13 sections; S210 consistently tilted 0.4°",
         "Not a defect, but fatal to any unregistered location matching — independent sampling would "
         "have compared different tissue for Philips throughout."],
        ["<strong>Scanner panel</strong>",
         "AT2, GT450, VERSA, AKOYA, S60, S360",
         "AT2, GT450, S60, S360, S210, SQ, Philips",
         "Four shared. AKOYA, the most extreme signature in our panel at 0.335, has no counterpart "
         "here, so the external check covers the middle of the range and not its edge."],
        ["<strong>Availability</strong>",
         "not public",
         "CC BY 4.0, 180.8 GB, plus a MICCAI 2025 leaderboard on the same tiles",
         "PLISM claims are reproducible by anyone, which is worth more for a mechanism claim than "
         "for a benchmark claim."],
    ]
    head = "".join(f"<th>{h}</th>" for h in ["", "PanNormal", "PLISM", "why it matters"])
    body = "".join(
        f'<tr><td style="white-space:normal;width:9rem">{a}</td>'
        f'<td style="white-space:normal;text-align:left">{b}</td>'
        f'<td style="white-space:normal;text-align:left">{c}</td>'
        f'<td style="white-space:normal;text-align:left;color:var(--ink-soft)">{d}</td></tr>'
        for a, b, c, d in comparison
    )
    parts.append(
        f'<div class="scroll"><table style="font-family:var(--f-body);font-size:.86rem">'
        f"<caption>Cohort differences found while registering PLISM</caption>"
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )

    parts.append(
        '<div class="note"><strong>Where this leaves the two cohorts.</strong> PanNormal keeps the '
        "correction and benchmark claims: it has the replicate count, the registered pixel pairs and "
        "the audited render chain, and its results are locked. PLISM carries the mechanism claims "
        "PanNormal cannot identify — optics against sampling, and instrument against chemistry. "
        "Neither substitutes for the other, and the magnification difference is exactly why.</div>"
    )

    ert_path = Path(args.ert)
    if ert_path.exists():
        ert = pd.read_csv(ert_path, index_col=0)
        parts.append("<h3>Context: the native spectra measured on these locations</h3>")
        parts.append(
            "<p>Reported for completeness. The raw column is confounded by the contrast difference "
            "above and is not a sharpness ordering; see the results digest.</p>"
        )
        rows = [
            [str(index), f"{row['estimate']:+.3f}", f"{row['ci_low']:+.3f}", f"{row['ci_high']:+.3f}",
             f"{row['fold']:.3f}",
             "—" if pd.isna(row.get("locked_fold")) else f"{row['locked_fold']:.3f}",
             f"{row['normalised_log2']:+.3f}"]
            for index, row in ert.iterrows()
        ]
        parts.append(table(
            "Native high-band transfer against AT2, 0.10–0.99 cyc/µm, 13-section bootstrap",
            ["scanner", "log2 ERT", "CI low", "CI high", "fold", "PanNormal fold", "contrast-normalised log2"],
            rows))

    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PLISM section registration — alignment QC and differences from PanNormal</title>
<style>{STYLE}</style>
</head>
<body>
<div class="page">
<header class="top">
  <div class="kicker">PanNormal · external cohort · 2026-08-08</div>
  <h1>Aligning PLISM: what the seven scanners agree on,<br>and where this cohort differs from ours</h1>
  <p class="lede">Thirteen staining conditions, seven scanners, one tissue microarray block.
  Each section's scanners were brought into correspondence so that the same tissue could be read on
  every instrument without resampling a single analysed pixel. This page records the alignment, the
  quality metrics behind it, and the cohort differences the exercise exposed.</p>
</header>
{"".join(parts)}
<footer>
Contract docs/e9_plism_native_ert_contract.md · registration outputs/plism_section_registration ·
spectra outputs/plism_native_psd · PLISM (Ochi et al., Sci Data 11:330, 2024) CC BY 4.0 ·
generated by src/build_plism_registration_report.py
</footer>
</div>
</body>
</html>"""

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document)
    print(f"wrote {destination} ({destination.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()

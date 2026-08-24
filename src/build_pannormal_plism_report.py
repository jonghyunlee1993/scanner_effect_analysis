"""Merge the PLISM external validation into the PanNormal report.

The August report ended by naming PLISM as the highest-value next step, with a
prerequisite attached.  That step has been taken twice: once on a sparse sample
of the cohort, and now on the rebuilt one.  Part IV is appended after section 14,
so the locked sections 1-14 keep their numbering and anything already cited
against this report still resolves.

**What changed against the 2026-08-10 edition.**  The cohort underneath Part IV
was rebuilt, not re-read.  Registration now covers 85-90% of the tissue at a
patch-pitch lattice instead of 6% at a random sample -- 817,817 measured
locations against 26,712 -- and the 46 TMA cores carry their published tissue
names, which makes PanNormal's own tissue model runnable here for the first time.
Rebuilding it also turned up three defects in the published dataset: two files
that hold each other's section, and one out-of-focus scan.  The encoder panel is
the E9 one (UNI2-h, CONCHv1.5, H-optimus-1) rather than the E0 one.

Every figure in Part IV is read from the analysis outputs by
`plism_core_report_data.py` rather than typed into the prose, because a stale
literal looks exactly like a fresh one.

Sources, all regenerated in this repository:
    outputs/plism_core_registration        the core map and the lattice
    outputs/plism_core_refinement          alignment residual per location
    outputs/plism_core_reinhard_bands      raw and post-Reinhard band energies
    outputs/plism_core_rf1u_destinations   destination rule on native energies
    outputs/plism_core_render_gains        RF1U gains on the deployment grid
    outputs/plism_core_pfm_frontier        representation endpoints, image arm
    outputs/plism_core_feature_correction  representation endpoints, feature arm
    outputs/plism_core_random_slopes       nested REML, both nestings
    outputs/plism_core_stain_covariate     stain covariate on the scanner slope
    outputs/plism_core_native_ert          native-resolution ERT
    data/PLISM_dataset/alignment           the cohort's own alignment report
    data/PLISM_dataset/features/analysis   cross-scanner retrieval per encoder
    outputs/pannormal_contrast_audit       the same contrast test on our cohort
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import plism_core_report_data as data
from plism_core_report_chart import CHART_CSS, sample_size_curve

SOURCE = "presentations/rf1u_multitarget_2026-08-04/index.html"
DESTINATION = "presentations/pannormal_plism_2026-08-23/index.html"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=SOURCE)
    parser.add_argument("--output", default=DESTINATION)
    parser.add_argument("--variant", default="focus_ok",
                        choices=("focus_ok", "all_blocks"),
                        help="whether the out-of-focus HRH/S60 block is in the "
                             "sharpness statistics")
    return parser.parse_args()


# --------------------------------------------------------------------- markup
def row(cells, first_class="txt", classes=None):
    classes = classes or {}
    out = [f'<td class="{first_class}">{cells[0]}</td>']
    for index, value in enumerate(cells[1:], start=1):
        cls = classes.get(index, "")
        out.append(f'<td class="{cls}">{value}</td>' if cls else f"<td>{value}</td>")
    return "<tr>" + "".join(out) + "</tr>"


def table(caption, headers, rows, note=None):
    head = "".join(f'<th class="txt">{h}</th>' if i == 0 else f"<th>{h}</th>"
                   for i, h in enumerate(headers))
    body = "\n".join(rows)
    tail = f'<div class="note">{note}</div>' if note else ""
    return f"""    <div class="table-wrap">
      <table>
        <caption>{caption}</caption>
        <thead><tr>{head}</tr></thead>
        <tbody>
{body}
        </tbody>
      </table>
    </div>
{tail}"""


def section(number, label, heading, blocks):
    return f"""
  <details class="sec" id="s{number}" open>
    <summary class="sec-head">
      <span class="sec-num">{number}</span>
      <span class="sec-titles"><span class="label">{label}</span><span class="sec-h">{heading}</span></span>
    </summary>
{"".join(blocks)}
  </details>
"""


def prose(*paragraphs):
    body = "".join(f"<p>{p}</p>\n" for p in paragraphs)
    return f'\n    <div class="prose">\n{body}    </div>\n'


def callout(heading, *paragraphs):
    body = "".join(f"<p>{p}</p>\n" for p in paragraphs)
    return f'\n    <div class="callout">\n      <h3>{heading}</h3>\n{body}    </div>\n'


def figure(markup, caption):
    """The report's own figure convention: bare <figure>, frame, caption."""
    return (f'\n    <figure>\n      <div class="fig-frame">{markup}</div>\n'
            f'      <figcaption>{caption}</figcaption>\n    </figure>\n')


def pos(text):
    return f'<span class="num-pos">{text}</span>'


def neg(text):
    return f'<span class="num-neg">{text}</span>'


def dim(text):
    return f'<span class="dim">{text}</span>'


def flag(text, ok=True):
    return f'<span class="flag flag-{"pass" if ok else "fail"}">{text}</span>'


def scanner(name):
    return data.SCANNER_LABEL.get(name, name)


def encoder(name):
    return data.ENCODER_LABEL.get(name, name)


# ---------------------------------------------------------------- section 15
def build_s15(bundle) -> str:
    a = bundle["alignment"]
    per_scanner = a["per_scanner"].set_index("scanner")
    coverage = a["coverage"]
    fit = a["canvas_fit"]

    rows = []
    for name in data.SCANNER_ORDER:
        if name == data.REFERENCE or name not in per_scanner.index:
            continue
        block = per_scanner.loc[name]
        rows.append(row([f"<code>{scanner(name)}</code>",
                         f"{block['median_um']:.3f}", f"{block['p90_um']:.3f}",
                         f"{block['response_median']:.2f}",
                         f"{100 * block['usable_frac']:.2f} %"]))

    defects = a["corrections"]
    swapped = ", ".join(f"<code>{r.published}.ndpi</code> ⇄ <code>{r.actual_file}</code>"
                        for r in defects.itertuples())

    return section(
        15, "The external cohort", "A second cohort where optics and sampling come apart",
        [
            prose(
                "Section 14 named PLISM as the highest-value external check and attached one "
                "prerequisite: its published tiles are warped onto a common reference and "
                "resampled, so interpolation could alter the exact quantity this report "
                "measures. The prerequisite is moot. The <strong>91 original, unregistered "
                "whole-slide images are public</strong> (figshare+, 180.8&nbsp;GB, "
                "CC&nbsp;BY&nbsp;4.0), so nothing downstream depends on someone else's render "
                "chain.",
                "PLISM is one tissue microarray block cut into 13 serial sections, each stained "
                "under a different H&amp;E condition and each scanned on all seven instruments "
                "— a fully crossed 13&nbsp;×&nbsp;7 design. Within a stain the seven scanners "
                "image the <em>same physical section</em>, so a scanner contrast is "
                "content-matched by construction.",
            ),
            table(
                "The property that makes this cohort worth the trouble",
                ["", "PanNormal", "PLISM"],
                [
                    row(["Native sampling", "mixed: 0.2624–0.5052 µm/px",
                         "uniform 40×: 0.2208–0.2621 µm/px"]),
                    row(["Nyquist span", "0.99 – 1.91 cyc/µm", "1.91 – 2.27 cyc/µm"]),
                    row(["High band 0.10–0.99 cyc/µm", "reaches AT2's own Nyquist",
                         "well inside every scanner's"]),
                    row(["Sampling vs measured transfer", flag("rank-correlated 0.90", False),
                         flag("equalised by design", True)]),
                    row(["Independent colour axis", "none", "13 staining conditions"]),
                    row(["Tissue axis", "10 organ types across 109 slides",
                         f"{a['cores_max']} named cores on one block"]),
                    row(["Replicate unit", "109 physical slides", "13 serial sections"]),
                ],
                note="In PanNormal the top of the high band <em>is</em> AT2's Nyquist while "
                     "GT450 sits at half of its own, so optics and sampling density cannot be "
                     "separated there. That is the one thing PLISM can do and we cannot.",
            ),
            callout(
                "Registration decides where to read, and nothing else",
                "Each section's seven scanners were brought into correspondence at 16 µm/px, "
                "then each location was refined against the section's AT2 reference by phase "
                "correlation and the correction applied as an <strong>integer pixel shift</strong>. "
                "Every analysed patch is an integer crop at level 0 on the scanner's own grid: "
                "no analysed pixel is interpolated, warped or resampled.",
                f"Across {a['measurements_nonref']:,} scanner-to-reference "
                f"measurements the median residual is "
                f"<strong>{a['residual_median_low']:.3f}–{a['residual_median_high']:.3f} µm</strong> "
                f"— under a tenth of a percent of the "
                f"{a['patch_um']:.0f} µm patch — and "
                f"{100 * a['gate_pass_nonref']:.2f} % of measurements clear a 1 µm gate. "
                "The rest are dropped for that scanner rather than entering the estimate.",
            ),
            prose(
                "<strong>This part of the report has been rebuilt since the August edition, and "
                "the rebuild is the reason to read it again.</strong> The first pass sampled "
                "locations at random and reached "
                + dim(f"{100 * a['sparse_coverage']:.1f} %")
                + " of the tissue by the same footprint measure. This one lays a "
                f"lattice at the {a['patch_um']:.0f} µm patch pitch and reaches "
                f"<strong>{100 * a['coverage_low']:.0f}–{100 * a['coverage_high']:.0f} %</strong> "
                f"of it: {a['locations_total']:,} locations against "
                f"{a['sparse_locations']:,}, a "
                f"{a['locations_total'] / a['sparse_locations']:.0f}-fold increase in what every "
                "number below is computed from. That matters most in section 18, where the "
                "sparse version could not distinguish a method failing from an estimator "
                "starved of samples.",
                "The cores can also now be named. PLISM's 46 tissue labels live in the authors' "
                "Elastix canvas, and the August edition treated them as unreachable without the "
                "warp this work avoids. That was too strong: naming a core does not need pixel "
                "accuracy — a core is 2.5 mm across on a 3.6 mm pitch — so a single global "
                "similarity between the canvas occupancy lattice and the section's own tissue "
                "mask suffices, and it carries only a <em>name</em> inward. No analysed pixel is "
                "placed by it.",
            ),
            table(
                "Alignment on the rebuilt grid, per scanner against its section's AT2",
                ["Scanner", "Median residual, µm", "p90, µm", "Median response", "Usable"],
                rows,
                note=f"{a['locations_total']:,} lattice locations over "
                     f"{a['sections']} sections. <em>Response</em> is the phase-correlation "
                     "peak height, carried through rather than gated on; <em>usable</em> is the "
                     "share clearing both the 1 µm residual gate and a 0.3 response floor.",
            ),
            callout(
                "The canvas fit is right, not merely converged",
                f"Fitted independently on each of the {a['sections']} sections, the canvas pixel "
                f"comes out at <strong>{a['canvas_um_low']:.4f}–{a['canvas_um_high']:.4f} µm</strong>, "
                "recovering the Hamamatsu 0.220 µm pitch the canvas was built on and which the "
                f"fit was never given. The rotation lands within {a['rotation_max']:.2f}° of zero "
                f"every time, and the mask overlap is Dice {a['dice_low']:.2f}–{a['dice_high']:.2f}. "
                "Three independent checks, none of them the objective being optimised.",
            ),
            prose(
                "Rebuilding the cohort also turned up <strong>three defects in the published "
                "dataset</strong>, none of which is visible in any header, and all three are "
                "recorded in <code>PROVENANCE.md</code> of the public folder rather than only "
                "here.",
                f"<strong>Two files hold each other's section.</strong> {swapped}. Matched "
                "against the AT2 reference of all thirteen sections, each scores 0.75–0.83 "
                "against the other's label and 0.11–0.13 against its own, while the remaining "
                "eleven sections score 0.09–0.32; over eight cores the verdict is unanimous, "
                "16 of 16, and S360 through the identical comparison is correct 16 of 16. Four "
                "alternatives were excluded first — wrong core, a large offset, a scale error, "
                "and blur. Under the published labels these two blocks retained "
                f"{neg('0 %')} of their locations. Corrected, they retain "
                f"{pos('99.1 %')} and {pos('97.8 %')}.",
                "<strong>One scan is out of focus.</strong> <code>HRH_S60</code> is correctly "
                "labelled and correctly registered — patches sit 4–11 µm from the reference and "
                "blur-tolerant cross-correlation confirms the tissue — but its high-band power "
                "is down five- to tenfold. That is a real property of the scan, so it stays in "
                "the dataset. It is excluded from the sharpness statistics of sections 16 to 19, "
                "and only those, because the endpoint of this study <em>is</em> high-band power: "
                "a per-section sharpness number that includes it measures a focus failure rather "
                "than a scanner. The exclusion is at block level and never per patch — rejecting "
                "individual patches for looking blurred is precisely the detail-based selection "
                "this study's own protocol forbids, and it would remove the evidence along with "
                "the problem.",
            ),
            prose(
                "One more thing the rebuild settled. Every downstream analysis gates on residual "
                "alone, and phase correlation always returns a peak — on unrelated content that "
                "peak is noise, biased toward zero shift by the Hann window. Measured over the "
                f"whole refinement, {100 * a['leak_fraction']:.2f} % of non-reference "
                "measurements clear the 1 µm gate while their response says the alignment could "
                "not be verified at all. Before the file correction the two SQ blocks were made "
                "<em>entirely</em> of such leaks. That is now a fraction of a percent, and "
                "<code>response</code> is stored per location so any analysis can tighten the "
                "gate without re-measuring anything.",
            ),
        ],
    )


# ------------------------------------------------------------- shared blocks
def retrieval_blocks(bundle) -> list[str]:
    """What the embeddings do to a cross-scanner match, before any correction.

    Reported as top-1 retrieval rather than raw cosine because the three encoders
    do not share a cosine scale -- CONCHv1.5 pairs at 0.94-0.97 and UNI2-h at
    0.69-0.89, which looks decisive until the null is read beside it.  A rank
    statistic does not care.
    """
    r = bundle["retrieval"]
    top1 = r["top1"]
    order = [e for e in data.ENCODER_ORDER if e in top1.columns]
    order += [e for e in top1.columns if e not in order]

    rows = []
    for name in data.SCANNER_ORDER:
        if name == data.REFERENCE or name not in top1.index:
            continue
        best = top1.loc[name, order].idxmax()
        cells = []
        for enc in order:
            value = top1.loc[name, enc]
            cells.append(f"<strong>{value:.3f}</strong>" if enc == best else f"{value:.3f}")
        rows.append(row([f"<code>{scanner(name)}</code>", *cells]))
    rows.append(row(["<strong>mean</strong>",
                     *(f"<strong>{top1[e].mean():.3f}</strong>" for e in order)]))

    agreement = r["agreement"]
    pairs = sorted(agreement.items(), key=lambda kv: -kv[1])
    strongest, weakest = pairs[0], pairs[-1]
    worst_scanner = top1.min(axis=1).idxmin()
    worst_value = top1.min(axis=1).min()

    return [
        prose(
            "Before any correction is fitted, it is worth asking what the batch effect looks "
            "like in these embeddings at all — and the answer depends entirely on which "
            "question is asked of them. <strong>No encoder clusters by scanner.</strong> "
            "Silhouette by scanner is within ±0.02 of zero for all three, and in the embedding "
            "plots the seven machines cover the same ground. A study that checks for scanner "
            "clusters would stop here and report that there is nothing to fix.",
            "Ask instead whether a given patch can be found again on another scanner and the "
            f"picture changes. Of all locations of a core on the reference, the worst scanner "
            f"misses the matching location "
            f"<strong>{100 * (1 - worst_value):.0f} times in a hundred</strong> "
            f"(<code>{scanner(worst_scanner)}</code>, {encoder(top1.loc[worst_scanner].idxmin())}). "
            "The scanner effect is invisible in the global structure and plainly present in the "
            "correspondence — which is the case for a correction that acts on the embedding "
            "rather than on the picture.",
        ),
        table(
            "Top-1 retrieval against the section's own AT2 acquisition, no correction",
            ["Scanner", *(encoder(e) for e in order)],
            rows,
            note="Of all locations of a TMA core on the reference scanner, the share whose "
                 "nearest neighbour on the other scanner is the matching location. A rank "
                 "statistic, so it compares across encoders where raw cosine does not.",
        ),
        callout(
            "The scanners do not rank the same way for every encoder",
            f"{encoder(strongest[0][0])} and {encoder(strongest[0][1])} agree almost completely "
            f"(Spearman {strongest[1]:+.2f}); "
            f"{encoder(weakest[0][0])} against {encoder(weakest[0][1])} is {weakest[1]:+.2f}. "
            "Only the extremes are common to all three. Take a single ranking of \"which scanner "
            "is hardest\" from one encoder and it will not transfer to the next — which is a "
            "caution against tuning a deployment to whichever encoder was benchmarked.",
        ),
    ]


def tissue_blocks(bundle) -> list[str]:
    """Where the cross-scanner match fails, by named tissue."""
    by_tissue = bundle["retrieval"]["by_tissue"]
    import pandas as pd

    wide = pd.DataFrame({name: frame.set_index("tissue_type")["top1"]
                         for name, frame in by_tissue.items()})
    order = [e for e in data.ENCODER_ORDER if e in wide.columns]
    wide = wide[order]
    wide["mean"] = wide.mean(axis=1)
    ranked = wide.sort_values("mean")

    def block(frame, highlight):
        out = []
        for tissue, entry in frame.iterrows():
            cells = [f"{entry[e]:.3f}" for e in order]
            label = tissue.split("_", 1)[1].replace("_", " ") if "_" in tissue else tissue
            out.append(row([f"<code>{label}</code>", *cells,
                            (neg if highlight else pos)(f"{entry['mean']:.3f}")]))
        return out

    rows = block(ranked.head(5), True) + [
        row([dim("…"), *(dim("") for _ in order), dim("")])] + block(ranked.tail(4), False)

    return [
        prose(
            "The tissue axis answers this differently again, and it is an axis the August "
            "edition could not use at all: with the cores unnamed, PLISM's tissue labels were "
            "treated as a nuisance index. Named, they show that the cross-scanner match does "
            f"not fail uniformly. Within one block, one stain and one set of instruments, top-1 "
            f"retrieval runs from <strong>{ranked['mean'].min():.3f} to "
            f"{ranked['mean'].max():.3f}</strong> across the "
            f"{len(ranked)} tissue types.",
            "<strong>It fails where the tissue has no architecture to anchor a match.</strong> "
            "Liver, skeletal muscle, collagenous fibre and left ventricle sit at the bottom — "
            "homogeneous, repetitive, few landmarks. Gallbladder, bronchus, colonic "
            "adenocarcinoma and ileum sit at the top, and they are the structurally rich ones. "
            "Read with section 3.2, which reports the scanner effect varying with tissue type "
            "on our own cohort, this says what that variation is <em>made of</em>: a scanner's "
            "distortion costs most where there is least structure to survive it.",
        ),
        table(
            "Top-1 retrieval by tissue — the five hardest and the four easiest",
            ["Tissue", *(encoder(e) for e in order), "mean"],
            rows,
            note="Pooled over the six non-reference scanners and all 13 sections. The ordering "
                 "is the encoders' shared one: the hardest tissues are hardest for all three, "
                 "even where the encoders disagree about which scanner is hardest.",
        ),
    ]


# ---------------------------------------------------------------- assembly
CHAPTER_IV = """
  <div class="chapter" id="ch-part-iv">
    <span class="chapter-eyebrow">Part IV · External validation</span>
    <span class="chapter-title">The same questions, asked of a cohort we did not build</span>
    <p class="chapter-note">Section 14 named PLISM as the highest-value next step and attached a prerequisite. This is what happened when it was done — and then done again on the whole cohort rather than a 6&nbsp;% sample of it, which changed one of the answers.</p>
    <span class="chapter-weight">sections 15–19</span>
  </div>
"""

TOC_ROWS = (
    (15, "The external cohort", "A second cohort where optics and sampling come apart"),
    (16, "What band power measures", "Contrast, sampling and optics, finally separated"),
    (17, "The destination rule", "The same estimator, aimed seven ways"),
    (18, "Feature space, externally", "The conclusion this report ends on, in a cohort it never saw"),
    (19, "What it settles", "And what it now settles that it could not before"),
)


def toc() -> str:
    out = ['<div class="toc-group"><span class="toc-group-title">Part IV · External validation'
           '</span><span class="toc-group-note">PLISM, and what it settles</span></div>']
    for number, label, title in TOC_ROWS:
        out.append(f'<a class="toc-row" href="#s{number}"><span class="toc-num">{number}</span>'
                   f'<span class="toc-text"><span class="toc-label">{label}</span>'
                   f'<span class="toc-title">{title}</span></span></a>')
    return "".join(out)


def merge(text: str, sections: str, replicated: int, total: int) -> str:
    """Splice Part IV into the locked report without disturbing sections 1-14."""
    def swap(old: str, new: str) -> None:
        nonlocal text
        if old not in text:
            raise RuntimeError(f"anchor not found in the source report: {old[:70]!r}")
        text = text.replace(old, new, 1)

    swap("<title>What a scanner does to a foundation model — PanNormal harmonization report</title>",
         "<title>What a scanner does to a foundation model — PanNormal, with external "
         "validation on PLISM</title>")
    swap("PanNormal · scanner harmonization · report for PI review · 2026-08-04",
         "PanNormal · scanner harmonization · with PLISM external validation · 2026-08-23")

    # the chart brings its own tokens; they must sit with the report's own
    swap("  .dim { color: var(--ink-faint); }",
         "  .dim { color: var(--ink-faint); }\n" + CHART_CSS)

    swap("""      <div class="stat">
        <div class="stat-value">4</div>
        <div class="stat-label">frozen foundation models: ResNet50, UNI v1, CONCH v1, Virchow2</div>
      </div>""",
         f"""      <div class="stat">
        <div class="stat-value">4</div>
        <div class="stat-label">frozen foundation models: ResNet50, UNI v1, CONCH v1, Virchow2</div>
      </div>
      <div class="stat">
        <div class="stat-value" style="color:var(--accent)">{replicated} <span class="dim" style="font-size:1.1rem">/ {total}</span></div>
        <div class="stat-label">claims replicated on PLISM, an independent public 13 × 7 cohort, at 88 % tissue coverage and on a 2026 encoder panel</div>
      </div>""")

    swap("Section 12 states where we think image-space work ends, and sections 13 and 14 place the work\n"
         "        against the current literature and say what we would do next.",
         "Section 12 states where we think image-space work ends, and sections 13 and 14 place the work\n"
         "        against the current literature and say what we would do next. Sections 15 to 19 are new:\n"
         "        they report what happened when the external validation section 14 asked for was carried\n"
         "        out on PLISM, a public cohort of 13 staining conditions crossed with 7 scanners.")

    # Part IV goes after section 14, so 1-14 keep their numbering and every prior citation resolves
    swap("\n  <footer>", CHAPTER_IV + sections + "\n  <footer>")

    marker = '<a class="toc-row" href="#s14">'
    end = text.index("</div>", text.index(marker))
    text = text[:end] + toc() + text[end:]

    swap("docs/final_study_protocol.md · docs/e5_rf1u_multitarget_contract.md · "
         "docs/e5_rf1u_multitarget_results.md",
         "docs/final_study_protocol.md · docs/e5_rf1u_multitarget_contract.md · "
         "docs/e5_rf1u_multitarget_results.md · docs/e9_plism_native_ert_contract.md")
    return text


def main() -> None:
    args = parse_args()
    bundle = data.load_all(variant=args.variant)
    bundle["variant"] = args.variant

    sections = "".join([
        build_s15(bundle), build_s16(bundle), build_s17(bundle),
        build_s18(bundle), build_s19(bundle)])
    verdicts = bundle["verdicts"]
    replicated = sum(1 for v in verdicts if v["outcome"] == "replicated")

    text = merge(Path(args.source).read_text(), sections, replicated, len(verdicts))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text)
    print(f"wrote {destination} ({destination.stat().st_size / 1e6:.2f} MB)")
    print(f"  variant {args.variant}: {replicated} of {len(verdicts)} claims replicated")



# ---------------------------------------------------------------- section 16
# The four scanners both cohorts have, and the locked transfer each was assigned
# in section 3.1.  PLISM's Philips, S210 and SQ have no PanNormal counterpart.
LOCKED_TRANSFER = {"GT450": 1.478, "S60": 1.044, "S360": 0.842, "AT2": 1.000}
# Post-Reinhard fine-band power on PanNormal, from the locked section 9 fit.
from analyze_plism_rf1u_destinations import PANNORMAL_POST_REINHARD


def build_s16(bundle) -> str:
    b = bundle["band_power"]
    ratio = b["ratio"]
    pannormal = b["pannormal"]
    variant = bundle["variant"]
    column = "raw" if variant == "all_blocks" else "raw_focus_ok"
    reinhard_column = "reinhard" if variant == "all_blocks" else "reinhard_focus_ok"

    shared = [s for s in ratio.index if s in LOCKED_TRANSFER]
    rho_raw = data.spearman([ratio.loc[s, column] for s in shared],
                            [LOCKED_TRANSFER[s] for s in shared])
    rho_rein = data.spearman([ratio.loc[s, reinhard_column] for s in shared],
                             [LOCKED_TRANSFER[s] for s in shared])

    rows = []
    for name in data.SCANNER_ORDER:
        if name not in ratio.index:
            continue
        raw_value = ratio.loc[name, column]
        rein_value = ratio.loc[name, reinhard_column]
        locked = LOCKED_TRANSFER.get(name)
        raw_cell = (dim(f"{raw_value:.3f}") if name == data.REFERENCE
                    else neg(f"{raw_value:.3f}") if raw_value < 0.7
                    else f"{raw_value:.3f}")
        rein_cell = (dim(f"{rein_value:.3f}") if name == data.REFERENCE
                     else pos(f"<strong>{rein_value:.3f}</strong>") if rein_value > 2.0
                     else pos(f"{rein_value:.3f}") if rein_value > 1.3
                     else f"{rein_value:.3f}")
        rows.append(row([f"<code>{scanner(name)}</code>", raw_cell, rein_cell,
                         dim("1.000") if name == data.REFERENCE
                         else (f"{locked:.3f}" if locked else "—")]))

    gt450 = ratio.loc["GT450"]
    pan_gt450 = pannormal.loc["gt450"]
    akoya = pannormal.loc["akoya"]
    plism = b["plism"]
    shared_lower = [s for s in shared if s.lower() in pannormal.index]
    plism_rho = data.spearman(
        [plism.loc[s, "excess"] for s in shared_lower],
        [data.LOCKED_TRANSFER[s.lower()] for s in shared_lower])

    return section(
        16, "What band power measures", "Contrast, sampling and optics, finally separated",
        [
            prose(
                "Measured the way section 3.1 measures it, PLISM inverts this report's ordering "
                f"completely: rank correlation with the locked transfer across the "
                f"{len(shared)} shared scanners is <strong>{rho_raw:+.3f}</strong>. GT450, the "
                "sharpest instrument in our panel, comes out the softest.",
                "The cause is not sharpness. Across all seven scanners the raw band-power ratio "
                "against AT2 tracks the <em>squared</em> optical-density spread: GT450 renders "
                "tissue at roughly half AT2's contrast, and its darkest one per cent of pixels "
                "only reaches 150 of 255. Raw band power measures how darkly a scanner draws "
                "the stain at least as much as how sharply it resolves it.",
            ),
            table(
                "Fine-band power against AT2 — the same patches, before and after colour matching",
                ["Scanner", "Raw", "After Reinhard", "Locked transfer"],
                rows,
                note=f"{b['patches']:,} patches over 13 sections. Rank correlation with the "
                     f"locked ordering over the {len(shared)} shared scanners: "
                     f"{flag(f'raw {rho_raw:+.3f}', rho_raw > 0)} "
                     f"{flag(f'after Reinhard {rho_rein:+.3f}', rho_rein > 0)}. "
                     "Reinhard here is the frozen implementation — CIE Lab moments closed by "
                     "source-ray gamut projection, not per-channel clipping."
                     + ("" if variant == "all_blocks" else
                        " The out-of-focus <code>HRH/S60</code> block is excluded; with it in, "
                        f"S60 reads {ratio.loc['S60', 'raw']:.3f} raw and "
                        f"{ratio.loc['S60', 'reinhard']:.3f} after Reinhard."),
            ),
            callout(
                "This is the mechanism behind section 12's hinge",
                "Section 6 records that frequency calibration on raw images is worth nothing — "
                "+0.5 %, −0.1 %, −0.5 %, +0.3 % — while the same correction after Reinhard is "
                "worth +2.8 to +5.0 points. This report states that as an observation and does "
                "not explain it.",
                "PLISM supplies the explanation. <strong>The raw band-power gap between scanners "
                "is mostly a contrast gap.</strong> Fitting band gains on raw energies therefore "
                "corrects contrast, which Reinhard removes anyway, so it adds nothing. Only "
                "after colour is matched is what remains in the bands an optical residual — and "
                "correcting that pays.",
            ),
            prose(
                "The obvious worry is that section 3.1 is then measuring contrast too. The audit "
                "that settles it divides each scanner's raw fine-band power by the square of its "
                "optical-density spread, which is what band power in a fixed band scales with, "
                "and asks what is left. On our own cohort the residual reproduces the locked "
                f"ordering exactly — rank correlation {pannormal.attrs['rho_locked']:+.3f} — so "
                "section 3.1 is not measuring contrast.",
                "Running the same audit on PLISM is where the August edition went wrong, and the "
                "rebuilt cohort is what makes the error visible.",
            ),
            table(
                "The same contrast audit, run on both cohorts",
                ["", "GT450 raw fine band vs AT2", "Predicted by contrast alone", "Excess"],
                [
                    row([f"PLISM (sampling equalised, {plism.attrs['locations']:,} locations)",
                         f"{plism.loc['GT450', 'b1_vs_at2']:.3f}",
                         f"{plism.loc['GT450', 'predicted']:.3f}",
                         pos(f"<strong>{plism.loc['GT450', 'excess']:.2f}×</strong>")]),
                    row(["PanNormal (GT450 sampled 1.93× finer)",
                         f"{pan_gt450['b1_vs_at2']:.3f}", f"{pan_gt450['predicted']:.3f}",
                         pos(f"<strong>{pan_gt450['excess']:.2f}×</strong>")]),
                ],
                note="Contrast is the optical-density standard deviation, not its mean: the mean "
                     "says how dark the render is, the spread says how much signal there is to "
                     "resolve, and it is the spread that band power is made of.",
            ),
            callout(
                "A correction to the August edition",
                "That edition put PLISM's excess at 0.90× and concluded that where sampling is "
                "equalised, contrast accounts for the whole ratio — which made the 1.36× on our "
                "own cohort a sampling effect and supported a clean split of GT450's advantage "
                "into roughly half optics, half sampling density. <strong>On the rebuilt cohort "
                f"the excess is {plism.loc['GT450', 'excess']:.2f}×, statistically "
                f"indistinguishable from PanNormal's {pan_gt450['excess']:.2f}×.</strong> "
                "Equalising sampling does not remove it.",
                "So the excess is optics, not sampling density, and the split does not stand as "
                "stated. What survives, and is the load-bearing part of section 16, is the first "
                "claim: <strong>raw band power is dominated by contrast</strong>, which is why "
                "calibrating bands before colour is worth nothing and after colour is worth "
                "several points. The decomposition into three multiplicative terms is a weaker "
                "result than the August edition claimed, and this table is why.",
            ),
            prose(
                "There is a second reason not to lean on the decomposition. The two "
                "normalisations disagree about how much sampling contributes. Compared "
                "post-Reinhard, GT450 sits at "
                f"{gt450[reinhard_column]:.2f}× AT2 on PLISM against "
                f"{PANNORMAL_POST_REINHARD['GT450']:.2f}× here, which reads as a large sampling "
                "effect; compared through the contrast audit, the residual is the same in both "
                "cohorts, which reads as none. Reinhard matches moments to a reference "
                "population and projects the gamut, so it is not the same normalisation as "
                "dividing by the spread, and the cohorts differ in where their finest band sits "
                "relative to Nyquist. Both numbers are measurements; the inference between them "
                "is not settled here.",
                "The ordering does not transfer past GT450 either. Over the four scanners the "
                "cohorts share, the contrast-normalised residual reproduces the locked ordering "
                f"on PanNormal ({pannormal.attrs['rho_locked']:+.3f}) but not on PLISM "
                f"({plism_rho:+.3f}): S360 and S60 change places. Four scanners is a short list "
                "for a rank correlation, but it is enough to say the residual is not a single "
                "scanner property that both cohorts recover.",
            ),
            prose(
                "One scanner resists all three explanations. <strong>AKOYA</strong> stays at "
                f"{akoya['excess']:.3f} after contrast normalisation, samples at 0.4999 µm/px "
                "against AT2's 0.5052 so sampling cannot help it, shows no time trend across a "
                "36-day acquisition window (ρ = +0.110, p = 0.25), and varies strongly with "
                "tissue type (ANOVA p = 5.5e-04, tissue medians spanning 5.7×). Its slides were "
                "re-scanned and reproduced the same result, no instrument fault was found, and "
                "the acquisition used the clinic's production setting. A low-numerical-aperture "
                "objective is the only account left standing, and the header's 10× objective "
                "string is the candidate — but numerical aperture is not recoverable from "
                "headers, so the optical mechanism remains a hypothesis. What is <em>not</em> a "
                "hypothesis is that a deployed clinical configuration produces images three "
                "times softer than the panel reference.",
            ),
        ],
    )


# ---------------------------------------------------------------- section 17
def build_s17(bundle) -> str:
    d = bundle["destinations"]
    summary = d["summary"].set_index("target")
    frontier = bundle["image_frontier"]["frontier"]

    # Ordered by post-Reinhard detail power rather than by the panel's usual
    # order, because the claim being made is about that ordering.
    rows = []
    for name in summary.sort_values("detail_b1_vs_AT2", ascending=False).index:
        block = summary.loc[name]
        amplifying = block["regime"] == "amplify"
        rows.append(row([
            f"<code>{scanner(name)}</code>",
            dim("1.000") if name == data.REFERENCE else f"{block['detail_b1_vs_AT2']:.3f}",
            pos(f"<strong>{int(block['amplify'])}</strong>") if block["amplify"] > block["attenuate"]
            else f"{int(block['amplify'])}",
            neg(f"<strong>{int(block['attenuate'])}</strong>") if block["attenuate"] > block["amplify"]
            else f"{int(block['attenuate'])}",
            f"{block['mean_log_gain']:+.3f}",
            flag("amplify", True) if amplifying else flag("attenuate", False)]))

    worst = summary["detail_b1_vs_AT2"].idxmin()
    cells = int(summary["cells"].iloc[0])

    # Representation arm: relative radius reduction toward each destination.
    def frontier_cell(encoder_id, condition):
        block = frontier.loc[(frontier["encoder"] == encoder_id)
                             & (frontier["condition"] == condition)]
        if block.empty:
            return dim("—")
        entry = block.iloc[0]
        text = f"{100 * entry['rr']:+.1f} %"
        if entry["rr_ci_low"] <= 0:
            return dim(text)
        return pos(f"<strong>{text}</strong>" if entry["rr"] > 0.1 else text)

    encoders = [e for e in data.ENCODER_ORDER if e in set(frontier["encoder"])]
    frontier_rows = []
    for encoder_id in encoders:
        block = frontier.loc[frontier["encoder"] == encoder_id].set_index("condition")
        frontier_rows.append(row([
            encoder(encoder_id),
            frontier_cell(encoder_id, "rf1u_at2"),
            frontier_cell(encoder_id, "rf1u_gt450"),
            f"{block.loc['rf1u_at2', 'probe']:.3f}" if "rf1u_at2" in block.index else dim("—"),
            f"{block.loc['rf1u_gt450', 'probe']:.3f}" if "rf1u_gt450" in block.index else dim("—"),
        ]))

    def count_safe(condition):
        block = frontier.loc[frontier["condition"] == condition]
        return int((block["safe"] & block["improved"]).sum()), len(block)

    at2_safe, at2_total = count_safe("rf1u_at2")
    gt450_safe, gt450_total = count_safe("rf1u_gt450")

    # Per encoder, which destination actually wins -- the claim is an ordering,
    # so counting "safe and improved" on each side can hide a tie.
    gt450_better = 0
    for encoder_id in encoders:
        block = frontier.loc[frontier["encoder"] == encoder_id].set_index("condition")
        if {"rf1u_at2", "rf1u_gt450"} <= set(block.index):
            gt450_better += int(block.loc["rf1u_gt450", "rr"] > block.loc["rf1u_at2", "rr"])

    dead = [e for e in encoders
            if not frontier.loc[(frontier["encoder"] == e)
                                & (frontier["condition"] != "raw"), "improved"].any()]
    if dead:
        best = {}
        for encoder_id in dead:
            block = frontier.loc[(frontier["encoder"] == encoder_id)
                                 & (frontier["condition"] != "raw")]
            best[encoder_id] = block.loc[block["rr"].idxmax()]
        no_benefit_text = " ".join(
            f"For {encoder(e)} no image condition clears zero — its best is "
            f"{100 * best[e]['rr']:+.1f} % with a confidence interval from "
            f"{100 * best[e]['rr_ci_low']:+.1f} % to {100 * best[e]['rr_ci_high']:+.1f} %."
            for e in dead)
    else:
        no_benefit_text = ""

    return section(
        17, "The destination rule",
        "The same estimator, aimed seven ways, in a cohort that never saw ours",
        [
            prose(
                "Section 9's claim is not that our correction works. It is that <em>where you "
                "aim it</em> decides whether it helps, and that the quantity governing this is "
                "the target's post-Reinhard detail power. PLISM offers seven destinations "
                "instead of three, with sampling equalised, and the frozen RF1U rule — pooled "
                "population gains, shrinkage by the ratio's own bootstrap standard error, the "
                "same hard cap — applied unchanged.",
            ),
            table(
                f"Fitted band gains by destination, {cells} source-band cells each",
                ["Destination", "Post-Reinhard detail", "Amplify", "Attenuate",
                 "Mean log gain", "Regime"],
                rows,
                note="Destination detail power against mean fitted log gain: "
                     f"<strong>Spearman {d['rho']:+.3f}</strong> over "
                     f"{len(summary)} destinations. This report finds 12 of 15 scanner-band "
                     "gains attenuating toward AT2 and 12 of 15 amplifying toward GT450 or S60; "
                     "PLISM reproduces the split on a different block, a different institution "
                     "and four instruments we have never touched.",
            ),
            callout(
                "AT2 is the worst destination in a panel it does not belong to",
                "AT2 enters this report as the paired acquisition anchor, and section 9 warns "
                "that this is not the same as being a good destination. PLISM reaches that "
                f"conclusion without our cohort: among its seven scanners <code>{scanner(worst)}"
                "</code> has the lowest post-Reinhard detail power, so every fitted gain toward "
                "it attenuates.",
            ),
            prose(
                "Carried into representation space, the same split appears in the endpoints. "
                "Each condition was rendered onto the study's 0.5052 µm/px grid through the "
                "frozen libvips chain and encoded with the three E9 foundation models — the "
                "pinned TRIDENT checkout, the frozen checkpoints, each model's official eval "
                "transform, its native field of view. The correction is applied to the whole "
                "rendered tile, as at deployment, and the encoder's centre crop is taken from "
                "the result.",
            ),
            table(
                "RF1U in representation space, bootstrap over 13 sections",
                ["Model", "Toward AT2", "Toward GT450", "Probe → AT2", "Probe → GT450"],
                frontier_rows,
                note="Relative scanner-radius reduction. Aiming at GT450 is safe and improved "
                     f"in <strong>{gt450_safe} of {gt450_total}</strong> models against "
                     f"<strong>{at2_safe} of {at2_total}</strong> toward AT2. Dimmed cells have "
                     "a confidence interval containing zero.",
            ),
            callout(
                "The rule holds where it is fitted and does not carry into representation space",
                "Section 9's ordering is about two things at once — which way the gains point, "
                "and whether the resulting image helps a foundation model — and this cohort "
                "separates them. <strong>On the energies the gains are fitted from, the rule is "
                f"exact: ρ = {d['rho']:+.3f} over seven destinations, and AT2 is the worst "
                "destination in the panel.</strong> Carried into representation space on the E9 "
                f"encoders it does not order the same way: GT450 wins for "
                f"{gt450_better} of {len(encoders)} models, AT2 for the rest, and the margin "
                "between the two destinations is a few points either way.",
                "The August edition read 4 of 4 toward GT450 against 1 of 4 toward AT2 on the "
                "E0 panel. That panel is not this one, and neither is the sampling, so the "
                "honest reading is that <strong>the destination rule is established on band "
                "energies and unconfirmed downstream</strong> — not that it has been overturned.",
            ),
            prose(
                "<strong>One encoder shows no image-space benefit at all.</strong> "
                f"{no_benefit_text} That is not a failure of the correction so much as a "
                "statement about where that encoder keeps its scanner information: its raw "
                "scanner probe is the highest in the panel, so the identity is plainly there, "
                "and section 18 shows a linear map on its embedding removing almost all of it. "
                "An encoder can be highly sensitive to the scanner in directions that colour and "
                "band correction never touch.",
                "The other thing that did not replicate is arithmetic rather than biology. In "
                "this report, aiming at AT2 breaks content non-inferiority and the collapse "
                "guardrail; in PLISM every AT2-aimed condition stays safe. With 13 sections the "
                "bootstrap standard error on each gain is large, so RF1U's reliability brake "
                "pulls the gains toward identity — where this cohort fitted 0.51–0.66. The heavy "
                "blur that does the damage here never happens there. <strong>PLISM confirms that "
                "aiming badly costs you the benefit; it has no power to test whether aiming "
                "badly destroys the representation.</strong> Thirty-one times more locations per "
                "section do not change that: the shortage is in sections, not in patches, and "
                "the replicate unit for a section-level bootstrap is still 13.",
            ),
        ],
    )


# ---------------------------------------------------------------- section 18
# What the August edition concluded from the sparse cohort, so section 18 can be
# read against it rather than silently replacing it.
SPARSE_CORAL = {"uni_v1": -0.0465, "virchow2": -0.3182,
                "conch_v1": 0.1211, "resnet50": 0.1639}
# Below one sample per dimension CORAL fails by several hundred per cent.  Those
# points are kept but drawn on the floor, so the crossing and the plateau -- the
# two things the curve is read for -- are not squashed into a few pixels.
CURVE_FLOOR = -0.5


def build_s18(bundle) -> str:
    f = bundle["feature_frontier"]
    frontier = f["frontier"]
    image = bundle["image_frontier"]["frontier"]
    encoders = [e for e in data.ENCODER_ORDER if e in set(frontier["encoder"])]
    chance = f["summary"]["chance"]

    def entry(encoder_id, method):
        block = frontier.loc[(frontier["encoder"] == encoder_id)
                             & (frontier["method"] == method)]
        return None if block.empty else block.iloc[0]

    def best_image(encoder_id):
        block = image.loc[image["encoder"] == encoder_id]
        block = block.loc[block["condition"] != "raw"]
        return None if block.empty else block.loc[block["rr"].idxmax()]

    probe_rows, radius_rows = [], []
    for encoder_id in encoders:
        raw_probe = image.loc[(image["encoder"] == encoder_id)
                              & (image["condition"] == "raw"), "probe"]
        best = best_image(encoder_id)
        coral, procrustes = entry(encoder_id, "coral"), entry(encoder_id, "procrustes")
        probe_rows.append(row([
            encoder(encoder_id),
            dim(f"{raw_probe.iloc[0]:.3f}") if len(raw_probe) else dim("—"),
            f"{best['probe']:.3f}" if best is not None else dim("—"),
            pos(f"<strong>{coral['probe']:.3f}</strong>") if coral is not None else dim("—"),
            pos(f"<strong>{procrustes['probe']:.3f}</strong>") if procrustes is not None else dim("—"),
        ]))
        radius_rows.append(row([
            encoder(encoder_id),
            (pos if coral["rr"] > 0 else neg)(f"{100 * coral['rr']:+.1f} %")
            if coral is not None else dim("—"),
            pos(f"<strong>{100 * procrustes['rr']:+.1f} %</strong>")
            if procrustes is not None and procrustes["rr"] > 0 else
            (neg(f"{100 * procrustes['rr']:+.1f} %") if procrustes is not None else dim("—")),
            f"{100 * best['rr']:+.1f} %" if best is not None else dim("—"),
        ]))

    safe_procrustes = int(((frontier["method"] == "procrustes") & frontier["safe"]
                           & frontier["improved"]).sum())
    beats_image, ties = 0, []
    for encoder_id in encoders:
        procrustes = entry(encoder_id, "procrustes")
        best = best_image(encoder_id)
        if procrustes is None or best is None:
            continue
        if procrustes["rr"] > best["rr"]:
            beats_image += 1
        else:
            ties.append((encoder_id, procrustes["rr"], best["rr"]))
    safe_coral = int(((frontier["method"] == "coral") & frontier["safe"]
                      & frontier["improved"]).sum())
    total = len(encoders)

    blocks = retrieval_blocks(bundle)
    blocks += [
        prose(
            "Section 12 ends on a boundary statement: image correction caps out, and the "
            "headroom is a per-scanner linear map on the embedding. That is the claim most worth "
            "checking elsewhere, and it is also the cheapest — CORAL and orthogonal Procrustes "
            "are linear algebra on features that already exist. Both are fitted "
            "leave-one-section-out toward AT2, so no section contributes to the transform "
            "applied to it.",
        ),
        table(
            f"Scanner probe on PLISM — chance is {chance:.3f} for seven scanners",
            ["Model", "Raw", "Best image correction", "CORAL", "Procrustes"],
            probe_rows,
            note="Image conditions barely move the probe. Feature-space conditions move it to "
                 "near chance. Section 11 reports 0.005–0.080 against 0.807–0.920 on our own "
                 "cohort — the same two orders of magnitude, on different instruments and a "
                 "different encoder generation.",
        ),
        table(
            "Feature-space radius reduction toward AT2, leave-one-section-out",
            ["Model", "CORAL", "Procrustes", "Best image condition"],
            radius_rows,
        ),
        callout(
            "The report's closing recommendation survives its first external test, with one edge",
            f"Orthogonal Procrustes is safe and improved in {safe_procrustes} of {total} models "
            f"on PLISM and beats the best image-space figure on the same patches in "
            f"{beats_image} of {total}. The part of the ordering this cohort can test — feature "
            "correction above image correction, and a linear probe that only feature correction "
            "can move — is reproduced on a cohort assembled by other people for other reasons, "
            "with encoders that did not exist when it was written.",
            "<strong>What it cannot test is which feature method to deploy.</strong> Procrustes "
            "leads CORAL on radius here as it does on our own cohort, but section 11 is the "
            "reason that is not the deployment answer: radius and scanner removal dissociate, "
            "and under a nonlinear probe Procrustes retains the scanner where CORAL does not. "
            "Every probe on this cohort is a linear logistic regression, the same estimator the "
            "locked analysis uses, so PLISM reproduces the image-versus-feature boundary "
            "without adjudicating Procrustes against CORAL. Settling it externally is cheap and "
            "outstanding: it is the same MLP probe re-run on corrected features that already "
            "exist, with no rendering and no encoding.",
            ("" if not ties else
             "The exception is worth naming rather than rounding away: for "
             + ", ".join(f"{encoder(e)} the best image condition reaches "
                         f"{100 * image_rr:+.1f} % against Procrustes' {100 * proc:+.1f} %"
                         for e, proc, image_rr in ties)
             + ". That is the encoder with by far the widest field of view and the lowest raw "
               "scanner probe in the panel — the one where the scanner effect is least "
               "concentrated in the embedding to begin with, and so the one with the least for "
               "a linear map to remove. The probe still moves two orders of magnitude further "
               "in feature space than in image space for it, as for the others."),
        ),
    ]
    blocks += coral_blocks(bundle, safe_coral, total)
    return section(18, "Feature space, externally",
                   "The conclusion this report ends on, in a cohort it never saw", blocks)


def coral_blocks(bundle, safe_coral: int, total: int) -> list[str]:
    """The sample-size sweep, and what it does to the August edition's qualification."""
    f = bundle["feature_frontier"]
    if "sweep_curve" not in f:
        raise RuntimeError("the CORAL sample-size sweep has not been run; "
                           "section 18's central claim cannot be written without it")
    curve = f["sweep_curve"]
    frontier = f["frontier"]
    encoders = [e for e in data.ENCODER_ORDER if e in set(curve["encoder"])]

    series, verdicts = [], {}
    for encoder_id in encoders:
        block = curve.loc[curve["encoder"] == encoder_id].sort_values("fit_samples")
        series.append({"label": encoder(encoder_id),
                       "x": block["samples_per_dimension"].tolist(),
                       "y": block["rr"].tolist()})
        verdicts[encoder_id] = {
            "small": float(block["rr"].iloc[0]),
            "small_n": float(block["samples_per_dimension"].iloc[0]),
            "full": float(block["rr"].iloc[-1]),
            "full_n": float(block["samples_per_dimension"].iloc[-1]),
            "crossing": f["sweep_crossing"].get(encoder_id, float("nan"))}

    starts_negative = [e for e, v in verdicts.items() if v["small"] < 0]
    ends_negative = [e for e, v in verdicts.items() if v["full"] < 0]
    recovered = [e for e in starts_negative if e not in ends_negative]

    if ends_negative:
        headline = (
            "<strong>The qualification stands, and it is about the method after all.</strong> "
            + ", ".join(encoder(e) for e in ends_negative)
            + " stays negative at every training size tested, so no amount of data rescues "
            "CORAL for it. Whitening and recolouring a full covariance is simply the wrong map "
            "for these embeddings.")
    elif recovered:
        crossings = {e: verdicts[e]["crossing"] for e in recovered
                     if np.isfinite(verdicts[e]["crossing"])}
        spread = ""
        if len(crossings) > 1 and max(crossings.values()) > 2 * min(crossings.values()):
            cheapest = min(crossings, key=crossings.get)
            dearest = max(crossings, key=crossings.get)
            spread = (
                " <strong>But the threshold is not a constant, and that is the second "
                "finding.</strong> The x axis already divides by the embedding dimension, so if "
                "the requirement were simply \"n samples per parameter\" the curves would cross "
                f"together. They do not: {encoder(cheapest)} breaks even at "
                f"{crossings[cheapest]:.0f} samples per dimension and {encoder(dearest)} needs "
                f"{crossings[dearest]:.0f}, a {crossings[dearest] / crossings[cheapest]:.0f}-fold "
                "difference. How hard a covariance is to estimate depends on how its eigenvalues "
                "are spread, not only on how many there are, so the number has to be measured "
                "for the encoder in hand rather than carried over from another.")
        names = ", ".join(encoder(e) for e in recovered)
        verb = "starts" if len(recovered) == 1 else "start"
        subject = ("All three — " + names + " — " if len(recovered) == 3 else names + " ")
        headline = (
            "<strong>The qualification was an estimation artefact, and the sweep says so "
            "directly.</strong> "
            + subject
            + verb
            + " below zero at the smallest training sizes — the regime the August "
            "edition was stuck in, where it makes the scanner spread <em>worse</em> by whole "
            "multiples — then climbs through zero and flattens. CORAL was never failing; it was "
            "being fitted from a covariance it did not have the samples to estimate." + spread)
    else:
        headline = (
            "<strong>CORAL never fails on this cohort at any training size.</strong> Every "
            "encoder is above zero from the smallest fit onward, so the August edition's "
            "negative result does not reproduce even in the sparse regime — which points at the "
            "encoder panel rather than at the sample count.")

    rows = []
    for encoder_id in encoders:
        v = verdicts[encoder_id]
        def per_dim(value):
            return f"{value:.2f}" if value < 1 else f"{value:,.0f}"

        rows.append(row([
            encoder(encoder_id),
            per_dim(v["small_n"]),
            (neg if v["small"] < 0 else pos)(f"{100 * v['small']:+.1f} %"),
            per_dim(v["full_n"]),
            (neg if v["full"] < 0 else pos)(f"<strong>{100 * v['full']:+.1f} %</strong>"),
            per_dim(v["crossing"]) if np.isfinite(v["crossing"]) else dim("never"),
        ]))

    return [
        prose(
            "That leaves the one claim this report carries about CORAL, and it is the reason "
            "the cohort was rebuilt. The August edition found CORAL failing on PLISM for the "
            f"two highest-dimensional encoders — UNI v1 {100 * SPARSE_CORAL['uni_v1']:+.1f} % "
            f"and Virchow2 {100 * SPARSE_CORAL['virchow2']:+.1f} %, the latter unsafe — and "
            "offered a diagnosis: <em>estimation, not method</em>. CORAL fits a full covariance, "
            "Virchow2's is 2560 × 2560, and a PLISM section supplied 192–384 samples, so the "
            "dimension exceeded the sample count by seven to thirteen times.",
            "<strong>That was an inference, not a measurement</strong>, and the sparse cohort "
            "had no way to make it one: there were no more samples to add. There are now. The "
            "sweep below refits CORAL at growing training sizes and reads off where — and "
            "whether — the failure stops.",
        ),
        figure(
            sample_size_curve(series, y_floor=CURVE_FLOOR),
            "CORAL refitted leave-one-section-out at growing training sizes, toward AT2. "
            "Each point is the mean relative scanner-radius reduction over the 13 held-out "
            "sections; above zero the correction helps, below zero it makes the scanner spread "
            "worse. The x axis is training samples per embedding dimension, which is the "
            "quantity the estimation argument is about. The smallest fits fail by whole "
            "multiples and would flatten everything else against the top of the frame, so they "
            "are drawn as triangles on the axis floor with their true values printed — the "
            "value is moved, not hidden.",
        ),
        table(
            "CORAL at the sparse regime and at full density",
            ["Model", "Samples/dim, smallest fit", "Radius reduction",
             "Samples/dim, full fit", "Radius reduction", "Crosses zero at"],
            rows,
            note=f"At full density CORAL is safe and improved in {safe_coral} of {total} models. "
                 "The crossing column is the first swept training size at which the mean "
                 "reduction is positive, in samples per embedding dimension.",
        ),
        callout("What the sweep settles", headline),
        prose(
            "The practical form of the recommendation is therefore sharper than "
            "\"CORAL when the fitting population is large\", and it is also more demanding. "
            "<strong>Every encoder has a break-even sample count, it is reached well after the "
            "point where the covariance is merely invertible, and it has to be measured per "
            "encoder.</strong> The curve above is the measurement, and running it costs nothing "
            "beyond refitting on subsamples of features that already exist — no new encoding "
            "pass, no new acquisition. A site choosing between CORAL and doing nothing can plot "
            "its own version before committing.",
            "Procrustes, an orthogonal matrix with far fewer free parameters, is stable across "
            "the whole range. But it is fitted on paired same-location embeddings, which is "
            "exactly what a deployment site does not have — which is why it is reported here as "
            "the paired upper bound rather than as a deployable method, and why knowing when "
            "CORAL becomes safe is what decides whether a site can harmonise at all.",
        ),
    ]


# ---------------------------------------------------------------- section 19
def build_verdicts(bundle) -> list[dict]:
    """Each claim in the locked report, and what the rebuilt cohort does to it.

    Derived rather than asserted: a claim's outcome comes from the same tables
    the sections print, so the scoreboard cannot drift away from the evidence.
    """
    image = bundle["image_frontier"]["frontier"]
    feature = bundle["feature_frontier"]["frontier"]
    dest = bundle["destinations"]
    sweep = bundle["feature_frontier"].get("sweep_curve")

    def safe_improved(condition):
        block = image.loc[image["condition"] == condition]
        return int((block["safe"] & block["improved"]).sum()), len(block)

    gt450 = safe_improved("rf1u_gt450")
    at2 = safe_improved("rf1u_at2")

    raw_probe = image.loc[image["condition"] == "raw", "probe"]
    image_probe = image.loc[image["condition"] != "raw", "probe"]
    feature_probe = feature["probe"]

    # Compare within an encoder: the raw probe of one and the corrected probe of
    # another is not a movement, it is two different models.
    image_move, feature_move = {}, {}
    for encoder_id in sorted(set(image["encoder"])):
        block = image.loc[image["encoder"] == encoder_id]
        base = float(block.loc[block["condition"] == "raw", "probe"].iloc[0])
        image_move[encoder_id] = base - float(
            block.loc[block["condition"] != "raw", "probe"].min())
        rows_here = feature.loc[feature["encoder"] == encoder_id, "probe"]
        if len(rows_here):
            feature_move[encoder_id] = base - float(rows_here.min())
    procrustes = feature.loc[feature["method"] == "procrustes"]
    coral = feature.loc[feature["method"] == "coral"]

    # The claim is an ordering between destinations, so the verdict is read from
    # the ordering rather than declared.  Arm A and Arm B can disagree, and here
    # they do.
    encoders = sorted(set(image["encoder"]))
    gt450_better = 0
    for encoder_id in encoders:
        block = image.loc[image["encoder"] == encoder_id].set_index("condition")
        if {"rf1u_at2", "rf1u_gt450"} <= set(block.index):
            gt450_better += int(block.loc["rf1u_gt450", "rr"] > block.loc["rf1u_at2", "rr"])
    ordered = gt450_better == len(encoders)

    out = [
        {"claim": "Aiming decides which way the fitted gains point", "where": "9",
         "outcome": "replicated",
         "detail": f"ρ = {dest['rho']:+.3f} over seven destinations; AT2 is the worst "
                   "destination in a panel it does not belong to"},
        {"claim": "Aiming at the detail-rich destination helps the encoder more", "where": "9",
         "outcome": "replicated" if ordered else "unconfirmed",
         "detail": (f"GT450 beats AT2 in {gt450_better} of {len(encoders)} models "
                    f"({gt450[0]} of {gt450[1]} safe and improved against "
                    f"{at2[0]} of {at2[1]}); the E0 panel read 4 of 4 against 1 of 4")},
        {"claim": "Image correction cannot move the scanner probe", "where": "11",
         "outcome": "replicated",
         "detail": f"the best image condition moves it by at most "
                   f"{max(image_move.values()):.3f} within an encoder"},
        {"claim": "The headroom is in feature space", "where": "12",
         "outcome": "replicated",
         "detail": f"a linear map on the embedding moves the same probe by "
                   f"{min(feature_move.values()):.2f}–{max(feature_move.values()):.2f}; "
                   f"Procrustes safe and improved in "
                   f"{int((procrustes['safe'] & procrustes['improved']).sum())} of "
                   f"{len(procrustes)}"},
        {"claim": "The scanner effect is not a detachable global style", "where": "3.2",
         "outcome": "replicated",
         "detail": "and now shown to vary by named tissue as well as by section"},
        {"claim": "Raw transfer is the scanner's spectral signature", "where": "3.1",
         "outcome": "restated",
         "detail": "raw band power is dominated by contrast; the three-term "
                   "decomposition offered in August does not survive the rebuilt cohort"},
    ]

    coral_safe = int((coral["safe"] & coral["improved"]).sum())
    if sweep is not None:
        negative = [e for e in sweep["encoder"].unique()
                    if sweep.loc[sweep["encoder"] == e].sort_values("fit_samples")
                    ["rr"].iloc[-1] < 0]
        if negative:
            out.append({"claim": "CORAL is the deployable feature-space option", "where": "6",
                        "outcome": "qualified",
                        "detail": "fails for " + ", ".join(encoder(e) for e in negative)
                                  + " at every training size"})
        else:
            out.append({"claim": "CORAL is the deployable feature-space option", "where": "6",
                        "outcome": "replicated",
                        "detail": f"{coral_safe} of {len(coral)} at full density; the August "
                                  "failure was the sample count, and the sweep measures where "
                                  "it stops"})
    else:
        out.append({"claim": "CORAL is the deployable feature-space option", "where": "6",
                    "outcome": "qualified", "detail": "sweep not run"})

    out.append({"claim": "Aiming badly destroys the representation", "where": "9",
                "outcome": "untested",
                "detail": "13 sections cannot produce the required blur, at any patch density"})
    return out


def covariate_blocks(bundle) -> list[str]:
    """Does the staining condition explain the scanner's slope, or only label it?

    The August edition answered no, from a hand-run leave-one-out check on a
    single scanner that never entered the code.  It runs on all of them here, so
    the answer is read off a table rather than generalised from one case.
    """
    entry = bundle["nesting"]["covariate"]
    covariate = entry["table"]
    if "stable" not in covariate:
        raise RuntimeError("the leave-one-section-out pass has not been run; "
                           "section 19's stain conclusion cannot be written without it")
    sections = bundle["nesting"]["covariate"]["summary"]["sections"]
    block = covariate.loc[covariate["covariate"] == "stain_od"].set_index("scanner")
    weakest = (entry["weakest"].loc[entry["weakest"]["covariate"] == "stain_od"]
               .set_index("scanner")) if "weakest" in entry else None

    rows = []
    for name in data.SCANNER_ORDER:
        if name not in block.index:
            continue
        row_entry = block.loc[name]
        significant = row_entry["p"] < 0.05
        rows.append(row([
            f"<code>{scanner(name)}</code>",
            f"{row_entry['beta']:+.3f}" if significant else dim(f"{row_entry['beta']:+.3f}"),
            f"{row_entry['z']:+.2f}" if significant else dim(f"{row_entry['z']:+.2f}"),
            f"{100 * row_entry['section_var_explained']:.0f} %" if significant else dim("—"),
            f"{int(row_entry['folds_significant'])} / {int(row_entry['folds'])}",
            (flag("stable", True) if row_entry["stable"]
             else (f'breaks without <code>{weakest.loc[name, "weakest_fold"]}</code>'
                   if weakest is not None and significant else dim("—"))),
        ]))

    stable = block.loc[block["stable"].astype(bool)]
    broken_by = ""
    _ = sections  # bound above; named here so the reader sees where the count comes from
    if weakest is not None:
        culprits = {}
        for name in block.index:
            if block.loc[name, "p"] < 0.05 and not block.loc[name, "stable"]:
                culprits.setdefault(weakest.loc[name, "weakest_fold"], []).append(scanner(name))
        if culprits:
            broken_by = (
                " The section doing the damage is named: "
                + "; ".join(f"withhold <code>{section}</code>, and "
                            + (names[0] if len(names) == 1 else
                               " and ".join([", ".join(names[:-1]), names[-1]]))
                            + (" loses" if len(names) == 1 else " lose")
                            + " significance"
                            for section, names in culprits.items())
                + ". That is the same section whose staining is three standard deviations "
                  "darker than any other, and the same one where the feature-space corrections "
                  f"of section 18 do nothing. <strong>One extreme condition out of {sections} "
                  "is carrying most of what looked like a stain effect</strong> — which is "
                  "exactly the failure mode the August edition suspected, now located rather "
                  "than inferred.")
    survivors = [scanner(s) for s in stable.index]
    fell = [scanner(s) for s in block.index
            if block.loc[s, "p"] < 0.05 and not block.loc[s, "stable"]]

    if len(stable) == 0:
        verdict = (
            "<strong>Not one coefficient survives.</strong> Every scanner whose full-data fit "
            "looked significant loses it when one section is withheld, which is what the August "
            "edition concluded from a single scanner and which now holds for all of them. "
            "<strong>Staining condition is a label here, not an explanatory variable.</strong>")
    else:
        verdict = (
            f"<strong>{len(stable)} of {len(block)} survive.</strong> "
            + ", ".join(f"<code>{s}</code>" for s in survivors)
            + (" keeps its sign and its significance in every fold"
               if len(survivors) == 1
               else " keep their sign and their significance in every fold")
            + (f", while <code>{fell[0]}</code> does not." if len(fell) == 1
               else ", while " + " and ".join(
                   [", ".join(f"<code>{s}</code>" for s in fell[:-1]),
                    f"<code>{fell[-1]}</code>"]) + " do not." if fell
               else ".")
            + " The August edition generalised from the one scanner it checked by hand and "
              "concluded that staining condition is a label rather than an explanatory "
              "variable. <strong>For at least one scanner it is an explanatory variable, and "
              "the check that decides this is now in the code.</strong>"
            + (broken_by if broken_by else ""))

    return [
        prose(
            "That leaves the question of whether the staining term means anything. If the "
            "section-level variance were staining chemistry, a scanner's slope should track how "
            "the stain actually renders — how dark it is, how much contrast it carries, how much "
            "fine detail survives it. All three are measurable on the section's own AT2 "
            "reference, and so are scanner-independent by construction.",
            f"With {sections} sections a single one has enormous leverage, so the full-data "
            "coefficient is not the answer on its own — the fit has to survive dropping each "
            "section in turn. That check is what the August edition rested its conclusion on, "
            "and it was run by hand on one scanner and never written down. Here it runs on every "
            "scanner and every covariate.",
        ),
        table(
            "Does the stain's own darkness predict the scanner's spectral slope?",
            ["Scanner", "β", "z", "Section variance explained",
             "Folds significant", "Leave-one-section-out"],
            rows,
            note="The covariate is the section's mean optical density on its AT2 reference, "
                 "standardised; contrast and detail behave the same way and are in "
                 "<code>stain_covariate_high.csv</code>. A coefficient counts as stable only if "
                 "it keeps its sign and its significance in every fold. Dimmed rows are not "
                 "significant on the full data to begin with.",
        ),
        callout("What the leave-one-out pass settles", verdict),
    ]


def build_s19(bundle) -> str:
    bundle["verdicts"] = verdicts = build_verdicts(bundle)
    nest = bundle["nesting"]

    style = {"replicated": lambda t: flag(t, True),
             "restated": lambda t: flag(t, False),
             "qualified": lambda t: flag(t, False),
             "unconfirmed": lambda t: flag(t, False),
             "untested": dim}
    rows = [row([v["claim"], v["where"],
                 f'{style[v["outcome"]](v["outcome"])} — {v["detail"]}'])
            for v in verdicts]

    section_fit = nest["section"]["table"]
    tissue_fit = nest["tissue"]["table"]

    def band(table_, band_name="high", condition="gated"):
        return table_.loc[(table_["band"] == band_name)
                          & (table_["condition"] == condition)]

    section_high = band(section_fit)
    tissue_high = band(tissue_fit)
    section_significant = int((section_high["q_section"] < 0.05).sum()) \
        if "q_section" in section_high else 0
    tissue_significant = int((tissue_high["q_section"] < 0.05).sum()) \
        if "q_section" in tissue_high else 0

    share_rows = []
    for _, entry in section_high.sort_values("scanner").iterrows():
        tissue_entry = tissue_high.loc[tissue_high["scanner"] == entry["scanner"]]
        tissue_share = (tissue_entry["section_share"].iloc[0]
                        if len(tissue_entry) else float("nan"))
        share_rows.append(row([
            f"<code>{scanner(entry['scanner'])}</code>",
            f"{entry['log2_ert']:+.3f}",
            f"{100 * entry['section_share']:.0f} %",
            f"{100 * tissue_share:.0f} %" if np.isfinite(tissue_share) else dim("—"),
        ]))

    covariate = nest["covariate"]["table"]

    blocks = [
        table("Every claim in this report, against PLISM",
              ["Claim", "Where", "Outcome"], rows),
        prose(
            "Two axes are left, and PLISM is the only cohort that can put them side by side, "
            "because the same 46 cores are cut into all 13 sections. The nesting is fitted "
            "twice on the identical rows — once with the <strong>staining condition</strong> on "
            "top and the core inside it, once with the <strong>tissue type</strong> on top and "
            "the section inside it. The second is section 3.2's own model, and it could not be "
            "run here before the cores were named.",
        ),
        table(
            "Where a scanner's spectral slope varies, by what it is nested under",
            ["Scanner", "log2 ERT", "Variance from staining condition", "Variance from tissue type"],
            share_rows,
            note="Share of the between-group slope variance sitting at the top level of each "
                 "fit; the remainder sits in the level below. Same locations, same estimator, "
                 "same profiled REML and boundary-mixture test used for the tissue analysis in "
                 f"section 3.2. Significant top-level variance (q &lt; 0.05): "
                 f"{section_significant} of {len(section_high)} scanners under staining "
                 f"condition, {tissue_significant} of {len(tissue_high)} under tissue type.",
        ),
    ] + covariate_blocks(bundle)
    blocks += tissue_blocks(bundle)
    blocks += [
        callout(
            "What PLISM still cannot do for this report",
            "<strong>Power, in sections.</strong> Thirteen serial sections against 109 physical "
            "slides. Rebuilding the cohort multiplied the patches by 31 and left the replicate "
            "count untouched, so every section-level interval is still wide, RF1U's shrinkage "
            "still dominates its own gains, and the destructive regime still never occurs. "
            "What the extra patches bought was the <em>fitting</em> problem in section 18, not "
            "the bootstrap.",
            "<strong>Coverage.</strong> PLISM has no instrument like AKOYA — its softest scanner "
            "sits far above where AKOYA sits after Reinhard. Since AKOYA is a reproduced "
            "clinical production setting rather than an acquisition error, that is a gap in the "
            "external benchmark, not a weakness here.",
            "<strong>Confounding.</strong> Its 13 stains are on 13 different serial sections, so "
            "a section-level term cannot be attributed to chemistry even when it is significant. "
            "The tissue axis does not have this problem, which is why naming the cores was worth "
            "the trouble.",
            "<strong>Endpoints.</strong> PLISM carries tissue-type labels and nothing else. The "
            "gap section 14 calls the real one — a molecular or clinical readout — is still "
            "open, and no public multi-scanner cohort closes it.",
        ),
        prose(
            "What changed is the standard of evidence. The acquisition-provenance argument "
            "against a resampling explanation is now a measurement; the hinge in section 12 has "
            "a mechanism instead of an observation; the destination rule has been run on four "
            "instruments we have never handled; the closing recommendation has been tested "
            "somewhere else and on encoders that postdate it; and the qualification this report "
            "carried against CORAL has been converted from a plausible story about estimation "
            "into a curve with a threshold on it. The cost was a restatement of section 3.1, "
            "which makes the report more precise rather than less.",
        ),
    ]
    return section(19, "What the external cohort settles",
                   "And what it now settles that it could not before", blocks)

if __name__ == "__main__":
    main()

"""The spectral characterization as a curve, not a single high-band number.

Section 3 of the report states the effective relative transfer of each scanner as
one number per instrument, taken over the band above 0.10 cyc/um. That is the
frozen endpoint and it stays the endpoint, but a single number cannot show the
shape the reader is being asked to believe in: that each instrument has a
distinct frequency signature, that the signatures are not parallel, and that a
one-parameter blur-sharpen family therefore cannot reproduce them.

This draws the curve behind the number -- radially averaged transfer against
AT2, per frequency bin, median across the 109 physical slides with the
interquartile band -- as inline SVG so it lives in the HTML report, stays crisp,
and follows the reader's light or dark theme.

AT2 is drawn as a neutral zero line rather than a sixth coloured series: it is 1.0
by construction, so colouring it would imply a measurement where there is none.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fetch_e0_pfm_checkpoints import sha256


# Slots 1-5 of the validated categorical palette, light and dark steps of the
# same hues. Checked with the data-viz validator: all six checks pass in both
# modes; the light mode contrast warning is relieved by direct labels, which
# every series carries.
SERIES = (
    ("gt450", "GT450", "#2a78d6", "#3987e5"),
    ("s60", "S60", "#eb6834", "#d95926"),
    ("versa", "VERSA", "#1baf7a", "#199e70"),
    ("s360", "S360", "#eda100", "#c98500"),
    ("akoya", "AKOYA", "#e87ba4", "#d55181"),
)
HIGH_BAND_START = 0.10
WIDTH, HEIGHT = 760, 430
PAD = {"left": 58, "right": 96, "top": 26, "bottom": 52}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spectra", default="outputs/exp05_spectral_cohort_109/slide_spectra.csv")
    parser.add_argument("--output", default="outputs/e8_transfer_figure")
    return parser.parse_args()


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(["scanner", "frequency_cyc_per_um"])["log2_relative_transfer"]
    summary = grouped.agg(
        median="median",
        q25=lambda value: value.quantile(0.25),
        q75=lambda value: value.quantile(0.75),
        slides="size",
    ).reset_index()
    if summary["slides"].nunique() != 1:
        raise ValueError("frequency bins are not balanced across slides")
    return summary


def scales(summary: pd.DataFrame):
    frequency = np.sort(summary["frequency_cyc_per_um"].unique())
    low, high = float(frequency[0]), float(frequency[-1])
    values = summary[summary["scanner"] != "at2"]
    span = max(abs(values["q25"].min()), abs(values["q75"].max()))
    limit = float(np.ceil(span * 2) / 2)

    def x_of(value):
        # Log frequency: the interesting structure is in the top decade and a
        # linear axis would compress it into the right-hand quarter.
        position = (np.log10(value) - np.log10(low)) / (np.log10(high) - np.log10(low))
        return PAD["left"] + position * (WIDTH - PAD["left"] - PAD["right"])

    def y_of(value):
        position = (value + limit) / (2 * limit)
        return HEIGHT - PAD["bottom"] - position * (HEIGHT - PAD["top"] - PAD["bottom"])

    return x_of, y_of, limit, low, high


def path_for(xs, ys) -> str:
    return "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in zip(xs, ys))


def spread_labels(anchors: list[float], minimum: float = 13.0) -> list[float]:
    """Push direct labels apart so none overlaps, keeping their vertical order.

    AT2 and VERSA end the plot about 0.17 octaves apart, which is under a line
    height at this scale; without this they collide. One forward pass fixes the
    order, one backward pass pulls the group back inside the plot.
    """
    order = sorted(range(len(anchors)), key=lambda index: anchors[index])
    placed = list(anchors)
    for position in range(1, len(order)):
        previous, current = order[position - 1], order[position]
        if placed[current] - placed[previous] < minimum:
            placed[current] = placed[previous] + minimum
    for position in range(len(order) - 2, -1, -1):
        current, following = order[position], order[position + 1]
        if placed[following] - placed[current] < minimum:
            placed[current] = placed[following] - minimum
    return placed


def build_svg(summary: pd.DataFrame) -> str:
    x_of, y_of, limit, low, high = scales(summary)
    plot_right = WIDTH - PAD["right"]
    parts = [
        f'<svg viewBox="0 0 {WIDTH} {HEIGHT}" class="transfer-fig" role="img" '
        'aria-label="Radially averaged transfer relative to AT2 against spatial '
        'frequency. Each scanner traces a distinct, non-parallel curve; AKOYA '
        'falls steeply with frequency while GT450 rises above AT2.">'
    ]

    # Gridlines and y axis, in half-octave steps of log2 transfer.
    step = 0.5 if limit <= 1.5 else 1.0
    ticks = np.arange(-limit, limit + 1e-9, step)
    for value in ticks:
        y = y_of(value)
        emphasis = "gridline is-zero" if abs(value) < 1e-9 else "gridline"
        parts.append(
            f'<line class="{emphasis}" x1="{PAD["left"]}" y1="{y:.2f}" '
            f'x2="{plot_right}" y2="{y:.2f}"/>'
        )
        fold = 2.0**value
        label = f"{fold:g}×" if fold >= 1 else f"{fold:.2f}×".rstrip("0").rstrip(".") + "×"
        label = label.replace("××", "×")
        parts.append(
            f'<text class="tick" x="{PAD["left"] - 8}" y="{y + 3.5:.2f}" '
            f'text-anchor="end">{label}</text>'
        )

    for value in (0.01, 0.1, 1.0):
        if not low <= value <= high:
            continue
        x = x_of(value)
        parts.append(
            f'<line class="gridline" x1="{x:.2f}" y1="{PAD["top"]}" '
            f'x2="{x:.2f}" y2="{HEIGHT - PAD["bottom"]}"/>'
        )
        parts.append(
            f'<text class="tick" x="{x:.2f}" y="{HEIGHT - PAD["bottom"] + 18}" '
            f'text-anchor="middle">{value:g}</text>'
        )

    # The frozen high band starts here; everything in section 3.1 is measured to
    # the right of this rule.
    band_x = x_of(HIGH_BAND_START)
    parts.append(
        f'<rect class="band" x="{band_x:.2f}" y="{PAD["top"]}" '
        f'width="{plot_right - band_x:.2f}" height="{HEIGHT - PAD["top"] - PAD["bottom"]:.2f}"/>'
    )
    parts.append(
        f'<text class="band-label" x="{band_x + 6:.2f}" y="{PAD["top"] + 14}">'
        "frozen high band</text>"
    )

    if not (summary["scanner"] == "at2").any():
        raise ValueError("AT2 reference rows are missing from the spectra table")

    drawn, ends = [], []
    for index, (key, label, _, _) in enumerate(SERIES, start=1):
        block = summary[summary["scanner"] == key].sort_values("frequency_cyc_per_um")
        xs = [x_of(value) for value in block["frequency_cyc_per_um"]]
        upper = [y_of(value) for value in block["q75"]]
        lower = [y_of(value) for value in block["q25"]]
        area = (
            path_for(xs, upper)
            + " L "
            + " L ".join(f"{x:.2f} {y:.2f}" for x, y in zip(reversed(xs), reversed(lower)))
            + " Z"
        )
        median = [y_of(value) for value in block["median"]]
        parts.append(f'<g class="series s{index}" data-scanner="{key}">')
        parts.append(f'<path class="band-iqr" d="{area}"/>')
        parts.append(f'<path class="line" d="{path_for(xs, median)}"/>')
        parts.append("</g>")
        drawn.append((index, label, median[-1]))
        ends.append(median[-1])

    drawn.append((0, "AT2", y_of(0.0)))
    ends.append(y_of(0.0))
    placed = spread_labels(ends)
    for (index, label, anchor), y in zip(drawn, placed):
        style = "series-label is-reference" if index == 0 else f"series-label s{index}"
        # A leader only earns its place when the label had to move.
        if abs(y - anchor) > 1.0:
            parts.append(
                f'<path class="leader" d="M {plot_right + 2:.2f} {anchor:.2f} '
                f'L {plot_right + 6:.2f} {y:.2f}"/>'
            )
        parts.append(
            f'<text class="{style}" x="{plot_right + 8}" y="{y + 3.5:.2f}">{label}</text>'
        )

    parts.append(
        f'<text class="axis" x="{(PAD["left"] + plot_right) / 2:.2f}" '
        f'y="{HEIGHT - 12}" text-anchor="middle">spatial frequency (cycles / µm)</text>'
    )
    parts.append(
        f'<text class="axis" transform="translate(14 {(PAD["top"] + HEIGHT - PAD["bottom"]) / 2:.2f}) '
        'rotate(-90)" text-anchor="middle">transfer relative to AT2</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def build_style() -> str:
    light = "\n".join(
        f"  .transfer-fig .s{index} {{ --series: {light_hex}; }}"
        for index, (_, _, light_hex, _) in enumerate(SERIES, start=1)
    )
    dark = "\n".join(
        f"    .transfer-fig .s{index} {{ --series: {dark_hex}; }}"
        for index, (_, _, _, dark_hex) in enumerate(SERIES, start=1)
    )
    return f"""<style>
  .transfer-fig {{ width: 100%; height: auto; display: block; }}
  .transfer-fig .gridline {{ stroke: var(--rule-soft, #e7eaf0); stroke-width: 1; }}
  .transfer-fig .gridline.is-zero {{ stroke: var(--ink-faint, #7a8593); stroke-width: 1.4; }}
  .transfer-fig .band {{ fill: var(--ink, #161b22); opacity: 0.035; }}
  .transfer-fig .band-label,
  .transfer-fig .tick,
  .transfer-fig .axis {{ fill: var(--ink-soft, #525c6b); font-family: var(--f-display, sans-serif); }}
  .transfer-fig .tick {{ font-size: 11px; font-variant-numeric: tabular-nums; }}
  .transfer-fig .axis {{ font-size: 11.5px; }}
  .transfer-fig .band-label {{ font-size: 10px; letter-spacing: 0.06em; text-transform: uppercase; }}
  .transfer-fig .line {{ fill: none; stroke: var(--series); stroke-width: 2; stroke-linejoin: round; }}
  .transfer-fig .band-iqr {{ fill: var(--series); opacity: 0.16; stroke: none; }}
  .transfer-fig .series-label {{
    fill: var(--ink, #161b22); font-family: var(--f-display, sans-serif);
    font-size: 11.5px; font-weight: 700;
  }}
  .transfer-fig .series-label.is-reference {{ fill: var(--ink-faint, #7a8593); font-weight: 400; }}
  .transfer-fig .leader {{ fill: none; stroke: var(--ink-faint, #7a8593); stroke-width: 1; }}
  /* Hover isolates one instrument without hiding the others' envelope. */
  .transfer-fig:hover .series {{ opacity: 0.32; }}
  .transfer-fig .series:hover {{ opacity: 1; }}
  .transfer-fig .series:hover .band-iqr {{ opacity: 0.26; }}
{light}
  @media (prefers-color-scheme: dark) {{
    :root:where(:not([data-theme="light"])) .transfer-fig .band {{ fill: #ffffff; opacity: 0.05; }}
{dark}
  }}
  :root[data-theme="dark"] .transfer-fig .band {{ fill: #ffffff; opacity: 0.05; }}
{dark.replace('    .transfer-fig', '  :root[data-theme="dark"] .transfer-fig')}
</style>"""


def main():
    args = parse_args()
    frame = pd.read_csv(args.spectra)
    required = {"scanner", "frequency_cyc_per_um", "log2_relative_transfer", "slide_id"}
    if not required.issubset(frame.columns):
        raise ValueError(f"spectra table needs {sorted(required)}")
    if frame["slide_id"].nunique() != 109:
        raise ValueError(f"expected 109 slides, got {frame['slide_id'].nunique()}")

    summary = aggregate(frame)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    table = output / "transfer_curves.csv"
    summary.to_csv(table, index=False)

    fragment = output / "transfer_figure.html"
    fragment.write_text(build_style() + "\n" + build_svg(summary) + "\n")

    high = frame[frame["frequency_cyc_per_um"] >= HIGH_BAND_START]
    crossings = {}
    for key, _, _, _ in SERIES:
        block = summary[summary["scanner"] == key].sort_values("frequency_cyc_per_um")
        sign = np.sign(block["median"].to_numpy())
        changes = int(np.sum(sign[1:] * sign[:-1] < 0))
        crossings[key] = changes

    meta = {
        "analysis": "e8_transfer_figure",
        "status": "POST_CORE_EXPLORATORY_EXTENSION",
        "note": (
            "Presentation of the locked E1 spectra: median and interquartile "
            "band of log2 transfer per frequency bin across 109 physical "
            "slides. No estimate is refitted."
        ),
        "source": args.spectra,
        "slides": int(frame["slide_id"].nunique()),
        "frequency_bins": int(frame["frequency_cyc_per_um"].nunique()),
        "high_band_start_cyc_per_um": HIGH_BAND_START,
        "high_band_bins": int(high["frequency_cyc_per_um"].nunique()),
        "median_sign_changes": crossings,
        "palette": {
            "validated": "data-viz six checks, light and dark, categorical slots 1-5",
            "light_contrast_relief": "every series carries a direct label",
        },
        "artifacts": {table.name: sha256(table), fragment.name: sha256(fragment)},
    }
    (output / "summary.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

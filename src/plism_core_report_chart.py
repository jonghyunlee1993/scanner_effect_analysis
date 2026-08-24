"""Inline SVG for the one PLISM figure that a table cannot carry.

The CORAL sample-size sweep is a curve, not a comparison of a few cells: what has
to be read off it is *where* the method stops hurting and whether it then
flattens, and a table of nine rows per encoder hides exactly that shape.

Self-contained SVG with no external dependency, coloured from its own tokens so
it follows the report's light/dark toggle.  The three hues are a validated
categorical triple, not the report's `--amplify`/`--attenuate` pair -- those two
are a diverging pair for one axis of meaning, and reusing them for three
independent encoders puts two series 6.4 CVD units apart.
"""

from __future__ import annotations

import math

# Validated all-pairs in both modes (OKLab dE: worst normal 21.0 light / 19.8
# dark, worst CVD 12.5 / 11.3).  Assigned to encoders in a fixed order, so a
# series keeps its hue whatever the ranking does.
SERIES_LIGHT = ("#0d8b7a", "#b25e10", "#7d3ba3")
SERIES_DARK = ("#2c9f8b", "#c87f2e", "#9a68d6")

CHART_CSS = """
  .viz { --viz-1: %s; --viz-2: %s; --viz-3: %s; }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) .viz { --viz-1: %s; --viz-2: %s; --viz-3: %s; }
  }
  :root[data-theme="dark"] .viz { --viz-1: %s; --viz-2: %s; --viz-3: %s; }
  .viz { padding: 0.25rem 0 0; }
  .viz svg { display: block; width: 100%%; height: auto; overflow: visible; }
  .viz .grid { stroke: var(--rule-soft); stroke-width: 1; }
  .viz .zero { stroke: var(--ink-faint); stroke-width: 1; stroke-dasharray: 3 3; }
  .viz .axis { stroke: var(--rule); stroke-width: 1; }
  .viz .tick { fill: var(--ink-faint); font-family: var(--f-data); font-size: 11px; }
  .viz .axis-title { fill: var(--ink-soft); font-family: var(--f-display);
                     font-size: 12px; font-weight: 600; letter-spacing: 0.02em; }
  .viz .series { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
  .viz .dot { stroke: var(--surface); stroke-width: 2; }
  .viz .series-label { font-family: var(--f-display); font-size: 12px; font-weight: 600; }
  .viz .note { fill: var(--ink-faint); font-family: var(--f-body); font-size: 11.5px; }
  .viz-legend { display: flex; flex-wrap: wrap; gap: 1.1rem; margin-top: 0.6rem;
                font-family: var(--f-display); font-size: 0.82rem; color: var(--ink-soft); }
  .viz-legend span { display: inline-flex; align-items: center; gap: 0.4rem; }
  .viz-legend i { width: 14px; height: 3px; border-radius: 2px; display: inline-block; }
""" % (SERIES_LIGHT + SERIES_DARK + SERIES_DARK)


def _ticks(low: float, high: float, count: int = 5) -> list[float]:
    span = high - low
    if span <= 0:
        return [low]
    raw = span / count
    magnitude = 10 ** math.floor(math.log10(raw))
    step = min((m * magnitude for m in (1, 2, 2.5, 5, 10)),
               key=lambda s: abs(s - raw))
    start = math.ceil(low / step) * step
    out, value = [], start
    while value <= high + 1e-9:
        out.append(round(value, 10))
        value += step
    return out


def sample_size_curve(series: list[dict], *, width: int = 720, height: int = 380,
                      x_title: str = "training samples per embedding dimension",
                      y_title: str = "scanner-radius reduction",
                      y_floor: float | None = None) -> str:
    """Log-x line chart.  `series` is [{'label', 'x': [...], 'y': [...]}, ...].

    Marks are 2 px lines with >=8 px end markers; the zero line is drawn because
    the sign of y is the whole reading.  Every series is direct-labelled as well
    as legended, so identity never rests on colour alone.

    `y_floor` clips the vertical range.  The smallest training sizes here fail by
    several hundred per cent, and drawing to them squashes the part that has to
    be read -- where the curve crosses zero and where it flattens -- into a few
    pixels.  A clipped point is drawn as a downward triangle sitting on the floor
    with its true value printed beside it, so nothing is hidden, only moved.  This
    is legitimate on a line chart, where position carries the value; it would not
    be on a bar chart, where length does.
    """
    if not 1 <= len(series) <= 3:
        raise ValueError("this chart carries one to three series; more needs facets")
    pad = {"l": 62, "r": 118, "t": 18, "b": 54}
    plot_w = width - pad["l"] - pad["r"]
    plot_h = height - pad["t"] - pad["b"]

    xs = [v for s in series for v in s["x"] if v > 0]
    ys = [v for s in series for v in s["y"]]
    x_low, x_high = math.log10(min(xs)), math.log10(max(xs))
    x_low -= 0.05 * (x_high - x_low)
    x_high += 0.05 * (x_high - x_low)
    visible = [v for v in ys if y_floor is None or v >= y_floor]
    y_low = min(min(visible), 0.0) if visible else 0.0
    y_high = max(max(ys), 0.0)
    margin = 0.12 * (y_high - y_low or 1.0)
    y_low, y_high = y_low - margin, y_high + margin
    if y_floor is not None:
        y_low = max(y_low, y_floor)
    clipped = y_floor is not None and any(v < y_floor for v in ys)

    def px(value: float) -> float:
        return pad["l"] + plot_w * (math.log10(value) - x_low) / (x_high - x_low)

    def py(value: float) -> float:
        clamped = max(value, y_low) if y_floor is not None else value
        return pad["t"] + plot_h * (1 - (clamped - y_low) / (y_high - y_low))

    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" '
             f'aria-label="{y_title} against {x_title}, one line per encoder">']

    for value in _ticks(y_low, y_high):
        y = py(value)
        parts.append(f'<line class="grid" x1="{pad["l"]}" y1="{y:.1f}" '
                     f'x2="{pad["l"] + plot_w}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{pad["l"] - 10:.0f}" y="{y + 4:.1f}" '
                     f'text-anchor="end">{value * 100:+.0f}%</text>')
    parts.append(f'<line class="zero" x1="{pad["l"]}" y1="{py(0):.1f}" '
                 f'x2="{pad["l"] + plot_w}" y2="{py(0):.1f}"/>')

    decade = math.ceil(x_low)
    while decade <= x_high:
        for multiple in (1, 3):
            value = multiple * 10 ** decade
            if not 10 ** x_low <= value <= 10 ** x_high:
                continue
            x = px(value)
            label = (f"{value:g}" if value < 1000 else
                     f"{value / 1000:g}k" if value < 1e6 else f"{value / 1e6:g}M")
            parts.append(f'<text class="tick" x="{x:.1f}" y="{pad["t"] + plot_h + 20:.0f}" '
                         f'text-anchor="middle">{label}</text>')
        decade += 1
    parts.append(f'<line class="axis" x1="{pad["l"]}" y1="{pad["t"] + plot_h:.0f}" '
                 f'x2="{pad["l"] + plot_w}" y2="{pad["t"] + plot_h:.0f}"/>')
    parts.append(f'<text class="axis-title" x="{pad["l"] + plot_w / 2:.0f}" '
                 f'y="{height - 12}" text-anchor="middle">{x_title}</text>')
    parts.append(f'<text class="axis-title" transform="translate(16,'
                 f'{pad["t"] + plot_h / 2:.0f}) rotate(-90)" '
                 f'text-anchor="middle">{y_title}</text>')

    for index, entry in enumerate(series):
        colour = f"var(--viz-{index + 1})"
        points = " ".join(f"{px(x):.1f},{py(y):.1f}"
                          for x, y in zip(entry["x"], entry["y"]))
        parts.append(f'<polyline class="series" points="{points}" stroke="{colour}"/>')
        for x, y in zip(entry["x"], entry["y"]):
            below = y_floor is not None and y < y_low
            marker = (f'<path class="dot" d="M {px(x) - 5.5:.1f} {py(y) - 7:.1f} '
                      f'h 11 l -5.5 8 z" fill="{colour}">'
                      if below else
                      f'<circle class="dot" cx="{px(x):.1f}" cy="{py(y):.1f}" '
                      f'r="4.5" fill="{colour}">')
            close = "</path>" if below else "</circle>"
            parts.append(marker
                         + f'<title>{entry["label"]}: {y * 100:+.1f}% at '
                         f'{x:,.1f} samples per dimension'
                         + (" (below the axis floor)" if below else "")
                         + f'</title>{close}')
            if below:
                parts.append(f'<text class="note" x="{px(x):.1f}" '
                             f'y="{py(y) + 14:.1f}" text-anchor="middle" '
                             f'fill="{colour}">{y * 100:.0f}%</text>')
        last_x, last_y = entry["x"][-1], entry["y"][-1]
        parts.append(f'<text class="series-label" x="{px(last_x) + 12:.1f}" '
                     f'y="{py(last_y) + 4:.1f}" fill="{colour}">{entry["label"]}</text>')

    if clipped:
        parts.append(f'<text class="note" x="{pad["l"] + 6}" '
                     f'y="{pad["t"] + plot_h - 8:.0f}">'
                     '▼ below the axis floor; true value printed</text>')
    parts.append("</svg>")
    legend = "".join(
        f'<span><i style="background:var(--viz-{i + 1})"></i>{s["label"]}</span>'
        for i, s in enumerate(series))
    return ('<div class="viz">' + "".join(parts)
            + f'<div class="viz-legend">{legend}</div></div>')

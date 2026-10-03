#!/usr/bin/env python3
"""Figure: band weighting against the AT2 share of continued pretraining (RV18 round 3, RV19).

Epoch-1 checkpoints of continued self-supervised pretraining on AT2/GT450 mixtures:
UNI (11 AT2 shares x 2 data-order seeds, `results/uni_continued/dose_e001/`) and Virchow2
(6 shares x 2 seeds, `results/virchow2_continued/dose_e001/`). Per model:
A. band ratio R (high / low-mid normalized shift, dose 0.25) against the AT2 share, the original
   model as a dashed line, pooled slope with its 95% CI inside the panel;
B. percent change of the low-mid and high-band normalized shifts from the GT450-only run of the
   same seed (0:100), so continued training as such cancels and only the scanner mix remains.
Lines and points are the mean of the two seeds and shading spans them. Each seed's value is the
model mean over the 21 held-out slides; the slope (pooled over seeds, equal to the slope of the
seed means) and its CI come from the 2,000 shared slide resamples in `slopes.csv`. The original
model's values are recomputed from its epoch-0 embeddings with the same resamples
(`uni_continued_metrics.model_values`) and cached in
`results/manuscript_figures/pretraining_composition_start.csv`.

Writes `00_manuscript/figures/fig_pretraining_composition_uni.{png,pdf}` (main) and
`fig_pretraining_composition_virchow2_supp.{png,pdf}`.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REVISION = Path(__file__).resolve().parent
PROJECT = REVISION.parents[1]
if str(REVISION) not in sys.path:
    sys.path.insert(0, str(REVISION))

FIGURES = Path(os.environ.get("PAPER_FIGURES_DIR", str(PROJECT / "00_manuscript/figures")))
CACHE = REVISION / "results/manuscript_figures/pretraining_composition_start.csv"
# model -> (results root, output name, base font size, corner of the slope note in panel A)
MODELS = {
    "uni": (REVISION / "results/uni_continued", "fig_pretraining_composition_uni", 12, "lower left"),
    "virchow2": (REVISION / "results/virchow2_continued", "fig_pretraining_composition_virchow2_supp", 11, "upper left"),
}
BAND_COLORS = {"shift_low_mid": "#5A7DA5", "shift_high": "#D46B43"}  # as the band-sensitivity figures
BAND_LABELS = {"shift_low_mid": "Low–mid band", "shift_high": "High band"}
RATIO_COLOR = "#3A3F45"
REFERENCE = "#8A9099"
ENDPOINTS = ("R", "shift_low_mid", "shift_high", "S6", "knn_unseen")  # S6, kNN: quoted in the text
TICKS = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
XLABEL = "Pretraining data, AT2 : GT450 (%)"


def start_values() -> pd.DataFrame:
    """Original-model values on the held-out slides, with the dose script's resamples."""
    if CACHE.exists():
        return pd.read_csv(CACHE)
    import scanner_composition_metrics as M
    import scanner_mixture_metrics as XM
    import uni_continued_metrics as UM

    rows = []
    for model, (root, *_rest) in MODELS.items():
        held, bank, models, names, band = XM.load_tags(root, ["init_e000"])
        slides = np.asarray(sorted(held.slide_id.unique()))
        draws = np.random.default_rng(M.SEED).integers(0, len(slides), size=(M.N_BOOT, len(slides)))
        values = UM.model_values(models["init_e000"], names, held, bank, band, slides, draws)
        rows += [{"model": model, "endpoint": e, **M.summarize(*values[e])} for e in ENDPOINTS]
    frame = pd.DataFrame(rows)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(CACHE, index=False)
    return frame


def seed_summary(points: pd.DataFrame, endpoint: str, relative: bool = False) -> pd.DataFrame:
    """Mean, minimum and maximum over seeds at each share; `relative`: percent change from the
    same seed's GT450-only run (p = 0)."""
    part = points[points.endpoint == endpoint].assign(value=lambda d: d["mean"])
    if relative:
        gt450 = part[part.p_at2 == 0].set_index("seed")["mean"]
        part = part.assign(value=100 * (part["mean"] / part.seed.map(gt450) - 1))
    return part.groupby("p_at2")["value"].agg(["mean", "min", "max"]).reset_index()


def draw(ax, summary: pd.DataFrame, color: str, label: str | None = None) -> None:
    ax.fill_between(summary.p_at2, summary["min"], summary["max"], color=color, alpha=0.15, linewidth=0)
    ax.plot(summary.p_at2, summary["mean"], color=color, marker="o", markersize=5, linewidth=1.4, label=label)


def reference(ax, y: float, font_size: int) -> None:
    ax.axhline(y, color=REFERENCE, linestyle="--", linewidth=1.0, zorder=0)
    ax.text(1.03, y, "Original", color=REFERENCE, ha="right", va="bottom", fontsize=font_size - 2)


def render(model: str, start: pd.DataFrame) -> None:
    root, name, font_size, note_corner = MODELS[model]
    points = pd.read_csv(root / "dose_e001/points.csv")
    slopes = pd.read_csv(root / "dose_e001/slopes.csv", dtype={"seed": str})
    begin = start[start.model == model].set_index("endpoint")["mean"]

    plt.rcParams.update({"font.size": font_size, "pdf.fonttype": 42})
    figure, axes = plt.subplots(1, 2, figsize=(8.8, 4.0), constrained_layout=True)
    figure.get_layout_engine().set(w_pad=0.08)

    ax = axes[0]
    reference(ax, begin["R"], font_size)
    draw(ax, seed_summary(points, "R"), RATIO_COLOR)
    pooled = slopes[(slopes.endpoint == "R") & (slopes.seed == "pooled")].iloc[0]
    top = note_corner.startswith("upper")
    ax.text(0.04, 0.95 if top else 0.05,
            f"Slope {pooled['mean']:+.3f}\n(95% CI {pooled.ci_low:+.3f} to {pooled.ci_high:+.3f})".replace("-", "\u2212"),
            transform=ax.transAxes, fontsize=font_size - 2, va="top" if top else "bottom")
    low, high = ax.get_ylim()
    pad = 0.35 * (high - low)  # room for the note
    ax.set_ylim(low, high + pad) if top else ax.set_ylim(low - pad, high)
    ax.set(xlabel=XLABEL, ylabel="Band ratio\n(high / low–mid sensitivity)")

    ax = axes[1]
    ax.axhline(0, color=REFERENCE, linestyle="--", linewidth=1.0, zorder=0)
    for endpoint, color in BAND_COLORS.items():
        draw(ax, seed_summary(points, endpoint, relative=True), color, BAND_LABELS[endpoint])
    ax.set(xlabel=XLABEL, ylabel="Sensitivity change vs GT450 only (%)")
    # above the axes: the seed range of Virchow2 fills most of the panel
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=font_size - 1,
              borderaxespad=0.2, handlelength=1.6, columnspacing=1.2)

    for ax, letter in zip(axes, "AB"):
        ax.set_xticks(TICKS, [f"{round(100 * t)}:{round(100 * (1 - t))}" for t in TICKS])
        ax.set_xlim(-0.05, 1.05)
        ax.grid(axis="y", alpha=0.2)
        ax.set_title(letter, loc="left", fontsize=font_size + 6, fontweight="bold", x=-0.2, y=1.0, pad=6)

    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(FIGURES / f"{name}.{suffix}", dpi=300, facecolor="white")
    plt.close(figure)
    print(f"wrote {FIGURES / name}.png/.pdf", flush=True)


def main() -> None:
    start = start_values()
    print(start.to_string(index=False), flush=True)
    for model in MODELS:
        render(model, start)


if __name__ == "__main__":
    main()

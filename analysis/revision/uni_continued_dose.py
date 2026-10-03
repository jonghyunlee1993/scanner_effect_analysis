#!/usr/bin/env python3
"""RV18 round 3 (CPU): dose-response of the band ratio over the AT2 share at epoch 1.

Protocol: ``uni_continued_protocol.md`` (round 3). Eleven AT2 shares (0.0-1.0 in steps of 0.1)
x data-order seeds 0 and 1, epoch-1 checkpoints of the 3-epoch schedule, plus UNI. Endpoints per
model from ``uni_continued_metrics.model_values`` (shared slide resamples). Per seed: least-squares
slope of each endpoint on p with a 95% slide-bootstrap CI and Spearman rho; pooled slope over both
seeds.

Outputs (``results/uni_continued/dose_e001/``): ``points.csv``, ``slopes.csv``, ``summary.md``,
``band_ratio_vs_share.png`` (primary), ``secondary_vs_share.png``.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from pathlib import Path

import numpy as np
import pandas as pd

REVISION = Path(__file__).resolve().parent
if str(REVISION) not in sys.path:
    sys.path.insert(0, str(REVISION))

import scanner_composition_metrics as M  # noqa: E402
import scanner_mixture_metrics as XM  # noqa: E402
import uni_continued_metrics as UM  # noqa: E402

import os

ROOT = Path(os.environ.get("DOSE_ROOT", REVISION / "results/uni_continued"))  # RV19: results/virchow2_continued
EPOCH = int(os.environ.get("DOSE_EPOCH", "1"))
OUT = ROOT / os.environ.get("DOSE_OUTPUT", f"dose_e{EPOCH:03d}")
MODEL_LABEL = os.environ.get("DOSE_MODEL", "UNI")
SHARES = ([float(x) for x in os.environ["DOSE_SHARES"].split(",")] if os.environ.get("DOSE_SHARES")
          else [round(0.1 * i, 1) for i in range(11)])  # DOSE_SHARES: code check on a subset only
SEEDS = {s: ("" if s == 0 else f"_s{s}") for s in map(int, os.environ.get("DOSE_SEEDS", "0,1").split(","))}
ENDPOINTS = ("R", "shift_low_mid", "shift_high", "S2", "S6", "S3", "P3", "knn_unseen")
LABELS = {"R": "Band ratio R (high / low–mid)", "shift_low_mid": "Low–mid sensitivity",
          "shift_high": "High-band sensitivity", "S2": "AT2–GT450 distance (normalized)",
          "S6": "AT2→target distance (normalized)", "S3": "AKOYA colour+frequency increment",
          "P3": "GT450 colour+frequency increment", "knn_unseen": "Tissue kNN, unseen scanners"}
COLORS = {0: "#2a78d6", 1: "#eb6834"}  # validated categorical slots 1-2 (dataviz validator: all checks pass)
MARKERS = {0: "o", 1: "s"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def run_name(p: float, seed: int) -> str:
    return f"p{int(round(p * 100)):03d}{SEEDS[seed]}"


def main() -> None:
    tags = ["init_e000"] + [f"{run_name(p, s)}_e{EPOCH:03d}" for s in SEEDS for p in SHARES]
    held, bank, models, names, band = XM.load_tags(ROOT, tags)
    slides = np.asarray(sorted(held.slide_id.unique()))
    draws = np.random.default_rng(M.SEED).integers(0, len(slides), size=(M.N_BOOT, len(slides)))
    values = {}
    for tag in tags:
        values[tag] = UM.model_values(models[tag], names, held, bank, band, slides, draws)
        print(tag, round(float(values[tag]["R"][0]), 4), flush=True)
        del models[tag]

    rows = []
    for s in SEEDS:
        for p in SHARES:
            v = values[f"{run_name(p, s)}_e{EPOCH:03d}"]
            rows += [{"seed": s, "p_at2": p, "endpoint": e, **M.summarize(*v[e])} for e in ENDPOINTS]
    uni = {e: M.summarize(*values["init_e000"][e]) for e in ENDPOINTS}
    points = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    points.to_csv(OUT / "points.csv", index=False)

    p = np.array(SHARES)
    slopes = []
    for e in ENDPOINTS:
        per_seed_boot = {}
        for s in SEEDS:
            pts = np.array([values[f"{run_name(q, s)}_e{EPOCH:03d}"][e][0] for q in SHARES])
            boots = np.stack([values[f"{run_name(q, s)}_e{EPOCH:03d}"][e][1] for q in SHARES], axis=1)
            pc = p - p.mean()
            slope, boot = float(pc @ pts / (pc @ pc)), boots @ pc / (pc @ pc)
            rho = float(pd.Series(pts).corr(pd.Series(p), method="spearman"))
            per_seed_boot[s] = (pts, boots)
            slopes.append({"endpoint": e, "seed": str(s), **M.summarize(slope, boot), "spearman_rho": rho})
        pts = np.concatenate([per_seed_boot[s][0] for s in SEEDS])
        boots = np.concatenate([per_seed_boot[s][1] for s in SEEDS], axis=1)
        pp = np.concatenate([p for _ in SEEDS])
        pc = pp - pp.mean()
        slopes.append({"endpoint": e, "seed": "pooled", **M.summarize(float(pc @ pts / (pc @ pc)), boots @ pc / (pc @ pc)),
                       "spearman_rho": float(pd.Series(pts).corr(pd.Series(pp), method="spearman"))})
    slopes = pd.DataFrame(slopes)
    slopes.to_csv(OUT / "slopes.csv", index=False)

    r = slopes[slopes.endpoint.eq("R")].set_index("seed")
    reproduced = all(r.loc[str(s), "ci_high"] < 0 for s in SEEDS) and len(SEEDS) == 2  # rule needs both seeds
    lines = [f"# Band ratio over the AT2 share (epoch {EPOCH}), continued {MODEL_LABEL}", "",
             f"{MODEL_LABEL}: R = {uni['R']['mean']:.3f} [{uni['R']['ci_low']:.3f}, {uni['R']['ci_high']:.3f}]", "",
             "| endpoint | seed | slope over p [95% CI] | Spearman ρ |", "| --- | --- | --- | --- |"]
    for row in slopes.to_dict("records"):
        lines.append(f"| {row['endpoint']} | {row['seed']} | {row['mean']:+.4f} [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] | {row['spearman_rho']:+.2f} |")
    lines += ["", f"**Reproduced (both per-seed R slopes negative with CIs excluding zero): {reproduced}**", "",
              "## R by share", "", "| p | seed 0 | seed 1 |", "| --- | --- | --- |"]
    for q in SHARES:
        a = points[(points.endpoint == "R") & (points.p_at2 == q)].set_index("seed")["mean"]
        lines.append(f"| {q:.1f} | " + " | ".join(f"{a.loc[s]:.3f}" if s in a.index else "–" for s in (0, 1)) + " |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    plot_primary(points, slopes, uni)
    plot_secondary(points, uni)


def style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def plot_primary(points: pd.DataFrame, slopes: pd.DataFrame, uni: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 4.0), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    style(ax)
    ax.axhline(uni["R"]["mean"], color=MUTED, lw=1.0, ls="--", zorder=1)
    ax.text(1.02, uni["R"]["mean"], MODEL_LABEL, color=MUTED, fontsize=8, va="center", transform=ax.get_yaxis_transform())
    xs = np.linspace(0, 1, 50)
    for s in SEEDS:
        part = points[(points.endpoint == "R") & (points.seed == s)].sort_values("p_at2")
        offset = -0.008 if s == 0 else 0.008
        ax.scatter(part.p_at2 + offset, part["mean"], s=42, marker=MARKERS[s], color=COLORS[s],
                   edgecolor="#fcfcfb", linewidth=1.2, zorder=3, label=f"seed {s}")
        sl = slopes[(slopes.endpoint == "R") & (slopes.seed == str(s))].iloc[0]
        intercept = part["mean"].mean() - sl["mean"] * part.p_at2.mean()
        ax.plot(xs, intercept + sl["mean"] * xs, color=COLORS[s], lw=1.2, alpha=0.8, zorder=2)
    pooled = slopes[(slopes.endpoint == "R") & (slopes.seed == "pooled")].iloc[0]
    ax.text(0.03, 0.05, f"pooled slope {pooled['mean']:+.3f} [{pooled['ci_low']:+.3f}, {pooled['ci_high']:+.3f}]\n"
            f"Spearman ρ {pooled['spearman_rho']:+.2f} ({int((points.endpoint == 'R').sum())} models)",
            transform=ax.transAxes, fontsize=8, color=INK)
    ax.set_xlabel("AT2 share of continued-pretraining images (rest GT450)", fontsize=9, color=INK)
    ax.set_ylabel("Band ratio R  (high ÷ low–mid sensitivity)", fontsize=9, color=INK)
    ax.set_title(f"Band ratio vs AT2 share of continued pretraining ({MODEL_LABEL})", fontsize=9.5, color=INK, loc="left")
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlim(-0.05, 1.05)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "band_ratio_vs_share.png", dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_secondary(points: pd.DataFrame, uni: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [e for e in ENDPOINTS if e != "R"]
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6), facecolor="#fcfcfb")
    for ax, e in zip(axes.ravel(), keys):
        ax.set_facecolor("#fcfcfb")
        style(ax)
        ax.axhline(uni[e]["mean"], color=MUTED, lw=1.0, ls="--")
        for s in SEEDS:
            part = points[(points.endpoint == e) & (points.seed == s)].sort_values("p_at2")
            ax.scatter(part.p_at2, part["mean"], s=26, marker=MARKERS[s], color=COLORS[s], edgecolor="#fcfcfb",
                       linewidth=1.0, label=f"seed {s}")
        ax.set_title(LABELS[e], fontsize=8.5, color=INK, loc="left")
        ax.set_xticks([0, 0.5, 1])
    axes.ravel()[-1].axis("off")
    axes.ravel()[0].legend(frameon=False, fontsize=7)
    fig.supxlabel(f"AT2 share of continued-pretraining images (dashed = {MODEL_LABEL})", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "secondary_vs_share.png", dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()

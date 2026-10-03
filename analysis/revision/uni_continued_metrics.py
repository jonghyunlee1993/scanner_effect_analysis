#!/usr/bin/env python3
"""RV18 stage 3 (CPU): endpoints of UNI and its continued checkpoints, trends over the AT2 share.

Protocol: ``uni_continued_protocol.md``. Per model: the RV17 endpoints (S1 home kNN, S2
AT2-GT450 distance, S3 AKOYA and P3 GT450 colour + frequency increments, S5 AT2-vs-GT450 probe,
R band ratio) plus normalized band shifts (low-mid, mid, high; dose 0.25), S6 (mean normalized
AT2 -> target distance over the five targets), between-tissue distance and tissue kNN on unseen
scanners. Per epoch: slope over p and mixture contrast (as RV17), and every arm minus UNI.
Shared slide resamples (seed 20260924).

Outputs (``results/uni_continued/<name>/``): ``model_endpoints.csv``, ``effects.csv``, ``summary.md``,
``endpoints_by_share.png``.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REVISION = Path(__file__).resolve().parent
if str(REVISION) not in sys.path:
    sys.path.insert(0, str(REVISION))

import scanner_composition_metrics as M  # noqa: E402
import scanner_mixture_metrics as XM  # noqa: E402
from scanner_composition_metrics import C  # noqa: E402

OUTPUT = REVISION / "results/uni_continued"
ARMS = XM.ARMS
KEYS = ("S1", "S2", "S3", "P3", "S5", "R", "S6", "shift_low_mid", "shift_mid", "shift_high", "between", "knn_unseen")
TREND_KEYS = ("S1", "S2", "S3", "P3", "S5", "R", "S6", "shift_low_mid", "shift_mid", "shift_high", "knn_unseen")


def model_values(feats, names, held, bank, band, slides, draws) -> dict:
    values = XM.endpoints_of(feats, names, held, bank, band, slides, draws)
    between, between_boot = values["between"]
    vec = M.slide_vectors(M.derived(M.location_values(feats["held"], names, held, band)), slides)
    for key, stat in (("shift_mid", ("shift_mid_d0.25", "all")), ("S6", ("raw_target_distance", "all"))):
        v = vec[stat]
        values[key] = (v.mean() / between, v[draws].mean(axis=1) / between_boot)
    # knn_unseen from XM has no bootstrap; recompute with resamples
    raw = {s: names.index("source_at2" if s == "at2" else f"target_{s}") for s in M.SCANNERS}
    points, boots = [], []
    for s in M.GATE_SCANNERS:
        correct = M.knn_correct(feats["held"][:, raw[s]], held.tissue_type.to_numpy(),
                                feats["bank"][:, M.SCANNERS.index(s)], bank.tissue_type.to_numpy())
        point, boot = M.balanced(correct, held, slides, draws)
        points.append(point)
        boots.append(boot)
    values["knn_unseen"] = (float(np.mean(points)), np.mean(np.stack(boots), axis=0))
    return values


def main() -> None:
    global ARMS
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--epochs", default="1,3")
    parser.add_argument("--output-name", default="metrics")
    parser.add_argument("--runs", default="p100,p080,p020,p000", help="run names; AT2 share from the pXXX prefix")
    args = parser.parse_args()
    root, out = args.output_root, args.output_root / args.output_name
    epochs = [int(e) for e in args.epochs.split(",")]
    ARMS = {run: int(run[1:4]) / 100 for run in args.runs.split(",")}
    tags = ["init_e000"] + [f"{a}_e{e:03d}" for e in epochs for a in ARMS]
    held, bank, models, names, band = XM.load_tags(root, tags)
    slides = np.asarray(sorted(held.slide_id.unique()))
    draws = np.random.default_rng(M.SEED).integers(0, len(slides), size=(M.N_BOOT, len(slides)))
    values = {}
    for tag in tags:
        values[tag] = model_values(models[tag], names, held, bank, band, slides, draws)
        print(tag, {k: round(float(values[tag][k][0]), 4) for k in KEYS}, flush=True)

    rows = []
    for tag, v in values.items():
        arm, epoch = tag.split("_e")
        for key in KEYS:
            rows.append({"model": tag, "arm": arm, "epoch": int(epoch), "p_at2": ARMS.get(arm, np.nan),
                         "endpoint": key, **M.summarize(*v[key])})
    model_frame = pd.DataFrame(rows)
    C.write_frame(out / "model_endpoints.csv", model_frame)

    p = np.array(list(ARMS.values()), dtype=float)
    pc = p - p.mean()
    effects = []
    for epoch in epochs:
        for key in TREND_KEYS:
            points = np.array([values[f"{a}_e{epoch:03d}"][key][0] for a in ARMS])
            boots = np.stack([values[f"{a}_e{epoch:03d}"][key][1] for a in ARMS], axis=1)
            slope, slope_boot = float(pc @ points / (pc @ pc)), boots @ pc / (pc @ pc)
            mixed = [i for i, v in enumerate(p) if 0 < v < 1]
            pure = [i for i, v in enumerate(p) if v in (0.0, 1.0)]
            contrast = float(points[mixed].mean() - points[pure].mean()) if mixed and pure else float("nan")
            contrast_boot = (boots[:, mixed].mean(axis=1) - boots[:, pure].mean(axis=1)) if mixed and pure else np.full(len(boots), np.nan)
            effects.append({"epoch": epoch, "endpoint": key, "effect": "slope_over_p", **M.summarize(slope, slope_boot)})
            effects.append({"epoch": epoch, "endpoint": key, "effect": "mixture_contrast", **M.summarize(contrast, contrast_boot)})
            for arm in ARMS:
                v, u = values[f"{arm}_e{epoch:03d}"][key], values["init_e000"][key]
                effects.append({"epoch": epoch, "endpoint": key, "effect": f"{arm}_minus_uni",
                                **M.summarize(v[0] - u[0], v[1] - u[1])})
    effects = pd.DataFrame(effects)
    C.write_frame(out / "effects.csv", effects)
    write_summary(out, values, effects, epochs)
    plot(out, model_frame, epochs)
    print(json.dumps({"status": "complete"}), flush=True)


def write_summary(out: Path, values: dict, effects: pd.DataFrame, epochs: list[int]) -> None:
    fmt = lambda r: f"{r['mean']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"  # noqa: E731
    lines = ["# RV18 continued UNI pretraining on AT2/GT450 mixtures: results", "",
             "Protocol: `analysis/revision/uni_continued_protocol.md`. Single seed (round 1).", "",
             "## Levels", "", "| model | " + " | ".join(KEYS) + " |", "|" + " --- |" * (len(KEYS) + 1)]
    for tag, v in values.items():
        lines.append(f"| {tag} | " + " | ".join(f"{float(v[k][0]):.4f}" for k in KEYS) + " |")
    for epoch in epochs:
        lines += ["", f"## Epoch {epoch}: trend over AT2 share, mixture contrast, arms minus UNI", "",
                  "| endpoint | slope over p | mixture contrast | " + " | ".join(f"{a} − UNI" for a in ARMS) + " |",
                  "|" + " --- |" * (3 + len(ARMS))]
        part = effects[effects.epoch.eq(epoch)].set_index(["endpoint", "effect"])
        for key in TREND_KEYS:
            cells = [fmt(part.loc[(key, "slope_over_p")]), fmt(part.loc[(key, "mixture_contrast")])]
            cells += [fmt(part.loc[(key, f"{a}_minus_uni")]) for a in ARMS]
            lines.append(f"| {key} | " + " | ".join(cells) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def plot(out: Path, frame: pd.DataFrame, epochs: list[int]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = ("S2", "S3", "S6", "R", "shift_high", "knn_unseen")
    fig, axes = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 3.2))
    uni = frame[frame.model.eq("init_e000")].set_index("endpoint")
    colors = {1: "#9DB4CF", 2: "#6E8FB5", 3: "#2F5D8C"}
    for ax, key in zip(axes, keys):
        ax.axhline(uni.loc[key, "mean"], color="#888888", lw=1, ls="--", label="UNI")
        for epoch in epochs:
            part = frame[frame.endpoint.eq(key) & frame.epoch.eq(epoch) & frame.arm.ne("init")].sort_values("p_at2")
            ax.errorbar(part.p_at2, part["mean"], yerr=[part["mean"] - part.ci_low, part.ci_high - part["mean"]],
                        marker="o", ms=4, lw=1.3, capsize=2, color=colors.get(epoch, "k"), label=f"epoch {epoch}")
        ax.set_title(key, fontsize=9)
        ax.set_xlabel("AT2 share")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "endpoints_by_share.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()

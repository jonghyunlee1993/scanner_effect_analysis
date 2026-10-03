#!/usr/bin/env python3
"""RV17 stage 3 (CPU): per-arm endpoints, trend over the AT2 share, mixture contrast, RV16 noise reference.

Protocol: ``scanner_mixture_protocol.md``. Endpoints per model (normalized by between-tissue
distance as RV16 amendment D1): S1 home-scanner tissue kNN (AT2 - GT450 accuracy), S2 AT2-GT450
distance of raw pairs, S3 AKOYA and P3 GT450 colour + frequency increment over Reinhard, S5
AT2-vs-GT450 linear-probe detectability, band ratio R (high / low-mid shift, dose 0.25). Trend =
least-squares slope over p in {1.0, 0.8, 0.2, 0.0}; mixture contrast M = mean(p 0.8, 0.2) -
mean(p 1.0, 0.0); 95% CIs from the shared slide resamples (seed 20260924). Noise reference: the
larger of |A' - A| and |B' - B| for the same endpoint in RV16 at epoch 25.

Outputs (``results/scanner_mixture/metrics/``): ``arm_endpoints.csv``, ``effects.csv``,
``decision.json``, ``summary.md``, ``endpoints_by_share.png``.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REVISION = Path(__file__).resolve().parent
if str(REVISION) not in sys.path:
    sys.path.insert(0, str(REVISION))

import scanner_composition_metrics as M  # noqa: E402
from scanner_composition_metrics import C  # noqa: E402

OUTPUT = REVISION / "results/scanner_mixture"
RV16 = REVISION / "results/scanner_composition_pilot"
ARMS = {"p100": 1.0, "p080": 0.8, "p020": 0.2, "p000": 0.0}
EPOCHS = (10, 25)
PRIMARY_EPOCH = 25
ENDPOINTS = ("S1", "S2", "S3", "P3", "S5", "R")
Q2_ENDPOINTS = ("S1", "S2", "S3", "P3", "S5")
EXPECTED_SLOPE = {"S1": 1, "S2": -1, "S3": 1, "P3": 1, "S5": None}
SHORTCUT = {"S2": 1, "S5": 1}
LABELS = {"S1": "Home-scanner tissue kNN (AT2 − GT450 accuracy)", "S2": "AT2–GT450 distance / between-tissue",
          "S3": "AKOYA colour+frequency increment / between-tissue",
          "P3": "GT450 colour+frequency increment / between-tissue",
          "S5": "AT2 vs GT450 linear-probe detectability", "R": "Band ratio high / low-mid (dose 0.25)"}


def load_tags(root: Path, tags: list[str]):
    held = pd.read_csv(root / "embeddings/held_out_locations.csv", dtype={"slide_id": str})
    bank = pd.read_csv(root / "embeddings/bank_locations.csv", dtype={"slide_id": str})
    models, names = {}, None
    for tag in tags:
        with h5py.File(root / "embeddings" / f"{tag}.h5", "r") as store:
            models[tag] = {"held": np.asarray(store["held_out"], np.float32), "bank": np.asarray(store["bank"], np.float32)}
            names = C.decode(store["condition_names"][:])
    band = pd.concat([pd.read_csv(p, dtype={"slide_id": str}) for p in sorted((root / "render").glob("*_band.csv.gz"))],
                     ignore_index=True)
    return held, bank, models, names, band


def probe_detectability(feats: np.ndarray, raw: dict, held: pd.DataFrame, slides: np.ndarray) -> np.ndarray:
    """Per-slide accuracy of a slide-grouped 5-fold linear probe, AT2 vs GT450 raw embeddings."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = np.concatenate([feats[:, raw["at2"]], feats[:, raw["gt450"]]])
    y = np.r_[np.zeros(len(held)), np.ones(len(held))]
    groups = np.r_[held.slide_id.to_numpy(), held.slide_id.to_numpy()]
    correct = np.zeros(len(y))
    for train, test in GroupKFold(n_splits=5).split(x, y, groups):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, solver="liblinear", max_iter=1000))
        model.fit(x[train], y[train])
        correct[test] = model.predict(x[test]) == y[test]
    # classes are balanced within every slide (each location contributes one AT2 and one GT450 image)
    return pd.Series(correct).groupby(groups).mean().reindex(slides).to_numpy()


def endpoints_of(feats: dict, names: list[str], held, bank, band, slides, draws) -> dict[str, tuple[float, np.ndarray]]:
    raw = {s: names.index("source_at2" if s == "at2" else f"target_{s}") for s in M.SCANNERS}
    codes = pd.Categorical(held.slide_id, categories=slides).codes
    slide_tissue = held.groupby("slide_id").tissue_type.first().reindex(slides).to_numpy()
    pair_means = np.zeros((len(slides), len(slides)))
    for s in ("at2", "gt450"):
        x = feats["held"][:, raw[s]]
        d = pd.DataFrame(1.0 - x @ x.T, index=codes, columns=codes)
        pair_means += d.groupby(level=0).mean().T.groupby(level=0).mean().reindex(
            index=range(len(slides)), columns=range(len(slides))).to_numpy() / 2
    between, between_boot = M.normalizer(pair_means, slide_tissue, draws)
    vec = M.slide_vectors(M.derived(M.location_values(feats["held"], names, held, band)), slides)

    def norm(v):
        return v.mean() / between, v[draws].mean(axis=1) / between_boot

    knn = {}
    for s in ("at2", "gt450"):
        correct = M.knn_correct(feats["held"][:, raw[s]], held.tissue_type.to_numpy(),
                                feats["bank"][:, M.SCANNERS.index(s)], bank.tissue_type.to_numpy())
        knn[s] = M.balanced(correct, held, slides, draws)
    pair = M.distance(feats["held"][:, raw["at2"]], feats["held"][:, raw["gt450"]])
    pair = pd.Series(pair).groupby(held.slide_id.to_numpy()).mean().reindex(slides).to_numpy()
    probe = probe_detectability(feats["held"], raw, held, slides)
    high, low = vec[("shift_high_d0.25", "all")], vec[("shift_low_mid_d0.25", "all")]
    return {"S1": (knn["at2"][0] - knn["gt450"][0], knn["at2"][1] - knn["gt450"][1]),
            "S2": norm(pair),
            "S3": norm(vec[("increment_combined_over_reinhard", "akoya")]),
            "P3": norm(vec[("increment_combined_over_reinhard", "gt450")]),
            "S5": (probe.mean(), probe[draws].mean(axis=1)),
            "R": (high.mean() / low.mean(), high[draws].mean(axis=1) / low[draws].mean(axis=1)),
            "shift_low_mid": norm(low), "shift_high": norm(high), "between": (between, between_boot),
            "knn_unseen": (float(np.mean([M.balanced(M.knn_correct(
                feats["held"][:, raw[s]], held.tissue_type.to_numpy(), feats["bank"][:, M.SCANNERS.index(s)],
                bank.tissue_type.to_numpy()), held, slides, draws)[0] for s in M.GATE_SCANNERS])), np.zeros(1))}


def rv16_noise() -> dict[str, dict]:
    tags = ["A_e025", "A1_e025", "B_e025", "B1_e025"]
    held, bank, models, names, band = load_tags(RV16, tags)
    slides = np.asarray(sorted(held.slide_id.unique()))
    draws = np.random.default_rng(M.SEED).integers(0, len(slides), size=(M.N_BOOT, len(slides)))
    values = {t: endpoints_of(models[t], names, held, bank, band, slides, draws) for t in tags}
    out = {}
    for e in ENDPOINTS:
        a = abs(values["A1_e025"][e][0] - values["A_e025"][e][0])
        b = abs(values["B1_e025"][e][0] - values["B_e025"][e][0])
        arm0 = values["B_e025"][e][0] - values["A_e025"][e][0]
        arm1 = values["B1_e025"][e][0] - values["A1_e025"][e][0]
        out[e] = {"noise_at2": a, "noise_gt450": b, "reference": max(a, b),
                  "rv16_arm_effect_seed0": arm0, "rv16_arm_effect_seed1": arm1}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    args = parser.parse_args()
    root = args.output_root
    out = root / "metrics"
    tags = [f"{a}_e{e:03d}" for e in EPOCHS for a in ARMS]
    held, bank, models, names, band = load_tags(root, tags)
    slides = np.asarray(sorted(held.slide_id.unique()))
    draws = np.random.default_rng(M.SEED).integers(0, len(slides), size=(M.N_BOOT, len(slides)))
    values = {t: endpoints_of(models[t], names, held, bank, band, slides, draws) for t in tags}
    noise = rv16_noise()

    rows = []
    for tag, v in values.items():
        arm, epoch = tag.split("_e")
        for e, (point, boot) in v.items():
            rows.append({"epoch": int(epoch), "arm": arm, "p_at2": ARMS[arm], "endpoint": e,
                         **(M.summarize(point, boot) if boot.size > 1 else {"mean": point})})
    arm_frame = pd.DataFrame(rows)
    C.write_frame(out / "arm_endpoints.csv", arm_frame)

    p = np.array(list(ARMS.values()))
    pc = p - p.mean()
    effects, decision = [], {"primary_epoch": PRIMARY_EPOCH, "q2": {}, "q1": {}}
    for epoch in EPOCHS:
        for e in ENDPOINTS:
            points = np.array([values[f"{a}_e{epoch:03d}"][e][0] for a in ARMS])
            boots = np.stack([values[f"{a}_e{epoch:03d}"][e][1] for a in ARMS], axis=1)  # [draws, arms]
            slope = float(pc @ points / (pc @ pc))
            slope_boot = boots @ pc / (pc @ pc)
            contrast = float(points[[1, 2]].mean() - points[[0, 3]].mean())
            contrast_boot = boots[:, [1, 2]].mean(axis=1) - boots[:, [0, 3]].mean(axis=1)
            ends = float(points[0] - points[3])
            for name, point, boot in (("slope", slope, slope_boot), ("mixture_contrast", contrast, contrast_boot)):
                effects.append({"epoch": epoch, "endpoint": e, "effect": name, **M.summarize(point, boot),
                                "noise_reference": noise[e]["reference"], "pure_end_difference": ends})
            if epoch == PRIMARY_EPOCH and e in Q2_ENDPOINTS:
                s = M.summarize(slope, slope_boot)
                m = M.summarize(contrast, contrast_boot)
                excl = lambda r: r["ci_low"] > 0 or r["ci_high"] < 0  # noqa: E731
                trend = excl(s) and np.sign(ends) == np.sign(slope)
                decision["q2"][e] = {
                    "label": LABELS[e], "levels": dict(zip(ARMS, points.round(5).tolist())),
                    "slope": s, "trend_present": bool(trend),
                    "trend_exceeds_rv16_noise": bool(trend and abs(slope) > noise[e]["reference"]),
                    "slope_sign_matches_rv16_expectation": None if EXPECTED_SLOPE[e] is None else bool(np.sign(slope) == EXPECTED_SLOPE[e]),
                    "mixture_contrast": m, "contrast_present": bool(excl(m)),
                    "contrast_exceeds_rv16_noise": bool(excl(m) and abs(contrast) > noise[e]["reference"]),
                    "shortcut_prediction_met": None if e not in SHORTCUT else bool(excl(m) and np.sign(contrast) == SHORTCUT[e]),
                    "rv16_noise_reference": noise[e]["reference"]}
        if epoch == PRIMARY_EPOCH:
            r = {a: values[f"{a}_e{epoch:03d}"]["R"][0] for a in ARMS}
            decision["q1"] = {"R_by_arm": r, "mean_R": float(np.mean(list(r.values()))),
                              "band_endpoints_testable": bool(np.mean(list(r.values())) >= 0.5),
                              "rv16_R_epoch25": {k: v for k, v in noise["R"].items()}}
    decision["go"] = any(d["trend_exceeds_rv16_noise"] or d["contrast_exceeds_rv16_noise"] for d in decision["q2"].values())
    decision["rv16_noise"] = noise
    effects = pd.DataFrame(effects)
    C.write_frame(out / "effects.csv", effects)
    C.write_json(out / "decision.json", decision)
    write_summary(out, arm_frame, effects, decision)
    plot(out, arm_frame)
    print(json.dumps({"status": "complete", "go": decision["go"]}), flush=True)


def write_summary(out: Path, arms: pd.DataFrame, effects: pd.DataFrame, decision: dict) -> None:
    fmt = lambda r: f"{r['mean']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"  # noqa: E731
    lines = ["# RV17 AT2/GT450 mixtures, blur off (ViT-Tiny): results", "",
             "Protocol: `analysis/revision/scanner_mixture_protocol.md`.", "",
             f"## Q2 (epoch {decision['primary_epoch']}): levels by AT2 share, trend and mixture contrast", "",
             "| Endpoint | p=1.0 | 0.8 | 0.2 | 0.0 | slope [95% CI] | trend > RV16 noise | M [95% CI] | M > RV16 noise | RV16 noise |",
             "|" + " --- |" * 10]
    for e, d in decision["q2"].items():
        lv = d["levels"]
        lines.append(f"| {e}: {d['label']} | {lv['p100']:+.4f} | {lv['p080']:+.4f} | {lv['p020']:+.4f} | {lv['p000']:+.4f} | "
                     f"{fmt(d['slope'])} | {d['trend_exceeds_rv16_noise']} | {fmt(d['mixture_contrast'])} | "
                     f"{d['contrast_exceeds_rv16_noise']} | {d['rv16_noise_reference']:.4f} |")
    q1 = decision["q1"]
    lines += ["", f"**Decision: {'GO' if decision['go'] else 'NO-GO'}**", "",
              "## Q1 (descriptive): band ratio R = high / low-mid shift", "",
              "| p=1.0 | 0.8 | 0.2 | 0.0 | mean | testable (≥ 0.5) |", "|" + " --- |" * 6,
              "| " + " | ".join(f"{q1['R_by_arm'][a]:.3f}" for a in ARMS) + f" | {q1['mean_R']:.3f} | {q1['band_endpoints_testable']} |",
              "", "## Per-arm values, every epoch", "", "| epoch | arm | endpoint | value [95% CI] |", "|" + " --- |" * 4]
    for r in arms.to_dict("records"):
        value = fmt(r) if np.isfinite(r.get("ci_low", np.nan)) else f"{r['mean']:+.4f}"
        lines.append(f"| {r['epoch']} | {r['arm']} | {r['endpoint']} | {value} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def plot(out: Path, arms: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(ENDPOINTS), figsize=(3.2 * len(ENDPOINTS), 3.2))
    for ax, e in zip(axes, ENDPOINTS):
        for epoch, color in ((10, "#9DB4CF"), (25, "#2F5D8C")):
            part = arms[arms.endpoint.eq(e) & arms.epoch.eq(epoch)].sort_values("p_at2")
            yerr = None
            if "ci_low" in part and part.ci_low.notna().all():
                yerr = [part["mean"] - part.ci_low, part.ci_high - part["mean"]]
            ax.errorbar(part.p_at2, part["mean"], yerr=yerr, marker="o", ms=4, lw=1.3, capsize=2,
                        color=color, label=f"epoch {epoch}")
        ax.set_title(LABELS[e], fontsize=8)
        ax.set_xlabel("AT2 share in pretraining")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "endpoints_by_share.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()

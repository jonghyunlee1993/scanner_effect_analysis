#!/usr/bin/env python3
"""RV18 round-2 reading: is each round-1 trend over the AT2 share replicated with data-order seed 1?

Reads ``metrics/`` (seed 0) and ``metrics_s1/`` (seed 1) from ``uni_continued_metrics.py``. Per
endpoint and epoch: slope over p in each seed (with its slide-bootstrap CI), training noise =
mean over the four arms of |E(seed 1) - E(seed 0)|, and the round-2 rule of
``uni_continued_protocol.md``. Writes ``replicate/summary.md`` and ``replicate/replicate.csv``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent / "results/uni_continued"
ENDPOINTS = ("R", "S2", "S6", "S3", "P3", "shift_low_mid", "shift_mid", "shift_high", "knn_unseen")
RULE_ENDPOINTS = ("R", "S2", "S6", "S3", "P3")
ARMS = ("p100", "p080", "p020", "p000")


def main() -> None:
    eff0 = pd.read_csv(ROOT / "metrics/effects.csv")
    eff1 = pd.read_csv(ROOT / "metrics_s1/effects.csv")
    lev0 = pd.read_csv(ROOT / "metrics/model_endpoints.csv")
    lev1 = pd.read_csv(ROOT / "metrics_s1/model_endpoints.csv")
    lev1 = lev1.assign(arm=lev1.arm.str.replace("_s1", "", regex=False))
    rows = []
    for epoch in (1, 3):
        for e in ENDPOINTS:
            s0 = eff0[(eff0.epoch == epoch) & (eff0.endpoint == e) & (eff0.effect == "slope_over_p")].iloc[0]
            s1 = eff1[(eff1.epoch == epoch) & (eff1.endpoint == e) & (eff1.effect == "slope_over_p")].iloc[0]
            a0 = lev0[(lev0.epoch == epoch) & (lev0.endpoint == e)].set_index("arm")["mean"].reindex(ARMS)
            a1 = lev1[(lev1.epoch == epoch) & (lev1.endpoint == e)].set_index("arm")["mean"].reindex(ARMS)
            noise = float((a1 - a0).abs().mean())
            rows.append({"epoch": epoch, "endpoint": e, "slope_seed0": s0["mean"], "ci0": (s0.ci_low, s0.ci_high),
                         "slope_seed1": s1["mean"], "ci1": (s1.ci_low, s1.ci_high), "noise_mean_abs_arm_diff": noise,
                         "same_sign": bool(np.sign(s0["mean"]) == np.sign(s1["mean"])),
                         "seed1_ci_excludes_zero": bool(s1.ci_low > 0 or s1.ci_high < 0),
                         "min_abs_slope_gt_noise": bool(min(abs(s0["mean"]), abs(s1["mean"])) > noise)})
    frame = pd.DataFrame(rows)
    out = ROOT / "replicate"
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "replicate.csv", index=False)
    lines = ["# RV18 round 2: replication of the round-1 trends (data-order seed 1)", "",
             "Slope over the AT2 share p (full range of p = 1). Noise = mean over arms of |E(seed 1) − E(seed 0)|.", "",
             "| epoch | endpoint | slope seed 0 [CI] | slope seed 1 [CI] | noise | same sign | seed-1 CI ≠ 0 | > noise |",
             "|" + " --- |" * 8]
    for r in frame.to_dict("records"):
        lines.append(f"| {r['epoch']} | {r['endpoint']} | {r['slope_seed0']:+.4f} [{r['ci0'][0]:+.4f}, {r['ci0'][1]:+.4f}] | "
                     f"{r['slope_seed1']:+.4f} [{r['ci1'][0]:+.4f}, {r['ci1'][1]:+.4f}] | {r['noise_mean_abs_arm_diff']:.4f} | "
                     f"{r['same_sign']} | {r['seed1_ci_excludes_zero']} | {r['min_abs_slope_gt_noise']} |")
    verdict = []
    for e in RULE_ENDPOINTS:
        r3 = frame[(frame.epoch == 3) & (frame.endpoint == e)].iloc[0]
        r1 = frame[(frame.epoch == 1) & (frame.endpoint == e)].iloc[0]
        ok = r3.same_sign and r3.seed1_ci_excludes_zero and r3.min_abs_slope_gt_noise and r1.same_sign
        verdict.append(f"| {e} | {'replicated' if ok else 'not replicated'} |")
    lines += ["", "## Round-2 rule (epoch 3 seed-1 CI ≠ 0, same sign at epochs 1 and 3, |slope| > noise)", "",
              "| endpoint | verdict |", "| --- | --- |", *verdict]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

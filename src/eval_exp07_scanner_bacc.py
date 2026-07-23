"""Exp-07 Stage-1 co-primary: low-band scanner separability (invariance).

Paired low-band MAE falling toward AT2 says the *rendering* moved closer; it does
NOT by itself say the scanner *signature* was removed.  The project's real target
is invariance, so we measure it directly: can a classifier still tell the scanners
apart from the coarse low band, before vs after the range-safe contract-A
correction?

Two protocols, each reported for a 4-class ({at2,gt450,versa,s60}) and a 3-class
({gt450,versa,s60}) task, split by slide (train+val slides fit, test slides
score), balanced-accuracy scored:

  P1  residual separability -- fit on corrected, score on corrected.  A re-fit
      adversary; the honest invariance number.  raw baseline = fit/score on raw.
  P2  transfer -- fit on raw, score on corrected.  Does the correction move
      features off the raw manifold (and toward AT2)?

Pre-committed reading (plan §7.5): MAE down but BACC ~ raw = "rendering-closer,
invariance-not-improved" (partial); BACC falling toward chance = real invariance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prenorm.exp06.data import BalancedPairDataset
from prenorm.exp06.frequency import FixedLaplacianPyramid
from prenorm.exp07.projection import project_correction, low_correction_delta
from utils.config import load_config

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

CLASSES4 = ["at2", "gt450", "versa", "s60"]
CLASSES3 = ["gt450", "versa", "s60"]


def apply_affine(low, affine):
    x = low.permute(0, 2, 3, 1)
    x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
    return (x @ affine).permute(0, 3, 1, 2)


def coarse_feats(coarse):
    """[N,3,h,w] -> [N,15] per-channel mean/std/q10/q50/q90."""
    mean = coarse.mean(dim=(-2, -1))
    std = coarse.std(dim=(-2, -1), unbiased=False)
    flat = coarse.flatten(2)
    qs = torch.quantile(flat, torch.tensor([0.1, 0.5, 0.9]), dim=2)   # [3,N,3]
    qs = qs.permute(1, 2, 0).reshape(coarse.shape[0], -1)            # [N,9]
    return torch.cat([mean, std, qs], dim=1).numpy()


@torch.no_grad()
def collect(cfg, pyramid, affines, split_slides, max_per_scanner, seed):
    """Return dict rep -> label -> [feat] for one slide split.

    ``BalancedPairDataset`` already filters records to its ``split`` arg, so we
    walk train/val/test and keep only records whose slide is in ``split_slides``.
    """
    rng = np.random.default_rng(seed)
    rows = {"raw": {c: [] for c in CLASSES4}, "corr": {c: [] for c in CLASSES4}}
    at2_seen = set()

    for phase in ("train", "val", "test"):
        if not any(s in split_slides for s in getattr(cfg.split.slides, phase)):
            continue
        ds = BalancedPairDataset(cfg, phase, False)
        for scanner in [str(s) for s in cfg.target_scanners]:
            idx = [i for i, r in enumerate(ds.records)
                   if r["scanner"] == scanner and r["slide_id"] in split_slides]
            if len(idx) > max_per_scanner:
                idx = sorted(rng.choice(idx, max_per_scanner, replace=False).tolist())
            affine = affines[scanner]
            for start in range(0, len(idx), 16):
                items = [ds[i] for i in idx[start:start + 16]]
                source = torch.stack([it["source"] for it in items])
                reference = torch.stack([it["reference"] for it in items])
                low, _ = pyramid.decompose(source)
                corrected_unsafe = apply_affine(low, affine)
                d, _, _ = low_correction_delta(pyramid, source, corrected_unsafe)
                _, alpha, _, _ = project_correction(source, d)
                safe = low + alpha.reshape(-1, 1, 1, 1) * (corrected_unsafe - low)

                fr = coarse_feats(low)
                fc = coarse_feats(safe)
                rows["raw"][scanner].extend(fr.tolist())
                rows["corr"][scanner].extend(fc.tolist())

                # AT2 reference (bypassed => raw == corr), dedup per tuple.
                ref_low = pyramid.coarsest(reference)
                far = coarse_feats(ref_low)
                for it, f in zip(items, far):
                    key = (it["slide_id"], int(it["tuple_id"]))
                    if key in at2_seen:
                        continue
                    at2_seen.add(key)
                    rows["raw"]["at2"].append(f.tolist())
                    rows["corr"]["at2"].append(f.tolist())
    return rows


def _xy(rows, rep, classes):
    x, y = [], []
    for c in classes:
        feats = rows[rep][c]
        x.extend(feats)
        y.extend([c] * len(feats))
    return np.asarray(x, np.float64), np.asarray(y)


def fit_score(train_rows, test_rows, classes, fit_rep, score_rep, seed):
    xtr, ytr = _xy(train_rows, fit_rep, classes)
    xte, yte = _xy(test_rows, score_rep, classes)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0),
    )
    clf.fit(xtr, ytr)
    pred = clf.predict(xte)
    bacc = balanced_accuracy_score(yte, pred)
    cm = confusion_matrix(yte, pred, labels=classes)
    return float(bacc), cm, yte


def counts(rows, classes):
    return {rep: {c: len(rows[rep][c]) for c in classes} for rep in ("raw", "corr")}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/exp06a_lf16_10slide.yaml")
    parser.add_argument("--baselines", default="outputs/exp06_baselines")
    parser.add_argument("--max-per-scanner", type=int, default=800)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    pyramid = FixedLaplacianPyramid(int(cfg.pyramid.levels))
    affines = {
        s: torch.tensor(json.loads(
            (Path(args.baselines) / f"{s}.json").read_text())["affine_matrix_with_bias"],
            dtype=torch.float32)
        for s in [str(x) for x in cfg.target_scanners]
    }

    fit_slides = list(cfg.split.slides.train) + list(cfg.split.slides.val)
    test_slides = list(cfg.split.slides.test)
    train_rows = collect(cfg, pyramid, affines, set(fit_slides), args.max_per_scanner, args.seed)
    test_rows = collect(cfg, pyramid, affines, set(test_slides), args.max_per_scanner, args.seed + 1)

    report = {"fit_slides": fit_slides, "test_slides": test_slides,
              "train_counts": counts(train_rows, CLASSES4),
              "test_counts": counts(test_rows, CLASSES4), "tasks": {}}

    for name, classes in (("4class", CLASSES4), ("3class", CLASSES3)):
        chance = 1.0 / len(classes)
        raw_b, raw_cm, _ = fit_score(train_rows, test_rows, classes, "raw", "raw", args.seed)
        p1_b, p1_cm, _ = fit_score(train_rows, test_rows, classes, "corr", "corr", args.seed)
        p2_b, p2_cm, yte = fit_score(train_rows, test_rows, classes, "raw", "corr", args.seed)
        report["tasks"][name] = {
            "chance": chance,
            "raw_bacc": raw_b,
            "p1_corrected_bacc": p1_b,           # retrain-on-corrected (residual)
            "p2_transfer_bacc": p2_b,            # raw-trained, corrected-scored
            "raw_confusion": raw_cm.tolist(),
            "p1_confusion": p1_cm.tolist(),
            "p2_confusion": p2_cm.tolist(),
            "labels": classes,
        }
        print(f"\n=== {name}  (chance={chance:.3f}) ===")
        print(f"  raw BACC                 = {raw_b:.3f}")
        print(f"  P1 corrected BACC (refit)= {p1_b:.3f}   (drop {raw_b - p1_b:+.3f})")
        print(f"  P2 transfer  BACC        = {p2_b:.3f}")

    out_dir = Path(cfg.paths.repo) / "outputs" / "exp07_stage1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "scanner_bacc.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\n[exp07-bacc] wrote {out_path}")


if __name__ == "__main__":
    main()

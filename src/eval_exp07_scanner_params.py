"""Exp-07 zero-shot probe: can a source-only fingerprint EXPLAIN each scanner's
correction parameters, and does the correction (low-band efficacy + invariance)
survive when we drive it from the fingerprint instead of the paired oracle?

Because the split slides are IMAGED BY EVERY SCANNER (registered), a per-slide
fingerprint's scanner differences are tissue-controlled -- scanner id is a
well-posed target.  We test three things the user asked for:

  (1) parameter explainability -- does a tissue-robust fingerprint (sharpness +
      colour moments) identify the scanner (leave-slide-out), which feature
      family carries it, and is the paired oracle affine stable across a
      scanner's slides (so scanner-id ~ affine)?
  (2) correction efficacy maintained -- apply the affine of the scanner the
      fingerprint PREDICTS (closed-set source-id contract) and, as a harder
      exploratory bound, a leave-one-scanner-out ridge fingerprint->affine
      regressor; compare range-safe low-band reduction to the paired oracle.
  (3) other correction effect maintained -- recompute the low-band scanner-BACC
      (Piece 3) on the fingerprint-driven correction vs the oracle correction.

Writes ``outputs/exp07_stage1/estimated_affines.json`` (closed-set source-id
per (scanner, slide)) for the UNI job to reuse.
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

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score

# fingerprint layout: [sharp_b0(3), sharp_b1(3), low_mean(3), low_std(3)]
SHARP_IDX = list(range(0, 6))
COLOR_IDX = list(range(6, 12))


def apply_affine(low, affine):
    x = low.permute(0, 2, 3, 1)
    x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
    return (x @ affine).permute(0, 3, 1, 2)


@torch.no_grad()
def fingerprint(pyramid, source):
    """[N,3,H,W] source in [-1,1] -> [N,12] tissue-robust fingerprint."""
    low, bands = pyramid.decompose(source)
    sharp0 = bands[0].abs().mean(dim=(2, 3))
    sharp1 = bands[1].abs().mean(dim=(2, 3))
    low_mean = low.mean(dim=(2, 3))
    low_std = low.std(dim=(2, 3), unbiased=False)
    return torch.cat([sharp0, sharp1, low_mean, low_std], dim=1)


@torch.no_grad()
def coarse_feats(coarse):
    """[N,3,h,w] -> [N,15] per-channel mean/std/q10/q50/q90 (matches Piece 3)."""
    mean = coarse.mean(dim=(-2, -1))
    std = coarse.std(dim=(-2, -1), unbiased=False)
    qs = torch.quantile(coarse.flatten(2), torch.tensor([0.1, 0.5, 0.9]), dim=2)
    qs = qs.permute(1, 2, 0).reshape(coarse.shape[0], -1)
    return torch.cat([mean, std, qs], dim=1)


def scanner_slide_index(dataset):
    groups = {}
    for i, r in enumerate(dataset.records):
        groups.setdefault((r["scanner"], r["slide_id"]), []).append(i)
    return groups


@torch.no_grad()
def slide_pass(dataset, indices, pyramid, cap, rng):
    """Accumulate slide fingerprint + paired oracle affine over a slide's tiles."""
    if len(indices) > cap:
        indices = sorted(rng.choice(indices, cap, replace=False).tolist())
    fp_sum = torch.zeros(12, dtype=torch.float64)
    n = 0
    xtx = torch.zeros(4, 4, dtype=torch.float64)
    xty = torch.zeros(4, 3, dtype=torch.float64)
    for start in range(0, len(indices), 16):
        items = [dataset[i] for i in indices[start:start + 16]]
        source = torch.stack([it["source"] for it in items])
        reference = torch.stack([it["reference"] for it in items])
        q = torch.tensor([it["q_reg"] for it in items], dtype=torch.float64)
        fp_sum += fingerprint(pyramid, source).double().sum(0)
        n += len(items)
        s_low = pyramid.coarsest(source).permute(0, 2, 3, 1).double()
        t_low = pyramid.coarsest(reference).permute(0, 2, 3, 1).double()
        x = torch.cat([s_low, torch.ones_like(s_low[..., :1])], -1)
        w = q[:, None, None, None].sqrt()
        xf = (x * w).reshape(-1, 4)
        yf = (t_low * w).reshape(-1, 3)
        xtx += xf.T @ xf
        xty += xf.T @ yf
    ridge = torch.diag(torch.tensor([1e-4, 1e-4, 1e-4, 0.0], dtype=torch.float64))
    affine = torch.linalg.solve(xtx + ridge, xty).float()
    return (fp_sum / n).float().numpy(), affine.numpy(), indices


@torch.no_grad()
def efficacy(dataset, indices, pyramid, affine):
    """Range-safe (local-omitted, scalar) low-band reduction + corrected feats."""
    affine = torch.as_tensor(affine, dtype=torch.float32)
    raw_e, safe_e, feats = [], [], []
    for start in range(0, len(indices), 16):
        items = [dataset[i] for i in indices[start:start + 16]]
        source = torch.stack([it["source"] for it in items])
        reference = torch.stack([it["reference"] for it in items])
        low, _ = pyramid.decompose(source)
        target_low = pyramid.coarsest(reference)
        corrected = apply_affine(low, affine)
        d, _, _ = low_correction_delta(pyramid, source, corrected)
        _, alpha, _, _ = project_correction(source, d)
        safe = low + alpha.reshape(-1, 1, 1, 1) * (corrected - low)
        raw_e += (low - target_low).abs().flatten(1).mean(1).tolist()
        safe_e += (safe - target_low).abs().flatten(1).mean(1).tolist()
        feats.append(coarse_feats(safe).numpy())
    raw = float(np.mean(raw_e)); safe = float(np.mean(safe_e))
    return {"raw": raw, "safe": safe, "reduction": (raw - safe) / raw,
            "feats": np.concatenate(feats)}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/exp06a_lf16_10slide.yaml")
    parser.add_argument("--baselines", default="outputs/exp06_baselines")
    parser.add_argument("--fp-cap", type=int, default=300)
    parser.add_argument("--eff-cap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    pyramid = FixedLaplacianPyramid(int(cfg.pyramid.levels))
    scanners = [str(s) for s in cfg.target_scanners]
    baseline = {s: np.asarray(json.loads(
        (Path(args.baselines) / f"{s}.json").read_text())["affine_matrix_with_bias"],
        np.float32) for s in scanners}
    rng = np.random.default_rng(args.seed)

    # pool all splits (slides are the deployment unit; tissue is shared across scanners)
    rows = []          # (scanner, slide, fingerprint[12], oracle_affine[4,3], indices)
    for split in ("train", "val", "test"):
        ds = BalancedPairDataset(cfg, split, False)
        for (scanner, slide), idx in scanner_slide_index(ds).items():
            fp, aff, kept = slide_pass(ds, idx, pyramid, args.fp_cap, rng)
            rows.append({"scanner": scanner, "slide": slide, "split": split,
                         "fp": fp, "affine": aff, "ds": ds, "idx": kept})

    fp = np.stack([r["fp"] for r in rows])
    y = np.array([r["scanner"] for r in rows])
    slide = np.array([r["slide"] for r in rows])

    # ---- (1a) explainability: scanner-id from fingerprint, leave-slide-out ----
    def loso_slide_classify(cols):
        pred = np.empty_like(y)
        for s in np.unique(slide):
            te = slide == s
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=3000, class_weight="balanced"))
            clf.fit(fp[~te][:, cols], y[~te])
            pred[te] = clf.predict(fp[te][:, cols])
        return balanced_accuracy_score(y, pred), pred
    id_all, pred_all = loso_slide_classify(list(range(12)))
    id_sharp, _ = loso_slide_classify(SHARP_IDX)
    id_color, _ = loso_slide_classify(COLOR_IDX)

    # ---- (1b) oracle affine stability across a scanner's slides ----
    stability = {}
    for s in scanners:
        A = np.stack([r["affine"] for r in rows if r["scanner"] == s])
        stability[s] = float(np.mean(A.std(0) / (np.abs(A.mean(0)) + 1e-6)))

    # ---- (2a) closed-set source-id efficacy: apply PREDICTED scanner's affine ----
    # ---- (2b) LOSO ridge fingerprint->affine (exploratory, N=2 train scanners) ----
    Y = np.stack([r["affine"].reshape(-1) for r in rows])
    loso_pred_affine = {}
    for s in scanners:
        te = y == s
        reg = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        reg.fit(fp[~te], Y[~te])
        for i in np.flatnonzero(te):
            loso_pred_affine[i] = reg.predict(fp[i:i + 1])[0].reshape(4, 3).astype(np.float32)

    est_affines = {s: {} for s in scanners}
    eff = {s: {"oracle": [], "closedset": [], "loso": []} for s in scanners}
    bacc_feats = {"oracle": {s: [] for s in scanners},
                  "closedset": {s: [] for s in scanners}}
    for i, r in enumerate(rows):
        s, sl, ds, idx = r["scanner"], r["slide"], r["ds"], r["idx"]
        if len(idx) > args.eff_cap:
            idx = sorted(rng.choice(idx, args.eff_cap, replace=False).tolist())
        oracle = efficacy(ds, idx, pyramid, baseline[s])
        predicted = pred_all[i]                       # fingerprint-predicted scanner
        closed = efficacy(ds, idx, pyramid, baseline[predicted])
        loso = efficacy(ds, idx, pyramid, loso_pred_affine[i])
        eff[s]["oracle"].append(oracle["reduction"])
        eff[s]["closedset"].append(closed["reduction"])
        eff[s]["loso"].append(loso["reduction"])
        est_affines[s][sl] = baseline[predicted].tolist()
        bacc_feats["oracle"][s].append(oracle["feats"])
        bacc_feats["closedset"][s].append(closed["feats"])

    def maintained(kind):
        out = {}
        for s in scanners:
            o = float(np.mean(eff[s]["oracle"]))
            e = float(np.mean(eff[s][kind]))
            out[s] = {"oracle_red": o, f"{kind}_red": e,
                      "maintained": e / o if o > 0 else float("nan")}
        return out

    # ---- (3) other effect maintained: low-band scanner-BACC (3-class) ----
    def bacc_3class(source):
        X, lab = [], []
        for s in scanners:
            f = np.concatenate(bacc_feats[source][s])
            X.append(f); lab += [s] * len(f)
        X = np.concatenate(X); lab = np.array(lab)
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=3000, class_weight="balanced"))
        # simple slide-agnostic 5-fold via shuffled split is overkill; report resubstitution-free
        from sklearn.model_selection import cross_val_predict
        pred = cross_val_predict(clf, X, lab, cv=5)
        return balanced_accuracy_score(lab, pred)

    report = {
        "explainability": {
            "scanner_id_bacc_leave_slide_out": {
                "all_features": id_all, "sharpness_only": id_sharp,
                "color_only": id_color, "chance": 1.0 / len(scanners),
            },
            "oracle_affine_slide_stability_relstd": stability,
        },
        "efficacy_maintained": {
            "closed_set_source_id": maintained("closedset"),
            "loso_unseen_scanner_exploratory": maintained("loso"),
        },
        "other_effect_maintained_low_band_bacc": {
            "oracle": bacc_3class("oracle"),
            "closed_set_source_id": bacc_3class("closedset"),
            "chance": 1.0 / len(scanners),
        },
    }
    out_dir = Path(cfg.paths.repo) / "outputs" / "exp07_stage1"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "estimated_affines.json").write_text(
        json.dumps(est_affines, indent=2, sort_keys=True) + "\n")
    (out_dir / "scanner_params.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")

    e = report["explainability"]["scanner_id_bacc_leave_slide_out"]
    print(f"\n=== (1) parameter explainability ===")
    print(f"  scanner-id BACC (leave-slide-out, chance {e['chance']:.2f}): "
          f"all={e['all_features']:.3f}  sharpness={e['sharpness_only']:.3f}  "
          f"color={e['color_only']:.3f}")
    print(f"  oracle affine slide-stability (rel std, lower=stable): "
          + "  ".join(f"{s}={v:.3f}" for s, v in stability.items()))
    print(f"\n=== (2) correction efficacy maintained (range-safe low-band reduction) ===")
    for kind in ("closed_set_source_id", "loso_unseen_scanner_exploratory"):
        print(f"  [{kind}]")
        for s in scanners:
            m = report["efficacy_maintained"][kind][s]
            key = [k for k in m if k.endswith("_red") and k != "oracle_red"][0]
            print(f"    {s:6s} oracle={m['oracle_red']*100:5.1f}%  "
                  f"est={m[key]*100:5.1f}%  maintained={m['maintained']*100:5.0f}%")
    b = report["other_effect_maintained_low_band_bacc"]
    print(f"\n=== (3) other effect: low-band scanner-BACC (chance {b['chance']:.2f}) ===")
    print(f"  oracle-corrected={b['oracle']:.3f}  closedset-corrected={b['closed_set_source_id']:.3f}")
    print(f"\n[exp07-params] wrote {out_dir/'scanner_params.json'} + estimated_affines.json")


if __name__ == "__main__":
    main()

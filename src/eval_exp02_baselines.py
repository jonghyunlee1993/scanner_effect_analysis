"""Exp-02 classical low-band correction baselines on identical tuples.

All methods correct only the coarsest low band (the copied detail is untouched),
are fit on the train slides per scanner, and are scored on the *same* held-out
test tuples. Every requested low correction is reconstructed with source detail
and explicitly clipped to the valid RGB range. This makes the comparison
like-for-like and answers two questions:

  1. Is a plain global affine already near the classical ceiling, or does a
     spatially-varying (quadratic) affine field capture illumination/vignetting
     structure it cannot?  (Decides whether a deep residual has any headroom.)
  2. How do OD-space and global-statistic stain methods (Reinhard, Macenko) --
     the tools that motivated this project by *not* solving cross-scanner batch
     effect -- compare, as a reference floor?

Reported per (scanner, method): low-band MAE (raw / requested / operational
clipped image), reduction, pre-clip violation rate, and clipping distortion.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prenorm.exp01.data import BalancedPairDataset
from prenorm.exp01.frequency import FixedLaplacianPyramid
from utils.config import load_config

EPS = 1e-6
LINEAR_METHODS = ("affine", "affine_ridge", "od_affine", "spatial_affine")


# ---------------------------------------------------------------- feature maps
def _coords(h, w):
    v = torch.linspace(-1, 1, h)
    u = torch.linspace(-1, 1, w)
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    return uu, vv                                        # each [h, w]


def rgb_features(low):
    """[N,3,h,w] coarse in [-1,1] -> per-pixel [..., 4] design [R,G,B,1]."""
    x = low.permute(0, 2, 3, 1)
    return torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)


def od_of(low):
    """Coarse [-1,1] -> optical density OD = -log(clamp(rgb01))."""
    rgb01 = ((low + 1) * 0.5).clamp(EPS, 1.0)
    return (-torch.log(rgb01)).permute(0, 2, 3, 1)       # [N,h,w,3]


def od_features(low):
    od = od_of(low)
    return torch.cat([od, torch.ones_like(od[..., :1])], dim=-1)


def spatial_features(low):
    n, _, h, w = low.shape
    uu, vv = _coords(h, w)
    phi = torch.stack([torch.ones_like(uu), uu, vv, uu * uu, vv * vv, uu * vv], -1)
    phi = phi.reshape(1, h, w, 6).expand(n, h, w, 6)
    rgb1 = rgb_features(low)                              # [N,h,w,4]
    return (rgb1[..., :, None] * phi[..., None, :]).reshape(n, h, w, 24)


def features_for(method, low):
    if method in ("affine", "affine_ridge"):
        return rgb_features(low)
    if method == "od_affine":
        return od_features(low)
    if method == "spatial_affine":
        return spatial_features(low)
    raise ValueError(method)


def target_for(method, target_low):
    if method == "od_affine":
        return od_of(target_low)
    return target_low.permute(0, 2, 3, 1)


# --------------------------------------------------------------------- fitting
def scanner_indices(dataset, scanner):
    return [i for i, r in enumerate(dataset.records) if r["scanner"] == scanner]


@torch.no_grad()
def fit_linear(dataset, indices, pyramid, method, ridge, max_images, seed):
    rng = np.random.default_rng(seed)
    if len(indices) > max_images:
        indices = sorted(rng.choice(indices, max_images, replace=False).tolist())
    xtx = xty = None
    for start in range(0, len(indices), 16):
        items = [dataset[i] for i in indices[start:start + 16]]
        source = torch.stack([it["source"] for it in items])
        reference = torch.stack([it["reference"] for it in items])
        q = torch.tensor([it["q_reg"] for it in items], dtype=torch.float64)
        low = pyramid.coarsest(source)
        target_low = pyramid.coarsest(reference)
        feat = features_for(method, low).double()
        tgt = target_for(method, target_low).double()
        wsqrt = q[:, None, None, None].sqrt()
        k = feat.shape[-1]
        x = (feat * wsqrt).reshape(-1, k)
        y = (tgt * wsqrt).reshape(-1, 3)
        xtx = x.T @ x if xtx is None else xtx + x.T @ x
        xty = x.T @ y if xty is None else xty + x.T @ y
    k = xtx.shape[0]
    reg = ridge * torch.eye(k, dtype=torch.float64)
    reg[-1, -1] = 0.0 if method in ("affine", "affine_ridge") else reg[-1, -1]
    coeff = torch.linalg.solve(xtx + reg, xty).float()   # [k, 3]
    return coeff, len(indices)


@torch.no_grad()
def fit_reference_stats(dataset, indices, pyramid, max_images, seed):
    """Pooled AT2 coarse statistics (Reinhard target) + Macenko target stains."""
    rng = np.random.default_rng(seed)
    if len(indices) > max_images:
        indices = sorted(rng.choice(indices, max_images, replace=False).tolist())
    ref_pixels = []
    for start in range(0, len(indices), 16):
        items = [dataset[i] for i in indices[start:start + 16]]
        reference = torch.stack([it["reference"] for it in items])
        low = pyramid.coarsest(reference)
        ref_pixels.append(low.permute(0, 2, 3, 1).reshape(-1, 3))
    ref = torch.cat(ref_pixels)
    mean = ref.mean(0)
    std = ref.std(0, unbiased=False).clamp_min(1e-4)
    stains, max_c = macenko_target(ref)
    return {"mean": mean, "std": std, "stains": stains, "max_c": max_c}


# --------------------------------------------------------------------- Macenko
def _od_pixels(rgb):
    return (-torch.log(((rgb + 1) * 0.5).clamp(EPS, 1.0)))


def macenko_target(ref_low_pixels, beta=0.15, alpha=1.0):
    """Estimate AT2 (target) stain matrix [3,2] and 99th-pct concentrations [2]."""
    od = _od_pixels(ref_low_pixels)
    od = od[od.sum(1) > beta]
    if od.shape[0] < 50:
        return None, None
    try:
        _, eig = torch.linalg.eigh(torch.cov(od.T))
    except Exception:
        return None, None
    plane = eig[:, 1:3]                                   # top-2 eigenvectors
    proj = od @ plane
    angle = torch.atan2(proj[:, 1], proj[:, 0])
    lo = torch.quantile(angle, alpha / 100.0)
    hi = torch.quantile(angle, 1 - alpha / 100.0)
    v1 = plane @ torch.stack([torch.cos(lo), torch.sin(lo)])
    v2 = plane @ torch.stack([torch.cos(hi), torch.sin(hi)])
    stains = torch.stack([v1, v2] if v1[0] > v2[0] else [v2, v1], dim=1)  # [3,2]
    stains = stains / stains.norm(dim=0, keepdim=True).clamp_min(1e-6)
    conc = torch.linalg.lstsq(stains, od.T).solution     # [2, P]
    max_c = torch.quantile(conc, 0.99, dim=1)
    return stains, max_c.clamp_min(1e-4)


def macenko_apply(low, tgt_stains, tgt_maxc):
    """Per-image Macenko normalisation of a coarse band to the AT2 target."""
    if tgt_stains is None:
        return low
    out = low.clone()
    for i in range(low.shape[0]):
        try:
            pix = low[i].permute(1, 2, 0).reshape(-1, 3)
            od = _od_pixels(pix)
            if (od.sum(1) > 0.15).sum() < 50:
                continue
            s, mc = macenko_target(pix)
            if s is None:
                continue
            conc = torch.linalg.lstsq(s, od.T).solution
            conc = conc * (tgt_maxc[:, None] / mc[:, None].clamp_min(1e-4))
            rgb01 = torch.exp(-(tgt_stains @ conc).T).clamp(0, 1)
            corrected = (rgb01 * 2 - 1).reshape(
                low.shape[-2], low.shape[-1], 3).permute(2, 0, 1)
            if torch.isfinite(corrected).all():
                out[i] = corrected
        except Exception:
            continue
    return out


# ------------------------------------------------------------------- transform
def transform(method, low, params):
    if method == "identity":
        return low
    if method in ("affine", "affine_ridge"):
        return (rgb_features(low) @ params["coeff"]).permute(0, 3, 1, 2)
    if method == "od_affine":
        od_out = od_features(low) @ params["coeff"]      # [N,h,w,3]
        rgb01 = torch.exp(-od_out).clamp(0, 1)
        return (rgb01 * 2 - 1).permute(0, 3, 1, 2)
    if method == "spatial_affine":
        return (spatial_features(low) @ params["coeff"]).permute(0, 3, 1, 2)
    if method == "reinhard_rgb":
        m = low.mean(dim=(-2, -1), keepdim=True)
        s = low.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(1e-4)
        tm = params["mean"][None, :, None, None]
        ts = params["std"][None, :, None, None]
        return (low - m) / s * ts + tm
    if method == "macenko":
        return macenko_apply(low, params.get("stains"), params.get("max_c"))
    raise ValueError(method)


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def evaluate(scanner, dataset, indices, pyramid, method_params):
    methods = list(method_params)
    acc = {m: {k: [] for k in (
        "raw_mae",
        "method_mae",
        "clipped_mae",
        "clip",
        "clip_pixel_mae",
    )}
           for m in methods}
    per_slide = {m: {} for m in methods}
    for start in range(0, len(indices), 16):
        items = [dataset[i] for i in indices[start:start + 16]]
        source = torch.stack([it["source"] for it in items])
        reference = torch.stack([it["reference"] for it in items])
        low, bands = pyramid.decompose(source)
        target_low = pyramid.coarsest(reference)
        raw_mae = (low - target_low).abs().flatten(1).mean(1)
        for m in methods:
            corrected = transform(m, low, method_params[m])
            preclip = pyramid.reconstruct(corrected, bands)
            clip = ((preclip < -1 - EPS) | (preclip > 1 + EPS)).float().flatten(1).mean(1)
            output = preclip.clamp(-1.0, 1.0)
            clipped_coarse = pyramid.coarsest(output)
            acc[m]["raw_mae"] += raw_mae.tolist()
            acc[m]["method_mae"] += (corrected - target_low).abs().flatten(1).mean(1).tolist()
            acc[m]["clipped_mae"] += (
                (clipped_coarse - target_low).abs().flatten(1).mean(1).tolist()
            )
            acc[m]["clip"] += clip.tolist()
            acc[m]["clip_pixel_mae"] += (
                (output - preclip).abs().flatten(1).mean(1).tolist()
            )
            for it, e in zip(
                items,
                (clipped_coarse - target_low).abs().flatten(1).mean(1),
            ):
                per_slide[m].setdefault(it["slide_id"], []).append(float(e))

    report = {"scanner": scanner, "n_images": int(len(indices)), "methods": {}}
    for m in methods:
        a = {k: np.asarray(v, np.float64) for k, v in acc[m].items()}
        raw = a["raw_mae"].mean()
        method_mae = a["method_mae"].mean()
        clipped = a["clipped_mae"].mean()
        requested_reduction = (raw - method_mae) / raw
        clipped_reduction = (raw - clipped) / raw
        report["methods"][m] = {
            "raw_mae": float(raw), "method_mae": float(method_mae),
            "clipped_mae": float(clipped),
            "method_reduction": float(requested_reduction),
            "clipped_reduction": float(clipped_reduction),
            "clip_fraction_mean": float(a["clip"].mean()),
            "clip_fraction_max": float(a["clip"].max()),
            "clip_pixel_mae": float(a["clip_pixel_mae"].mean()),
            "slide_clipped_mae": {
                s: float(np.mean(v)) for s, v in per_slide[m].items()
            },
        }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/exp01a_lf16_10slide.yaml")
    parser.add_argument("--max-train-images", type=int, default=2048)
    parser.add_argument("--max-test-images", type=int, default=512)
    parser.add_argument(
        "--output", default="outputs/exp02_stage1/baselines.json"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    pyramid = FixedLaplacianPyramid(int(cfg.pyramid.levels))
    train = BalancedPairDataset(cfg, "train", False)
    test = BalancedPairDataset(cfg, "test", False)
    rng = np.random.default_rng(args.seed)
    seed = int(cfg.runtime.seed)

    reports = {}
    for scanner in [str(s) for s in cfg.target_scanners]:
        tr = scanner_indices(train, scanner)
        params = {"identity": {}, "reinhard_rgb": {}, "macenko": {}}
        for method in LINEAR_METHODS:
            ridge = 1e-4 if method != "affine_ridge" else 1e-1
            coeff, _ = fit_linear(train, tr, pyramid, method, ridge,
                                  args.max_train_images, seed)
            params[method] = {"coeff": coeff}
        stats = fit_reference_stats(train, tr, pyramid, args.max_train_images, seed)
        params["reinhard_rgb"] = {"mean": stats["mean"], "std": stats["std"]}
        params["macenko"] = {"stains": stats["stains"], "max_c": stats["max_c"]}

        te = scanner_indices(test, scanner)
        if len(te) > args.max_test_images:
            te = sorted(rng.choice(te, args.max_test_images, replace=False).tolist())
        ordered = {m: params[m] for m in (
            "identity", "affine", "affine_ridge", "od_affine",
            "spatial_affine", "reinhard_rgb", "macenko",
        )}
        report = evaluate(scanner, test, te, pyramid, ordered)
        reports[scanner] = report

        print(f"\n=== {scanner}  (n={report['n_images']}) ===")
        print(f"  {'method':14s} {'clip_mae':>9s} {'m.red':>7s} {'clip.red':>9s} "
              f"{'clip%':>7s} {'clipΔ':>8s}")
        for m, r in report["methods"].items():
            print(
                f"  {m:14s} {r['clipped_mae']:9.4f} "
                f"{r['method_reduction']*100:6.1f}% "
                f"{r['clipped_reduction']*100:8.1f}% "
                f"{r['clip_fraction_mean']*100:6.2f}% "
                f"{r['clip_pixel_mae']:8.5f}"
            )

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = Path(cfg.paths.repo) / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(reports, indent=2, sort_keys=True) + "\n")
    print(f"\n[exp02-baselines] wrote {out_path}")


if __name__ == "__main__":
    main()

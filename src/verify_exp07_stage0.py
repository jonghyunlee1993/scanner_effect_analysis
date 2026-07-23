"""Exp-07 Stage-0 verification: does range-safe scalar projection retain the
scanner-specific affine's low-band efficacy, and what binds ``alpha``?

Reuses the already-fitted Exp-06 classical affine coefficients
(``outputs/exp06_baselines/<scanner>.json``, contract A: scanner-known,
train-fit) and applies §5.3 identity-direction composite projection on held-out
test tuples.  Reports, per scanner:

- raw / unsafe-affine / safe-projected low-band MAE and efficacy survival,
- alpha distribution and the fraction of range-binding images,
- the §5.3 binding decomposition (pre-existing |x|~1 vs correction magnitude),
- architectural safety (projected clip fraction, copied-detail MAE).

No model, no training -- pure numerics on a modest CPU sample.
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


def apply_affine(low: torch.Tensor, affine: torch.Tensor) -> torch.Tensor:
    x = low.permute(0, 2, 3, 1)
    x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
    return (x @ affine).permute(0, 3, 1, 2)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def summarize(values) -> dict:
    v = np.asarray(values, np.float64)
    return {
        "mean": float(v.mean()), "median": float(np.median(v)),
        "q05": float(np.quantile(v, 0.05)), "q95": float(np.quantile(v, 0.95)),
        "max": float(v.max()), "min": float(v.min()),
    }


@torch.no_grad()
def evaluate_scanner(scanner, dataset, indices, affine, pyramid):
    fields = {k: [] for k in (
        "raw_mae", "unsafe_mae", "safe_mae", "alpha", "alpha_raw",
        "bind_abs_x", "bind_abs_d", "d_inf", "frac_extreme",
        "safe_clip_frac", "copied_detail_mae",
    )}
    for start in range(0, len(indices), 16):
        items = [dataset[i] for i in indices[start:start + 16]]
        source = torch.stack([it["source"] for it in items])
        reference = torch.stack([it["reference"] for it in items])

        low, bands = pyramid.decompose(source)
        target_low = pyramid.coarsest(reference)
        corrected_unsafe = apply_affine(low, affine)

        d, _, _ = low_correction_delta(pyramid, source, corrected_unsafe)
        output, alpha, alpha_raw, binding = project_correction(source, d)
        a = alpha.reshape(-1, 1, 1, 1)
        corrected_safe = low + a * (corrected_unsafe - low)

        # architectural safety of the projected output
        clip = ((output < -1.0 - 1e-6) | (output > 1.0 + 1e-6)).float().mean(dim=(1, 2, 3))
        zeros = [torch.zeros_like(b) for b in bands]
        low_only = pyramid.reconstruct(corrected_safe, zeros)
        input_detail = pyramid.reconstruct(torch.zeros_like(low), bands)
        copied_detail_mae = (output - low_only - input_detail).abs().flatten(1).mean(1)

        def img_mae(pred):
            return (pred - target_low).abs().flatten(1).mean(1)

        fields["raw_mae"] += img_mae(low).tolist()
        fields["unsafe_mae"] += img_mae(corrected_unsafe).tolist()
        fields["safe_mae"] += img_mae(corrected_safe).tolist()
        fields["alpha"] += alpha.reshape(-1).tolist()
        fields["alpha_raw"] += alpha_raw.tolist()
        fields["bind_abs_x"] += binding["abs_x"].tolist()
        fields["bind_abs_d"] += binding["abs_d"].tolist()
        fields["d_inf"] += binding["d_inf"].tolist()
        fields["frac_extreme"] += binding["frac_extreme"].tolist()
        fields["safe_clip_frac"] += clip.tolist()
        fields["copied_detail_mae"] += copied_detail_mae.tolist()

    a = {k: np.asarray(v, np.float64) for k, v in fields.items()}
    raw_m, unsafe_m, safe_m = a["raw_mae"].mean(), a["unsafe_mae"].mean(), a["safe_mae"].mean()
    unsafe_red = (raw_m - unsafe_m) / raw_m
    safe_red = (raw_m - safe_m) / raw_m
    binding_mask = a["alpha_raw"] < 1.0

    report = {
        "scanner": scanner,
        "n_images": int(len(indices)),
        "low_mae": {
            "raw": float(raw_m), "unsafe_affine": float(unsafe_m),
            "safe_projected": float(safe_m),
            "unsafe_reduction": float(unsafe_red),
            "safe_reduction": float(safe_red),
            "efficacy_survival": float(safe_red / unsafe_red) if unsafe_red > 0 else float("nan"),
        },
        "alpha": {
            **summarize(a["alpha"]),
            "frac_lt_0.05": float((a["alpha"] < 0.05).mean()),
            "frac_lt_0.5": float((a["alpha"] < 0.5).mean()),
            "frac_range_binding": float(binding_mask.mean()),
        },
        "binding_decomposition": {
            "n_binding": int(binding_mask.sum()),
            "median_abs_x_at_bind": float(np.median(a["bind_abs_x"][binding_mask]))
            if binding_mask.any() else float("nan"),
            "median_abs_d_at_bind": float(np.median(a["bind_abs_d"][binding_mask]))
            if binding_mask.any() else float("nan"),
            "corr_alpha_vs_d_inf": pearson(a["alpha"], a["d_inf"]),
            "corr_alpha_vs_frac_extreme": pearson(a["alpha"], a["frac_extreme"]),
        },
        "safety": {
            "max_clip_fraction": float(a["safe_clip_frac"].max()),
            "mean_clip_fraction": float(a["safe_clip_frac"].mean()),
            "max_copied_detail_mae": float(a["copied_detail_mae"].max()),
        },
    }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/exp06a_lf16_10slide.yaml")
    parser.add_argument("--baselines", default="outputs/exp06_baselines")
    parser.add_argument("--max-images", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    pyramid = FixedLaplacianPyramid(int(cfg.pyramid.levels))
    dataset = BalancedPairDataset(cfg, "test", False)
    rng = np.random.default_rng(args.seed)

    reports = {}
    for scanner in [str(s) for s in cfg.target_scanners]:
        affine_path = Path(args.baselines) / f"{scanner}.json"
        affine = torch.tensor(
            json.loads(affine_path.read_text())["affine_matrix_with_bias"],
            dtype=torch.float32,
        )
        indices = [i for i, r in enumerate(dataset.records) if r["scanner"] == scanner]
        if len(indices) > args.max_images:
            indices = sorted(rng.choice(indices, args.max_images, replace=False).tolist())
        report = evaluate_scanner(scanner, dataset, indices, affine, pyramid)
        reports[scanner] = report

        m = report["low_mae"]; al = report["alpha"]; bd = report["binding_decomposition"]
        sf = report["safety"]
        print(f"\n=== {scanner}  (n={report['n_images']}) ===")
        print(f"  low MAE   raw={m['raw']:.4f}  unsafe={m['unsafe_affine']:.4f}  "
              f"safe={m['safe_projected']:.4f}")
        print(f"  reduction unsafe={m['unsafe_reduction']*100:5.1f}%  "
              f"safe={m['safe_reduction']*100:5.1f}%  "
              f"survival={m['efficacy_survival']*100:5.1f}%")
        print(f"  alpha     mean={al['mean']:.3f} median={al['median']:.3f} q05={al['q05']:.3f}  "
              f"frac<0.05={al['frac_lt_0.05']*100:.1f}%  binding={al['frac_range_binding']*100:.1f}%")
        print(f"  binding   |x|_bind={bd['median_abs_x_at_bind']:.3f}  "
              f"corr(alpha,d_inf)={bd['corr_alpha_vs_d_inf']:+.2f}  "
              f"corr(alpha,frac_extreme)={bd['corr_alpha_vs_frac_extreme']:+.2f}")
        print(f"  safety    max_clip={sf['max_clip_fraction']*100:.4f}%  "
              f"copied_detail_mae<={sf['max_copied_detail_mae']:.2e}")

    out_dir = Path(cfg.paths.repo) / "outputs" / "exp07_stage0"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "range_safe_affine.json"
    out_path.write_text(json.dumps(reports, indent=2, sort_keys=True) + "\n")
    print(f"\n[exp07-stage0] wrote {out_path}")


if __name__ == "__main__":
    main()

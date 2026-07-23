"""Exp-07 Stage-1: does the coefficient-wise (local) range projection recover the
efficacy the global-scalar projection loses on VERSA/S60?

Stage-0 showed the scalar identity-direction projection stays provably range-safe
but its single per-image ``alpha`` is choked by one near-saturated copied-detail
pixel -- S60 kept only 43% of the affine's low-band efficacy.  §5.3's second step
replaces the scalar with the coarse-space QP

    min || c - l' ||^2  s.t.  R(c, bands) in [-1, 1],

solved by ``local_range_project`` (penalised GD + exact scalar backstop).  This
script applies BOTH projections to the *same* held-out test tuples with the same
contract-A affine and reports, per scanner, the range-safe low-band MAE and the
efficacy survival of each -- so the comparison is like-for-like.

No training; pure numerics on a CPU sample.
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
from prenorm.exp07.projection import (
    project_correction,
    low_correction_delta,
    local_range_project,
)
from utils.config import load_config


def apply_affine(low: torch.Tensor, affine: torch.Tensor) -> torch.Tensor:
    x = low.permute(0, 2, 3, 1)
    x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
    return (x @ affine).permute(0, 3, 1, 2)


def summarize(values) -> dict:
    v = np.asarray(values, np.float64)
    return {
        "mean": float(v.mean()), "median": float(np.median(v)),
        "q05": float(np.quantile(v, 0.05)), "q95": float(np.quantile(v, 0.95)),
    }


@torch.no_grad()
def _batch_low(pyramid, dataset, indices, affine):
    """Yield (source, low, target_low, corrected_unsafe) per 16-image batch."""
    for start in range(0, len(indices), 16):
        items = [dataset[i] for i in indices[start:start + 16]]
        source = torch.stack([it["source"] for it in items])
        reference = torch.stack([it["reference"] for it in items])
        low = pyramid.coarsest(source)
        target_low = pyramid.coarsest(reference)
        corrected_unsafe = apply_affine(low, affine)
        yield source, low, target_low, corrected_unsafe


def evaluate_scanner(scanner, dataset, indices, affine, pyramid, steps, lr, penalty):
    fields = {k: [] for k in (
        "raw_mae", "unsafe_mae", "scalar_mae", "local_mae",
        "scalar_alpha", "local_alpha", "local_pre_maxviol", "local_pre_feasible",
    )}
    for source, low, target_low, corrected_unsafe in _batch_low(
        pyramid, dataset, indices, affine
    ):
        with torch.no_grad():
            d, _, _ = low_correction_delta(pyramid, source, corrected_unsafe)
            _, alpha_s, _, _ = project_correction(source, d)
            scalar_coarse = low + alpha_s.reshape(-1, 1, 1, 1) * (corrected_unsafe - low)

        local = local_range_project(
            pyramid, source, corrected_unsafe, steps=steps, lr=lr, penalty=penalty
        )

        def img_mae(pred):
            return (pred - target_low).abs().flatten(1).mean(1)

        fields["raw_mae"] += img_mae(low).tolist()
        fields["unsafe_mae"] += img_mae(corrected_unsafe).tolist()
        fields["scalar_mae"] += img_mae(scalar_coarse).tolist()
        fields["local_mae"] += img_mae(local["coarse"]).tolist()
        fields["scalar_alpha"] += alpha_s.reshape(-1).tolist()
        fields["local_alpha"] += local["alpha"].reshape(-1).tolist()
        fields["local_pre_maxviol"] += local["pre_max_violation"].tolist()
        fields["local_pre_feasible"] += local["pre_feasible"].float().tolist()

    a = {k: np.asarray(v, np.float64) for k, v in fields.items()}
    raw = a["raw_mae"].mean()
    unsafe = a["unsafe_mae"].mean()
    scalar = a["scalar_mae"].mean()
    local = a["local_mae"].mean()
    unsafe_red = (raw - unsafe) / raw

    def survival(safe_mean):
        red = (raw - safe_mean) / raw
        return red, (red / unsafe_red if unsafe_red > 0 else float("nan"))

    scalar_red, scalar_surv = survival(scalar)
    local_red, local_surv = survival(local)
    return {
        "scanner": scanner, "n_images": int(len(indices)),
        "low_mae": {
            "raw": float(raw), "unsafe_affine": float(unsafe),
            "scalar_safe": float(scalar), "local_safe": float(local),
            "unsafe_reduction": float(unsafe_red),
            "scalar_reduction": float(scalar_red), "local_reduction": float(local_red),
            "scalar_survival": float(scalar_surv), "local_survival": float(local_surv),
            "survival_gain": float(local_surv - scalar_surv),
        },
        "scalar_alpha": summarize(a["scalar_alpha"]),
        "local_alpha": summarize(a["local_alpha"]),
        "local_feasibility": {
            "frac_feasible_pre_backstop": float(a["local_pre_feasible"].mean()),
            "median_pre_max_violation": float(np.median(a["local_pre_maxviol"])),
            "q95_pre_max_violation": float(np.quantile(a["local_pre_maxviol"], 0.95)),
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/exp06a_lf16_10slide.yaml")
    parser.add_argument("--baselines", default="outputs/exp06_baselines")
    parser.add_argument("--max-images", type=int, default=512)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--penalty", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    pyramid = FixedLaplacianPyramid(int(cfg.pyramid.levels))
    dataset = BalancedPairDataset(cfg, "test", False)
    rng = np.random.default_rng(args.seed)

    reports = {}
    for scanner in [str(s) for s in cfg.target_scanners]:
        affine = torch.tensor(
            json.loads((Path(args.baselines) / f"{scanner}.json").read_text())
            ["affine_matrix_with_bias"], dtype=torch.float32,
        )
        indices = [i for i, r in enumerate(dataset.records) if r["scanner"] == scanner]
        if len(indices) > args.max_images:
            indices = sorted(rng.choice(indices, args.max_images, replace=False).tolist())
        report = evaluate_scanner(
            scanner, dataset, indices, affine, pyramid,
            args.steps, args.lr, args.penalty,
        )
        reports[scanner] = report
        m = report["low_mae"]
        f = report["local_feasibility"]
        print(f"\n=== {scanner}  (n={report['n_images']}) ===")
        print(f"  low MAE   raw={m['raw']:.4f}  unsafe={m['unsafe_affine']:.4f}  "
              f"scalar={m['scalar_safe']:.4f}  local={m['local_safe']:.4f}")
        print(f"  survival  scalar={m['scalar_survival']*100:5.1f}%  "
              f"local={m['local_survival']*100:5.1f}%  "
              f"(gain {m['survival_gain']*100:+.1f} pts)")
        print(f"  alpha     scalar_mean={report['scalar_alpha']['mean']:.3f}  "
              f"local_mean={report['local_alpha']['mean']:.3f}")
        print(f"  local QP  feasible_pre_backstop={f['frac_feasible_pre_backstop']*100:.1f}%  "
              f"med_viol={f['median_pre_max_violation']:.4f}")

    out_dir = Path(cfg.paths.repo) / "outputs" / "exp07_stage1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "local_projection.json"
    out_path.write_text(json.dumps(reports, indent=2, sort_keys=True) + "\n")
    print(f"\n[exp07-local] wrote {out_path}")


if __name__ == "__main__":
    main()

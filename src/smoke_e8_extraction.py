"""Exercise the encoding-side paths before the encoding array spends GPU hours.

Two things only run at extraction time and nowhere else: loading a trained
checkpoint back, and rebuilding the frozen RF1U base at an encoder's native FOV
that is not the 256 px the network trained on.  Both are cheap to test and
expensive to discover broken after four encoders have queued.

Uses whatever checkpoints already exist, so it can run as soon as one fold of
one arm has finished.  No PFM encoder is loaded and no feature is written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from e8_paired_residual import FrozenBase, ResidualCorrector, load_base_state
from train_e8_residual import configuration_tag, grid_configurations, run_directory


FOVS = (224, 256, 512)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="gt450")
    parser.add_argument("--runs", default="outputs/e8_residual/runs")
    parser.add_argument("--fold", type=int, default=0)
    return parser.parse_args()


def find_checkpoints(runs: Path, target: str, fold: int):
    found = {}
    for configuration in grid_configurations():
        directory = run_directory(
            runs, target, configuration["arm"], fold, configuration_tag(configuration)
        )
        summary = directory / "summary.json"
        if not summary.exists():
            continue
        path = Path(json.loads(summary.read_text())["checkpoint"])
        if path.exists():
            found.setdefault(configuration["arm"], path)
    return found


def main():
    args = parse_args()
    checkpoints = find_checkpoints(Path(args.runs), args.target, args.fold)
    if not checkpoints:
        raise SystemExit(f"no finished checkpoint under {args.runs}")

    for arm, path in checkpoints.items():
        model = ResidualCorrector.load_from_checkpoint(path, map_location="cpu")
        model.eval()
        print(f"\n{arm}: loaded {path.name} "
              f"(arm={model.hparams.arm}, fold={model.hparams.outer_fold})")
        if model.hparams.arm != arm:
            raise SystemExit(f"{arm}: checkpoint carries arm {model.hparams.arm}")

        for fov in FOVS:
            if fov != 256:
                model.base = FrozenBase(
                    load_base_state(args.target, model.hparams.outer_fold, fov=fov)
                )
            rgb = torch.rand(2, fov, fov, 3).clamp(0.05, 0.95)
            position = torch.tensor([0, 3])
            with torch.no_grad():
                report = model.correct(rgb, position)
            output = report["output"]
            drift = float((output - report["base"]).abs().mean())
            if output.shape != rgb.shape:
                raise SystemExit(f"{arm}@{fov}: output shape {output.shape}")
            if not torch.isfinite(output).all():
                raise SystemExit(f"{arm}@{fov}: non-finite output")
            if float(output.min()) < 0.0 or float(output.max()) > 1.0:
                raise SystemExit(f"{arm}@{fov}: output outside [0, 1]")
            print(f"  fov {fov:3d}: output {tuple(output.shape)}  "
                  f"mean |output - base| {drift:.5f}  "
                  f"projection {float(report['projection_fraction'].mean()):.5f}")
            if drift <= 0.0:
                raise SystemExit(f"{arm}@{fov}: trained model reproduces the base exactly")

    print("\nevery checkpoint loads and corrects at every native FOV")


if __name__ == "__main__":
    main()

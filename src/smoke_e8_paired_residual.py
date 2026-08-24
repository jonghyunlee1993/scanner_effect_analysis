"""End-to-end exercise of every E8 arm: train step, backward, validation, epoch end.

This lives outside `tests/` because it depends on the materialized RF1U
statistics rather than on synthetic input alone, and it needs more memory than
the login node has.  Its job is to touch every code path the GPU array will
touch, for **both** arms, before any GPU time is spent.

That distinction is why it exists.  The first grid submission ran only `gainfield`
configurations in its opening wave, so a batch-axis error on the `free` arm's
pixel term survived a green test suite and a green first wave, and only
surfaced when the whole array failed.  A smoke run that exercises every arm is
cheaper than a wave of GPU jobs.
"""

from __future__ import annotations

import argparse

import torch

from build_e8_cache import E8_CROP
from e8_paired_residual import E8_ARMS, ResidualCorrector


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="gt450")
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--batch", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(0)
    count = args.batch

    for arm in E8_ARMS:
        model = ResidualCorrector(
            arm=arm,
            target=args.target,
            outer_fold=args.outer_fold,
            lambda_pixel=0.1,
            lambda_identity=0.03 if arm == "gainfield" else 0.0,
        )
        source = torch.randint(20, 235, (count, E8_CROP, E8_CROP, 3), dtype=torch.uint8)
        target = torch.randint(20, 235, (count, E8_CROP, E8_CROP, 3), dtype=torch.uint8)
        position = torch.arange(count) % 5
        shift = torch.tensor([[0, 0], [1, -2], [-3, 3], [2, 1]])[:count]
        batch = (source, target, position, shift)

        model.train()
        loss = model.training_step(batch, 0)
        loss.backward()
        gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.net.parameters()
            if parameter.grad is not None
        )

        model.eval()
        model.on_validation_epoch_start()
        with torch.no_grad():
            model.validation_step(batch, 0)
        model.on_validation_epoch_end()

        counts = model._paired_counts.clamp_min(1.0)
        output_mae = float((model._paired["output"] / counts).mean())
        base_mae = float((model._paired["base"] / counts).mean())

        print(
            f"{arm:9s} loss {float(loss):.4f}  gradient-sum {gradient:.3e}  "
            f"paired_mae output {output_mae:.6f} base {base_mae:.6f}"
        )
        if not torch.isfinite(loss):
            raise SystemExit(f"{arm}: non-finite loss")
        if gradient <= 0:
            raise SystemExit(f"{arm}: no gradient reached the network")
        if abs(output_mae - base_mae) > 1e-6:
            raise SystemExit(f"{arm}: the zero-initialised head is not the identity")

    print("\nboth arms complete: train step, backward, validation, epoch end")


if __name__ == "__main__":
    main()

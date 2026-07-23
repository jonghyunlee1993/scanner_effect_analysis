"""Fixed one-slide/three-scanner overfit check for Exp-06."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prenorm.exp06.data import Exp06DataModule
from prenorm.exp06.module import Exp06Module
from utils.config import load_config


def move(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()}


@torch.no_grad()
def measure(module, batches):
    module.eval()
    report = {}
    for batch in batches:
        terms = module.losses(batch)
        source = batch["source"]
        n = len(source)
        context = batch["context_source"].unsqueeze(0).expand(n, -1, -1, -1, -1)
        mask = batch["context_mask"].unsqueeze(0).expand(n, -1)
        result = module.model(source, context, mask, clip=False)
        input_detail = module.model.pyramid.reconstruct(
            torch.zeros_like(result["source_low"]), result["copied_bands"]
        )
        low_only = module.model.pyramid.reconstruct(
            result["corrected_low"], [torch.zeros_like(band) for band in result["copied_bands"]]
        )
        detail_error = ((result["preclip"] - low_only) - input_detail).abs().max()
        report[str(batch["scanner"])] = {
            key: float(value) for key, value in terms.items()
        }
        report[str(batch["scanner"])]["preclip_copied_detail_max_error"] = float(detail_error)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/exp06a_lf16_10slide.yaml")
    parser.add_argument("--slide", default="12.5_17")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--idempotence-weight", type=float, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    final_idempotence_weight = float(
        args.idempotence_weight if args.idempotence_weight is not None
        else cfg.loss.idempotence
    )
    if args.warmup_steps:
        cfg.loss.idempotence = 0.5
    else:
        cfg.loss.idempotence = final_idempotence_weight
    cfg.loader.num_workers = 0
    cfg.loader.batches_per_epoch = 3
    data = Exp06DataModule(cfg)
    data.setup("fit")
    for scanner in cfg.target_scanners:
        if args.slide not in data.train_ds.scanner_to_slides[str(scanner)]:
            raise ValueError(f"{scanner} has no training pairs on {args.slide}")
        data.train_ds.scanner_to_slides[str(scanner)] = [args.slide]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batches = [move(batch, device) for batch in data.train_dataloader()]
    if {str(batch["scanner"]) for batch in batches} != set(cfg.target_scanners):
        raise RuntimeError("overfit fixture did not cover every target scanner")
    module = Exp06Module(cfg).to(device)
    initial = measure(module, batches)
    module.train()
    optimizer = module.configure_optimizers()
    for step in range(args.steps):
        if step == args.warmup_steps and args.warmup_steps:
            cfg.loss.idempotence = final_idempotence_weight
            print(f"[overfit] idempotence weight -> {final_idempotence_weight}", flush=True)
        batch = batches[step % len(batches)]
        optimizer.zero_grad(set_to_none=True)
        loss = module.losses(batch)["total"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), float(cfg.optim.grad_clip))
        optimizer.step()
        if (step + 1) % 50 == 0:
            print(f"[overfit] step={step + 1} total={float(loss):.6f}", flush=True)
    final = measure(module, batches)
    report = {"slide": args.slide, "steps": args.steps,
              "warmup_steps": args.warmup_steps,
              "idempotence_weight": final_idempotence_weight,
              "initial": initial, "final": final}
    output = Path(args.output or (Path(cfg.paths.output_dir) / "overfit"))
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    torch.save({"state_dict": module.state_dict(), "exp06_metadata": module.checkpoint_metadata()},
               output / "model.pt")
    print(f"[overfit] report -> {output}")


if __name__ == "__main__":
    main()

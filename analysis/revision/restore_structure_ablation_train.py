#!/usr/bin/env python3
"""RV-P0d (iii): Pix2Pix structure-loss ablation matched to the existing GT450 baseline.

Recovered verbatim from `.Trash/2026-09-29_paper_code_cleanup/src/manuscript_completion/discussion_structure_train.py`
(RV-P0d iii). Run-time byte-code cache: `.Trash/2026-09-26_paper_refactor/generated/src/
manuscript_completion/__pycache__/discussion_structure_train.cpython-39.pyc` (same source size and mtime).
Paths are relative to the repository root. The caller passes an output location under
`analysis/revision/results/provenance_restoration/`; see `restore_structure_ablation.py`.
Only the module docstring differs from the recovered file.
"""


from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
import torch.nn.functional as F

from scanner_gan.data import PairedScannerDataset, SlideBalancedSampler, assign_split_roles, load_sample_index
from scanner_gan.models import GANLoss, build_pix2pix_models
from scanner_gan.train import (
    Pix2PixConfig, _amp_dtype, _loader, central_crop, pass_learning_rate,
    save_checkpoint, seed_everything, set_learning_rate, set_requires_grad,
    sha256, utc_now, validate, write_json,
)


BASE = Path(
    "outputs/scanner_batch_effect_analysis_2026-09-17/06_learned_baselines/"
    "09_bidirectional_full_training/01_training/runs/at2_to_gt450"
)


def gradient_l1(generated: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    weights = generated.new_tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
    g = (generated.float() * weights).sum(dim=1)
    s = (source.float() * weights).sum(dim=1)
    return 0.5 * (
        F.l1_loss(g[:, :, 1:] - g[:, :, :-1], s[:, :, 1:] - s[:, :, :-1])
        + F.l1_loss(g[:, 1:, :] - g[:, :-1, :], s[:, 1:, :] - s[:, :-1, :])
    )


def train(fold: int, output: Path, lambda_gradient: float) -> None:
    source_manifest = json.loads(
        next((BASE / f"fold_{fold}").glob("seed_*/run_manifest.json")).read_text()
    )
    config = replace(Pix2PixConfig(**source_manifest["config"]), output_dir=str(output.resolve()))
    if config.source_scanner != "at2" or config.target_scanner != "gt450":
        raise ValueError("baseline direction mismatch")
    if config.validation_fold != (fold + 1) % 5 or config.test_fold != fold:
        raise ValueError("fold mismatch")
    if lambda_gradient <= 0:
        raise ValueError("positive gradient loss required")
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    amp_dtype = _amp_dtype(config.amp)

    index = load_sample_index(config.sample_index, verify_caches=False)
    split = assign_split_roles(index, config.test_fold, config.validation_fold)
    train_data = PairedScannerDataset(
        split, "at2", target_scanner="gt450", split_role="train", augment=True,
        seed=config.seed, crop_size=config.crop_size,
        allow_bidirectional_reference_pair=True,
    )
    val_data = PairedScannerDataset(
        split, "at2", target_scanner="gt450", split_role="validation", augment=False,
        seed=config.seed, crop_size=config.crop_size,
        allow_bidirectional_reference_pair=True,
    )
    train_sampler = SlideBalancedSampler(
        train_data.sample_index, locations_per_slide=config.locations_per_train_slide,
        seed=config.seed,
    )
    val_sampler = SlideBalancedSampler(
        val_data.sample_index, locations_per_slide=config.locations_per_validation_slide,
        seed=config.seed + 1,
    )
    train_loader = _loader(
        train_data, train_sampler, batch_size=config.batch_size,
        workers=config.num_workers, device=device,
    )
    val_loader = _loader(
        val_data, val_sampler, batch_size=config.validation_batch_size,
        workers=config.num_workers, device=device,
    )
    generator, discriminator = build_pix2pix_models(
        ngf=config.ngf, ndf=config.ndf, device=device,
    )
    adversarial = GANLoss("vanilla").to(device)
    optimizer_g = torch.optim.Adam(
        generator.parameters(), lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(), lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
    )
    manifest = {
        "training_version": "pix2pix_scanner_reference_v2",
        "direction": "at2->gt450", "status": "running", "started_utc": utc_now(),
        "config": asdict(config), "sample_index": str(Path(config.sample_index).resolve()),
        "sample_index_sha256": sha256(config.sample_index),
        "discussion_ablation": {
            "comparison": "existing_no_explicit_gradient_loss_baseline",
            "lambda_source_gradient_l1": lambda_gradient,
            "checkpoint_selection": "inner_validation_target_l1_only",
        },
    }
    write_json(output / "run_manifest.json", manifest)
    best_l1 = float("inf")
    best_path = None
    global_step = 0
    started = time.monotonic()
    for pass_index in range(config.max_passes):
        train_sampler.set_epoch(pass_index)
        val_sampler.set_epoch(0)
        set_learning_rate((optimizer_g, optimizer_d), pass_learning_rate(config, pass_index))
        generator.train()
        discriminator.train()
        losses = []
        for batch in train_loader:
            source = batch["source_image"].to(device, non_blocking=True)
            source_valid = batch["source_valid"].to(device, non_blocking=True)
            target_valid = batch["target_valid"].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None
            ):
                generated = generator(source)
                generated_valid = central_crop(generated, config.crop_size)
            set_requires_grad(discriminator, True)
            optimizer_d.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None
            ):
                real_logits = discriminator(torch.cat((source_valid, target_valid), dim=1))
                fake_logits = discriminator(
                    torch.cat((source_valid, generated_valid.detach()), dim=1)
                )
                loss_d = 0.5 * (
                    adversarial(real_logits, True) + adversarial(fake_logits, False)
                )
            loss_d.backward()
            optimizer_d.step()
            set_requires_grad(discriminator, False)
            optimizer_g.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None
            ):
                fake_logits = discriminator(torch.cat((source_valid, generated_valid), dim=1))
                loss_gan = adversarial(fake_logits, True)
                loss_pixel = F.l1_loss(generated_valid, target_valid)
            loss_edge = gradient_l1(generated_valid, source_valid)
            loss_g = loss_gan + config.lambda_l1 * loss_pixel + lambda_gradient * loss_edge
            loss_g.backward()
            optimizer_g.step()
            global_step += 1
            losses.append(float(loss_edge.detach()))
        completed = pass_index + 1
        if completed % config.selection_every == 0 or completed == config.max_passes:
            validation = validate(generator, val_loader, device, amp_dtype)
            row = {
                "completed_passes": completed, "global_step": global_step,
                "mean_train_gradient_l1": float(sum(losses) / len(losses)),
                "elapsed_seconds": time.monotonic() - started,
                **{f"validation_{key}": value for key, value in validation.items()},
            }
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps(row, sort_keys=True), flush=True)
            if validation["l1_normalized"] < best_l1:
                best_l1 = validation["l1_normalized"]
                best_path = output / "checkpoints/best_image_only.pt"
                save_checkpoint(
                    best_path, generator=generator, discriminator=discriminator,
                    optimizer_g=None, optimizer_d=None, config=config,
                    completed_passes=completed, global_step=global_step,
                    validation=validation,
                )
    manifest.update({
        "status": "complete", "completed_utc": utc_now(),
        "completed_passes": config.max_passes, "global_step": global_step,
        "elapsed_seconds": time.monotonic() - started,
        "best_image_only_checkpoint": str(best_path),
        "best_validation_l1": best_l1,
        "best_checkpoint_sha256": sha256(best_path),
    })
    write_json(output / "run_manifest.json", manifest)
    train_data.close()
    val_data.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lambda-gradient", type=float, default=100.0)
    args = parser.parse_args()
    train(args.fold, args.output, args.lambda_gradient)

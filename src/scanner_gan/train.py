#!/usr/bin/env python3
"""Train paired Pix2Pix scanner-reference baselines.

This module deliberately contains no UNI calls.  Checkpoint choice is based on
inner-validation image losses and safety diagnostics only; frozen UNI is opened
after predictions and checkpoints have been locked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .data import (
    DEFAULT_VALID_CROP_SIZE,
    PRIMARY_SOURCE_SCANNERS,
    PairedScannerDataset,
    SlideBalancedSampler,
    assign_split_roles,
    load_sample_index,
)
from .models import GANLoss, build_pix2pix_models, count_parameters


TRAINING_VERSION = "pix2pix_scanner_reference_v2"
LEGACY_TRAINING_VERSION = "pix2pix_scanner_to_at2_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def training_version(config: "Pix2PixConfig") -> str:
    if (
        config.target_scanner.strip().lower() == "at2"
        and config.source_scanner.strip().lower() in PRIMARY_SOURCE_SCANNERS
        and config.selection_policy == "source_fidelity_then_l1"
    ):
        return LEGACY_TRAINING_VERSION
    return TRAINING_VERSION


@dataclass(frozen=True)
class Pix2PixConfig:
    sample_index: str
    source_scanner: str
    target_scanner: str
    test_fold: int
    validation_fold: int
    output_dir: str
    seed: int = 20260917
    max_passes: int = 8
    max_steps: int = 0
    locations_per_train_slide: int = 100
    locations_per_validation_slide: int = 40
    batch_size: int = 16
    validation_batch_size: int = 32
    num_workers: int = 8
    learning_rate: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    lambda_l1: float = 100.0
    ngf: int = 64
    ndf: int = 64
    crop_size: int = DEFAULT_VALID_CROP_SIZE
    amp: str = "bfloat16"
    save_every: int = 2
    constant_passes: int = 4
    selection_policy: str = "source_fidelity_then_l1"
    selection_every: int = 1


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def central_crop(image: Tensor, size: int) -> Tensor:
    if image.ndim != 4:
        raise ValueError(f"expected NCHW tensor, got {tuple(image.shape)}")
    height, width = image.shape[-2:]
    if size <= 0 or size > height or size > width:
        raise ValueError(f"invalid central crop {size} for {height}x{width}")
    top = (height - size) // 2
    left = (width - size) // 2
    return image[..., top : top + size, left : left + size]


def _gradient_vector(image: Tensor) -> Tensor:
    gray = 0.2989 * image[:, 0] + 0.5870 * image[:, 1] + 0.1140 * image[:, 2]
    dx = gray[..., :, 1:] - gray[..., :, :-1]
    dy = gray[..., 1:, :] - gray[..., :-1, :]
    # Remove unmatched edge rows/columns before concatenation.
    return torch.cat((dx[..., :-1, :].flatten(1), dy[..., :, :-1].flatten(1)), dim=1)


def gradient_ncc(first: Tensor, second: Tensor, eps: float = 1e-8) -> Tensor:
    a = _gradient_vector(first.float())
    b = _gradient_vector(second.float())
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    numerator = (a * b).sum(dim=1)
    denominator = torch.sqrt(
        (a.square().sum(dim=1) * b.square().sum(dim=1)).clamp_min(eps)
    )
    return numerator / denominator


def saturation_fraction(image: Tensor) -> Tensor:
    unit = image.float().add(1.0).mul(0.5)
    return ((unit <= 1.0 / 255.0) | (unit >= 254.0 / 255.0)).float().mean(dim=(1, 2, 3))


def pass_learning_rate(config: Pix2PixConfig, pass_index: int) -> float:
    constant_passes = min(config.constant_passes, config.max_passes)
    if pass_index < constant_passes or config.max_passes <= constant_passes:
        factor = 1.0
    else:
        factor = (config.max_passes - pass_index) / (
            config.max_passes - constant_passes
        )
    return config.learning_rate * max(float(factor), 0.0)


def set_learning_rate(
    optimizers: tuple[torch.optim.Optimizer, ...], value: float
) -> None:
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["lr"] = value


def set_requires_grad(module: nn.Module, value: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(value)


def _amp_dtype(name: str) -> torch.dtype | None:
    normalized = name.strip().lower()
    if normalized in {"none", "off", "float32"}:
        return None
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    raise ValueError(f"unknown amp mode {name!r}")


def _loader(
    dataset: PairedScannerDataset,
    sampler: SlideBalancedSampler,
    *,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        drop_last=False,
    )


@torch.inference_mode()
def validate(
    generator: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> dict[str, float]:
    generator.eval()
    totals = {
        "l1": 0.0,
        "mse": 0.0,
        "source_gradient_ncc": 0.0,
        "target_gradient_ncc": 0.0,
        "saturation": 0.0,
    }
    count = 0
    for batch in loader:
        source = batch["source_image"].to(device, non_blocking=True)
        source_valid = batch["source_valid"].to(device, non_blocking=True)
        target_valid = batch["target_valid"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_dtype is not None,
        ):
            generated = generator(source)
        generated_valid = central_crop(generated.float(), target_valid.shape[-1])
        batch_count = source.shape[0]
        totals["l1"] += (
            F.l1_loss(generated_valid, target_valid, reduction="mean").item()
            * batch_count
        )
        totals["mse"] += (
            F.mse_loss(generated_valid, target_valid, reduction="mean").item()
            * batch_count
        )
        totals["source_gradient_ncc"] += (
            gradient_ncc(generated_valid, source_valid).sum().item()
        )
        totals["target_gradient_ncc"] += (
            gradient_ncc(generated_valid, target_valid).sum().item()
        )
        totals["saturation"] += saturation_fraction(generated_valid).sum().item()
        count += batch_count
    if count == 0:
        raise ValueError("validation loader yielded no samples")
    l1 = totals["l1"] / count
    mse_unit = (totals["mse"] / count) / 4.0
    return {
        "samples": int(count),
        "l1_normalized": float(l1),
        "psnr_db": float(-10.0 * math.log10(max(mse_unit, 1e-12))),
        "source_gradient_ncc": float(totals["source_gradient_ncc"] / count),
        "target_gradient_ncc": float(totals["target_gradient_ncc"] / count),
        "saturation_fraction": float(totals["saturation"] / count),
    }


def save_checkpoint(
    path: Path,
    *,
    generator: nn.Module,
    discriminator: nn.Module,
    optimizer_g: torch.optim.Optimizer | None,
    optimizer_d: torch.optim.Optimizer | None,
    config: Pix2PixConfig,
    completed_passes: int,
    global_step: int,
    validation: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    payload = {
        "training_version": training_version(config),
        "mapping": f"{config.source_scanner}->{config.target_scanner}",
        "source_scanner": config.source_scanner,
        "target_scanner": config.target_scanner,
        "test_fold": config.test_fold,
        "validation_fold": config.validation_fold,
        "completed_passes": completed_passes,
        "global_step": global_step,
        "validation": validation,
        "config": asdict(config),
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
    }
    if optimizer_g is not None and optimizer_d is not None:
        payload["optimizer_g"] = optimizer_g.state_dict()
        payload["optimizer_d"] = optimizer_d.state_dict()
    torch.save(payload, temporary)
    temporary.replace(path)


def train(config: Pix2PixConfig) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("formal Pix2Pix training requires CUDA")
    source_scanner = config.source_scanner.strip().lower()
    target_scanner = config.target_scanner.strip().lower()
    legacy_direction = (
        target_scanner == "at2" and source_scanner in PRIMARY_SOURCE_SCANNERS
    )
    bidirectional_reference = (source_scanner, target_scanner) in {
        ("gt450", "at2"),
        ("at2", "gt450"),
    }
    if not legacy_direction and not bidirectional_reference:
        raise ValueError(
            "supported mappings are primary scanner->AT2 and the bidirectional "
            f"GT450<->AT2 reference pair; got {source_scanner}->{target_scanner}"
        )
    if config.validation_fold != (config.test_fold + 1) % 5:
        raise ValueError("validation fold must equal (test_fold + 1) mod 5")
    if (
        config.max_passes < 1
        or config.batch_size < 1
        or config.lambda_l1 <= 0
        or config.constant_passes < 0
        or config.save_every < 0
        or config.selection_every < 1
    ):
        raise ValueError("invalid training configuration")
    if config.selection_policy not in {"source_fidelity_then_l1", "target_l1"}:
        raise ValueError(f"unknown selection policy {config.selection_policy!r}")

    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    amp_dtype = _amp_dtype(config.amp)

    full_index = load_sample_index(config.sample_index, verify_caches=False)
    split = assign_split_roles(full_index, config.test_fold, config.validation_fold)
    train_dataset = PairedScannerDataset(
        split,
        source_scanner,
        target_scanner=target_scanner,
        split_role="train",
        augment=True,
        seed=config.seed,
        crop_size=config.crop_size,
        allow_bidirectional_reference_pair=bidirectional_reference,
    )
    validation_dataset = PairedScannerDataset(
        split,
        source_scanner,
        target_scanner=target_scanner,
        split_role="validation",
        augment=False,
        seed=config.seed,
        crop_size=config.crop_size,
        allow_bidirectional_reference_pair=bidirectional_reference,
    )
    train_sampler = SlideBalancedSampler(
        train_dataset.sample_index,
        locations_per_slide=config.locations_per_train_slide,
        seed=config.seed,
    )
    validation_sampler = SlideBalancedSampler(
        validation_dataset.sample_index,
        locations_per_slide=config.locations_per_validation_slide,
        seed=config.seed + 1,
    )
    train_loader = _loader(
        train_dataset,
        train_sampler,
        batch_size=config.batch_size,
        workers=config.num_workers,
        device=device,
    )
    validation_loader = _loader(
        validation_dataset,
        validation_sampler,
        batch_size=config.validation_batch_size,
        workers=config.num_workers,
        device=device,
    )

    generator, discriminator = build_pix2pix_models(
        ngf=config.ngf,
        ndf=config.ndf,
        device=device,
    )
    adversarial_loss = GANLoss("vanilla").to(device)
    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
    )

    manifest = {
        "training_version": training_version(config),
        "status": "running",
        "started_utc": utc_now(),
        "direction": f"{source_scanner}->{target_scanner}",
        "selection_boundary": (
            "image-only inner-validation target L1; UNI forbidden until lock"
            if config.selection_policy == "target_l1"
            else "image-only inner validation; UNI forbidden until lock"
        ),
        "sample_index": str(Path(config.sample_index).resolve()),
        "sample_index_sha256": sha256(config.sample_index),
        "train_slides": int(train_dataset.sample_index["slide_id"].nunique()),
        "validation_slides": int(validation_dataset.sample_index["slide_id"].nunique()),
        "test_slides": int(
            split.loc[split["split_role"] == "test", "slide_id"].nunique()
        ),
        "train_samples_per_pass": len(train_sampler),
        "validation_samples_per_pass": len(validation_sampler),
        "generator_parameters": count_parameters(generator),
        "discriminator_parameters": count_parameters(discriminator),
        "config": asdict(config),
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(device),
    }
    write_json(output_dir / "run_manifest.json", manifest)

    global_step = 0
    best_l1 = float("inf")
    best_checkpoint = ""
    history: list[dict[str, Any]] = []
    started = time.monotonic()
    reached_max_steps = False

    for pass_index in range(config.max_passes):
        train_sampler.set_epoch(pass_index)
        validation_sampler.set_epoch(
            0
        )  # exactly the same inner-val locations each pass
        learning_rate = pass_learning_rate(config, pass_index)
        set_learning_rate((optimizer_g, optimizer_d), learning_rate)
        generator.train()
        discriminator.train()
        sums = {"d": 0.0, "g": 0.0, "g_gan": 0.0, "g_l1": 0.0}
        seen = 0
        pass_started = time.monotonic()

        for batch in train_loader:
            source = batch["source_image"].to(device, non_blocking=True)
            source_valid = batch["source_valid"].to(device, non_blocking=True)
            target_valid = batch["target_valid"].to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_dtype is not None,
            ):
                generated = generator(source)
                generated_valid = central_crop(generated, config.crop_size)

            set_requires_grad(discriminator, True)
            optimizer_d.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_dtype is not None,
            ):
                real_logits = discriminator(
                    torch.cat((source_valid, target_valid), dim=1)
                )
                fake_logits = discriminator(
                    torch.cat((source_valid, generated_valid.detach()), dim=1)
                )
                loss_d = 0.5 * (
                    adversarial_loss(real_logits, True)
                    + adversarial_loss(fake_logits, False)
                )
            loss_d.backward()
            optimizer_d.step()

            set_requires_grad(discriminator, False)
            optimizer_g.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_dtype is not None,
            ):
                fake_logits_for_g = discriminator(
                    torch.cat((source_valid, generated_valid), dim=1)
                )
                loss_g_gan = adversarial_loss(fake_logits_for_g, True)
                loss_g_l1 = F.l1_loss(generated_valid, target_valid)
                loss_g = loss_g_gan + config.lambda_l1 * loss_g_l1
            loss_g.backward()
            optimizer_g.step()

            batch_count = source.shape[0]
            seen += batch_count
            global_step += 1
            sums["d"] += loss_d.detach().float().item() * batch_count
            sums["g"] += loss_g.detach().float().item() * batch_count
            sums["g_gan"] += loss_g_gan.detach().float().item() * batch_count
            sums["g_l1"] += loss_g_l1.detach().float().item() * batch_count
            if config.max_steps > 0 and global_step >= config.max_steps:
                reached_max_steps = True
                break

        validation = validate(generator, validation_loader, device, amp_dtype)
        completed_passes = pass_index + 1
        row = {
            "completed_passes": completed_passes,
            "global_step": global_step,
            "learning_rate": learning_rate,
            "train_samples": seen,
            "train_d_loss": sums["d"] / max(seen, 1),
            "train_g_loss": sums["g"] / max(seen, 1),
            "train_g_gan": sums["g_gan"] / max(seen, 1),
            "train_g_l1": sums["g_l1"] / max(seen, 1),
            "pass_seconds": time.monotonic() - pass_started,
            "elapsed_seconds": time.monotonic() - started,
            "max_memory_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            **{f"validation_{key}": value for key, value in validation.items()},
        }
        history.append(row)
        with (output_dir / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)

        eligible = config.selection_policy == "target_l1" or (
            validation["source_gradient_ncc"] >= 0.90
            and validation["saturation_fraction"] <= 0.10
        )
        is_selection_candidate = (
            completed_passes % config.selection_every == 0
            or completed_passes == config.max_passes
            or reached_max_steps
        )
        if (
            is_selection_candidate
            and eligible
            and validation["l1_normalized"] < best_l1
        ):
            best_l1 = validation["l1_normalized"]
            best_path = output_dir / "checkpoints" / "best_image_only.pt"
            save_checkpoint(
                best_path,
                generator=generator,
                discriminator=discriminator,
                optimizer_g=None,
                optimizer_d=None,
                config=config,
                completed_passes=completed_passes,
                global_step=global_step,
                validation=validation,
            )
            best_checkpoint = str(best_path)
        if config.save_every > 0 and (
            completed_passes % config.save_every == 0 or reached_max_steps
        ):
            save_checkpoint(
                output_dir / "checkpoints" / f"pass_{completed_passes:02d}.pt",
                generator=generator,
                discriminator=discriminator,
                optimizer_g=optimizer_g,
                optimizer_d=optimizer_d,
                config=config,
                completed_passes=completed_passes,
                global_step=global_step,
                validation=validation,
            )
        if reached_max_steps:
            break

    final_validation = {
        key.removeprefix("validation_"): value
        for key, value in history[-1].items()
        if key.startswith("validation_")
    }
    final_path = output_dir / "checkpoints" / "final.pt"
    save_checkpoint(
        final_path,
        generator=generator,
        discriminator=discriminator,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        config=config,
        completed_passes=len(history),
        global_step=global_step,
        validation=final_validation,
    )
    manifest.update(
        {
            "status": "complete",
            "completed_utc": utc_now(),
            "completed_passes": len(history),
            "global_step": global_step,
            "elapsed_seconds": time.monotonic() - started,
            "best_image_only_checkpoint": best_checkpoint,
            "best_validation_l1": None if not math.isfinite(best_l1) else best_l1,
            "final_checkpoint": str(final_path),
            "final_checkpoint_sha256": sha256(final_path),
            "note": (
                "The best checkpoint minimizes inner-validation target L1 under the configured "
                "image-only policy. It remains provisional until the locked outer-test image and "
                "hallucination audit. No UNI result was consulted."
            ),
        }
    )
    write_json(output_dir / "run_manifest.json", manifest)
    train_dataset.close()
    validation_dataset.close()
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-index", type=Path, required=True)
    parser.add_argument(
        "--source-scanner",
        choices=tuple((*PRIMARY_SOURCE_SCANNERS, "at2")),
        required=True,
    )
    parser.add_argument("--target-scanner", choices=("at2", "gt450"), default="at2")
    parser.add_argument("--test-fold", type=int, choices=range(5), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--max-passes", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--locations-per-train-slide", type=int, default=100)
    parser.add_argument("--locations-per-validation-slide", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--validation-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lambda-l1", type=float, default=100.0)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument(
        "--amp", choices=("bfloat16", "float16", "none"), default="bfloat16"
    )
    parser.add_argument("--save-every", type=int, default=2)
    parser.add_argument("--constant-passes", type=int, default=4)
    parser.add_argument("--selection-every", type=int, default=1)
    parser.add_argument(
        "--selection-policy",
        choices=("source_fidelity_then_l1", "target_l1"),
        default="source_fidelity_then_l1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validation_fold = (args.test_fold + 1) % 5
    config = Pix2PixConfig(
        sample_index=str(args.sample_index.resolve()),
        source_scanner=args.source_scanner,
        target_scanner=args.target_scanner,
        test_fold=args.test_fold,
        validation_fold=validation_fold,
        output_dir=str(args.output_dir.resolve()),
        seed=args.seed,
        max_passes=args.max_passes,
        max_steps=args.max_steps,
        locations_per_train_slide=args.locations_per_train_slide,
        locations_per_validation_slide=args.locations_per_validation_slide,
        batch_size=args.batch_size,
        validation_batch_size=args.validation_batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        lambda_l1=args.lambda_l1,
        ngf=args.ngf,
        ndf=args.ndf,
        amp=args.amp,
        save_every=args.save_every,
        constant_passes=args.constant_passes,
        selection_policy=args.selection_policy,
        selection_every=args.selection_every,
    )
    result = train(config)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train unpaired bidirectional CycleGAN on scanner domains.

The training loader never exposes same-location pairs. Source and target
patches are independently, slide-balanced sampled from different physical
slides. Checkpoint selection is UNI-blind and pair-blind: it minimizes a
robust marginal distance over the ten locked image endpoint families in both
directions. Paired targets are opened only after a checkpoint is frozen.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from final_image_study import (
    BANDS_CYC_PER_UM,
    PROVISIONAL_MPP,
    SPECTRAL_BINS,
    frequency_geometry,
    image_metrics,
    radial_mean,
)

from .data import (
    UnpairedScannerDataset,
    UnpairedSlideBalancedSampler,
    assign_split_roles,
    load_sample_index,
)
from .models import (
    GANLoss,
    ImagePool,
    ResnetGenerator,
    build_cyclegan_models,
    count_parameters,
)
from .train import gradient_ncc, saturation_fraction, set_learning_rate, set_requires_grad


TRAINING_VERSION = "cyclegan_unpaired_bidirectional_v1"
ENDPOINT_NAMES = (
    "lab_l_mean",
    "lab_a_mean",
    "lab_b_mean",
    "log_mean_od",
    "log_lab_l_sd",
    "log_od_sd",
    "log_gradient_rms",
    "frequency_low_mid",
    "frequency_mid",
    "frequency_high",
)


@dataclass(frozen=True)
class CycleGANConfig:
    sample_index: str
    domain_a: str
    domain_b: str
    test_fold: int
    validation_fold: int
    output_dir: str
    seed: int = 20261917
    max_passes: int = 200
    max_steps: int = 0
    locations_per_train_slide: int = 100
    locations_per_validation_slide: int = 40
    batch_size: int = 16
    validation_batch_size: int = 32
    num_workers: int = 4
    learning_rate: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    lambda_cycle: float = 10.0
    lambda_identity: float = 5.0
    pool_size: int = 50
    ngf: int = 64
    ndf: int = 64
    n_blocks: int = 9
    amp: str = "bfloat16"
    constant_passes: int = 100
    selection_every: int = 5
    save_every: int = 0
    crop_size: int = 252


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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def amp_dtype(name: str) -> torch.dtype | None:
    normalized = name.strip().lower()
    if normalized in {"none", "off", "float32"}:
        return None
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float16", "fp16"}:
        return torch.float16
    raise ValueError(f"unknown AMP mode {name!r}")


def pass_learning_rate(config: CycleGANConfig, pass_index: int) -> float:
    constant = min(config.constant_passes, config.max_passes)
    if pass_index < constant or config.max_passes <= constant:
        factor = 1.0
    else:
        factor = (config.max_passes - pass_index) / (config.max_passes - constant)
    return config.learning_rate * max(float(factor), 0.0)


def normalized_to_uint8(images: Tensor) -> np.ndarray:
    value = (
        images.detach()
        .to(device="cpu", dtype=torch.float32)
        .clamp(-1.0, 1.0)
        .add(1.0)
        .mul(127.5)
        .round()
        .to(torch.uint8)
    )
    return np.ascontiguousarray(value.permute(0, 2, 3, 1).numpy())


def absolute_endpoint_features(images: np.ndarray) -> np.ndarray:
    """Measure the ten image properties used for pair-blind selection."""

    if images.ndim != 4 or images.shape[1:] != (256, 256, 3):
        raise ValueError(f"expected uint8 NHWC 256px images, got {images.shape}")
    scalar, od, _ = image_metrics(images)
    window_1d = np.hanning(256).astype(np.float32)
    window = window_1d[:, None] * window_1d[None, :]
    centered = od - od.mean(axis=(1, 2), keepdims=True)
    power = np.abs(np.fft.fft2(centered * window[None], axes=(-2, -1))) ** 2
    radial_index, valid, counts, frequency_px = frequency_geometry(
        256, SPECTRAL_BINS
    )
    frequency_um = frequency_px / PROVISIONAL_MPP
    radial = np.empty((len(images), SPECTRAL_BINS), dtype=np.float64)
    for row in range(len(images)):
        radial[row] = radial_mean(power[row], radial_index, valid, counts)
    log_amplitude = 0.5 * np.log(np.maximum(radial, 1e-20))
    normalization = (frequency_um >= 0.03) & (frequency_um <= 0.10)
    log_amplitude -= log_amplitude[:, normalization].mean(axis=1, keepdims=True)
    bands = []
    for bounds in BANDS_CYC_PER_UM.values():
        selected = (frequency_um >= bounds[0]) & (frequency_um < bounds[1])
        if not selected.any():
            raise ValueError(f"empty frequency band {bounds}")
        bands.append(log_amplitude[:, selected].mean(axis=1))
    return np.column_stack(
        [
            scalar[:, 0],
            scalar[:, 1],
            scalar[:, 2],
            np.log(np.maximum(scalar[:, 4], 1e-12)),
            np.log(np.maximum(scalar[:, 3], 1e-12)),
            np.log(np.maximum(scalar[:, 5], 1e-12)),
            np.log(np.maximum(scalar[:, 6], 1e-12)),
            *bands,
        ]
    ).astype(np.float32)


def marginal_endpoint_distance(
    generated: np.ndarray, real: np.ndarray
) -> dict[str, Any]:
    """Robust location-and-spread distance between two unpaired domains."""

    if generated.shape != real.shape or generated.ndim != 2:
        raise ValueError("generated and real endpoint matrices must match")
    q_generated = np.quantile(generated, [0.25, 0.5, 0.75], axis=0)
    q_real = np.quantile(real, [0.25, 0.5, 0.75], axis=0)
    real_iqr = np.maximum(q_real[2] - q_real[0], 1e-6)
    generated_iqr = np.maximum(q_generated[2] - q_generated[0], 1e-6)
    scale = np.maximum(real_iqr / 1.349, 1e-6)
    location = (q_generated[1] - q_real[1]) / scale
    spread = np.log(generated_iqr / real_iqr)
    component = np.sqrt(location**2 + spread**2)
    return {
        "distance": float(np.sqrt(np.mean(component**2))),
        "location_rms": float(np.sqrt(np.mean(location**2))),
        "spread_rms": float(np.sqrt(np.mean(spread**2))),
        "endpoint_components": {
            name: float(value)
            for name, value in zip(ENDPOINT_NAMES, component)
        },
    }


def make_loader(
    dataset: UnpairedScannerDataset,
    sampler: UnpairedSlideBalancedSampler,
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
    generator_a_to_b: nn.Module,
    generator_b_to_a: nn.Module,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype | None,
) -> dict[str, Any]:
    generator_a_to_b.eval()
    generator_b_to_a.eval()
    generated_b_features: list[np.ndarray] = []
    real_b_features: list[np.ndarray] = []
    generated_a_features: list[np.ndarray] = []
    real_a_features: list[np.ndarray] = []
    totals = {
        "cycle_a_l1": 0.0,
        "cycle_b_l1": 0.0,
        "identity_a_l1": 0.0,
        "identity_b_l1": 0.0,
        "a_to_b_source_gradient_ncc": 0.0,
        "b_to_a_source_gradient_ncc": 0.0,
        "a_to_b_saturation": 0.0,
        "b_to_a_saturation": 0.0,
    }
    count = 0
    same_slide = 0
    same_location = 0
    for batch in loader:
        real_a = batch["source_image"].to(device, non_blocking=True)
        real_b = batch["target_image"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=dtype,
            enabled=dtype is not None,
        ):
            fake_b = generator_a_to_b(real_a)
            fake_a = generator_b_to_a(real_b)
            reconstructed_a = generator_b_to_a(fake_b)
            reconstructed_b = generator_a_to_b(fake_a)
            identity_a = generator_b_to_a(real_a)
            identity_b = generator_a_to_b(real_b)
        batch_count = int(real_a.shape[0])
        count += batch_count
        totals["cycle_a_l1"] += F.l1_loss(
            reconstructed_a, real_a, reduction="mean"
        ).item() * batch_count
        totals["cycle_b_l1"] += F.l1_loss(
            reconstructed_b, real_b, reduction="mean"
        ).item() * batch_count
        totals["identity_a_l1"] += F.l1_loss(
            identity_a, real_a, reduction="mean"
        ).item() * batch_count
        totals["identity_b_l1"] += F.l1_loss(
            identity_b, real_b, reduction="mean"
        ).item() * batch_count
        totals["a_to_b_source_gradient_ncc"] += gradient_ncc(
            fake_b.float(), real_a
        ).sum().item()
        totals["b_to_a_source_gradient_ncc"] += gradient_ncc(
            fake_a.float(), real_b
        ).sum().item()
        totals["a_to_b_saturation"] += saturation_fraction(fake_b).sum().item()
        totals["b_to_a_saturation"] += saturation_fraction(fake_a).sum().item()
        same_slide += int(np.asarray(batch["accidental_same_slide"]).sum())
        same_location += int(np.asarray(batch["accidental_same_location"]).sum())
        generated_b_features.append(
            absolute_endpoint_features(normalized_to_uint8(fake_b))
        )
        real_b_features.append(
            absolute_endpoint_features(normalized_to_uint8(real_b))
        )
        generated_a_features.append(
            absolute_endpoint_features(normalized_to_uint8(fake_a))
        )
        real_a_features.append(
            absolute_endpoint_features(normalized_to_uint8(real_a))
        )
    if count == 0:
        raise ValueError("validation loader yielded no samples")
    a_to_b = marginal_endpoint_distance(
        np.concatenate(generated_b_features), np.concatenate(real_b_features)
    )
    b_to_a = marginal_endpoint_distance(
        np.concatenate(generated_a_features), np.concatenate(real_a_features)
    )
    return {
        "samples": count,
        "accidental_same_slide_pairs": same_slide,
        "accidental_same_location_pairs": same_location,
        "marginal_score": float((a_to_b["distance"] + b_to_a["distance"]) / 2.0),
        "a_to_b_marginal": a_to_b,
        "b_to_a_marginal": b_to_a,
        **{key: float(value / count) for key, value in totals.items()},
    }


def save_checkpoint(
    path: Path,
    *,
    networks: tuple[nn.Module, nn.Module, nn.Module, nn.Module],
    optimizers: tuple[torch.optim.Optimizer, torch.optim.Optimizer] | None,
    config: CycleGANConfig,
    completed_passes: int,
    global_step: int,
    validation: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    generator_a_to_b, generator_b_to_a, discriminator_a, discriminator_b = networks
    payload: dict[str, Any] = {
        "training_version": TRAINING_VERSION,
        "mapping": f"{config.domain_a}<->{config.domain_b}",
        "domain_a": config.domain_a,
        "domain_b": config.domain_b,
        "test_fold": config.test_fold,
        "validation_fold": config.validation_fold,
        "completed_passes": completed_passes,
        "global_step": global_step,
        "validation": dict(validation),
        "config": asdict(config),
        "generator_a_to_b": generator_a_to_b.state_dict(),
        "generator_b_to_a": generator_b_to_a.state_dict(),
        "discriminator_a": discriminator_a.state_dict(),
        "discriminator_b": discriminator_b.state_dict(),
    }
    if optimizers is not None:
        payload["optimizer_g"] = optimizers[0].state_dict()
        payload["optimizer_d"] = optimizers[1].state_dict()
    torch.save(payload, temporary)
    temporary.replace(path)


def train(config: CycleGANConfig) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("formal CycleGAN training requires CUDA")
    domain_a = config.domain_a.strip().lower()
    domain_b = config.domain_b.strip().lower()
    if (domain_a, domain_b) != ("gt450", "at2"):
        raise ValueError("the formal bidirectional domains are GT450 and AT2")
    if config.validation_fold != (config.test_fold + 1) % 5:
        raise ValueError("validation fold must equal (test_fold + 1) mod 5")
    if min(
        config.max_passes,
        config.batch_size,
        config.locations_per_train_slide,
        config.selection_every,
    ) < 1:
        raise ValueError("invalid positive training configuration")
    if config.lambda_cycle <= 0 or config.lambda_identity < 0:
        raise ValueError("CycleGAN loss weights are invalid")

    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dtype = amp_dtype(config.amp)

    full_index = load_sample_index(config.sample_index, verify_caches=False)
    split = assign_split_roles(full_index, config.test_fold, config.validation_fold)
    train_dataset = UnpairedScannerDataset(
        split,
        domain_a,
        domain_b,
        split_role="train",
        augment=True,
        seed=config.seed,
    )
    validation_dataset = UnpairedScannerDataset(
        split,
        domain_a,
        domain_b,
        split_role="validation",
        augment=False,
        seed=config.seed + 1,
    )
    train_sampler = UnpairedSlideBalancedSampler(
        train_dataset.sample_index,
        locations_per_slide=config.locations_per_train_slide,
        seed=config.seed,
    )
    validation_sampler = UnpairedSlideBalancedSampler(
        validation_dataset.sample_index,
        locations_per_slide=config.locations_per_validation_slide,
        seed=config.seed + 1,
    )
    train_loader = make_loader(
        train_dataset,
        train_sampler,
        batch_size=config.batch_size,
        workers=config.num_workers,
        device=device,
    )
    validation_loader = make_loader(
        validation_dataset,
        validation_sampler,
        batch_size=config.validation_batch_size,
        workers=config.num_workers,
        device=device,
    )

    networks = build_cyclegan_models(
        ngf=config.ngf,
        ndf=config.ndf,
        n_blocks=config.n_blocks,
        device=device,
    )
    generator_a_to_b, generator_b_to_a, discriminator_a, discriminator_b = networks
    adversarial_loss = GANLoss("lsgan").to(device)
    optimizer_g = torch.optim.Adam(
        list(generator_a_to_b.parameters()) + list(generator_b_to_a.parameters()),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
    )
    optimizer_d = torch.optim.Adam(
        list(discriminator_a.parameters()) + list(discriminator_b.parameters()),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
    )
    fake_a_pool = ImagePool(config.pool_size, seed=config.seed + 101)
    fake_b_pool = ImagePool(config.pool_size, seed=config.seed + 202)

    manifest: dict[str, Any] = {
        "training_version": TRAINING_VERSION,
        "status": "running",
        "started_utc": utc_now(),
        "direction": f"{domain_a}<->{domain_b}",
        "selection_boundary": (
            "pair-blind marginal image endpoints on inner validation; "
            "same-location keys and UNI forbidden until lock"
        ),
        "sample_index": str(Path(config.sample_index).resolve()),
        "sample_index_sha256": sha256(config.sample_index),
        "train_slides": int(train_dataset.sample_index["slide_id"].nunique()),
        "validation_slides": int(
            validation_dataset.sample_index["slide_id"].nunique()
        ),
        "test_slides": int(
            split.loc[split["split_role"] == "test", "slide_id"].nunique()
        ),
        "train_samples_per_pass_per_domain": len(train_sampler),
        "validation_samples_per_check_per_domain": len(validation_sampler),
        "generator_parameters_each": count_parameters(generator_a_to_b),
        "discriminator_parameters_each": count_parameters(discriminator_a),
        "config": asdict(config),
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(device),
    }
    write_json(output_dir / "run_manifest.json", manifest)

    global_step = 0
    best_score = float("inf")
    best_checkpoint = ""
    history: list[dict[str, Any]] = []
    started = time.monotonic()
    reached_max_steps = False

    for pass_index in range(config.max_passes):
        train_sampler.set_epoch(pass_index)
        learning_rate = pass_learning_rate(config, pass_index)
        set_learning_rate((optimizer_g, optimizer_d), learning_rate)
        for network in networks:
            network.train()
        sums = {
            "d_a": 0.0,
            "d_b": 0.0,
            "g": 0.0,
            "g_a_to_b": 0.0,
            "g_b_to_a": 0.0,
            "cycle_a": 0.0,
            "cycle_b": 0.0,
            "identity_a": 0.0,
            "identity_b": 0.0,
        }
        seen = 0
        accidental_same_slide = 0
        accidental_same_location = 0
        pass_started = time.monotonic()

        for batch in train_loader:
            real_a = batch["source_image"].to(device, non_blocking=True)
            real_b = batch["target_image"].to(device, non_blocking=True)
            accidental_same_slide += int(
                np.asarray(batch["accidental_same_slide"]).sum()
            )
            accidental_same_location += int(
                np.asarray(batch["accidental_same_location"]).sum()
            )

            set_requires_grad(discriminator_a, False)
            set_requires_grad(discriminator_b, False)
            optimizer_g.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=dtype is not None,
            ):
                fake_b = generator_a_to_b(real_a)
                reconstructed_a = generator_b_to_a(fake_b)
                fake_a = generator_b_to_a(real_b)
                reconstructed_b = generator_a_to_b(fake_a)
                identity_a = generator_b_to_a(real_a)
                identity_b = generator_a_to_b(real_b)
                loss_g_a_to_b = adversarial_loss(discriminator_b(fake_b), True)
                loss_g_b_to_a = adversarial_loss(discriminator_a(fake_a), True)
                loss_cycle_a = F.l1_loss(reconstructed_a, real_a)
                loss_cycle_b = F.l1_loss(reconstructed_b, real_b)
                loss_identity_a = F.l1_loss(identity_a, real_a)
                loss_identity_b = F.l1_loss(identity_b, real_b)
                loss_g = (
                    loss_g_a_to_b
                    + loss_g_b_to_a
                    + config.lambda_cycle * (loss_cycle_a + loss_cycle_b)
                    + config.lambda_identity
                    * (loss_identity_a + loss_identity_b)
                )
            loss_g.backward()
            optimizer_g.step()

            set_requires_grad(discriminator_a, True)
            set_requires_grad(discriminator_b, True)
            optimizer_d.zero_grad(set_to_none=True)
            historical_a = fake_a_pool.query(fake_a)
            historical_b = fake_b_pool.query(fake_b)
            with torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=dtype is not None,
            ):
                loss_d_a = 0.5 * (
                    adversarial_loss(discriminator_a(real_a), True)
                    + adversarial_loss(discriminator_a(historical_a), False)
                )
                loss_d_b = 0.5 * (
                    adversarial_loss(discriminator_b(real_b), True)
                    + adversarial_loss(discriminator_b(historical_b), False)
                )
                loss_d = loss_d_a + loss_d_b
            loss_d.backward()
            optimizer_d.step()

            batch_count = int(real_a.shape[0])
            seen += batch_count
            global_step += 1
            for key, value in (
                ("d_a", loss_d_a),
                ("d_b", loss_d_b),
                ("g", loss_g),
                ("g_a_to_b", loss_g_a_to_b),
                ("g_b_to_a", loss_g_b_to_a),
                ("cycle_a", loss_cycle_a),
                ("cycle_b", loss_cycle_b),
                ("identity_a", loss_identity_a),
                ("identity_b", loss_identity_b),
            ):
                sums[key] += float(value.detach().float().item()) * batch_count
            if config.max_steps > 0 and global_step >= config.max_steps:
                reached_max_steps = True
                break

        completed_passes = pass_index + 1
        row: dict[str, Any] = {
            "completed_passes": completed_passes,
            "global_step": global_step,
            "learning_rate": learning_rate,
            "train_samples_per_domain": seen,
            "accidental_same_slide_pairs": accidental_same_slide,
            "accidental_same_location_pairs": accidental_same_location,
            **{f"train_{key}": value / max(seen, 1) for key, value in sums.items()},
            "pass_seconds": time.monotonic() - pass_started,
            "elapsed_seconds": time.monotonic() - started,
            "max_memory_allocated_gib": torch.cuda.max_memory_allocated(device)
            / 2**30,
        }
        selection_point = (
            completed_passes % config.selection_every == 0
            or completed_passes == config.max_passes
            or reached_max_steps
        )
        validation: dict[str, Any] = {}
        if selection_point:
            validation_sampler.set_epoch(0)
            validation = validate(
                generator_a_to_b,
                generator_b_to_a,
                validation_loader,
                device,
                dtype,
            )
            row["validation"] = validation
            score = float(validation["marginal_score"])
            if score < best_score:
                best_score = score
                best_path = output_dir / "checkpoints/best_image_only.pt"
                save_checkpoint(
                    best_path,
                    networks=networks,
                    optimizers=None,
                    config=config,
                    completed_passes=completed_passes,
                    global_step=global_step,
                    validation=validation,
                )
                best_checkpoint = str(best_path)
        history.append(row)
        with (output_dir / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)

        if config.save_every > 0 and (
            completed_passes % config.save_every == 0 or reached_max_steps
        ):
            save_checkpoint(
                output_dir / "checkpoints" / f"pass_{completed_passes:03d}.pt",
                networks=networks,
                optimizers=(optimizer_g, optimizer_d),
                config=config,
                completed_passes=completed_passes,
                global_step=global_step,
                validation=validation,
            )
        if reached_max_steps:
            break

    final_validation = (
        history[-1].get("validation")
        or validate(
            generator_a_to_b,
            generator_b_to_a,
            validation_loader,
            device,
            dtype,
        )
    )
    final_path = output_dir / "checkpoints/final.pt"
    save_checkpoint(
        final_path,
        networks=networks,
        optimizers=(optimizer_g, optimizer_d),
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
            "best_validation_marginal_score": (
                None if not math.isfinite(best_score) else best_score
            ),
            "final_checkpoint": str(final_path),
            "final_checkpoint_sha256": sha256(final_path),
            "note": (
                "The best checkpoint minimizes a pair-blind two-direction marginal "
                "endpoint score. Same-location targets and UNI were not consulted."
            ),
        }
    )
    write_json(output_dir / "run_manifest.json", manifest)
    train_dataset.close()
    validation_dataset.close()
    return manifest


def load_validated_cyclegan_generator(
    checkpoint_path: str | Path,
    *,
    source_scanner: str,
    target_scanner: str,
    test_fold: int,
    device: torch.device,
) -> tuple[ResnetGenerator, dict[str, Any], dict[str, Any]]:
    """Load one direction from a frozen bidirectional CycleGAN checkpoint."""

    path = Path(checkpoint_path).resolve()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("CycleGAN checkpoint root must be a mapping")
    required = {
        "training_version",
        "domain_a",
        "domain_b",
        "test_fold",
        "validation_fold",
        "completed_passes",
        "global_step",
        "config",
        "generator_a_to_b",
        "generator_b_to_a",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"CycleGAN checkpoint is missing fields: {missing}")
    if payload["training_version"] != TRAINING_VERSION:
        raise ValueError("unsupported CycleGAN training version")
    source = source_scanner.strip().lower()
    target = target_scanner.strip().lower()
    domain_a = str(payload["domain_a"]).strip().lower()
    domain_b = str(payload["domain_b"]).strip().lower()
    if (source, target) == (domain_a, domain_b):
        state_key = "generator_a_to_b"
    elif (source, target) == (domain_b, domain_a):
        state_key = "generator_b_to_a"
    else:
        raise ValueError(
            f"requested {source}->{target} is outside {domain_a}<->{domain_b}"
        )
    fold = int(test_fold)
    if int(payload["test_fold"]) != fold or int(payload["validation_fold"]) != (
        fold + 1
    ) % 5:
        raise ValueError("CycleGAN checkpoint fold provenance differs")
    config = dict(payload["config"])
    # Keep the immutable split/domain provenance available to the generic
    # prediction validator.  ``validation_fold`` is stored at checkpoint root
    # because it is derived from ``test_fold`` rather than supplied by CLI.
    config.update(
        {
            "domain_a": domain_a,
            "domain_b": domain_b,
            "test_fold": fold,
            "validation_fold": int(payload["validation_fold"]),
        }
    )
    generator = ResnetGenerator(
        input_nc=3,
        output_nc=3,
        ngf=int(config["ngf"]),
        n_blocks=int(config["n_blocks"]),
    )
    incompatible = generator.load_state_dict(payload[state_key], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"incompatible CycleGAN generator state: {incompatible}")
    generator.to(device).eval()
    for parameter in generator.parameters():
        parameter.requires_grad_(False)
    metadata = {
        "training_version": payload["training_version"],
        "mapping": f"{source}->{target}",
        "source_scanner": source,
        "target_scanner": target,
        "test_fold": fold,
        "validation_fold": int(payload["validation_fold"]),
        "completed_passes": int(payload["completed_passes"]),
        "global_step": int(payload["global_step"]),
        "validation": payload.get("validation", {}),
    }
    del payload
    return generator, metadata, config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-index", type=Path, required=True)
    parser.add_argument("--domain-a", choices=("gt450",), default="gt450")
    parser.add_argument("--domain-b", choices=("at2",), default="at2")
    parser.add_argument("--test-fold", type=int, choices=range(5), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261917)
    parser.add_argument("--max-passes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--locations-per-train-slide", type=int, default=100)
    parser.add_argument("--locations-per-validation-slide", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--validation-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lambda-cycle", type=float, default=10.0)
    parser.add_argument("--lambda-identity", type=float, default=5.0)
    parser.add_argument("--pool-size", type=int, default=50)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--n-blocks", type=int, default=9)
    parser.add_argument(
        "--amp", choices=("bfloat16", "float16", "none"), default="bfloat16"
    )
    parser.add_argument("--constant-passes", type=int, default=100)
    parser.add_argument("--selection-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CycleGANConfig(
        sample_index=str(args.sample_index.resolve()),
        domain_a=args.domain_a,
        domain_b=args.domain_b,
        test_fold=args.test_fold,
        validation_fold=(args.test_fold + 1) % 5,
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
        lambda_cycle=args.lambda_cycle,
        lambda_identity=args.lambda_identity,
        pool_size=args.pool_size,
        ngf=args.ngf,
        ndf=args.ndf,
        n_blocks=args.n_blocks,
        amp=args.amp,
        constant_passes=args.constant_passes,
        selection_every=args.selection_every,
        save_every=args.save_every,
    )
    print(json.dumps(train(config), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

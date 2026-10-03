#!/usr/bin/env python3
"""Generate locked outer-fold Pix2Pix predictions for image/UNI audits.

This command intentionally stops at images.  It validates that the checkpoint,
training manifest, frozen sample index, requested scanner pair, and outer fold
all describe the same experiment. It then writes one
compact HDF5 shard per test slide.  A later, separately frozen stage may read
those shards to extract UNI embeddings; this module neither loads UNI nor
selects a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .data import (
    PRIMARY_SOURCE_SCANNERS,
    PRIMARY_TARGET_SCANNER,
    LockedEvaluationDataset,
    PairedScannerDataset,
    assign_split_roles,
    load_sample_index,
)
from .models import UnetGenerator


PREDICTION_VERSION = "pix2pix_scanner_reference_predictions_v2"
CYCLEGAN_PREDICTION_VERSION = "cyclegan_unpaired_bidirectional_predictions_v1"
EXPECTED_TRAINING_VERSIONS = {
    "pix2pix_scanner_to_at2_v1",
    "pix2pix_scanner_reference_v2",
}
REQUIRED_CHECKPOINT_FIELDS = (
    "training_version",
    "mapping",
    "source_scanner",
    "target_scanner",
    "test_fold",
    "validation_fold",
    "completed_passes",
    "global_step",
    "config",
    "generator",
)


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


def normalized_tensor_to_uint8(image: Tensor) -> np.ndarray:
    """Invert ``uint8 / 127.5 - 1`` deterministically on the CPU.

    The explicit float32 conversion and round-after-clamp rule is part of the
    prediction contract.  Both CHW and NCHW tensors are accepted and converted
    to HWC/NHWC NumPy arrays.
    """

    if image.ndim not in (3, 4):
        raise ValueError(f"expected CHW or NCHW tensor, got shape {tuple(image.shape)}")
    if image.shape[-3] != 3:
        raise ValueError(f"expected RGB tensor, got shape {tuple(image.shape)}")
    value = image.detach().to(device="cpu", dtype=torch.float32)
    value = value.clamp(-1.0, 1.0).add(1.0).mul(127.5).round().to(torch.uint8)
    if image.ndim == 3:
        return np.ascontiguousarray(value.permute(1, 2, 0).numpy())
    return np.ascontiguousarray(value.permute(0, 2, 3, 1).numpy())


def central_crop(image: Tensor, crop_size: int) -> Tensor:
    if image.ndim != 4:
        raise ValueError(f"expected NCHW tensor, got shape {tuple(image.shape)}")
    height, width = image.shape[-2:]
    if crop_size <= 0 or crop_size > height or crop_size > width:
        raise ValueError(f"invalid crop_size {crop_size} for {height}x{width}")
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    return image[..., top : top + crop_size, left : left + crop_size]


def _normalized_scanner(value: Any, field: str) -> str:
    result = str(value).strip().lower()
    if not result:
        raise ValueError(f"{field} must not be empty")
    return result


def validate_checkpoint_payload(
    payload: Mapping[str, Any],
    *,
    source_scanner: str,
    target_scanner: str = PRIMARY_TARGET_SCANNER,
    test_fold: int,
) -> dict[str, Any]:
    """Validate checkpoint direction, fold, and internal config consistency."""

    missing = [field for field in REQUIRED_CHECKPOINT_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"checkpoint is missing fields: {missing}")
    source = _normalized_scanner(source_scanner, "source_scanner")
    target = _normalized_scanner(target_scanner, "target_scanner")
    allowed = {
        *((value, PRIMARY_TARGET_SCANNER) for value in PRIMARY_SOURCE_SCANNERS),
        ("at2", "gt450"),
        # Formal manuscript Panel A models use AT2 as the source and each
        # non-reference scanner as the target.  Their checkpoint payloads use
        # the same scanner_gan.train format, so inference may validate these
        # mappings without weakening any fold or provenance checks below.
        ("at2", "akoya"),
        ("at2", "s360"),
        ("at2", "s60"),
        ("at2", "versa"),
    }
    if (source, target) not in allowed:
        raise ValueError(f"unsupported prediction mapping {source}->{target}")
    fold = int(test_fold)
    if fold not in range(5):
        raise ValueError(f"test_fold must be in 0..4, got {fold}")
    validation_fold = (fold + 1) % 5

    observed_source = _normalized_scanner(
        payload["source_scanner"], "checkpoint source"
    )
    observed_target = _normalized_scanner(
        payload["target_scanner"], "checkpoint target"
    )
    expected_mapping = f"{source}->{target}"
    observed_mapping = str(payload["mapping"]).strip().lower()
    if (
        observed_source != source
        or observed_target != target
        or observed_mapping != expected_mapping
    ):
        raise ValueError(
            "checkpoint direction mismatch: expected "
            f"{expected_mapping}, got source={observed_source}, target={observed_target}, "
            f"mapping={observed_mapping}"
        )
    if int(payload["test_fold"]) != fold:
        raise ValueError(
            f"checkpoint test_fold={payload['test_fold']} does not match requested {fold}"
        )
    if int(payload["validation_fold"]) != validation_fold:
        raise ValueError(
            "checkpoint validation fold does not follow the frozen rule: "
            f"expected {validation_fold}, got {payload['validation_fold']}"
        )
    if str(payload["training_version"]) not in EXPECTED_TRAINING_VERSIONS:
        raise ValueError(
            f"unsupported training_version {payload['training_version']!r}; "
            f"expected one of {sorted(EXPECTED_TRAINING_VERSIONS)!r}"
        )
    if int(payload["completed_passes"]) < 1 or int(payload["global_step"]) < 1:
        raise ValueError(
            "checkpoint must contain at least one completed pass and optimizer step"
        )

    config_value = payload["config"]
    if not isinstance(config_value, Mapping):
        raise TypeError("checkpoint config must be a mapping")
    config = dict(config_value)
    required_config = {
        "sample_index",
        "source_scanner",
        "target_scanner",
        "test_fold",
        "validation_fold",
        "ngf",
        "crop_size",
    }
    missing_config = sorted(required_config - set(config))
    if missing_config:
        raise ValueError(f"checkpoint config is missing fields: {missing_config}")
    comparisons = {
        "source_scanner": (str(config["source_scanner"]).strip().lower(), source),
        "target_scanner": (str(config["target_scanner"]).strip().lower(), target),
        "test_fold": (int(config["test_fold"]), fold),
        "validation_fold": (int(config["validation_fold"]), validation_fold),
    }
    inconsistent = {
        key: {"config": observed, "expected": expected}
        for key, (observed, expected) in comparisons.items()
        if observed != expected
    }
    if inconsistent:
        raise ValueError(
            f"checkpoint top-level/config provenance mismatch: {inconsistent}"
        )
    if int(config["ngf"]) <= 0:
        raise ValueError("checkpoint ngf must be positive")
    crop_size = int(config["crop_size"])
    if crop_size <= 0 or crop_size > 256:
        raise ValueError(f"checkpoint crop_size is invalid: {crop_size}")
    if not isinstance(payload["generator"], Mapping) or not payload["generator"]:
        raise ValueError("checkpoint generator state is empty or invalid")
    return config


def _infer_run_manifest(checkpoint_path: Path) -> Path:
    candidate = checkpoint_path.parent.parent / "run_manifest.json"
    if not candidate.is_file():
        raise FileNotFoundError(
            f"training run manifest not found at {candidate}; pass --run-manifest explicitly"
        )
    return candidate


def validate_run_manifest(
    manifest: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    source_scanner: str,
    target_scanner: str,
    test_fold: int,
    sample_index_sha256: str,
) -> None:
    """Cross-check the immutable data identity recorded by training."""

    source = source_scanner.strip().lower()
    target = target_scanner.strip().lower()
    expected_direction = f"{source}->{target}"
    if str(manifest.get("direction", "")).strip().lower() != expected_direction:
        raise ValueError("run manifest direction does not match checkpoint/request")
    if str(manifest.get("training_version", "")) not in EXPECTED_TRAINING_VERSIONS:
        raise ValueError("run manifest training_version does not match checkpoint")
    if (
        str(manifest.get("sample_index_sha256", "")).lower()
        != sample_index_sha256.lower()
    ):
        raise ValueError("sample index SHA256 does not match the training run manifest")
    manifest_config = manifest.get("config")
    if not isinstance(manifest_config, Mapping):
        raise ValueError("run manifest is missing its training config")
    for field in (
        "sample_index",
        "source_scanner",
        "target_scanner",
        "test_fold",
        "validation_fold",
        "ngf",
        "crop_size",
    ):
        if str(manifest_config.get(field)).lower() != str(config.get(field)).lower():
            raise ValueError(f"run manifest/checkpoint config mismatch for {field}")
    if int(manifest_config["test_fold"]) != int(test_fold):
        raise ValueError("run manifest test fold does not match request")
    if str(manifest.get("status", "")).strip().lower() != "complete":
        raise ValueError("training run manifest is not marked complete")


def validate_cyclegan_run_manifest(
    manifest: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    source_scanner: str,
    target_scanner: str,
    test_fold: int,
    sample_index_sha256: str,
) -> None:
    """Cross-check a pair-blind bidirectional CycleGAN run manifest."""

    from .train_cyclegan import TRAINING_VERSION

    if str(manifest.get("training_version", "")) != TRAINING_VERSION:
        raise ValueError("run manifest is not a supported CycleGAN experiment")
    domains = {
        str(config.get("domain_a", "")).strip().lower(),
        str(config.get("domain_b", "")).strip().lower(),
    }
    if domains != {
        source_scanner.strip().lower(),
        target_scanner.strip().lower(),
    }:
        raise ValueError("CycleGAN domains do not match the requested direction")
    if int(config.get("test_fold", -1)) != int(test_fold) or int(
        config.get("validation_fold", -1)
    ) != (int(test_fold) + 1) % 5:
        raise ValueError("CycleGAN run manifest fold provenance differs")
    if str(manifest.get("sample_index_sha256", "")).lower() != str(
        sample_index_sha256
    ).lower():
        raise ValueError("sample index SHA256 does not match the CycleGAN run")
    if "pair-blind" not in str(manifest.get("selection_boundary", "")).lower():
        raise ValueError("CycleGAN checkpoint selection was not pair-blind")
    if str(manifest.get("status", "")).strip().lower() != "complete":
        raise ValueError("CycleGAN training run is not marked complete")


def load_validated_generator(
    checkpoint_path: str | Path,
    *,
    source_scanner: str,
    target_scanner: str = PRIMARY_TARGET_SCANNER,
    test_fold: int,
    device: torch.device,
) -> tuple[UnetGenerator, dict[str, Any], dict[str, Any]]:
    """Safely load, validate, and instantiate the checkpoint generator."""

    checkpoint_path = Path(checkpoint_path).resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint root must be a mapping")
    config = validate_checkpoint_payload(
        payload,
        source_scanner=source_scanner,
        target_scanner=target_scanner,
        test_fold=test_fold,
    )
    generator = UnetGenerator(
        input_nc=3,
        output_nc=3,
        num_downs=8,
        ngf=int(config["ngf"]),
        use_dropout=False,
    )
    incompatible = generator.load_state_dict(payload["generator"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"incompatible generator state: {incompatible}")
    generator.to(device).eval()
    for parameter in generator.parameters():
        parameter.requires_grad_(False)
    # Do not retain the discriminator and two Adam states during inference;
    # together they are substantially larger than the generator itself.
    checkpoint_metadata = {
        field: payload.get(field)
        for field in (
            "training_version",
            "mapping",
            "source_scanner",
            "target_scanner",
            "test_fold",
            "validation_fold",
            "completed_passes",
            "global_step",
            "validation",
        )
    }
    del payload
    return generator, checkpoint_metadata, config


def _validate_locked_index(locked: pd.DataFrame, full: pd.DataFrame) -> pd.DataFrame:
    """Prove that a supplied locked index is an exact 40/slide subset."""

    if locked[["slide_id", "location_index"]].duplicated().any():
        raise ValueError("locked index contains duplicate slide/location rows")
    if (
        "evaluation_locked" in locked
        and not locked["evaluation_locked"].astype(bool).all()
    ):
        raise ValueError("locked index contains rows not marked evaluation_locked")
    identity = [
        "slide_id",
        "location_index",
        "fold",
        "cache_path",
        "source_index",
        "scanner_order",
    ]
    expected = full[identity].copy()
    observed = locked[identity].copy()
    for column in identity:
        expected[column] = expected[column].astype(str)
        observed[column] = observed[column].astype(str)
    merged = observed.merge(
        expected.drop_duplicates(), on=identity, how="left", indicator=True
    )
    if not (merged["_merge"] == "both").all():
        raise ValueError(
            "locked index contains rows absent from the frozen sample index"
        )
    return locked.sort_values(
        ["slide_id", "location_index"], kind="stable"
    ).reset_index(drop=True)


def _safe_slide_filename(slide_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", slide_id).strip("._") or "slide"
    token = hashlib.sha256(slide_id.encode("utf-8")).hexdigest()[:10]
    return f"{slug[:80]}__{token}.h5"


def _stack(records: Sequence[Mapping[str, np.ndarray]], key: str) -> np.ndarray:
    return np.stack([record[key] for record in records], axis=0)


def _write_slide_shard(
    path: Path,
    *,
    slide_id: str,
    records: Sequence[Mapping[str, np.ndarray]],
    rows: pd.DataFrame,
    provenance: Mapping[str, Any],
    prediction_version: str,
    overwrite: bool,
) -> dict[str, Any]:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing prediction shard: {path}"
        )
    if not records or len(records) != len(rows):
        raise ValueError(
            "slide records and metadata rows must be non-empty and aligned"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with h5py.File(temporary, "w") as store:
            for key, value in provenance.items():
                store.attrs[key] = value
            store.attrs["schema_version"] = prediction_version
            store.attrs["slide_id"] = slide_id
            store.attrs["tissue_type"] = str(rows["tissue_type"].iloc[0])
            store.attrs["cache_path"] = str(rows["cache_path"].iloc[0])
            store.attrs["samples"] = len(records)
            store.attrs["image_dtype"] = "uint8"
            store.attrs["residual_convention"] = (
                "target_crop_xy=q_target-q_source; "
                "q=gradient_shift_xy-applied_shift_xy; shift order=(x,y)"
            )
            images = store.create_group("images")
            valid = store.create_group("aligned_valid")
            image_options = {
                "compression": "gzip",
                "compression_opts": 4,
                "shuffle": True,
                "chunks": (1, 256, 256, 3),
            }
            for key in ("raw_source", "real_target", "generated"):
                images.create_dataset(key, data=_stack(records, key), **image_options)
            if str(provenance["target_scanner"]).lower() == "at2":
                images["real_at2"] = images["real_target"]
            valid_size = int(records[0]["raw_source_valid"].shape[0])
            valid_options = dict(image_options)
            valid_options["chunks"] = (1, valid_size, valid_size, 3)
            for key in ("raw_source_valid", "real_target_valid", "generated_valid"):
                valid.create_dataset(key, data=_stack(records, key), **valid_options)
            if str(provenance["target_scanner"]).lower() == "at2":
                valid["real_at2_valid"] = valid["real_target_valid"]

            metadata = store.create_group("metadata")
            for column in ("sample_index", "location_index", "source_index", "fold"):
                metadata.create_dataset(
                    column, data=rows[column].to_numpy(dtype=np.int64)
                )
            for key in (
                "target_crop_shift_xy",
                "source_gradient_shift_xy",
                "source_applied_shift_xy",
                "source_residual_offset_xy",
                "target_residual_offset_xy",
            ):
                metadata.create_dataset(key, data=_stack(records, key).astype(np.int16))
        temporary.replace(path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return {
        "slide_id": slide_id,
        "tissue_type": str(rows["tissue_type"].iloc[0]),
        "samples": len(records),
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def _resolve_device(value: str) -> torch.device:
    normalized = value.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _resolve_amp_dtype(value: str, device: torch.device) -> torch.dtype | None:
    normalized = value.strip().lower()
    if normalized == "auto":
        return torch.bfloat16 if device.type == "cuda" else None
    if normalized in {"none", "off", "float32"}:
        return None
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float16", "fp16"}:
        if device.type == "cpu":
            raise ValueError(
                "float16 autocast is not supported by this CPU inference path"
            )
        return torch.float16
    raise ValueError(f"unknown AMP mode {value!r}")


@torch.inference_mode()
def predict(
    *,
    checkpoint_path: str | Path,
    sample_index_path: str | Path,
    output_dir: str | Path,
    source_scanner: str,
    target_scanner: str = PRIMARY_TARGET_SCANNER,
    test_fold: int,
    metrics_root: str | Path | None = None,
    locked_index_path: str | Path | None = None,
    run_manifest_path: str | Path | None = None,
    batch_size: int = 32,
    num_workers: int = 8,
    device: str = "auto",
    amp: str = "auto",
    overwrite: bool = False,
    identity_input: bool = False,
    method: str = "pix2pix",
) -> dict[str, Any]:
    """Write locked test-fold predictions without running any evaluation model."""

    if (metrics_root is None) == (locked_index_path is None):
        raise ValueError("provide exactly one of metrics_root or locked_index_path")
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    source = source_scanner.strip().lower()
    target = target_scanner.strip().lower()
    method = method.strip().lower()
    if method not in {"pix2pix", "cyclegan"}:
        raise ValueError("method must be 'pix2pix' or 'cyclegan'")
    fold = int(test_fold)
    checkpoint_path = Path(checkpoint_path).resolve()
    sample_index_path = Path(sample_index_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_device = _resolve_device(device)
    amp_dtype = _resolve_amp_dtype(amp, resolved_device)
    if method == "pix2pix":
        generator, checkpoint, config = load_validated_generator(
            checkpoint_path,
            source_scanner=source,
            target_scanner=target,
            test_fold=fold,
            device=resolved_device,
        )
        prediction_version = PREDICTION_VERSION
    else:
        from .train_cyclegan import load_validated_cyclegan_generator

        generator, checkpoint, config = load_validated_cyclegan_generator(
            checkpoint_path,
            source_scanner=source,
            target_scanner=target,
            test_fold=fold,
            device=resolved_device,
        )
        prediction_version = CYCLEGAN_PREDICTION_VERSION

    sample_index_hash = sha256(sample_index_path)
    run_manifest_path = (
        _infer_run_manifest(checkpoint_path)
        if run_manifest_path is None
        else Path(run_manifest_path).resolve()
    )
    run_manifest = json.loads(run_manifest_path.read_text())
    if method == "pix2pix":
        validate_run_manifest(
            run_manifest,
            config=config,
            source_scanner=source,
            target_scanner=target,
            test_fold=fold,
            sample_index_sha256=sample_index_hash,
        )
    else:
        validate_cyclegan_run_manifest(
            run_manifest,
            config=config,
            source_scanner=source,
            target_scanner=target,
            test_fold=fold,
            sample_index_sha256=sample_index_hash,
        )

    full_index = load_sample_index(sample_index_path, verify_caches=False)
    split = assign_split_roles(full_index, fold, (fold + 1) % 5)
    locked_provenance: dict[str, Any]
    if locked_index_path is not None:
        locked_path = Path(locked_index_path).resolve()
        locked = load_sample_index(locked_path, verify_caches=False)
        locked = _validate_locked_index(locked, full_index)
        locked_split = locked.loc[pd.to_numeric(locked["fold"]) == fold].copy()
        if locked_split.empty:
            raise ValueError(f"locked index has no rows for outer test fold {fold}")
        locked_split["split_role"] = "test"
        dataset = PairedScannerDataset(
            locked_split,
            source,
            target_scanner=target,
            split_role="test",
            augment=False,
            crop_size=int(config["crop_size"]),
            allow_bidirectional_reference_pair=(source, target)
            in {("gt450", "at2"), ("at2", "gt450")},
        )
        locked_provenance = {
            "mode": "prebuilt_locked_index",
            "path": str(locked_path),
            "sha256": sha256(locked_path),
        }
    else:
        metrics_path = Path(metrics_root).resolve()  # type: ignore[arg-type]
        dataset = LockedEvaluationDataset(
            split.loc[split["split_role"] == "test"].copy(),
            metrics_path,
            source,
            target_scanner=target,
            split_role="test",
            crop_size=int(config["crop_size"]),
            allow_bidirectional_reference_pair=(source, target)
            in {("gt450", "at2"), ("at2", "gt450")},
        )
        locked_provenance = {"mode": "chooser", "metrics_root": str(metrics_path)}

    dataset.sample_index = dataset.sample_index.sort_values(
        ["slide_id", "location_index"], kind="stable"
    ).reset_index(drop=True)
    counts = dataset.sample_index.groupby("slide_id", sort=False).size()
    expected_test_slides = set(
        split.loc[split["split_role"] == "test", "slide_id"].astype(str).unique()
    )
    observed_slides = set(dataset.sample_index["slide_id"].astype(str).unique())
    if observed_slides != expected_test_slides:
        raise ValueError(
            "locked evaluation index does not cover exactly the outer test slides"
        )
    if not (counts == 40).all():
        offenders = counts[counts != 40].to_dict()
        raise ValueError(
            f"locked evaluation requires exactly 40 locations/slide: {offenders}"
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=resolved_device.type == "cuda",
        persistent_workers=num_workers > 0,
        drop_last=False,
    )
    checkpoint_hash = sha256(checkpoint_path)
    provenance = {
        "method": method,
        "mapping": f"{source}->{target}",
        "source_scanner": source,
        "target_scanner": target,
        "test_fold": fold,
        "validation_fold": (fold + 1) % 5,
        "checkpoint_sha256": checkpoint_hash,
        "sample_index_sha256": sample_index_hash,
        "valid_crop_size": int(config["crop_size"]),
        "conversion": "clamp[-1,1]; (x+1)*127.5; round; uint8",
    }
    shard_rows: list[dict[str, Any]] = []
    current_slide: str | None = None
    current_records: list[dict[str, np.ndarray]] = []
    current_positions: list[int] = []

    def flush_slide() -> None:
        nonlocal current_slide, current_records, current_positions
        if current_slide is None:
            return
        slide_rows = dataset.sample_index.iloc[current_positions].reset_index(drop=True)
        shard_rows.append(
            _write_slide_shard(
                output_dir / "slides" / _safe_slide_filename(current_slide),
                slide_id=current_slide,
                records=current_records,
                rows=slide_rows,
                provenance=provenance,
                prediction_version=prediction_version,
                overwrite=overwrite,
            )
        )
        current_slide = None
        current_records = []
        current_positions = []

    offset = 0
    try:
        for batch in loader:
            model_input = (
                batch["target_image"] if identity_input else batch["source_image"]
            )
            source_tensor = model_input.to(resolved_device, non_blocking=True)
            with torch.autocast(
                device_type=resolved_device.type,
                dtype=amp_dtype,
                enabled=amp_dtype is not None,
            ):
                generated = generator(source_tensor)
            generated = generated.float()
            crop_size = int(config["crop_size"])
            generated_valid = central_crop(generated, crop_size)
            raw_source = normalized_tensor_to_uint8(model_input)
            real_target = normalized_tensor_to_uint8(batch["target_image"])
            generated_uint8 = normalized_tensor_to_uint8(generated)
            if identity_input:
                central_target = central_crop(batch["target_image"], crop_size)
                raw_source_valid = normalized_tensor_to_uint8(central_target)
                real_target_valid = raw_source_valid.copy()
            else:
                raw_source_valid = normalized_tensor_to_uint8(batch["source_valid"])
                real_target_valid = normalized_tensor_to_uint8(batch["target_valid"])
            generated_valid_uint8 = normalized_tensor_to_uint8(generated_valid)
            batch_count = int(source_tensor.shape[0])
            rows = dataset.sample_index.iloc[offset : offset + batch_count]
            if len(rows) != batch_count:
                raise RuntimeError(
                    "DataLoader order/count diverged from the frozen dataset index"
                )
            shifts = {
                key: np.asarray(batch[key].cpu(), dtype=np.int64)
                for key in (
                    "target_crop_shift_xy",
                    "source_gradient_shift_xy",
                    "source_applied_shift_xy",
                    "source_residual_offset_xy",
                    "target_residual_offset_xy",
                )
            }
            for index, row in enumerate(rows.itertuples(index=False)):
                slide_id = str(row.slide_id)
                if current_slide is not None and slide_id != current_slide:
                    flush_slide()
                if current_slide is None:
                    current_slide = slide_id
                current_records.append(
                    {
                        "raw_source": raw_source[index],
                        "real_target": real_target[index],
                        "generated": generated_uint8[index],
                        "raw_source_valid": raw_source_valid[index],
                        "real_target_valid": real_target_valid[index],
                        "generated_valid": generated_valid_uint8[index],
                        **{key: value[index] for key, value in shifts.items()},
                    }
                )
                current_positions.append(offset + index)
            offset += batch_count
        flush_slide()
        if offset != len(dataset):
            raise RuntimeError(
                f"predicted {offset} rows but locked dataset contains {len(dataset)}"
            )
    finally:
        dataset.close()

    manifest = {
        "prediction_version": prediction_version,
        "method": method,
        "status": "complete",
        "created_utc": utc_now(),
        "direction": f"{source}->{target}",
        "source_scanner": source,
        "target_scanner": target,
        "inference_input_scanner": target if identity_input else source,
        "analysis_role": "target_identity_damage"
        if identity_input
        else "primary_translation",
        "test_fold": fold,
        "validation_fold": (fold + 1) % 5,
        "samples": offset,
        "slides": len(shard_rows),
        "locations_per_slide": 40,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_hash,
            "training_version": checkpoint["training_version"],
            "completed_passes": int(checkpoint["completed_passes"]),
            "global_step": int(checkpoint["global_step"]),
            "validation": checkpoint.get("validation", {}),
        },
        "run_manifest": {
            "path": str(run_manifest_path),
            "sha256": sha256(run_manifest_path),
        },
        "sample_index": {"path": str(sample_index_path), "sha256": sample_index_hash},
        "locked_evaluation": locked_provenance,
        "device": str(resolved_device),
        "amp": "none" if amp_dtype is None else str(amp_dtype).removeprefix("torch."),
        "image_contract": {
            "dtype": "uint8",
            "full_shape": [256, 256, 3],
            "valid_shape": [int(config["crop_size"]), int(config["crop_size"]), 3],
            "conversion": "clamp[-1,1]; (x+1)*127.5; round; uint8",
            "datasets": {
                "images/raw_source": (
                    "real target image passed through its non-identity generator"
                    if identity_input
                    else "uncorrected source-scanner image"
                ),
                "images/real_target": f"same-location stored {target.upper()} image",
                "images/generated": (
                    f"{method} {source.upper()}-to-{target.upper()} model output"
                ),
                "aligned_valid/raw_source_valid": "central source crop",
                "aligned_valid/real_target_valid": "residual-aligned target crop",
                "aligned_valid/generated_valid": "central generated crop",
            },
        },
        "selection_boundary": (
            "checkpoint supplied by caller; no checkpoint selection and no UNI computation here"
        ),
        "shards": shard_rows,
    }
    manifest_path = output_dir / "prediction_manifest.json"
    write_json(manifest_path, manifest)
    checksum_payload = {
        "prediction_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": sha256(manifest_path),
        },
        "checkpoint_sha256": checkpoint_hash,
        "shards": {row["path"]: row["sha256"] for row in shard_rows},
    }
    write_json(output_dir / "prediction_checksums.json", checksum_payload)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path)
    parser.add_argument("--sample-index", type=Path, required=True)
    lock_group = parser.add_mutually_exclusive_group(required=True)
    lock_group.add_argument("--metrics-root", type=Path)
    lock_group.add_argument("--locked-index", type=Path)
    parser.add_argument(
        "--source-scanner",
        choices=tuple((*PRIMARY_SOURCE_SCANNERS, "at2")),
        required=True,
    )
    parser.add_argument("--target-scanner", choices=("at2", "gt450"), default="at2")
    parser.add_argument("--test-fold", type=int, choices=range(5), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--amp", choices=("auto", "bfloat16", "float16", "none"), default="auto"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--identity-input", action="store_true")
    parser.add_argument("--method", choices=("pix2pix", "cyclegan"), default="pix2pix")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = predict(
        checkpoint_path=args.checkpoint,
        run_manifest_path=args.run_manifest,
        sample_index_path=args.sample_index,
        metrics_root=args.metrics_root,
        locked_index_path=args.locked_index,
        output_dir=args.output_dir,
        source_scanner=args.source_scanner,
        target_scanner=args.target_scanner,
        test_fold=args.test_fold,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        amp=args.amp,
        overwrite=args.overwrite,
        identity_input=args.identity_input,
        method=args.method,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

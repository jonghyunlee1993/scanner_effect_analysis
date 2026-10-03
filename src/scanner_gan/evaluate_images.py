#!/usr/bin/env python3
"""Evaluate locked Pix2Pix images before opening frozen UNI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity

from .uni import aggregate_physical_slides, summarize_physical_slides


IMAGE_EVALUATION_VERSION = "scanner_translation_locked_image_safety_v3"
EDGE_HIGH_THRESHOLD = 0.10
EDGE_LOW_THRESHOLD = 0.05
IMAGE_METRICS = (
    "target_l1_unit",
    "target_psnr_db",
    "target_ssim",
    "source_gradient_ncc",
    "target_gradient_ncc",
    "raw_source_target_gradient_ncc",
    "saturation_fraction",
    "invented_edge_fraction",
    "source_edge_deletion_fraction",
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _gray(images: np.ndarray) -> np.ndarray:
    value = images.astype(np.float32) / 255.0
    return 0.2989 * value[..., 0] + 0.5870 * value[..., 1] + 0.1140 * value[..., 2]


def _gradient(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx = gray[:, :-1, 1:] - gray[:, :-1, :-1]
    dy = gray[:, 1:, :-1] - gray[:, :-1, :-1]
    magnitude = np.sqrt(dx * dx + dy * dy)
    return dx, dy, magnitude


def _row_ncc(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first_gray = _gray(first)
    second_gray = _gray(second)
    first_dx, first_dy, _ = _gradient(first_gray)
    second_dx, second_dy, _ = _gradient(second_gray)
    left = np.concatenate(
        (first_dx.reshape(len(first), -1), first_dy.reshape(len(first), -1)), axis=1
    )
    right = np.concatenate(
        (second_dx.reshape(len(second), -1), second_dy.reshape(len(second), -1)), axis=1
    )
    left -= left.mean(axis=1, keepdims=True)
    right -= right.mean(axis=1, keepdims=True)
    numerator = np.einsum("ij,ij->i", left, right)
    denominator = np.sqrt(np.square(left).sum(axis=1) * np.square(right).sum(axis=1))
    return numerator / np.maximum(denominator, 1e-12)


def image_metrics(
    source: np.ndarray,
    generated: np.ndarray,
    at2: np.ndarray,
) -> pd.DataFrame:
    if source.shape != generated.shape or source.shape != at2.shape:
        raise ValueError("source, generated, and AT2 arrays must have identical shape")
    if source.ndim != 4 or source.shape[-1] != 3 or source.dtype != np.uint8:
        raise ValueError("images must be uint8 NHWC arrays")
    source_gray = _gray(source)
    generated_gray = _gray(generated)
    at2_gray = _gray(at2)
    _, _, source_edges = _gradient(source_gray)
    _, _, generated_edges = _gradient(generated_gray)
    _, _, at2_edges = _gradient(at2_gray)
    unit_error = (generated.astype(np.float32) - at2.astype(np.float32)) / 255.0
    mse = np.square(unit_error).mean(axis=(1, 2, 3))
    ssim = np.asarray(
        [
            structural_similarity(
                at2[index], generated[index], channel_axis=-1, data_range=255
            )
            for index in range(len(source))
        ],
        dtype=float,
    )
    invented = (
        (generated_edges >= EDGE_HIGH_THRESHOLD)
        & (source_edges < EDGE_LOW_THRESHOLD)
        & (at2_edges < EDGE_LOW_THRESHOLD)
    )
    deleted = (source_edges >= EDGE_HIGH_THRESHOLD) & (
        generated_edges < EDGE_LOW_THRESHOLD
    )
    saturation = ((generated <= 1) | (generated >= 254)).mean(axis=(1, 2, 3))
    return pd.DataFrame(
        {
            "target_l1_unit": np.abs(unit_error).mean(axis=(1, 2, 3)),
            "target_psnr_db": -10.0 * np.log10(np.maximum(mse, 1e-12)),
            "target_ssim": ssim,
            "source_gradient_ncc": _row_ncc(generated, source),
            "target_gradient_ncc": _row_ncc(generated, at2),
            "raw_source_target_gradient_ncc": _row_ncc(source, at2),
            "saturation_fraction": saturation,
            "invented_edge_fraction": invented.mean(axis=(1, 2)),
            "source_edge_deletion_fraction": deleted.mean(axis=(1, 2)),
        }
    )


def _load_manifests(
    prediction_dirs: Sequence[str | Path],
) -> tuple[str, str, str, str, str, list[tuple[Path, dict[str, Any]]]]:
    values = []
    for directory in prediction_dirs:
        path = Path(directory).resolve() / "prediction_manifest.json"
        manifest = json.loads(path.read_text())
        if manifest.get("status") != "complete":
            raise ValueError(f"prediction is incomplete: {path}")
        values.append((path, manifest))
    sources = {str(manifest["source_scanner"]) for _, manifest in values}
    targets = {str(manifest["target_scanner"]) for _, manifest in values}
    roles = {str(manifest.get("analysis_role", "")) for _, manifest in values}
    inputs = {
        str(manifest.get("inference_input_scanner", "")) for _, manifest in values
    }
    methods = {
        str(manifest.get("method", "pix2pix")).strip().lower()
        for _, manifest in values
    }
    folds = [int(manifest["test_fold"]) for _, manifest in values]
    if (
        len(sources) != 1
        or len(targets) != 1
        or len(roles) != 1
        or len(inputs) != 1
        or len(methods) != 1
        or len(folds) != len(set(folds))
    ):
        raise ValueError(
            "prediction directories must contain unique folds for one scanner pair, "
            "analysis role, and inference input"
        )
    source = sources.pop()
    target = targets.pop()
    role = roles.pop()
    inference_input = inputs.pop()
    method = methods.pop()
    if method not in {"pix2pix", "cyclegan"}:
        raise ValueError(f"unsupported prediction method {method!r}")
    expected_input = {
        "primary_translation": source,
        "target_identity_damage": target,
    }.get(role)
    if expected_input is None or inference_input != expected_input:
        raise ValueError(
            f"invalid analysis role/input contract: role={role!r}, "
            f"input={inference_input!r}, expected={expected_input!r}"
        )
    return (
        source,
        target,
        role,
        inference_input,
        method,
        sorted(values, key=lambda value: int(value[1]["test_fold"])),
    )


def evaluate(
    prediction_dirs: Sequence[str | Path],
    output_dir: str | Path,
    *,
    bootstrap_replicates: int = 20_000,
) -> dict[str, Any]:
    (
        source_scanner,
        target_scanner,
        analysis_role,
        inference_input_scanner,
        method,
        manifests,
    ) = _load_manifests(prediction_dirs)
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    frames = []
    seen: set[tuple[str, int]] = set()
    for manifest_path, manifest in manifests:
        fold = int(manifest["test_fold"])
        for shard in manifest["shards"]:
            path = Path(shard["path"])
            if sha256(path) != shard["sha256"]:
                raise ValueError(f"prediction shard changed: {path}")
            with h5py.File(path, "r") as store:
                source = np.asarray(
                    store["aligned_valid/raw_source_valid"], dtype=np.uint8
                )
                generated = np.asarray(
                    store["aligned_valid/generated_valid"], dtype=np.uint8
                )
                target_key = (
                    "aligned_valid/real_target_valid"
                    if "aligned_valid/real_target_valid" in store
                    else "aligned_valid/real_at2_valid"
                )
                target = np.asarray(store[target_key], dtype=np.uint8)
                locations = np.asarray(store["metadata/location_index"], dtype=int)
                slide_id = str(store.attrs["slide_id"])
                tissue = str(store.attrs["tissue_type"])
            metrics = image_metrics(source, generated, target)
            metrics["target_gradient_gain"] = (
                metrics["target_gradient_ncc"]
                - metrics["raw_source_target_gradient_ncc"]
            )
            for location in locations:
                key = (slide_id, int(location))
                if key in seen:
                    raise ValueError(f"duplicate locked location: {key}")
                seen.add(key)
            metrics.insert(0, "location_index", locations)
            metrics.insert(0, "fold", fold)
            metrics.insert(0, "tissue_type", tissue)
            metrics.insert(0, "slide_id", slide_id)
            metrics.insert(0, "method", method)
            metrics.insert(0, "source_scanner", source_scanner)
            metrics.insert(1, "target_scanner", target_scanner)
            metrics.insert(2, "analysis_role", analysis_role)
            metrics.insert(3, "inference_input_scanner", inference_input_scanner)
            frames.append(metrics)

    location = pd.concat(frames, ignore_index=True)
    location_path = output_root / "location_metrics.csv.gz"
    location.to_csv(location_path, index=False, compression="gzip")
    slide = aggregate_physical_slides(
        location,
        metric_columns=(*IMAGE_METRICS, "target_gradient_gain"),
    )
    slide_path = output_root / "slide_metrics.csv"
    slide.to_csv(slide_path, index=False)
    summary = summarize_physical_slides(
        slide,
        metric_columns=(*IMAGE_METRICS, "target_gradient_gain"),
        bootstrap_replicates=bootstrap_replicates,
    )
    summary_path = output_root / "summary.csv"
    summary.to_csv(summary_path, index=False)
    estimates = summary.set_index("metric")["estimate"]
    gates = {
        "saturation_fraction": {
            "threshold": 0.10,
            "estimate": float(estimates["saturation_fraction"]),
            "pass": bool(estimates["saturation_fraction"] <= 0.10),
        },
        "invented_edge_fraction": {
            "threshold": 0.001,
            "estimate": float(estimates["invented_edge_fraction"]),
            "pass": bool(estimates["invented_edge_fraction"] <= 0.001),
            "definition": (
                f"generated gradient >= {EDGE_HIGH_THRESHOLD}, while source and paired "
                f"target gradients are both < {EDGE_LOW_THRESHOLD}"
            ),
        },
    }
    payload = {
        "evaluation_version": IMAGE_EVALUATION_VERSION,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_scanner": source_scanner,
        "target_scanner": target_scanner,
        "analysis_role": analysis_role,
        "inference_input_scanner": inference_input_scanner,
        "method": method,
        "outer_folds": [int(manifest["test_fold"]) for _, manifest in manifests],
        "locations": int(len(location)),
        "physical_slides": int(location["slide_id"].nunique()),
        "prediction_manifests": [
            {"path": str(path), "sha256": sha256(path)} for path, _ in manifests
        ],
        "safety_gates": gates,
        "all_available_safety_gates_pass": all(
            value["pass"] for value in gates.values()
        ),
        "source_gradient_ncc_role": (
            "descriptive only; an absolute 0.90 gate was removed because the paired real "
            "source-target NCC can itself be far below 0.90"
        ),
        "pending_formal_gates": [
            "reverse ten-endpoint residual versus combined",
            "memorization/nearest-training-target audit",
        ],
        "outputs": {
            "location_metrics": {
                "path": str(location_path),
                "sha256": sha256(location_path),
            },
            "slide_metrics": {"path": str(slide_path), "sha256": sha256(slide_path)},
            "summary": {"path": str(summary_path), "sha256": sha256(summary_path)},
        },
        "uni_status": "not opened by this stage",
    }
    write_json(output_root / "evaluation_manifest.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(
                args.prediction_dir,
                args.output_dir,
                bootstrap_replicates=args.bootstrap_replicates,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

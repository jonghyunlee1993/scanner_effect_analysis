"""Extract the frozen four-PFM panel from the audited native-AA RGB grid."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from fetch_e0_pfm_checkpoints import TRIDENT_COMMIT, sha256
from smoke_e0_pfm_encoders import (
    encoder_kwargs,
    verify_runtime_contract,
    verify_trident_source,
)


SCANNERS = ("at2", "gt450", "versa", "akoya", "s60", "s360")
BATCH_SIZE = {"resnet50": 128, "uni_v1": 64, "conch_v1": 64, "virchow2": 32}
FEATURE_VERSION = "e0_native_aa_four_pfm_v1"


def parse_args():
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--encoder-id")
    selection.add_argument("--encoder-index", type=int)
    parser.add_argument("--slide-id")
    parser.add_argument(
        "--contract", default="outputs/e0_pfm_contract/checkpoint_manifest.json"
    )
    parser.add_argument(
        "--grid-audit", default="outputs/e0_native_aa_grid/audit/summary.json"
    )
    parser.add_argument("--grid", default="outputs/e0_native_aa_grid/shards")
    parser.add_argument("--output", default="outputs/e0_pfm_features")
    parser.add_argument(
        "--trident-root",
        default="/mnt/isilon/oldridge_lab/irinaz/github/env/trident",
    )
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def select_model(contract: dict, encoder_id: str | None, encoder_index: int | None):
    models = contract["models"]
    if encoder_index is not None:
        if encoder_index < 0 or encoder_index >= len(models):
            raise IndexError(f"encoder index {encoder_index} outside 0..{len(models) - 1}")
        selected = models[encoder_index]
    else:
        matches = [model for model in models if model["encoder_id"] == encoder_id]
        if len(matches) != 1:
            raise KeyError(f"encoder absent or duplicated in frozen contract: {encoder_id}")
        selected = matches[0]
    if selected["encoder_id"] not in BATCH_SIZE:
        raise ValueError(f"encoder is outside the frozen core panel: {selected['encoder_id']}")
    return selected


def centered_crop(array: np.ndarray, fov: int) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim != 4 or value.shape[1:] != (512, 512, 3):
        raise ValueError(f"expected N×512×512×3 RGB, got {value.shape}")
    if fov <= 0 or fov > 512 or fov % 2:
        raise ValueError(f"invalid centered model FOV: {fov}")
    start = (512 - fov) // 2
    return value[:, start : start + fov, start : start + fov]


def string_array(values):
    return np.asarray([str(value) for value in values], dtype=h5py.string_dtype("utf-8"))


def load_grid_contract(path: Path):
    summary_path = path.with_suffix(".summary.json")
    summary = json.loads(summary_path.read_text())
    if summary.get("pixel_source") != "native_wsi_only":
        raise ValueError(f"{path}: source grid is not native-only")
    if summary.get("output_sha256") is None:
        raise ValueError(f"{path}: source grid summary lacks SHA256")
    return summary


def existing_output_passes(path: Path, model: dict, source_summary: dict):
    summary_path = path.with_suffix(".summary.json")
    if not path.exists() or not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text())
        return bool(
            summary.get("analysis") == "e0_pfm_feature_shard"
            and summary.get("feature_version") == FEATURE_VERSION
            and summary.get("encoder_id") == model["encoder_id"]
            and summary.get("checkpoint_sha256") == model["checkpoint_sha256"]
            and summary.get("source_grid_sha256") == source_summary["output_sha256"]
            and summary.get("output_sha256") == sha256(path)
            and summary.get("shard_gate_pass") is True
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def transformed_batch(images: np.ndarray, transform):
    import torch

    tensors = [transform(Image.fromarray(image, mode="RGB")) for image in images]
    return torch.stack(tensors, dim=0)


def extract_shard(grid_path: Path, output_path: Path, model: dict, encoder, batch_size: int):
    import torch

    source_summary = load_grid_contract(grid_path)
    if existing_output_passes(output_path, model, source_summary):
        return json.loads(output_path.with_suffix(".summary.json").read_text()), True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    fov = int(model["native_fov_px"])
    feature_dim = int(model["feature_dim"])
    minimum = np.inf
    maximum = -np.inf
    with h5py.File(grid_path, "r") as source, h5py.File(temporary, "w") as target:
        slide_id = str(source.attrs["slide_id"])
        if source["rgb"].shape != (6, 100, 512, 512, 3):
            raise ValueError(f"{slide_id}: invalid source grid shape")
        if [value.decode() if isinstance(value, bytes) else str(value) for value in source["scanner"][:]] != list(SCANNERS):
            raise ValueError(f"{slide_id}: invalid source scanner order")
        features = target.create_dataset(
            "features",
            shape=(6, 100, feature_dim),
            dtype=np.float32,
            chunks=(1, 100, feature_dim),
            compression="lzf",
            shuffle=True,
        )
        for scanner_index in range(6):
            for start in range(0, 100, batch_size):
                stop = min(start + batch_size, 100)
                rgb = centered_crop(source["rgb"][scanner_index, start:stop], fov)
                batch = transformed_batch(rgb, encoder.eval_transforms)
                batch = batch.cuda(non_blocking=True)
                batch = batch.half() if encoder.precision == torch.float16 else batch.float()
                with torch.inference_mode():
                    output = encoder(batch)
                output = output.float().cpu().numpy()
                expected = (stop - start, feature_dim)
                if output.shape != expected or not np.isfinite(output).all():
                    raise ValueError(
                        f"{slide_id}/{model['encoder_id']}: invalid feature batch "
                        f"{output.shape}, expected {expected}"
                    )
                features[scanner_index, start:stop] = output
                minimum = min(minimum, float(output.min()))
                maximum = max(maximum, float(output.max()))

        for name in (
            "scanner",
            "location_id",
            "replicate_id",
            "canonical_center_x",
            "canonical_center_y",
        ):
            target.create_dataset(name, data=source[name][:])
        target.attrs["analysis"] = "e0_pfm_feature_shard"
        target.attrs["feature_version"] = FEATURE_VERSION
        target.attrs["slide_id"] = slide_id
        target.attrs["encoder_id"] = model["encoder_id"]
        target.attrs["feature_dim"] = feature_dim
        target.attrs["native_fov_px"] = fov
        target.attrs["checkpoint_sha256"] = model["checkpoint_sha256"]
        target.attrs["source_grid_sha256"] = source_summary["output_sha256"]
        target.attrs["pixel_source"] = "native_wsi_only"
        target.attrs["trident_role"] = "encoder_factory_and_official_eval_transform_only"
        target.flush()
    os.replace(temporary, output_path)
    summary = {
        "analysis": "e0_pfm_feature_shard",
        "feature_version": FEATURE_VERSION,
        "slide_id": slide_id,
        "encoder_id": model["encoder_id"],
        "feature_dim": feature_dim,
        "native_fov_px": fov,
        "scanners": 6,
        "locations_per_scanner": 100,
        "features": 600,
        "feature_min": minimum,
        "feature_max": maximum,
        "checkpoint_sha256": model["checkpoint_sha256"],
        "source_grid_sha256": source_summary["output_sha256"],
        "output": str(output_path.resolve()),
        "output_sha256": sha256(output_path),
        "shard_gate_pass": True,
    }
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary, False


def main():
    import torch

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("PFM extraction requires a visible CUDA device")
    grid_audit = json.loads(Path(args.grid_audit).read_text())
    if grid_audit.get("grid_gate_pass") is not True:
        raise RuntimeError("native-AA grid audit has not passed")
    contract = json.loads(Path(args.contract).read_text())
    if contract.get("trident_commit") != TRIDENT_COMMIT:
        raise RuntimeError("PFM contract does not use the frozen TRIDENT commit")
    verify_runtime_contract(contract)
    verify_trident_source(Path(args.trident_root))
    model = select_model(contract, args.encoder_id, args.encoder_index)
    if sha256(Path(model["checkpoint_path"])) != model["checkpoint_sha256"]:
        raise RuntimeError(f"checkpoint hash mismatch: {model['encoder_id']}")

    sys.path.insert(0, str(Path(args.trident_root).resolve()))
    from trident.patch_encoder_models.load import encoder_factory

    encoder = encoder_factory(
        model["encoder_id"],
        **encoder_kwargs(model["encoder_id"], model["checkpoint_path"]),
    ).eval().cuda()
    encoder = encoder.half() if encoder.precision == torch.float16 else encoder.float()
    batch_size = args.batch_size or BATCH_SIZE[model["encoder_id"]]
    if batch_size <= 0:
        raise ValueError("batch size must be positive")

    grid_root = Path(args.grid)
    paths = sorted(grid_root.glob("*.h5"))
    if args.slide_id:
        paths = [grid_root / f"{args.slide_id}.h5"]
    if len(paths) != (1 if args.slide_id else 109):
        raise ValueError(f"expected {1 if args.slide_id else 109} source grids, got {len(paths)}")
    output_root = Path(args.output) / model["encoder_id"] / "shards"
    summaries = []
    reused = 0
    for index, grid_path in enumerate(paths):
        output_path = output_root / grid_path.name
        summary, existing = extract_shard(grid_path, output_path, model, encoder, batch_size)
        summaries.append(summary)
        reused += int(existing)
        print(
            f"[{index + 1}/{len(paths)}] {model['encoder_id']} {summary['slide_id']} "
            f"{'reused' if existing else 'written'}",
            flush=True,
        )
    run_summary = {
        "analysis": "e0_pfm_feature_extraction",
        "feature_version": FEATURE_VERSION,
        "encoder_id": model["encoder_id"],
        "slides": len(summaries),
        "features": len(summaries) * 600,
        "new_shards": len(summaries) - reused,
        "reused_shards": reused,
        "all_shards_pass": all(summary["shard_gate_pass"] for summary in summaries),
        "checkpoint_sha256": model["checkpoint_sha256"],
        "trident_commit": TRIDENT_COMMIT,
        "device": torch.cuda.get_device_name(0),
    }
    run_output = Path(args.output) / model["encoder_id"]
    run_output.mkdir(parents=True, exist_ok=True)
    (run_output / "summary.json").write_text(json.dumps(run_summary, indent=2) + "\n")
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()

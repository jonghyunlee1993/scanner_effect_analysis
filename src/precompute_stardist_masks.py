"""Precompute per-scanner StarDist instance labels for the v3 RGB store."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import h5py
import numpy as np
from csbdeep.utils import normalize
from stardist.models import StarDist2D

from utils import store
from utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-slides", type=int)
    parser.add_argument("--slide-index", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _decode_batch(dataset, start, stop):
    images = []
    for index in range(start, stop):
        value = np.asarray(dataset[index])
        image = store.decode(value) if value.ndim == 1 else value
        images.append(np.ascontiguousarray(image))
    return images


def _predict_labels(model, images):
    normalized = np.stack(
        [normalize(image, 1, 99.8, axis=(0, 1)).astype(np.float32) for image in images]
    )
    probability, distances = model.keras_model.predict_on_batch(normalized)
    probability = np.asarray(probability)
    distances = np.asarray(distances)
    labels = []
    for image, prob, dist in zip(images, probability, distances):
        # Rare corrupted/near-empty patches can make the regression head emit
        # non-finite or astronomically long rays.  StarDist's polygon backend
        # then aborts the whole process (rather than raising Python error).
        # A nucleus cannot extend beyond this patch, so bound rays before NMS.
        max_ray = float(max(image.shape[:2]))
        dist = np.nan_to_num(dist, nan=0.0, posinf=max_ray, neginf=0.0)
        dist = np.clip(dist, 0.0, max_ray)
        label, _ = model._instances_from_prediction(
            image.shape[:2], prob[..., 0], dist, return_labels=True
        )
        if label.max(initial=0) > np.iinfo(np.uint16).max:
            raise ValueError("StarDist instance count exceeds uint16 capacity")
        labels.append(label.astype(np.uint16, copy=False))
    return labels


def _complete(path, scanners, expected_count):
    try:
        with h5py.File(path, "r") as handle:
            return (
                int(handle.attrs.get("schema_version", -1)) == 1
                and all(handle[scanner]["labels"].shape[0] == expected_count for scanner in scanners)
            )
    except (OSError, KeyError):
        return False


def process_slide(model, source_path, output_path, scanners, batch_size, overwrite=False):
    with h5py.File(source_path, "r") as source:
        count = int(source[scanners[0]]["rgb"].shape[0])
        if output_path.exists() and not overwrite and _complete(output_path, scanners, count):
            print(f"[stardist] complete, skipping {output_path.name}", flush=True)
            return
        first = store.decode(np.asarray(source[scanners[0]]["rgb"][0]))
        height, width = first.shape[:2]
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary.unlink(missing_ok=True)
        with h5py.File(temporary, "w") as target:
            target.attrs["schema_version"] = 1
            target.attrs["model"] = "2D_versatile_he"
            target.attrs["slide_id"] = source_path.stem
            target.attrs["scanners"] = scanners
            target.attrs["prob_thresh"] = float(model.thresholds.prob)
            target.attrs["nms_thresh"] = float(model.thresholds.nms)
            for scanner in scanners:
                rgb = source[scanner]["rgb"]
                if len(rgb) != count:
                    raise ValueError(f"scanner length mismatch in {source_path}: {scanner}")
                labels = target.create_group(scanner).create_dataset(
                    "labels",
                    shape=(count, height, width),
                    dtype=np.uint16,
                    chunks=(1, height, width),
                    compression="lzf",
                )
                for start in range(0, count, batch_size):
                    stop = min(start + batch_size, count)
                    images = _decode_batch(rgb, start, stop)
                    labels[start:stop] = np.stack(_predict_labels(model, images))
                    if start == 0 or stop == count or stop % (batch_size * 10) == 0:
                        print(
                            f"[stardist] {source_path.stem}/{scanner} {stop}/{count}",
                            flush=True,
                        )
        os.replace(temporary, output_path)


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    cfg = load_config(args.config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model = StarDist2D.from_pretrained("2D_versatile_he")
    source_paths = sorted(Path(cfg.paths.output_store).glob("*.h5"))
    if args.slide_index is not None:
        if not 0 <= args.slide_index < len(source_paths):
            raise ValueError(
                f"--slide-index {args.slide_index} outside [0, {len(source_paths) - 1}]"
            )
        source_paths = [source_paths[args.slide_index]]
    if args.max_slides is not None:
        source_paths = source_paths[: args.max_slides]
    if not source_paths:
        raise ValueError(f"no RGB slide stores found in {cfg.paths.output_store}")
    for source_path in source_paths:
        process_slide(
            model,
            source_path,
            output / source_path.name,
            list(cfg.scanners),
            args.batch_size,
            overwrite=args.overwrite,
        )
    print(f"[stardist] wrote {len(source_paths)} slides to {output}", flush=True)


if __name__ == "__main__":
    main()

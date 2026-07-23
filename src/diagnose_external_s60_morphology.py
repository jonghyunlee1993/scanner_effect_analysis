"""Diagnose where external-S60 morphology is lost in canonicalization.

The diagnostic separates three questions on co-located S60/AT2 patches:

1. Does the AB15 auxiliary nuclei head recover source/AT2 StarDist structure?
2. Does StarDist applied to canonical RGB recover the same structure?
3. Does AB15 improve that agreement over the AB13 baseline?

StarDist predictions are produced identically for raw S60, paired AT2, AB13
canonical RGB, and AB15 canonical RGB.  The AB15 head is evaluated both as
soft occupancy/boundary predictions and as thresholded connected instances.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import pandas as pd
import torch
from csbdeep.utils import normalize
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from stardist.models import StarDist2D
from torch.utils.data import DataLoader

from external_s60_eval import ExternalPairs
from prenorm.checkpoint import load_model_for_inference
from utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ab13-config", required=True)
    parser.add_argument("--ab13-checkpoint", required=True)
    parser.add_argument("--ab15-config", required=True)
    parser.add_argument("--ab15-checkpoint", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tiles", type=int)
    parser.add_argument("--head-threshold", type=float, default=0.5)
    parser.add_argument("--boundary-tolerance", type=int, default=2)
    parser.add_argument("--centroid-tolerance", type=float, default=8.0)
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument(
        "--one-per-slide",
        action="store_true",
        help="Evaluate the first present tuple from every slide for a visual cross-slide check.",
    )
    return parser.parse_args()


def tensor_to_rgb(batch: torch.Tensor) -> np.ndarray:
    return (
        ((batch.detach().float().cpu().clamp(-1, 1).permute(0, 2, 3, 1).numpy() + 1.0)
         * 127.5)
        .round()
        .astype(np.uint8)
    )


def predict_stardist(model, images: np.ndarray) -> list[np.ndarray]:
    normalized = np.stack([
        normalize(image, 1, 99.8, axis=(0, 1)).astype(np.float32)
        for image in images
    ])
    probability, distances = model.keras_model.predict_on_batch(normalized)
    labels = []
    for image, prob, dist in zip(images, np.asarray(probability), np.asarray(distances)):
        max_ray = float(max(image.shape[:2]))
        dist = np.nan_to_num(dist, nan=0.0, posinf=max_ray, neginf=0.0)
        dist = np.clip(dist, 0.0, max_ray)
        label, _ = model._instances_from_prediction(
            image.shape[:2], prob[..., 0], dist, return_labels=True
        )
        labels.append(label.astype(np.int32, copy=False))
    return labels


def instance_boundary(labels: np.ndarray) -> np.ndarray:
    boundary = np.zeros(labels.shape, dtype=bool)
    vertical = labels[1:] != labels[:-1]
    horizontal = labels[:, 1:] != labels[:, :-1]
    boundary[1:] |= vertical
    boundary[:-1] |= vertical
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    return boundary & (labels > 0)


def dice(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction, target = prediction.astype(bool), target.astype(bool)
    denominator = int(prediction.sum() + target.sum())
    if denominator == 0:
        return 1.0
    return float(2 * np.logical_and(prediction, target).sum() / denominator)


def soft_dice(probability: np.ndarray, target: np.ndarray) -> float:
    target = target.astype(np.float32)
    denominator = float(probability.sum() + target.sum())
    return float((2.0 * (probability * target).sum() + 1.0) / (denominator + 1.0))


def boundary_f1(prediction: np.ndarray, target: np.ndarray, tolerance: int) -> float:
    prediction, target = prediction.astype(bool), target.astype(bool)
    if not prediction.any() and not target.any():
        return 1.0
    if not prediction.any() or not target.any():
        return 0.0
    structure = ndimage.generate_binary_structure(2, 1)
    pred_neighborhood = ndimage.binary_dilation(prediction, structure, iterations=tolerance)
    target_neighborhood = ndimage.binary_dilation(target, structure, iterations=tolerance)
    precision = float(target_neighborhood[prediction].mean())
    recall = float(pred_neighborhood[target].mean())
    return 2.0 * precision * recall / max(precision + recall, 1e-12)


def instance_centroids(labels: np.ndarray) -> np.ndarray:
    count = int(labels.max(initial=0))
    if count == 0:
        return np.empty((0, 2), dtype=np.float32)
    flat = labels.ravel()
    areas = np.bincount(flat, minlength=count + 1).astype(np.float64)
    yy, xx = np.indices(labels.shape)
    sum_y = np.bincount(flat, weights=yy.ravel(), minlength=count + 1)
    sum_x = np.bincount(flat, weights=xx.ravel(), minlength=count + 1)
    valid = areas[1:] > 0
    return np.stack(
        (sum_y[1:][valid] / areas[1:][valid], sum_x[1:][valid] / areas[1:][valid]),
        axis=1,
    ).astype(np.float32)


def centroid_scores(prediction: np.ndarray, target: np.ndarray, tolerance: float):
    pred_centroids, target_centroids = instance_centroids(prediction), instance_centroids(target)
    pred_count, target_count = len(pred_centroids), len(target_centroids)
    if pred_count == 0 and target_count == 0:
        return 1.0, 1.0, 1.0
    if pred_count == 0 or target_count == 0:
        return 0.0, 0.0, 0.0
    distances = cdist(pred_centroids, target_centroids)
    pred_indices, target_indices = linear_sum_assignment(distances)
    matched = int((distances[pred_indices, target_indices] <= tolerance).sum())
    precision = matched / pred_count
    recall = matched / target_count
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return float(precision), float(recall), float(f1)


def label_sizes(labels: np.ndarray) -> np.ndarray:
    if labels.max(initial=0) == 0:
        return np.empty(0, dtype=np.float32)
    return np.bincount(labels.ravel())[1:].astype(np.float32)


def compare_instances(prediction: np.ndarray, target: np.ndarray, boundary_tolerance: int,
                      centroid_tolerance: float) -> dict[str, float]:
    pred_count, target_count = int(prediction.max(initial=0)), int(target.max(initial=0))
    precision, recall, centroid_f1 = centroid_scores(
        prediction, target, centroid_tolerance
    )
    pred_sizes, target_sizes = label_sizes(prediction), label_sizes(target)
    return {
        "occupancy_dice": dice(prediction > 0, target > 0),
        "boundary_f1": boundary_f1(
            instance_boundary(prediction), instance_boundary(target), boundary_tolerance
        ),
        "centroid_precision": precision,
        "centroid_recall": recall,
        "centroid_f1": centroid_f1,
        "prediction_count": float(pred_count),
        "target_count": float(target_count),
        "count_abs_error": float(abs(pred_count - target_count)),
        "prediction_median_area": float(np.median(pred_sizes)) if len(pred_sizes) else 0.0,
        "target_median_area": float(np.median(target_sizes)) if len(target_sizes) else 0.0,
    }


def head_instances(occupancy: np.ndarray, boundary: np.ndarray, threshold: float) -> np.ndarray:
    core = (occupancy >= threshold) & ~(boundary >= threshold)
    labels, count = ndimage.label(core)
    if count == 0:
        return labels.astype(np.int32)
    sizes = np.bincount(labels.ravel())
    remove = np.flatnonzero(sizes < 3)
    if len(remove):
        labels[np.isin(labels, remove)] = 0
        labels, _ = ndimage.label(labels > 0)
    return labels.astype(np.int32, copy=False)


def add_relation(row: dict, prefix: str, prediction: np.ndarray, target: np.ndarray,
                 boundary_tolerance: int, centroid_tolerance: float):
    for key, value in compare_instances(
        prediction, target, boundary_tolerance, centroid_tolerance
    ).items():
        row[f"{prefix}_{key}"] = value


def overlay(image: np.ndarray, labels: np.ndarray, color=(0, 255, 0)) -> np.ndarray:
    output = image.copy()
    output[instance_boundary(labels)] = np.asarray(color, dtype=np.uint8)
    return output


def sample_panel(images, labels, head_occ, head_boundary, path):
    tiles = [
        (overlay(images[0], labels[0]), "raw S60 + StarDist"),
        (overlay(images[1], labels[1]), "paired AT2 + StarDist"),
        (overlay(images[2], labels[2]), "AB13 canonical + StarDist"),
        (overlay(images[3], labels[3]), "AB15 canonical + StarDist"),
    ]
    head_image = images[3].copy()
    head_image[head_occ >= 0.5] = (
        0.65 * head_image[head_occ >= 0.5] + 0.35 * np.array([40, 220, 70])
    ).astype(np.uint8)
    head_image[head_boundary >= 0.5] = np.array([255, 210, 0], dtype=np.uint8)
    tiles.append((head_image, "AB15 head: green=occ, yellow=boundary"))
    canvas = Image.new("RGB", (256 * len(tiles), 282), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (image, title) in enumerate(tiles):
        canvas.paste(Image.fromarray(image), (256 * index, 0))
        draw.text((256 * index + 6, 262), title, fill="black")
    canvas.save(path)


def summarize(frame: pd.DataFrame, args, stardist) -> dict:
    numeric = frame.select_dtypes(include=[np.number])
    means = {key: float(value) for key, value in numeric.mean().items()}
    per_slide = {}
    for slide_id, slide in frame.groupby("slide_id", sort=True):
        slide_numeric = slide.select_dtypes(include=[np.number])
        per_slide[str(slide_id)] = {
            key: float(value) for key, value in slide_numeric.mean().items()
        }
        per_slide[str(slide_id)]["n_tiles"] = int(len(slide))
    return {
        "n_tiles": int(len(frame)),
        "n_slides": int(frame.slide_id.nunique()),
        "ab13_checkpoint": str(args.ab13_checkpoint),
        "ab15_checkpoint": str(args.ab15_checkpoint),
        "head_threshold": float(args.head_threshold),
        "boundary_tolerance_px": int(args.boundary_tolerance),
        "centroid_tolerance_px": float(args.centroid_tolerance),
        "stardist_model": "2D_versatile_he",
        "stardist_probability_threshold": float(stardist.thresholds.prob),
        "stardist_nms_threshold": float(stardist.thresholds.nms),
        "means": means,
        "per_slide": per_slide,
    }


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if not 0.0 < args.head_threshold < 1.0:
        raise ValueError("head-threshold must be in (0, 1)")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    samples = output / "samples"
    samples.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ExternalPairs(args.index, args.store, "s60", "at2", args.max_tiles)
    if args.one_per_slide:
        dataset.frame = (
            dataset.frame.groupby("slide_id", sort=True, as_index=False)
            .head(1)
            .reset_index(drop=True)
        )
    # Keep workers at zero: TensorFlow and PyTorch both own GPU contexts in this process.
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
    ab13 = load_model_for_inference(
        args.ab13_checkpoint, load_config(args.ab13_config)
    ).to(device).eval()
    ab15 = load_model_for_inference(
        args.ab15_checkpoint, load_config(args.ab15_config)
    ).to(device).eval()
    if ab15.G.nuclei_head is None:
        raise ValueError("AB15 checkpoint does not contain a nuclei head")
    stardist = StarDist2D.from_pretrained("2D_versatile_he")

    rows, saved_samples = [], 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            source = batch["source"].to(device)
            reference = batch["reference"].to(device)
            content13 = ab13.encode(source)
            canonical13 = ab13.decode_canonical(content13, source)
            content15 = ab15.encode(source)
            canonical15, head_logits = ab15.decode_canonical_with_nuclei(content15, source)
            head_prob = torch.sigmoid(head_logits).float().cpu().numpy()

            raw_rgb = tensor_to_rgb(source)
            at2_rgb = tensor_to_rgb(reference)
            canonical13_rgb = tensor_to_rgb(canonical13)
            canonical15_rgb = tensor_to_rgb(canonical15)
            combined = np.concatenate(
                (raw_rgb, at2_rgb, canonical13_rgb, canonical15_rgb), axis=0
            )
            predicted = predict_stardist(stardist, combined)
            size = len(raw_rgb)
            raw_labels = predicted[:size]
            at2_labels = predicted[size:2 * size]
            canonical13_labels = predicted[2 * size:3 * size]
            canonical15_labels = predicted[3 * size:]

            for index in range(size):
                source_label, at2_label = raw_labels[index], at2_labels[index]
                label13, label15 = canonical13_labels[index], canonical15_labels[index]
                occupancy, boundary = head_prob[index, 0], head_prob[index, 1]
                head_label = head_instances(occupancy, boundary, args.head_threshold)
                row = {
                    "slide_id": str(batch["slide_id"][index]),
                    "tuple_id": int(batch["tuple_id"][index]),
                    "q_reg_s60": float(batch["q_reg"][index]),
                    "head_source_occupancy_soft_dice": soft_dice(
                        occupancy, source_label > 0
                    ),
                    "head_at2_occupancy_soft_dice": soft_dice(occupancy, at2_label > 0),
                    "head_source_occupancy_dice": dice(
                        occupancy >= args.head_threshold, source_label > 0
                    ),
                    "head_at2_occupancy_dice": dice(
                        occupancy >= args.head_threshold, at2_label > 0
                    ),
                    "head_source_boundary_f1": boundary_f1(
                        boundary >= args.head_threshold,
                        instance_boundary(source_label),
                        args.boundary_tolerance,
                    ),
                    "head_at2_boundary_f1": boundary_f1(
                        boundary >= args.head_threshold,
                        instance_boundary(at2_label),
                        args.boundary_tolerance,
                    ),
                }
                for prefix, prediction, target in (
                    ("raw_to_at2", source_label, at2_label),
                    ("ab13_to_source", label13, source_label),
                    ("ab13_to_at2", label13, at2_label),
                    ("ab15_to_source", label15, source_label),
                    ("ab15_to_at2", label15, at2_label),
                    ("head_to_source", head_label, source_label),
                    ("head_to_at2", head_label, at2_label),
                    ("head_to_ab15_rgb", head_label, label15),
                ):
                    add_relation(
                        row, prefix, prediction, target,
                        args.boundary_tolerance, args.centroid_tolerance,
                    )
                rows.append(row)
                if saved_samples < args.sample_count:
                    sample_panel(
                        (raw_rgb[index], at2_rgb[index], canonical13_rgb[index],
                         canonical15_rgb[index]),
                        (source_label, at2_label, label13, label15),
                        occupancy,
                        boundary,
                        samples / f"{saved_samples:03d}.png",
                    )
                    saved_samples += 1
            if batch_index == 0 or (batch_index + 1) % 25 == 0:
                print(
                    f"[morphology] {min((batch_index + 1) * args.batch_size, len(dataset))}"
                    f"/{len(dataset)} tiles",
                    flush=True,
                )

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "per_tile.csv", index=False)
    metrics = summarize(frame, args, stardist)
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    print(f"[morphology] wrote {len(frame)} tiles -> {output}", flush=True)


if __name__ == "__main__":
    main()

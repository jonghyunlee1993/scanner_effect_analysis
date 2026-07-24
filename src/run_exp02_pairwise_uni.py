"""Extract pairwise scanner-transform, UNI, and matched-blur measurements.

Every comparison stays within one physical sampling lattice and uses identical
registered locations. Pairwise RGB affines are fitted on train-slide coarse
bands and evaluated on held-out test locations. The operational image policy is
explicit clipping; there is no fallback to the raw source.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prenorm.data.identity import matched_indices_by_lattice
from prenorm.embedding import embed_uni, load_uni
from prenorm.exp01.data import BalancedPairDataset
from prenorm.exp01.frequency import FixedLaplacianPyramid
from prenorm.exp02.pairwise import (
    clipped_detail_retention_frontier,
    clipped_low_transform,
    fit_pairwise_affines,
    load_lattice_batch,
)
from prenorm.metrics import per_image_ssim as ssim_score
from utils.config import load_config


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/experiments/exp01a_lf16_10slide.yaml"
    )
    parser.add_argument("--output", default="outputs/exp02_visualization")
    parser.add_argument("--fit-max-locations", type=int, default=600)
    parser.add_argument("--eval-max-locations", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument(
        "--detail-retention", default="1.0,0.75,0.5,0.25,0.0"
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sum(left * right, axis=1)


def per_image_mae(left: torch.Tensor, right: torch.Tensor) -> np.ndarray:
    return (left - right).abs().flatten(1).mean(1).cpu().numpy()


def per_image_ssim(left: torch.Tensor, right: torch.Tensor) -> np.ndarray:
    return ssim_score(left, right).cpu().numpy()


def range_fraction(images: torch.Tensor, tolerance: float = 1e-6) -> np.ndarray:
    outside = (images < -1.0 - tolerance) | (images > 1.0 + tolerance)
    return outside.float().flatten(1).mean(1).cpu().numpy()


def max_range_excess(images: torch.Tensor) -> np.ndarray:
    excess = (images.abs() - 1.0).clamp_min(0.0)
    return excess.flatten(1).amax(1).cpu().numpy()


def band_distortion_mae(
    pyramid,
    images: torch.Tensor,
    requested_bands,
) -> np.ndarray:
    """Per-image coefficient MAE after an operational image transform."""
    _, actual_bands = pyramid.decompose(images)
    numerator = sum(
        (actual - requested).abs().flatten(1).sum(1)
        for actual, requested in zip(actual_bands, requested_bands)
    )
    denominator = sum(band[0].numel() for band in requested_bands)
    return (numerator / denominator).cpu().numpy()


def select_representative_position(dataset, selection) -> int:
    """Choose a high-but-not-extreme tissue tile for qualitative panels."""
    first_scanner = selection["scanners"][0]
    indices = selection["indices"][first_scanner]
    density = np.asarray(
        [dataset.records[index]["tissue_density"] for index in indices]
    )
    target = np.quantile(density, 0.75)
    return int(np.argmin(np.abs(density - target)))


def direction_summary(
    pair_frame: pd.DataFrame,
    blur_frame: pd.DataFrame,
) -> dict:
    pair_metrics = (
        "low_mae_raw",
        "low_mae_affine_requested",
        "low_mae_clipped",
        "low_mae_reduction_fraction",
        "full_ssim_raw",
        "full_ssim_clipped",
        "uni_source_to_clipped_cos",
        "uni_source_to_target_cos",
        "uni_clipped_to_target_cos",
        "uni_target_cos_delta",
        "preclip_range_fraction",
        "preclip_max_excess",
        "clipped_detail_distortion_mae",
    )
    blur_metrics = (
        "uni_transformed_to_source_sharp_cos",
        "uni_transformed_to_target_sharp_cos",
        "uni_transformed_to_matched_target_cos",
        "uni_raw_to_matched_target_cos",
        "uni_matched_alignment_delta",
        "uni_raw_blurred_to_source_sharp_cos",
        "uni_target_blurred_to_target_sharp_cos",
        "image_ssim_transformed_to_matched_target",
        "image_ssim_raw_to_matched_target",
        "preclip_range_fraction",
        "clip_detail_distortion_mae",
    )
    report = {}
    for direction, group in pair_frame.groupby("direction", sort=False):
        row = {
            metric: {
                "mean": float(group[metric].mean()),
                "std": float(group[metric].std(ddof=0)),
            }
            for metric in pair_metrics
        }
        frontier = {}
        selected = blur_frame[blur_frame["direction"] == direction]
        for retention, part in selected.groupby(
            "detail_retention", sort=False
        ):
            frontier[f"{float(retention):.2f}"] = {
                metric: float(part[metric].mean())
                for metric in blur_metrics
            }
        row["matched_blur_frontier"] = frontier
        report[direction] = row
    return report


def clipped_frontier(pyramid, low, bands, retention):
    return clipped_detail_retention_frontier(
        pyramid, low, bands, retention
    )


@torch.no_grad()
def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("Exp-02 UNI extraction requires a CUDA allocation")

    retention = tuple(
        float(value.strip()) for value in args.detail_retention.split(",")
    )
    if tuple(sorted(retention, reverse=True)) != retention:
        raise ValueError("detail retention must be listed in descending order")
    if retention[0] != 1.0 or retention[-1] != 0.0:
        raise ValueError("detail retention must include endpoints 1.0 and 0.0")

    cfg = load_config(args.config)
    scanners = [str(scanner) for scanner in cfg.target_scanners]
    pyramid = FixedLaplacianPyramid(int(cfg.pyramid.levels))

    fit_dataset = BalancedPairDataset(cfg, "train", False)
    fit_selections = matched_indices_by_lattice(
        fit_dataset.records,
        scanners,
        args.fit_max_locations,
        np.random.default_rng(args.seed),
    )
    pairwise_affines = fit_pairwise_affines(
        fit_dataset,
        pyramid,
        fit_selections,
        batch_size=args.batch_size,
    )

    eval_dataset = BalancedPairDataset(cfg, "test", False)
    eval_selections = matched_indices_by_lattice(
        eval_dataset.records,
        scanners,
        args.eval_max_locations,
        np.random.default_rng(args.seed + 1),
    )

    device = torch.device("cuda")
    model, size, mean, std = load_uni(device)

    def embed(images):
        return embed_uni(
            model,
            images,
            size,
            mean,
            std,
            device,
            batch_size=args.embed_batch_size,
        ).numpy()

    pair_rows = []
    blur_rows = []
    exemplar_arrays = {}
    exemplar_manifest = {}

    for lattice_id, selection in eval_selections.items():
        expected_directions = pairwise_affines[lattice_id]
        representative_position = select_representative_position(
            eval_dataset, selection
        )
        location_count = len(selection["keys"])
        for start in range(0, location_count, args.batch_size):
            positions = list(
                range(start, min(start + args.batch_size, location_count))
            )
            images = load_lattice_batch(eval_dataset, selection, positions)
            decomposed = {
                scanner: pyramid.decompose(scanner_images)
                for scanner, scanner_images in images.items()
            }
            native_frontiers = {
                scanner: clipped_frontier(
                    pyramid, low, bands, retention
                )
                for scanner, (low, bands) in decomposed.items()
            }
            native_order = [
                (scanner, value)
                for scanner in images
                for value in retention
            ]
            native_chunks = np.split(
                embed(torch.cat([
                    native_frontiers[scanner]["clipped"][value]
                    for scanner, value in native_order
                ])),
                len(native_order),
            )
            native_embeddings = {scanner: {} for scanner in images}
            for (scanner, value), values in zip(
                native_order, native_chunks
            ):
                native_embeddings[scanner][value] = values
            raw_embeddings = {
                scanner: values[1.0]
                for scanner, values in native_embeddings.items()
            }

            for short_pair, affine in expected_directions.items():
                source_name, target_name = short_pair.split("_to_")
                direction = f"{lattice_id}__{short_pair}"
                source = images[source_name]
                target = images[target_name]
                source_low, source_bands = decomposed[source_name]
                target_low, target_bands = decomposed[target_name]

                transformed = clipped_low_transform(
                    pyramid, source, affine
                )
                transformed_frontier = clipped_frontier(
                    pyramid,
                    transformed["affine_low"],
                    source_bands,
                    retention,
                )
                raw_frontier = native_frontiers[source_name]
                target_frontier = native_frontiers[target_name]

                chunks = np.split(
                    embed(torch.cat([
                        transformed_frontier["clipped"][value]
                        for value in retention
                    ])),
                    len(retention),
                )
                transformed_embeddings = dict(
                    zip(retention, chunks)
                )
                raw_blurred_embeddings = native_embeddings[source_name]
                target_blurred_embeddings = native_embeddings[target_name]

                source_embedding = raw_embeddings[source_name]
                target_embedding = raw_embeddings[target_name]
                clipped_output = transformed_frontier["clipped"][1.0]
                clipped_embedding = transformed_embeddings[1.0]
                clipped_low = pyramid.coarsest(clipped_output)

                raw_low_mae = per_image_mae(source_low, target_low)
                affine_low_mae = per_image_mae(
                    transformed["affine_low"], target_low
                )
                clipped_low_mae = per_image_mae(clipped_low, target_low)
                raw_full_ssim = per_image_ssim(source, target)
                clipped_full_ssim = per_image_ssim(clipped_output, target)
                source_clipped_cos = cosine(
                    source_embedding, clipped_embedding
                )
                source_target_cos = cosine(
                    source_embedding, target_embedding
                )
                clipped_target_cos = cosine(
                    clipped_embedding, target_embedding
                )
                preclip_range = range_fraction(transformed["preclip"])
                preclip_excess = max_range_excess(transformed["preclip"])
                clipped_detail_distortion = band_distortion_mae(
                    pyramid, clipped_output, source_bands
                )

                for offset, position in enumerate(positions):
                    key = selection["keys"][position]
                    denominator = max(float(raw_low_mae[offset]), 1e-12)
                    pair_rows.append({
                        "lattice_id": lattice_id,
                        "direction": direction,
                        "source_scanner": source_name,
                        "target_scanner": target_name,
                        "slide_id": key.slide_id,
                        "tuple_id": key.tuple_id,
                        "low_mae_raw": float(raw_low_mae[offset]),
                        "low_mae_affine_requested": float(
                            affine_low_mae[offset]
                        ),
                        "low_mae_clipped": float(clipped_low_mae[offset]),
                        "low_mae_reduction_fraction": float(
                            (raw_low_mae[offset] - clipped_low_mae[offset])
                            / denominator
                        ),
                        "full_ssim_raw": float(raw_full_ssim[offset]),
                        "full_ssim_clipped": float(
                            clipped_full_ssim[offset]
                        ),
                        "uni_source_to_clipped_cos": float(
                            source_clipped_cos[offset]
                        ),
                        "uni_source_to_target_cos": float(
                            source_target_cos[offset]
                        ),
                        "uni_clipped_to_target_cos": float(
                            clipped_target_cos[offset]
                        ),
                        "uni_target_cos_delta": float(
                            clipped_target_cos[offset]
                            - source_target_cos[offset]
                        ),
                        "preclip_range_fraction": float(
                            preclip_range[offset]
                        ),
                        "preclip_max_excess": float(
                            preclip_excess[offset]
                        ),
                        "clipped_detail_distortion_mae": float(
                            clipped_detail_distortion[offset]
                        ),
                    })

                for value in retention:
                    transformed_image = transformed_frontier["clipped"][value]
                    raw_blurred_image = raw_frontier["clipped"][value]
                    target_blurred_image = target_frontier["clipped"][value]
                    transformed_embedding = transformed_embeddings[value]
                    raw_blurred_embedding = raw_blurred_embeddings[value]
                    target_blurred_embedding = target_blurred_embeddings[value]

                    transformed_to_matched = cosine(
                        transformed_embedding, target_blurred_embedding
                    )
                    raw_to_matched = cosine(
                        raw_blurred_embedding, target_blurred_embedding
                    )
                    requested_bands = [
                        float(value) * band for band in source_bands
                    ]
                    preclip_variant = transformed_frontier["preclip"][value]
                    arrays = {
                        "uni_transformed_to_source_sharp_cos": cosine(
                            transformed_embedding, source_embedding
                        ),
                        "uni_transformed_to_target_sharp_cos": cosine(
                            transformed_embedding, target_embedding
                        ),
                        "uni_transformed_to_matched_target_cos":
                            transformed_to_matched,
                        "uni_raw_to_matched_target_cos": raw_to_matched,
                        "uni_matched_alignment_delta":
                            transformed_to_matched - raw_to_matched,
                        "uni_raw_blurred_to_source_sharp_cos": cosine(
                            raw_blurred_embedding, source_embedding
                        ),
                        "uni_target_blurred_to_target_sharp_cos": cosine(
                            target_blurred_embedding, target_embedding
                        ),
                        "image_ssim_transformed_to_source_sharp":
                            per_image_ssim(transformed_image, source),
                        "image_ssim_transformed_to_target_sharp":
                            per_image_ssim(transformed_image, target),
                        "image_ssim_transformed_to_matched_target":
                            per_image_ssim(
                                transformed_image, target_blurred_image
                            ),
                        "image_ssim_raw_to_matched_target":
                            per_image_ssim(
                                raw_blurred_image, target_blurred_image
                            ),
                        "preclip_range_fraction":
                            range_fraction(preclip_variant),
                        "preclip_max_excess":
                            max_range_excess(preclip_variant),
                        "clip_pixel_mae":
                            per_image_mae(transformed_image, preclip_variant),
                        "clip_detail_distortion_mae":
                            band_distortion_mae(
                                pyramid,
                                transformed_image,
                                requested_bands,
                            ),
                    }
                    for offset, position in enumerate(positions):
                        key = selection["keys"][position]
                        blur_rows.append({
                            "lattice_id": lattice_id,
                            "direction": direction,
                            "source_scanner": source_name,
                            "target_scanner": target_name,
                            "slide_id": key.slide_id,
                            "tuple_id": key.tuple_id,
                            "detail_retention": value,
                            "blur_severity": 1.0 - value,
                            **{
                                name: float(metric[offset])
                                for name, metric in arrays.items()
                            },
                        })

                if representative_position in positions:
                    offset = positions.index(representative_position)
                    exemplar_id = direction.replace("/", "_")
                    key = selection["keys"][representative_position]
                    exemplar_manifest[direction] = {
                        "array_prefix": exemplar_id,
                        "slide_id": key.slide_id,
                        "tuple_id": key.tuple_id,
                        "source_scanner": source_name,
                        "target_scanner": target_name,
                    }
                    exemplar_arrays[f"{exemplar_id}__source"] = (
                        source[offset].numpy()
                    )
                    exemplar_arrays[f"{exemplar_id}__target"] = (
                        target[offset].numpy()
                    )
                    exemplar_arrays[f"{exemplar_id}__clipped"] = (
                        clipped_output[offset].numpy()
                    )
                    for value in retention:
                        suffix = f"retention_{value:.2f}".replace(".", "p")
                        exemplar_arrays[
                            f"{exemplar_id}__transformed_{suffix}"
                        ] = transformed_frontier["clipped"][value][
                            offset
                        ].numpy()
                        exemplar_arrays[
                            f"{exemplar_id}__raw_{suffix}"
                        ] = raw_frontier["clipped"][value][offset].numpy()
                        exemplar_arrays[
                            f"{exemplar_id}__target_{suffix}"
                        ] = target_frontier["clipped"][value][offset].numpy()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_frame = pd.DataFrame(pair_rows)
    blur_frame = pd.DataFrame(blur_rows)
    pair_frame.to_csv(output_dir / "pairwise_metrics.csv", index=False)
    blur_frame.to_csv(output_dir / "blur_frontier_metrics.csv", index=False)
    np.savez_compressed(output_dir / "exemplars.npz", **exemplar_arrays)

    affine_report = {
        lattice_id: {
            direction: affine.tolist()
            for direction, affine in lattice_affines.items()
        }
        for lattice_id, lattice_affines in pairwise_affines.items()
    }
    report = {
        "experiment": "Exp-02",
        "analysis_contract_version": 3,
        "config": args.config,
        "model": "UNI",
        "image_policy": "explicit_clipped_affine_low",
        "reference_policy": (
            "rotate every target scanner; compare transformed source and "
            "untransformed target at identical detail retention"
        ),
        "s60_role": (
            "independent S60 lattice; not unseen-scanner external validation"
        ),
        "pyramid_levels": int(cfg.pyramid.levels),
        "sample_identity": ["lattice_id", "slide_id", "tuple_id"],
        "fit_split": "train",
        "eval_split": "test",
        "fit_max_locations": args.fit_max_locations,
        "eval_max_locations": args.eval_max_locations,
        "detail_retention": list(retention),
        "pairwise_affines": affine_report,
        "exemplars": exemplar_manifest,
        "directions": direction_summary(pair_frame, blur_frame),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"[exp02-pairwise-uni] {len(pair_frame)} pair rows, "
        f"{len(blur_frame)} matched-blur rows -> {output_dir}"
    )


if __name__ == "__main__":
    main()

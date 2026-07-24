"""Extract UNI trajectories under controlled high-frequency perturbations.

Three well-registered internal-lattice locations are selected without looking
at embeddings. Each location contains AT2, GT450, VERSA, and Akoya. The same
raw patch is then blurred, sharpened, or mixed toward either a registered
leave-one-scanner-out high-frequency mean or a global train-derived mean.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prenorm.data.identity import INTERNAL_LATTICE_ID, matched_indices_by_lattice
from prenorm.embedding import embed_uni, load_uni
from prenorm.exp01.data import BalancedPairDataset
from prenorm.exp01.frequency import FixedLaplacianPyramid
from prenorm.exp02.pairwise import load_lattice_batch
from prenorm.exp02.trajectory import (
    gain_high_frequency,
    mix_high_frequency,
)
from utils.config import load_config


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/experiments/exp01a_lf16_10slide.yaml"
    )
    parser.add_argument(
        "--output", default="outputs/exp02_hf_trajectory"
    )
    parser.add_argument("--locations", type=int, default=3)
    parser.add_argument("--train-mean-locations", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--min-tissue-density", type=float, default=0.50)
    parser.add_argument("--max-tissue-density", type=float, default=0.90)
    parser.add_argument(
        "--max-blue-dominant-fraction", type=float, default=0.80
    )
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args(argv)


def internal_selection(dataset, scanners, max_locations, seed):
    selections = matched_indices_by_lattice(
        dataset.records,
        scanners,
        max_locations,
        np.random.default_rng(seed),
    )
    return selections[INTERNAL_LATTICE_ID]


def location_candidates(dataset, selection):
    candidates = []
    for position, key in enumerate(selection["keys"]):
        records = [
            dataset.records[selection["indices"][scanner][position]]
            for scanner in selection["scanners"]
        ]
        candidates.append({
            "position": position,
            "slide_id": key.slide_id,
            "tuple_id": key.tuple_id,
            "min_q_reg": min(record["q_reg"] for record in records),
            "mean_q_reg": float(np.mean([
                record["q_reg"] for record in records
            ])),
            "tissue_density": float(np.mean([
                record["tissue_density"] for record in records
            ])),
            "x": int(records[0]["x"]),
            "y": int(records[0]["y"]),
        })
    return candidates


def blue_dominant_fraction(images):
    """Fraction of pixels with a strong non-H&E blue/cyan color cast."""
    rgb = (images.float() + 1.0) / 2.0
    red, green, blue = rgb.unbind(dim=1)
    blue_dominant = (
        (blue - red > 0.10)
        & (blue - green > 0.06)
        & (blue > 0.25)
    )
    return blue_dominant.flatten(1).float().mean(dim=1)


def candidate_color_audit(dataset, selection, position):
    images = load_lattice_batch(dataset, selection, [position])
    return {
        scanner: float(blue_dominant_fraction(image)[0])
        for scanner, image in images.items()
    }


def select_registered_locations(
    dataset,
    selection,
    count,
    min_tissue_density=0.50,
    max_tissue_density=0.90,
    max_blue_dominant_fraction=0.80,
):
    """Select high-QC tissue patches with artifact, slide, and spatial QC."""
    candidates = [
        row for row in location_candidates(dataset, selection)
        if min_tissue_density
        <= row["tissue_density"]
        <= max_tissue_density
    ]
    candidates.sort(
        key=lambda row: (-row["min_q_reg"], -row["mean_q_reg"])
    )
    slide_limit = max(1, int(np.ceil(count / 2)))
    slide_counts = Counter()
    chosen = []
    for row in candidates:
        if slide_counts[row["slide_id"]] >= slide_limit:
            continue
        too_close = any(
            previous["slide_id"] == row["slide_id"]
            and np.hypot(
                previous["x"] - row["x"],
                previous["y"] - row["y"],
            ) < 2048
            for previous in chosen
        )
        if too_close:
            continue
        color_audit = candidate_color_audit(
            dataset, selection, row["position"]
        )
        if max(color_audit.values()) > max_blue_dominant_fraction:
            continue
        row = {
            **row,
            "blue_dominant_fraction": color_audit,
            "max_blue_dominant_fraction": max(color_audit.values()),
        }
        chosen.append(row)
        slide_counts[row["slide_id"]] += 1
        if len(chosen) == count:
            return chosen
    raise ValueError(
        f"could select only {len(chosen)} of {count} registered locations"
    )


@torch.no_grad()
def train_global_mean_bands(
    dataset,
    selection,
    pyramid,
    batch_size,
):
    scanner_order = ("at2", *selection["scanners"])
    band_sums = None
    band_square_sum = 0.0
    band_element_count = 0
    image_count = 0
    for start in range(0, len(selection["keys"]), batch_size):
        positions = list(range(
            start,
            min(start + batch_size, len(selection["keys"])),
        ))
        images = load_lattice_batch(dataset, selection, positions)
        for scanner in scanner_order:
            _, bands = pyramid.decompose(images[scanner])
            if band_sums is None:
                band_sums = [
                    torch.zeros_like(band[:1]) for band in bands
                ]
            for index, band in enumerate(bands):
                band_sums[index] += band.sum(dim=0, keepdim=True)
                band_square_sum += float(band.square().sum())
                band_element_count += band.numel()
            image_count += len(images[scanner])
    if band_sums is None or image_count == 0:
        raise ValueError("no train images available for global HF mean")
    mean_bands = [band_sum / image_count for band_sum in band_sums]
    global_square_sum = sum(
        float(band.square().sum()) for band in mean_bands
    )
    global_element_count = sum(band.numel() for band in mean_bands)
    return mean_bands, {
        "train_images": image_count,
        "individual_band_rms": float(
            np.sqrt(band_square_sum / band_element_count)
        ),
        "global_mean_band_rms": float(
            np.sqrt(global_square_sum / global_element_count)
        ),
    }


def variant_specs():
    yield {
        "family": "raw",
        "parameter": 0.0,
        "effective_hf_gain": 1.0,
        "label": "Raw",
    }
    for severity in (0.25, 0.50, 0.75, 1.00):
        yield {
            "family": "blur",
            "parameter": severity,
            "effective_hf_gain": 1.0 - severity,
            "label": f"Blur {severity:.2f}",
        }
    for strength in (0.25, 0.50, 1.00):
        yield {
            "family": "sharpen",
            "parameter": strength,
            "effective_hf_gain": 1.0 + strength,
            "label": f"Sharpen {strength:.2f}",
        }
    for amount in (0.25, 0.50, 0.75, 1.00):
        yield {
            "family": "registered_mean",
            "parameter": amount,
            "effective_hf_gain": np.nan,
            "label": f"Registered HF mean {amount:.2f}",
        }
    for amount in (0.25, 0.50, 0.75, 1.00):
        yield {
            "family": "global_mean",
            "parameter": amount,
            "effective_hf_gain": np.nan,
            "label": f"Global HF mean {amount:.2f}",
        }


def detail_rms(pyramid, low, bands):
    detail = pyramid.reconstruct(torch.zeros_like(low), list(bands))
    return float(detail.square().mean().sqrt())


def operational_detail_rms(pyramid, image):
    return float(pyramid.copied_detail(image).square().mean().sqrt())


@torch.no_grad()
def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("UNI trajectory extraction requires a CUDA allocation")
    if args.locations < 1:
        raise ValueError("--locations must be positive")
    if not (
        0.0
        <= args.min_tissue_density
        <= args.max_tissue_density
        <= 1.0
    ):
        raise ValueError("tissue-density bounds must satisfy 0 <= min <= max <= 1")
    if not 0.0 <= args.max_blue_dominant_fraction <= 1.0:
        raise ValueError("--max-blue-dominant-fraction must be in [0,1]")

    cfg = load_config(args.config)
    scanners = [str(scanner) for scanner in cfg.target_scanners]
    pyramid = FixedLaplacianPyramid(int(cfg.pyramid.levels))

    train_dataset = BalancedPairDataset(cfg, "train", False)
    train_selection = internal_selection(
        train_dataset,
        scanners,
        args.train_mean_locations,
        args.seed,
    )
    global_mean_bands, global_audit = train_global_mean_bands(
        train_dataset,
        train_selection,
        pyramid,
        args.batch_size,
    )

    test_dataset = BalancedPairDataset(cfg, "test", False)
    test_selection = internal_selection(
        test_dataset, scanners, None, args.seed + 1
    )
    chosen = select_registered_locations(
        test_dataset,
        test_selection,
        args.locations,
        min_tissue_density=args.min_tissue_density,
        max_tissue_density=args.max_tissue_density,
        max_blue_dominant_fraction=args.max_blue_dominant_fraction,
    )
    positions = [row["position"] for row in chosen]
    images = load_lattice_batch(test_dataset, test_selection, positions)
    scanner_order = ("at2", *test_selection["scanners"])
    images = {scanner: images[scanner] for scanner in scanner_order}
    decomposed = {
        scanner: pyramid.decompose(images[scanner])
        for scanner in scanner_order
    }

    device = torch.device("cuda")
    model, size, mean, std = load_uni(device)

    rows = []
    output_images = []
    for location_index, selected in enumerate(chosen):
        registered_mean = []
        for band_index in range(pyramid.levels):
            registered_mean.append({
                scanner: torch.stack([
                    decomposed[other][1][band_index][location_index]
                    for other in scanner_order
                    if other != scanner
                ]).mean(dim=0, keepdim=True)
                for scanner in scanner_order
            })

        for scanner in scanner_order:
            low = decomposed[scanner][0][location_index:location_index + 1]
            bands = [
                band[location_index:location_index + 1]
                for band in decomposed[scanner][1]
            ]
            loo_bands = [
                registered_mean[index][scanner]
                for index in range(pyramid.levels)
            ]
            for spec in variant_specs():
                family = spec["family"]
                if family == "raw":
                    result = gain_high_frequency(
                        pyramid, low, bands, gain=1.0
                    )
                    requested_bands = bands
                elif family in ("blur", "sharpen"):
                    gain = float(spec["effective_hf_gain"])
                    result = gain_high_frequency(
                        pyramid, low, bands, gain=gain
                    )
                    requested_bands = [gain * band for band in bands]
                elif family == "registered_mean":
                    amount = float(spec["parameter"])
                    result = mix_high_frequency(
                        pyramid, low, bands, loo_bands, amount
                    )
                    requested_bands = [
                        (1.0 - amount) * source + amount * reference
                        for source, reference in zip(bands, loo_bands)
                    ]
                elif family == "global_mean":
                    amount = float(spec["parameter"])
                    result = mix_high_frequency(
                        pyramid,
                        low,
                        bands,
                        global_mean_bands,
                        amount,
                    )
                    requested_bands = [
                        (1.0 - amount) * source
                        + amount * reference.to(source)
                        for source, reference
                        in zip(bands, global_mean_bands)
                    ]
                else:
                    raise ValueError(f"unknown variant family: {family}")

                image = result["image"]
                preclip = result["preclip"]
                row_id = len(rows)
                rows.append({
                    "row_id": row_id,
                    "location_index": location_index,
                    "slide_id": selected["slide_id"],
                    "tuple_id": selected["tuple_id"],
                    "x": selected["x"],
                    "y": selected["y"],
                    "min_q_reg": selected["min_q_reg"],
                    "mean_q_reg": selected["mean_q_reg"],
                    "tissue_density": selected["tissue_density"],
                    "max_blue_dominant_fraction": (
                        selected["max_blue_dominant_fraction"]
                    ),
                    "scanner": scanner,
                    **spec,
                    "preclip_range_fraction": float(
                        ((preclip < -1.0) | (preclip > 1.0))
                        .float().mean()
                    ),
                    "clip_pixel_mae": float((image - preclip).abs().mean()),
                    "requested_hf_rms": detail_rms(
                        pyramid, low, requested_bands
                    ),
                    "operational_hf_rms": operational_detail_rms(
                        pyramid, image
                    ),
                })
                uint8 = (
                    ((image[0].permute(1, 2, 0) + 1.0) * 127.5)
                    .round().clamp(0, 255).byte().numpy()
                )
                output_images.append(uint8)

    tensor_images = torch.from_numpy(
        np.stack(output_images)
    ).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    embeddings = embed_uni(
        model,
        tensor_images,
        size,
        mean,
        std,
        device,
        batch_size=args.embed_batch_size,
    ).numpy()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "trajectory_metadata.csv", index=False)
    np.savez_compressed(
        output_dir / "trajectory_arrays.npz",
        embeddings=embeddings.astype(np.float32),
        images=np.stack(output_images),
    )
    global_audit["mean_to_individual_rms_ratio"] = (
        global_audit["global_mean_band_rms"]
        / max(global_audit["individual_band_rms"], 1e-12)
    )
    report = {
        "experiment": "Exp-02",
        "analysis": "high_frequency_embedding_trajectory",
        "config": args.config,
        "model": "UNI",
        "lattice_id": INTERNAL_LATTICE_ID,
        "scanners": list(scanner_order),
        "selection_policy": (
            "top minimum registration-QC after tissue-density, blue-dominant "
            "color-artifact, slide-diversity, and coordinate-separation QC"
        ),
        "selection_thresholds": {
            "min_tissue_density": args.min_tissue_density,
            "max_tissue_density": args.max_tissue_density,
            "max_blue_dominant_fraction": (
                args.max_blue_dominant_fraction
            ),
            "maximum_locations_per_slide": max(
                1, int(np.ceil(args.locations / 2))
            ),
            "minimum_coordinate_separation": 2048,
        },
        "locations": [
            {key: value for key, value in row.items() if key != "position"}
            for row in chosen
        ],
        "global_high_frequency_reference": global_audit,
        "variant_families": {
            family: [
                {
                    key: (
                        None
                        if isinstance(value, float) and np.isnan(value)
                        else value
                    )
                    for key, value in spec.items()
                }
                for spec in variant_specs()
                if spec["family"] == family
            ]
            for family in (
                "raw",
                "blur",
                "sharpen",
                "registered_mean",
                "global_mean",
            )
        },
        "rows": len(frame),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"[exp02-hf-trajectory] {len(chosen)} locations, "
        f"{len(scanner_order)} scanners, {len(frame)} variants -> {output_dir}"
    )


if __name__ == "__main__":
    main()

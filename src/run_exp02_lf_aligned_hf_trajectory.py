"""Extract UNI blur/sharpen trajectories after pairwise LF affine alignment.

Directed RGB affines are fitted only on train-split coarse Laplacian bands.
For each held-out registered location, every scanner is independently aligned
to each of the four rotating LF reference scanners. Source detail bands are
then attenuated or amplified before one explicit output clamp.
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

from prenorm.data.identity import INTERNAL_LATTICE_ID, matched_indices_by_lattice
from prenorm.embedding import embed_uni, load_uni
from prenorm.exp01.data import BalancedPairDataset
from prenorm.exp01.frequency import FixedLaplacianPyramid
from prenorm.exp02.pairwise import (
    apply_affine,
    fit_pairwise_affines,
    load_lattice_batch,
)
from prenorm.exp02.trajectory import gain_high_frequency
from run_exp02_hf_trajectory import (
    internal_selection,
    select_registered_locations,
)
from utils.config import load_config


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/experiments/exp01a_lf16_10slide.yaml"
    )
    parser.add_argument(
        "--output", default="outputs/exp02_lf_aligned_hf_trajectory"
    )
    parser.add_argument("--locations", type=int, default=3)
    parser.add_argument("--fit-max-locations", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--min-tissue-density", type=float, default=0.50)
    parser.add_argument("--max-tissue-density", type=float, default=0.90)
    parser.add_argument(
        "--max-blue-dominant-fraction", type=float, default=0.80
    )
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args(argv)


def variant_specs():
    yield {
        "family": "aligned_raw",
        "parameter": 0.0,
        "effective_hf_gain": 1.0,
        "label": "LF-aligned raw HF",
    }
    for severity in (0.25, 0.50, 0.75, 1.00):
        yield {
            "family": "blur",
            "parameter": severity,
            "effective_hf_gain": 1.0 - severity,
            "label": f"LF-aligned blur {severity:.2f}",
        }
    for strength in (0.25, 0.50, 1.00):
        yield {
            "family": "sharpen",
            "parameter": strength,
            "effective_hf_gain": 1.0 + strength,
            "label": f"LF-aligned sharpen {strength:.2f}",
        }


def per_image_mae(left, right):
    return (
        (left - right).abs().flatten(1).mean(dim=1).cpu().numpy()
    )


def uint8_image(image):
    return (
        ((image.permute(1, 2, 0) + 1.0) * 127.5)
        .round()
        .clamp(0, 255)
        .byte()
        .numpy()
    )


@torch.no_grad()
def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("LF-aligned UNI trajectories require CUDA")
    if args.locations < 1:
        raise ValueError("--locations must be positive")

    cfg = load_config(args.config)
    configured_scanners = [str(scanner) for scanner in cfg.target_scanners]
    pyramid = FixedLaplacianPyramid(int(cfg.pyramid.levels))

    fit_dataset = BalancedPairDataset(cfg, "train", False)
    fit_selections = matched_indices_by_lattice(
        fit_dataset.records,
        configured_scanners,
        args.fit_max_locations,
        np.random.default_rng(args.seed),
    )
    fitted = fit_pairwise_affines(
        fit_dataset,
        pyramid,
        fit_selections,
        batch_size=args.batch_size,
    )[INTERNAL_LATTICE_ID]

    test_dataset = BalancedPairDataset(cfg, "test", False)
    test_selection = internal_selection(
        test_dataset,
        configured_scanners,
        None,
        args.seed + 1,
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
        scanner: pyramid.decompose(image)
        for scanner, image in images.items()
    }

    rows = []
    output_images = []
    for lf_reference in scanner_order:
        target_low = decomposed[lf_reference][0]
        for source_scanner in scanner_order:
            source_low, source_bands = decomposed[source_scanner]
            if source_scanner == lf_reference:
                affine_low = source_low
                affine_name = "identity"
            else:
                affine_name = (
                    f"{source_scanner}_to_{lf_reference}"
                )
                affine_low = apply_affine(
                    source_low, fitted[affine_name]
                )
            low_mae_raw = per_image_mae(source_low, target_low)
            low_mae_affine = per_image_mae(affine_low, target_low)

            for spec in variant_specs():
                result = gain_high_frequency(
                    pyramid,
                    affine_low,
                    source_bands,
                    gain=float(spec["effective_hf_gain"]),
                )
                operational_low = pyramid.coarsest(result["image"])
                low_mae_operational = per_image_mae(
                    operational_low, target_low
                )
                preclip_fraction = (
                    (result["preclip"] < -1.0)
                    | (result["preclip"] > 1.0)
                ).float().flatten(1).mean(dim=1).cpu().numpy()
                clip_mae = (
                    (result["image"] - result["preclip"])
                    .abs()
                    .flatten(1)
                    .mean(dim=1)
                    .cpu()
                    .numpy()
                )
                for location_index, selected in enumerate(chosen):
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
                        "lf_reference_scanner": lf_reference,
                        "source_scanner": source_scanner,
                        "affine_direction": affine_name,
                        **spec,
                        "low_mae_raw_to_reference": float(
                            low_mae_raw[location_index]
                        ),
                        "low_mae_affine_to_reference": float(
                            low_mae_affine[location_index]
                        ),
                        "low_mae_operational_to_reference": float(
                            low_mae_operational[location_index]
                        ),
                        "preclip_range_fraction": float(
                            preclip_fraction[location_index]
                        ),
                        "clip_pixel_mae": float(
                            clip_mae[location_index]
                        ),
                    })
                    output_images.append(uint8_image(
                        result["image"][location_index]
                    ))

    tensor_images = torch.from_numpy(
        np.stack(output_images)
    ).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    device = torch.device("cuda")
    model, size, mean, std = load_uni(device)
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
    affine_report = {
        name: affine.tolist() for name, affine in fitted.items()
    }
    report = {
        "experiment": "Exp-02",
        "analysis": "lf_affine_then_high_frequency_trajectory",
        "config": args.config,
        "model": "UNI",
        "lattice_id": INTERNAL_LATTICE_ID,
        "fit_split": "train",
        "eval_split": "test",
        "fit_max_locations": args.fit_max_locations,
        "scanners": list(scanner_order),
        "lf_references": list(scanner_order),
        "transform_contract": (
            "train-fitted directed affine on the source coarsest "
            "Laplacian band; preserve source detail coefficients; apply "
            "one requested HF gain; explicitly clamp once"
        ),
        "locations": [
            {key: value for key, value in row.items() if key != "position"}
            for row in chosen
        ],
        "pairwise_affines": affine_report,
        "variant_families": {
            family: [
                spec for spec in variant_specs()
                if spec["family"] == family
            ]
            for family in ("aligned_raw", "blur", "sharpen")
        },
        "rows": len(frame),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"[exp02-lf-aligned-hf] {len(chosen)} locations, "
        f"{len(scanner_order)} LF references, {len(frame)} variants "
        f"-> {output_dir}"
    )


if __name__ == "__main__":
    main()

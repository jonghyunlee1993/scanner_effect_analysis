#!/usr/bin/env python3
"""Score frozen augmentation-oracle choices with benchmark SSIM and LPIPS.

Candidate selection is read from the completed image-phenotype experiment; no
candidate is selected using SSIM, LPIPS, or UNI. Results use the same 2-pixel
valid crop and LPIPS-VGG16 definition as the manuscript correction benchmark.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from skimage.color import rgb2hed
from skimage.metrics import structural_similarity
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
from augmentation_ood_study import (  # noqa: E402
    OUTPUT_ROOT as AUGMENTATION_ROOT,
    SCANNERS,
    augmentation_seed,
    load_libraries,
    render_augmentation,
)


ARMS = ("identity", "default_oracle", "strong_oracle", "strong_unconstrained_oracle")
OUTPUT = Path(__file__).resolve().parent / "results/augmentation_paired_metrics"


def score_slide(
    slide: pd.Series,
    selected: pd.DataFrame,
    libraries: dict[str, np.ndarray],
    model: LearnedPerceptualImagePatchSimilarity,
    batch_size: int,
) -> pd.DataFrame:
    slide_id = str(slide.slide_id)
    selected = selected[selected.slide_id.eq(slide_id)]
    locations = np.sort(selected.location_index.unique().astype(int))
    if len(locations) != 40 or len(selected) != 40 * 5 * len(ARMS):
        raise ValueError(f"incomplete selected augmentation arms for {slide_id}")
    with h5py.File(str(slide.cache_path), "r") as store:
        images = np.asarray(store["images"][locations], dtype=np.uint8)
    if images.shape != (40, 6, 256, 256, 3):
        raise ValueError(f"unexpected cache image shape {slide_id}: {images.shape}")

    chosen = {
        (int(row.location_index), str(row.scanner), str(row.arm)):
        (str(row.library), int(row.candidate_index))
        for row in selected.itertuples(index=False)
    }
    if len(chosen) != len(selected):
        raise ValueError(f"duplicate selected augmentation rows for {slide_id}")

    generated: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    metadata: list[tuple[str, int, str, str]] = []
    for row_index, location in enumerate(locations):
        source = images[row_index, 0]
        source_hed = rgb2hed(source.astype(np.float32) / 255.0).astype(np.float32)
        rendered: dict[tuple[str, int], np.ndarray] = {}
        for scanner_index, scanner in enumerate(SCANNERS[1:], start=1):
            target = images[row_index, scanner_index, 2:-2, 2:-2]
            for arm in ARMS:
                library, candidate_index = chosen[(int(location), scanner, arm)]
                key = (library, candidate_index)
                if key not in rendered:
                    parameters = libraries[library][candidate_index]
                    rendered[key] = render_augmentation(
                        source,
                        source_hed,
                        parameters,
                        augmentation_seed(slide_id, int(location), parameters),
                    )[2:-2, 2:-2]
                generated.append(rendered[key])
                targets.append(target)
                metadata.append((slide_id, int(location), scanner, arm))
    left = np.stack(generated)
    right = np.stack(targets)
    ssim = np.asarray(
        [structural_similarity(right[i], left[i], channel_axis=-1, data_range=255)
         for i in range(len(left))], dtype=np.float64
    )
    lpips_values = []
    for start in range(0, len(left), batch_size):
        stop = min(start + batch_size, len(left))
        x = torch.from_numpy(left[start:stop].copy()).permute(0, 3, 1, 2).to("cuda", torch.float32).div_(255)
        y = torch.from_numpy(right[start:stop].copy()).permute(0, 3, 1, 2).to("cuda", torch.float32).div_(255)
        with torch.inference_mode():
            lpips_values.append(model(x, y).detach().float().cpu().numpy().reshape(-1))
    frame = pd.DataFrame(metadata, columns=["slide_id", "location_index", "scanner", "arm"])
    frame["ssim"] = ssim
    frame["lpips_vgg"] = np.concatenate(lpips_values)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--task-count", type=int, required=True)
    parser.add_argument("--max-slides", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("LPIPS scoring requires a GPU")
    if not 0 <= args.task_index < args.task_count:
        raise ValueError("invalid task index")
    cohort = pd.read_csv(AUGMENTATION_ROOT / "00_contract/cohort.csv", dtype={"slide_id": str})
    selected = pd.read_csv(
        AUGMENTATION_ROOT / "02_aggregate/selected_candidates.csv",
        dtype={"slide_id": str},
        usecols=["slide_id", "location_index", "scanner", "arm", "library", "candidate_index"],
    )
    selected = selected[selected.arm.isin(ARMS)]
    libraries = load_libraries(AUGMENTATION_ROOT)
    model = LearnedPerceptualImagePatchSimilarity(
        net_type="vgg", reduction="none", normalize=True
    ).to("cuda").eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    slides = cohort.iloc[args.task_index :: args.task_count]
    if args.max_slides:
        slides = slides.iloc[: args.max_slides]
    shard_root = OUTPUT / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    for slide in slides.itertuples(index=False):
        path = shard_root / f"{slide.slide_id}.csv.gz"
        if path.exists():
            print(f"skip completed slide {slide.slide_id}", flush=True)
            continue
        frame = score_slide(slide, selected, libraries, model, args.batch_size)
        temporary = path.with_name(f".{path.name}.tmp")
        frame.to_csv(temporary, index=False, compression="gzip")
        temporary.replace(path)
        print(f"scored {slide.slide_id}: {len(frame)} paired comparisons", flush=True)


if __name__ == "__main__":
    main()

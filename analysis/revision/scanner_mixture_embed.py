#!/usr/bin/env python3
"""RV17 stage 2 (GPU): render the held-out conditions once and embed them with every RV17 checkpoint.

Same images, QC and embedding path as RV16 (``scanner_composition_embed``: the 81 RV13
conditions of the fold-0 ``set20`` locations, band renders checked against RV02, raw six-scanner
bank of the pretraining slides; teacher layer-normed CLS, 256 -> 224 antialiased bicubic,
ImageNet normalization, fp32, unit norm). Models: the step-0 teacher (verified identical in the
four arms) and each arm after epochs 10 and 25.

Outputs (``results/scanner_mixture/``): ``embeddings/<model>.h5``, ``embeddings/*_locations.csv``,
``render/<slide>_band.csv.gz``, ``qc/embed.json``.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REVISION = Path(__file__).resolve().parent
if str(REVISION) not in sys.path:
    sys.path.insert(0, str(REVISION))

import scanner_composition_embed as E  # noqa: E402  (sets up project and DINOv2 paths)
from scanner_composition_embed import C, CONDITIONS, band_parity  # noqa: E402
from scanner_mixture_pretrain import ARM_ORDER, register_vit_tiny  # noqa: E402

OUTPUT = REVISION / "results/scanner_mixture"
EPOCHS = (10, 25)


def checkpoints(root: Path) -> dict[str, Path]:
    tags = {"init_e000": root / "train" / ARM_ORDER[0] / "teacher_e000.pt"}
    for arm in ARM_ORDER:
        for epoch in EPOCHS:
            tags[f"{arm}_e{epoch:03d}"] = root / "train" / arm / f"teacher_e{epoch:03d}.pt"
    return tags


class Teachers(E.Teachers):
    """RV16 embedding path; the backbone is rebuilt from the checkpoint's arch."""

    def __init__(self, paths: dict[str, Path], batch_size: int = 256):
        import torch
        from dinov2.models import vision_transformer as vits

        register_vit_tiny()
        self.torch = torch
        self.batch_size = batch_size
        self.models, self.meta = {}, {}
        for tag, path in paths.items():
            state = torch.load(path, map_location="cpu")
            model = getattr(vits, state["arch"])(**state["vit_kwargs"])
            model.load_state_dict(state["teacher_backbone"], strict=True)
            self.models[tag] = model.cuda().eval()
            self.meta[tag] = {"iteration": int(state["iteration"]), "epoch": float(state["epoch"]),
                              "arm": state["arm"], "arch": state["arch"]}
        self.mean = torch.tensor(E.IMAGENET_MEAN, device="cuda").view(1, 3, 1, 1)
        self.std = torch.tensor(E.IMAGENET_STD, device="cuda").view(1, 3, 1, 1)


def check_init(root: Path) -> dict:
    import torch

    reference = torch.load(root / "train" / ARM_ORDER[0] / "teacher_e000.pt", map_location="cpu")["teacher_backbone"]
    result = {}
    for arm in ARM_ORDER:
        other = torch.load(root / "train" / arm / "teacher_e000.pt", map_location="cpu")["teacher_backbone"]
        result[arm] = bool(reference.keys() == other.keys() and all(torch.equal(reference[k], other[k]) for k in reference))
    if not all(result.values()):
        raise ValueError(f"step-0 teachers differ across arms: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    parser.add_argument("--max-slides", type=int, default=0, help="smoke test only")
    parser.add_argument("--max-locations", type=int, default=0, help="smoke test only")
    args = parser.parse_args()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.set_num_threads(1)
    root = args.output_root
    smoke = bool(args.max_slides or args.max_locations)
    paths = {tag: path for tag, path in checkpoints(root).items() if path.exists() or not smoke}
    init_check = check_init(root) if not smoke else {}
    cohort = C.load_cohort()
    set20 = C.load_set(C.SET20, 20)
    fitted = json.loads(C.PARAMETERS.read_text())["fold"]["0"]
    held_out = cohort[cohort.fold.eq(0)].sort_values("slide_id", kind="stable")
    bank = cohort[cohort.fold.ne(0)].sort_values("slide_id", kind="stable")
    if args.max_slides:
        held_out, bank = held_out.iloc[:args.max_slides], bank.iloc[:args.max_slides]

    pool = mp.get_context("fork").Pool(max(1, args.workers), initializer=E._init_worker)
    teachers = Teachers(paths)
    print(f"loaded {len(paths)} checkpoints: {json.dumps(teachers.meta)}", flush=True)
    held_features = {tag: [] for tag in paths}
    held_rows, band_checks = [], []
    for slide in held_out.itertuples(index=False):
        started = time.time()
        rows20 = set20[set20.slide_id.eq(str(slide.slide_id))].sort_values("location_index")
        locations = rows20.location_index.to_numpy(np.int64)
        if args.max_locations:
            locations = locations[:args.max_locations]
        with h5py.File(slide.cache_path, "r") as cache:
            paired = np.asarray(cache["images"][locations], dtype=np.uint8)
            source_index = np.asarray(cache["source_index"][locations], dtype=np.int64)
        if not np.array_equal(source_index, rows20.source_index.to_numpy(np.int64)[:len(locations)]):
            raise ValueError(f"{slide.slide_id}: source_index differs from set20")
        results = pool.map(E._render, [(str(slide.slide_id), int(loc), paired[i], fitted) for i, loc in enumerate(locations)])
        band_rows = [row for r in results for row in r[2]]
        check = band_parity(str(slide.slide_id), band_rows)
        band_checks.append(check)
        if not check["values_identical"] or (not check["complete"] and not smoke):
            raise ValueError(f"{slide.slide_id}: band renders differ from RV02: {check}")
        C.write_frame(root / "render" / f"{slide.slide_id}_band.csv.gz", pd.DataFrame(band_rows))
        vectors = teachers(np.concatenate([r[1] for r in results]))
        for tag, value in vectors.items():
            held_features[tag].append(value.reshape(len(locations), len(CONDITIONS), -1))
        held_rows.append(pd.DataFrame({"slide_id": str(slide.slide_id), "tissue_type": str(slide.tissue_type),
                                       "location_index": locations, "source_index": source_index}))
        print(f"held-out {slide.slide_id}: {len(locations)} x {len(CONDITIONS)} in {time.time() - started:.0f}s", flush=True)
    pool.close()
    pool.join()

    bank_features = {tag: [] for tag in paths}
    bank_rows = []
    for slide in bank.itertuples(index=False):
        rows20 = set20[set20.slide_id.eq(str(slide.slide_id))].sort_values("location_index")
        locations = rows20.location_index.to_numpy(np.int64)
        if args.max_locations:
            locations = locations[:args.max_locations]
        with h5py.File(slide.cache_path, "r") as cache:
            paired = np.asarray(cache["images"][locations], dtype=np.uint8)
        vectors = teachers(paired.reshape(-1, 256, 256, 3).astype(np.float32) / 255.0)
        for tag, value in vectors.items():
            bank_features[tag].append(value.reshape(len(locations), len(C.SCANNERS), -1))
        bank_rows.append(pd.DataFrame({"slide_id": str(slide.slide_id), "tissue_type": str(slide.tissue_type),
                                       "location_index": locations}))

    held_frame, bank_frame = pd.concat(held_rows, ignore_index=True), pd.concat(bank_rows, ignore_index=True)
    C.write_frame(root / "embeddings" / "held_out_locations.csv", held_frame)
    C.write_frame(root / "embeddings" / "bank_locations.csv", bank_frame)
    for tag in paths:
        C.write_h5(root / "embeddings" / f"{tag}.h5",
                   {"held_out": np.concatenate(held_features[tag]).astype(np.float32),
                    "bank": np.concatenate(bank_features[tag]).astype(np.float32),
                    "condition_names": np.asarray(CONDITIONS), "bank_scanners": np.asarray(C.SCANNERS)},
                   {"model": tag, **teachers.meta[tag], "stage": "rv17_embed"})
    C.write_json(root / "qc" / "embed.json", {"tags": list(paths), "init_identical": init_check, "smoke": smoke,
                                               "held_out_locations": len(held_frame), "bank_locations": len(bank_frame),
                                               "band_parity": band_checks})
    print(json.dumps({"status": "complete", "models": len(paths), "held_out": len(held_frame)}), flush=True)


if __name__ == "__main__":
    main()

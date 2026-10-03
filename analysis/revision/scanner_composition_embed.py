#!/usr/bin/env python3
"""RV16 stage 2 (GPU): render the held-out conditions once and embed them with every checkpoint.

Protocol: ``scanner_composition_pilot_protocol.md``.

* Held-out images: ``set20`` locations of the 21 fold-0 slides, the 81 RV13 conditions
  (raw six scanners; Reinhard, frequency and colour + frequency AT2 -> each target; the 60 RV02
  band manipulations), rendered with the RV02/RV03 functions unchanged and the fold-0
  correction parameters. QC: the band metadata must equal the RV02 shards row for row and the
  colour-matched base must equal the Reinhard image (``anchor_frequency_diversity_embed``
  checks, imported).
* Bank images (tissue gate): raw six-scanner images at the ``set20`` locations of the 82
  pretraining slides.
* Models: the step-0 teacher (identical in every arm; verified here) and the teacher backbone of
  each arm (A, B, A1, B1) after epochs 10, 25, 50 and 100. Teacher layer-normed CLS token; the
  256-px field resized to 224 px (antialiased bicubic, float tensors), ImageNet normalization,
  fp32; unit-normalized.

Outputs (``results/scanner_composition_pilot/``): ``embeddings/<model>.h5`` (``held_out``
[420, 81, 384], ``bank`` [1640, 6, 384]), ``embeddings/held_out_locations.csv``,
``embeddings/bank_locations.csv``, ``render/<slide>_band.csv.gz``, ``qc/``.
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

PROJECT = Path(__file__).resolve().parents[2]
REVISION = Path(__file__).resolve().parent
DINOV2 = REVISION / "third_party/dinov2_7764ea0f"
for _path in (PROJECT, PROJECT / "src", PROJECT / "scripts", REVISION, DINOV2):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import corrected_embeddings_common as C  # noqa: E402
from anchor_frequency_diversity_embed import CONDITIONS, TARGETS, band_name, band_parity  # noqa: E402

OUTPUT = REVISION / "results/scanner_composition_pilot"
HELD_OUT_FOLD = 0
ARMS = ("A", "B", "A1", "B1")
EPOCHS = (10, 25, 50, 100)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def model_tags() -> list[str]:
    return ["init_e000"] + [f"{arm}_e{epoch:03d}" for arm in ARMS for epoch in EPOCHS]


def checkpoint_path(root: Path, tag: str) -> Path:
    if tag == "init_e000":
        return root / "train" / "A" / "teacher_e000.pt"
    arm, epoch = tag.split("_e")
    return root / "train" / arm / f"teacher_e{epoch}.pt"


def _init_worker() -> None:
    import torch

    torch.set_num_threads(1)


def _render(job):
    """The 81 RV13 conditions of one location (no Macenko), float32 in [0, 1]."""
    from band_manipulation_set20_features import render_location
    from scanner_batch_extensions import frequency_transform, reinhard_transform

    slide_id, location, paired, fitted = job
    source = paired[0]
    images = {"source_at2": source.astype(np.float32) / 255.0}
    for position, scanner in enumerate(TARGETS, start=1):
        images[f"target_{scanner}"] = paired[position].astype(np.float32) / 255.0
        reinhard = reinhard_transform(source, fitted[scanner])
        images[f"reinhard_to_{scanner}"] = reinhard.astype(np.float32) / 255.0
        images[f"frequency_to_{scanner}"] = frequency_transform(source, fitted[scanner]).astype(np.float32) / 255.0
        images[f"combined_to_{scanner}"] = frequency_transform(reinhard, fitted[scanner]).astype(np.float32) / 255.0
    rendered, slots = render_location((location, paired, fitted))
    rows = []
    for slot in slots:
        scanner = slot["scanner"]
        if not np.array_equal(rendered[slot["base"]], images[f"reinhard_to_{scanner}"]):
            raise ValueError(f"{slide_id}/{location}/{scanner}: RV02 colour-matched base != RV03 Reinhard")
        if not np.array_equal(rendered[slot["target"]], images[f"target_{scanner}"]):
            raise ValueError(f"{slide_id}/{location}/{scanner}: RV02 target != cache target")
        for item in slot["perturbations"]:
            name = band_name(item["band"], item["sign"], item["dose_fraction"], scanner)
            images[name] = rendered[item["index"]]
            rows.append({"slide_id": slide_id, "location_index": int(location), "scanner": scanner,
                         "condition": name, **{k: v for k, v in item.items() if k != "index"}})
    stack = np.stack([np.asarray(images[name], dtype=np.float32) for name in CONDITIONS])
    return int(location), stack, rows


class Teachers:
    """Every checkpoint on one GPU; embeds a list of HWC float images with each of them."""

    def __init__(self, root: Path, tags: list[str], batch_size: int = 256):
        import torch
        from dinov2.models.vision_transformer import vit_small

        self.torch = torch
        self.batch_size = batch_size
        self.models, self.meta = {}, {}
        for tag in tags:
            state = torch.load(checkpoint_path(root, tag), map_location="cpu")
            if state["arch"] != "vit_small":
                raise ValueError(f"{tag}: arch {state['arch']}")
            model = vit_small(**state["vit_kwargs"])
            model.load_state_dict(state["teacher_backbone"], strict=True)
            self.models[tag] = model.cuda().eval()
            self.meta[tag] = {"iteration": int(state["iteration"]), "epoch": float(state["epoch"]), "arm": state["arm"]}
        self.mean = torch.tensor(IMAGENET_MEAN, device="cuda").view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device="cuda").view(1, 3, 1, 1)

    def __call__(self, images: np.ndarray) -> dict[str, np.ndarray]:
        torch = self.torch
        import torch.nn.functional as F

        outputs = {tag: [] for tag in self.models}
        with torch.inference_mode():
            for start in range(0, len(images), self.batch_size):
                batch = torch.from_numpy(np.ascontiguousarray(images[start:start + self.batch_size])).cuda()
                batch = batch.permute(0, 3, 1, 2).float()
                batch = F.interpolate(batch, size=(224, 224), mode="bicubic", align_corners=False, antialias=True)
                batch = (batch - self.mean) / self.std
                for tag, model in self.models.items():
                    cls = model.forward_features(batch)["x_norm_clstoken"].float()
                    outputs[tag].append(F.normalize(cls, dim=-1).cpu().numpy())
        return {tag: np.concatenate(value) for tag, value in outputs.items()}


def check_init(root: Path) -> dict:
    """The step-0 teacher must be bit-identical in every arm."""
    import torch

    reference = torch.load(checkpoint_path(root, "init_e000"), map_location="cpu")["teacher_backbone"]
    result = {}
    for arm in ARMS:
        other = torch.load(root / "train" / arm / "teacher_e000.pt", map_location="cpu")["teacher_backbone"]
        same = reference.keys() == other.keys() and all(torch.equal(reference[k], other[k]) for k in reference)
        result[arm] = bool(same)
    if not all(result.values()):
        raise ValueError(f"step-0 teachers differ across arms: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    parser.add_argument("--max-slides", type=int, default=0, help="smoke test only")
    parser.add_argument("--max-locations", type=int, default=0, help="smoke test only")
    parser.add_argument("--tags", default="", help="comma-separated subset (smoke test only)")
    parser.add_argument("--epochs", default="", help="comma-separated epochs to embed (with the step-0 model)")
    parser.add_argument("--label", default="all", help="name of this embedding run in qc/")
    args = parser.parse_args()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    os.chdir(PROJECT)
    torch.set_num_threads(1)
    root = args.output_root
    smoke = bool(args.max_slides or args.max_locations or args.tags)
    tags = args.tags.split(",") if args.tags else model_tags()
    if args.epochs:
        wanted = {f"e{int(e):03d}" for e in args.epochs.split(",")}
        tags = [t for t in tags if t == "init_e000" or t.split("_")[-1] in wanted]
    init_check = {} if smoke else check_init(root)
    cohort = C.load_cohort()
    set20 = C.load_set(C.SET20, 20)
    fitted = json.loads(C.PARAMETERS.read_text())["fold"][str(HELD_OUT_FOLD)]
    held_out = cohort[cohort.fold.eq(HELD_OUT_FOLD)].sort_values("slide_id", kind="stable")
    bank = cohort[cohort.fold.ne(HELD_OUT_FOLD)].sort_values("slide_id", kind="stable")
    if args.max_slides:
        held_out, bank = held_out.iloc[:args.max_slides], bank.iloc[:args.max_slides]

    pool = mp.get_context("fork").Pool(max(1, args.workers), initializer=_init_worker)
    teachers = Teachers(root, tags)
    print(f"loaded {len(tags)} checkpoints: {json.dumps(teachers.meta)}", flush=True)
    held_features = {tag: [] for tag in tags}
    held_rows, band_checks = [], []
    for slide in held_out.itertuples(index=False):
        started = time.time()
        rows20 = set20[set20.slide_id.eq(str(slide.slide_id))].sort_values("location_index")
        locations = rows20.location_index.to_numpy(np.int64)
        if args.max_locations:
            locations = locations[:args.max_locations]
        with h5py.File(slide.cache_path, "r") as cache:
            if tuple(x.decode().lower() for x in cache["scanner_names"][:]) != C.SCANNERS:
                raise ValueError(f"{slide.slide_id}: scanner order")
            paired = np.asarray(cache["images"][locations], dtype=np.uint8)
            source_index = np.asarray(cache["source_index"][locations], dtype=np.int64)
        if not np.array_equal(source_index, rows20.source_index.to_numpy(np.int64)[:len(locations)]):
            raise ValueError(f"{slide.slide_id}: source_index differs from set20")
        results = pool.map(_render, [(str(slide.slide_id), int(loc), paired[i], fitted) for i, loc in enumerate(locations)])
        if [r[0] for r in results] != [int(v) for v in locations]:
            raise ValueError("worker results out of order")
        band_rows = [row for r in results for row in r[2]]
        check = band_parity(str(slide.slide_id), band_rows)
        band_checks.append(check)
        if not check["values_identical"] or (not check["complete"] and not smoke):
            raise ValueError(f"{slide.slide_id}: band renders differ from RV02: {check}")
        C.write_frame(root / "render" / f"{slide.slide_id}_band.csv.gz", pd.DataFrame(band_rows))
        stack = np.concatenate([r[1] for r in results])  # [n * 81, 256, 256, 3]
        vectors = teachers(stack)
        for tag, value in vectors.items():
            held_features[tag].append(value.reshape(len(locations), len(CONDITIONS), -1))
        held_rows.append(pd.DataFrame({"slide_id": str(slide.slide_id), "tissue_type": str(slide.tissue_type),
                                       "location_index": locations, "source_index": source_index}))
        print(f"held-out {slide.slide_id}: {len(locations)} x {len(CONDITIONS)} in {time.time() - started:.0f}s", flush=True)
    pool.close()
    pool.join()

    bank_features = {tag: [] for tag in tags}
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
    print("bank embedded", flush=True)

    held_frame = pd.concat(held_rows, ignore_index=True)
    bank_frame = pd.concat(bank_rows, ignore_index=True)
    C.write_frame(root / "embeddings" / "held_out_locations.csv", held_frame)
    C.write_frame(root / "embeddings" / "bank_locations.csv", bank_frame)
    for tag in tags:
        C.write_h5(root / "embeddings" / f"{tag}.h5",
                   {"held_out": np.concatenate(held_features[tag]).astype(np.float32),
                    "bank": np.concatenate(bank_features[tag]).astype(np.float32),
                    "condition_names": np.asarray(CONDITIONS), "bank_scanners": np.asarray(C.SCANNERS)},
                   {"model": tag, **{k: v for k, v in teachers.meta[tag].items()}, "stage": "rv16_embed"})
    C.write_json(root / "qc" / f"embed_{args.label}.json", {"tags": tags, "init_identical": init_check, "smoke": smoke,
                                               "held_out_locations": len(held_frame), "bank_locations": len(bank_frame),
                                               "band_parity": band_checks})
    print(json.dumps({"status": "complete", "models": len(tags), "held_out": len(held_frame), "bank": len(bank_frame)}), flush=True)


if __name__ == "__main__":
    main()

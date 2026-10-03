#!/usr/bin/env python3
"""RV18 stage 2 (GPU): embed the held-out conditions with UNI (epoch 0) and every continued checkpoint.

Images, QC and outputs as RV17 (``scanner_mixture_embed``); the embedding path is the paper's
UNI path (``prenorm.embedding.embed_uni``): bicubic 256 -> 224 without antialiasing, ImageNet
normalization, layer-normed CLS, fp32 (TF32 allowed), unit norm. Models: epoch 0 (= UNI,
verified identical in the four arms) and each arm after the requested epochs.
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

import scanner_composition_embed as E  # noqa: E402
import scanner_mixture_embed as XE  # noqa: E402
from scanner_composition_embed import C, CONDITIONS, band_parity  # noqa: E402
from uni_continued_pretrain import ARM_ORDER, OUTPUTS, register_vit_huge2  # noqa: E402

OUTPUT = REVISION / "results/uni_continued"


class UniPathTeachers(XE.Teachers):
    """RV17 loader; images resized as the paper's UNI/Virchow2 path (no antialiasing), TF32 matmuls.

    base "uni": layer-normed CLS, fp32. base "virchow2" (RV19): CLS + mean patch token (registers
    excluded), bf16 autocast, inputs clamped to [0, 1] -- the manuscript's Virchow2 path.
    """

    base = "uni"

    def __call__(self, images: np.ndarray) -> dict[str, np.ndarray]:
        if self.base == "virchow2":
            return self._virchow2(images)
        torch = self.torch
        import torch.nn.functional as F

        torch.backends.cuda.matmul.allow_tf32 = True
        outputs = {tag: [] for tag in self.models}
        with torch.inference_mode():
            for start in range(0, len(images), self.batch_size):
                batch = torch.from_numpy(np.ascontiguousarray(images[start:start + self.batch_size])).cuda()
                batch = batch.permute(0, 3, 1, 2).float()
                batch = F.interpolate(batch, size=(224, 224), mode="bicubic", align_corners=False)
                batch = (batch - self.mean) / self.std
                for tag, model in self.models.items():
                    cls = model.forward_features(batch)["x_norm_clstoken"].float()
                    outputs[tag].append(F.normalize(cls, dim=-1).cpu().numpy())
        return {tag: np.concatenate(value) for tag, value in outputs.items()}


    def _virchow2(self, images: np.ndarray) -> dict[str, np.ndarray]:
        torch = self.torch
        import torch.nn.functional as F

        torch.backends.cuda.matmul.allow_tf32 = True
        outputs = {tag: [] for tag in self.models}
        with torch.inference_mode():
            for start in range(0, len(images), self.batch_size):
                batch = torch.from_numpy(np.ascontiguousarray(images[start:start + self.batch_size])).cuda()
                batch = batch.permute(0, 3, 1, 2).float()
                batch = F.interpolate(batch, size=(224, 224), mode="bicubic", align_corners=False)
                batch = (batch.clamp(0, 1) - self.mean) / self.std
                for tag, model in self.models.items():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        out = model.forward_features(batch)
                        vector = torch.cat([out["x_norm_clstoken"], out["x_norm_patchtokens"].mean(dim=1)], dim=-1)
                    outputs[tag].append(F.normalize(vector.float(), dim=-1).cpu().numpy())
        return {tag: np.concatenate(value) for tag, value in outputs.items()}


def checkpoints(root: Path, epochs: list[int], runs: list[str]) -> dict[str, Path]:
    tags = {"init_e000": root / "train" / ARM_ORDER[0] / "teacher_e000.pt"}
    for run in runs:
        for epoch in epochs:
            tags[f"{run}_e{epoch:03d}"] = root / "train" / run / f"teacher_e{epoch:03d}.pt"
    return tags


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--base", choices=sorted(OUTPUTS), default="uni")
    parser.add_argument("--epochs", default="1,3")
    parser.add_argument("--runs", default=",".join(ARM_ORDER), help="run names under train/")
    parser.add_argument("--label", default="", help="qc file label")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    parser.add_argument("--max-slides", type=int, default=0, help="smoke test only")
    parser.add_argument("--max-locations", type=int, default=0, help="smoke test only")
    args = parser.parse_args()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.set_num_threads(1)
    root = args.output_root or OUTPUTS[args.base]
    register_vit_huge2()
    UniPathTeachers.base = args.base
    smoke = bool(args.max_slides or args.max_locations)
    paths = {t: p for t, p in checkpoints(root, [int(e) for e in args.epochs.split(",")], args.runs.split(",")).items() if p.exists() or not smoke}
    init_check = XE.check_init(root) if not smoke else {}
    cohort = C.load_cohort()
    set20 = C.load_set(C.SET20, 20)
    fitted = json.loads(C.PARAMETERS.read_text())["fold"]["0"]
    held_out = cohort[cohort.fold.eq(0)].sort_values("slide_id", kind="stable")
    bank = cohort[cohort.fold.ne(0)].sort_values("slide_id", kind="stable")
    if args.max_slides:
        held_out, bank = held_out.iloc[:args.max_slides], bank.iloc[:args.max_slides]

    pool = mp.get_context("fork").Pool(max(1, args.workers), initializer=E._init_worker)
    teachers = UniPathTeachers(paths, batch_size=128)
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
                   {"model": tag, **teachers.meta[tag], "stage": "rv18_embed"})
    C.write_json(root / "qc" / f"embed_{args.label or 'e' + args.epochs.replace(',', '_')}.json",
                 {"tags": list(paths), "init_identical": init_check, "smoke": smoke, "held_out_locations": len(held_frame),
                  "bank_locations": len(bank_frame), "band_parity": band_checks})
    print(json.dumps({"status": "complete", "models": len(paths), "held_out": len(held_frame)}), flush=True)


if __name__ == "__main__":
    main()

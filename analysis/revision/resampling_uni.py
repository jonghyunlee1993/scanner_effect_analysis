#!/usr/bin/env python3
"""RV10 Stage C (UNI v1): embed native re-renders, native AT2 and pipeline patches.

UNI v1 is applied exactly as for the stored ``03_uni`` embeddings
(``src/scanner_batch_extensions.py::uni_extract_task``): uint8 RGB / 255, bicubic
resize to the model input size, fp16 autocast, model mean/std, L2 normalisation.
The pipeline (cached registered) patches are embedded again so that parity with the
stored embeddings can be checked before the variant distances are used.  Without CUDA the
same model runs in fp32 on CPU (the parity check then also covers the precision change).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from resampling_geometry import COHORT, OUTPUT, PATCH, SCANNERS, TARGETS  # noqa: E402
from resampling_render import RENDER, VARIANTS  # noqa: E402

UNI = OUTPUT / "uni"
STORED_UNI = PROJECT / "outputs/scanner_batch_effect_analysis_2026-09-17/03_uni/shards"


def embed(model, size, mean, std, device, images: np.ndarray, batch_size: int) -> np.ndarray:
    import torch
    import torch.nn.functional as functional

    outputs = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size]
            tensor = torch.from_numpy(batch.astype(np.float32)).permute(0, 3, 1, 2).div_(255.0)
            tensor = tensor.to(device, non_blocking=True)
            tensor = functional.interpolate(tensor, size=(size, size), mode="bicubic",
                                            align_corners=False)
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    feature = model((tensor - mean) / std)
            else:  # CPU fallback when the GPU partition is saturated: fp32
                feature = model((tensor - mean) / std)
            outputs.append(functional.normalize(feature.float(), dim=1).cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def task(task_index: int, task_count: int, batch_size: int) -> None:
    import torch
    from prenorm.embedding import load_uni

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        device = torch.device("cpu")
        torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    print(json.dumps({"device": str(device)}), flush=True)
    model, size, mean, std = load_uni(device)
    cohort = pd.read_csv(COHORT, dtype={"slide_id": str})
    (UNI / "shards").mkdir(parents=True, exist_ok=True)
    for row in cohort.iloc[task_index::task_count].itertuples(index=False):
        slide_id = str(row.slide_id)
        with h5py.File(RENDER / "shards" / f"{slide_id}.h5", "r") as store:
            location_index = np.asarray(store["location_index"], dtype=int)
            rendered = np.asarray(store["rendered"], dtype=np.uint8)
            rendered_ok = np.asarray(store["rendered_ok"], dtype=bool)
            native_at2 = np.asarray(store["native_at2"], dtype=np.uint8)
        with h5py.File(row.cache_path, "r") as store:
            pipeline = np.asarray(store["images"][location_index.tolist()], dtype=np.uint8)
        n = len(location_index)
        pipeline_features = embed(model, size, mean, std, device,
                                  pipeline.reshape(-1, PATCH, PATCH, 3), batch_size
                                  ).reshape(n, len(SCANNERS), -1)
        at2_features = embed(model, size, mean, std, device, native_at2, batch_size)
        variant_features = embed(model, size, mean, std, device,
                                 rendered.reshape(-1, PATCH, PATCH, 3), batch_size
                                 ).reshape(n, len(TARGETS), len(VARIANTS), -1)
        variant_features[~rendered_ok] = np.nan

        with h5py.File(STORED_UNI / f"{slide_id}.h5", "r") as store:
            stored_names = [name.decode() for name in store["condition_names"][:]]
            stored_locations = np.asarray(store["location_index"], dtype=int)
            stored = np.asarray(store["features"], dtype=np.float32)
        if not np.array_equal(stored_locations, location_index):
            raise ValueError(f"{slide_id}: stored UNI locations differ from set20")
        parity = {"source": float(np.min(np.sum(
            stored[:, stored_names.index("source")] * pipeline_features[:, 0], axis=1)))}
        for scanner in TARGETS:
            parity[scanner] = float(np.min(np.sum(
                stored[:, stored_names.index(f"target:{scanner}")]
                * pipeline_features[:, SCANNERS.index(scanner)], axis=1)))

        output = UNI / "shards" / f"{slide_id}.h5"
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        with h5py.File(temporary, "w") as store:
            store.attrs["slide_id"] = slide_id
            store.attrs["tissue_type"] = str(row.tissue_type)
            store.create_dataset("location_index", data=location_index)
            store.create_dataset("scanner_names", data=np.asarray(SCANNERS, dtype="S8"))
            store.create_dataset("target_scanners", data=np.asarray(TARGETS, dtype="S8"))
            store.create_dataset("variants", data=np.asarray(VARIANTS, dtype="S16"))
            store.create_dataset("pipeline", data=pipeline_features, compression="lzf")
            store.create_dataset("native_at2", data=at2_features, compression="lzf")
            store.create_dataset("variant", data=variant_features, compression="lzf")
        temporary.replace(output)
        summary = {"slide_id": slide_id, "status": "pass", "min_cosine_to_stored": parity}
        output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-index", type=int,
                        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    parser.add_argument("--task-count", type=int,
                        default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1")))
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    task(args.task_index, args.task_count, args.batch_size)


if __name__ == "__main__":
    main()

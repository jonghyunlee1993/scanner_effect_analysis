"""Encode one PLISM render condition with one frozen PFM — Arm B.

Uses the study's own encoder path: the pinned TRIDENT checkout, the frozen
checkpoint manifest, each model's official eval transform, and the same
centre-crop to its native field of view.  Nothing about the encoder differs from
the locked PanNormal extraction, so a difference in the result is a difference in
the cohort and not in the runtime.

One task per (condition, encoder).  Conditions are the rendered tile stacks:
raw, Reinhard toward each destination, and RF1U toward each destination.

Contract: docs/e9_plism_native_ert_contract.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

from extract_e0_pfm_features import (
    BATCH_SIZE,
    select_model,
    transformed_batch,
)
from fetch_e0_pfm_checkpoints import TRIDENT_COMMIT
from render_e0_native_aa_shard import MODEL_FOV
from smoke_e0_pfm_encoders import encoder_kwargs, verify_trident_source

FEATURE_VERSION = "plism_arm_b_four_pfm_v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", default="outputs/plism_render")
    parser.add_argument("--condition", required=True)
    parser.add_argument("--encoder-index", type=int, required=True)
    parser.add_argument("--contract", default="outputs/e0_pfm_contract/checkpoint_manifest.json")
    parser.add_argument("--trident-root",
                        default="/mnt/isilon/oldridge_lab/irinaz/github/env/trident")
    parser.add_argument("--output", default="outputs/plism_pfm_features")
    parser.add_argument("--batch-size", type=int, default=0)
    return parser.parse_args()


def centered_crop(array: np.ndarray, fov: int) -> np.ndarray:
    height, width = array.shape[1:3]
    if fov > height or fov > width:
        raise ValueError(f"field of view {fov} exceeds the rendered tile {height}x{width}")
    top = (height - fov) // 2
    left = (width - fov) // 2
    return array[:, top:top + fov, left:left + fov]


def main() -> None:
    args = parse_args()
    contract = json.loads(Path(args.contract).read_text())
    if contract.get("trident_commit") != TRIDENT_COMMIT:
        raise RuntimeError("checkpoint manifest does not carry the frozen TRIDENT commit")
    verify_trident_source(Path(args.trident_root))
    model = select_model(contract, None, args.encoder_index)
    encoder_id = model["encoder_id"]
    fov = MODEL_FOV[encoder_id]

    import torch

    sys.path.insert(0, str(Path(args.trident_root).resolve()))
    from trident.patch_encoder_models.load import encoder_factory

    encoder = encoder_factory(
        encoder_id, **encoder_kwargs(encoder_id, model["checkpoint_path"])
    ).eval().cuda()
    encoder = encoder.half() if encoder.precision == torch.float16 else encoder.float()
    batch_size = args.batch_size or BATCH_SIZE[encoder_id]

    condition_dir = Path(args.render) / args.condition
    shards = sorted(condition_dir.glob("*.h5"))
    if not shards:
        raise RuntimeError(f"no rendered tiles in {condition_dir}")

    output_dir = Path(args.output) / args.condition / encoder_id
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"{args.condition} / {encoder_id}: {len(shards)} slides, fov {fov}, batch {batch_size}")

    for number, shard in enumerate(shards, start=1):
        destination = output_dir / shard.name
        if destination.exists():
            continue
        with h5py.File(shard, "r") as source:
            meta = json.loads(source.attrs["meta"])
            total = source["rgb"].shape[0]
            features = np.empty((total, model["feature_dim"]), dtype=np.float32)
            for start in range(0, total, batch_size):
                stop = min(start + batch_size, total)
                rgb = centered_crop(source["rgb"][start:stop], fov)
                batch = transformed_batch(rgb, encoder.eval_transforms).cuda(non_blocking=True)
                batch = batch.half() if encoder.precision == torch.float16 else batch.float()
                with torch.inference_mode():
                    output = encoder(batch)
                features[start:stop] = output.float().cpu().numpy()
            payload = {key: source[key][:] for key in ("location", "core", "residual_um")}

        with h5py.File(destination, "w") as target:
            target.create_dataset("features", data=features, compression="lzf")
            for key, value in payload.items():
                target.create_dataset(key, data=value)
            target.attrs["meta"] = json.dumps({
                **meta, "condition": args.condition, "encoder_id": encoder_id,
                "feature_dim": int(model["feature_dim"]), "fov": fov,
                "feature_version": FEATURE_VERSION, "trident_commit": TRIDENT_COMMIT,
                "checkpoint_sha": model.get("sha256", ""),
            })
        if number % 20 == 0 or number == len(shards):
            print(f"  {number}/{len(shards)} slides")

    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()

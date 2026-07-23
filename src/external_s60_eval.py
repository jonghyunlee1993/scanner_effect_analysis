"""External AT2/S60 evaluation with slide-context S60 reconstruction.

Canonicalization remains label-free and uses the learned AT2 prototype.  For
reconstruction only, 64 random patches from other areas of each S60 slide are
pooled by ``E_set`` into one frozen slide-style vector and injected into ``G``.
No S60 prototype is introduced and no model parameter is updated.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

from eval_report import embed_uni, load_uni, same_location_cosine
from prenorm.checkpoint import load_model_for_inference
from prenorm.losses import robust_image_distance
from prenorm.metrics import focus_score, ssim
from utils.config import load_config
from utils.store import decode


class ExternalPairs(Dataset):
    def __init__(self, index_path, store_dir, source, reference, limit=None):
        frame = pd.read_parquet(index_path)
        frame = frame[frame[f"present_{source}"] & frame[f"present_{reference}"]].copy()
        if limit is not None:
            frame = frame.iloc[:limit].copy()
        if frame.empty:
            raise ValueError("no present AT2/S60 pairs in external index")
        self.frame, self.store_dir = frame.reset_index(drop=True), Path(store_dir)
        self.source, self.reference = source, reference

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with h5py.File(self.store_dir / f"{row.slide_id}.h5", "r") as handle:
            source = decode(handle[self.source]["rgb"][int(row.tuple_id)])
            reference = decode(handle[self.reference]["rgb"][int(row.tuple_id)])
        to_tensor = lambda image: torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 127.5 - 1.0
        return {
            "source": to_tensor(source), "reference": to_tensor(reference),
            "slide_id": str(row.slide_id), "tuple_id": int(row.tuple_id),
            "q_reg": float(row.get(f"q_reg_{self.source}", np.nan)),
        }


def panel(images, labels, path):
    tiles = []
    for image, label in zip(images, labels):
        rgb = ((image.clamp(-1, 1).permute(1, 2, 0).numpy() + 1) * 127.5).round().astype(np.uint8)
        tile = Image.fromarray(rgb).resize((256, 256))
        canvas = Image.new("RGB", (256, 280), "white")
        canvas.paste(tile, (0, 0))
        ImageDraw.Draw(canvas).text((8, 260), label, fill="black")
        tiles.append(canvas)
    out = Image.new("RGB", (256 * len(tiles), 280), "white")
    for i, tile in enumerate(tiles): out.paste(tile, (256 * i, 0))
    out.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--style-context-size", type=int, default=64)
    parser.add_argument("--style-seed", type=int, default=1234)
    parser.add_argument("--max-tiles", type=int)
    parser.add_argument("--skip-uni", action="store_true")
    args = parser.parse_args()

    if args.style_context_size < 1:
        raise ValueError("--style-context-size must be positive")
    cfg = load_config(args.model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ExternalPairs(args.index, args.store, "s60", "at2", args.max_tiles)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=device.type == "cuda")
    model = load_model_for_inference(args.checkpoint, cfg).to(device).eval()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    random = np.random.default_rng(args.style_seed)
    slide_styles, slide_style_replacements, style_contexts = {}, {}, {}
    with torch.no_grad():
        for slide_id, positions in dataset.frame.groupby("slide_id", sort=True).groups.items():
            positions = np.asarray(list(positions), dtype=int)
            if len(positions) <= args.style_context_size:
                raise ValueError(
                    f"slide {slide_id} needs at least {args.style_context_size + 1} patches "
                    "to exclude each reconstruction target from its style context"
                )
            selected = random.choice(
                positions, size=args.style_context_size + 1, replace=False
            )
            context = torch.stack([
                dataset[int(index)]["source"] for index in selected
            ]).to(device)
            # E_set is the mean of per-patch codes. Keep one reserve patch so a
            # target that belongs to the base 64 can be replaced without another
            # style-encoder pass or any self-patch leakage.
            codes = model.E_set.patch_encoder(context)
            base_codes, reserve_code = codes[:-1], codes[-1]
            slide_key = str(slide_id)
            slide_styles[slide_key] = base_codes.mean(0)
            base_tuple_ids = [
                int(dataset.frame.iloc[int(index)].tuple_id) for index in selected[:-1]
            ]
            slide_style_replacements[slide_key] = {
                tuple_id: (base_codes.sum(0) - base_codes[offset] + reserve_code)
                          / args.style_context_size
                for offset, tuple_id in enumerate(base_tuple_ids)
            }
            style_contexts[slide_key] = {
                "base_tuple_ids": base_tuple_ids,
                "reserve_tuple_id": int(dataset.frame.iloc[int(selected[-1])].tuple_id),
            }
    (output / "style_contexts.json").write_text(json.dumps({
        "seed": args.style_seed,
        "requested_patches_per_slide": args.style_context_size,
        "selection": style_contexts,
        "target_exclusion": "replace a matching base patch with the reserve patch",
    }, indent=2, sort_keys=True) + "\n")
    rows, sample_rows = [], []
    uni = load_uni(device) if not args.skip_uni else None
    embeddings = {name: [] for name in (
        "raw_s60", "standard_s60", "recon_s60", "paired_at2", "standard_at2"
    )}
    with torch.no_grad():
        for batch in loader:
            source, reference = batch["source"].to(device), batch["reference"].to(device)
            content = model.encode(source)
            standard = model.decode_canonical(content, source)
            standard_at2 = model.decode_canonical(model.encode(reference), reference)
            style = torch.stack([
                slide_style_replacements[str(slide_id)].get(
                    int(tuple_id), slide_styles[str(slide_id)]
                )
                for slide_id, tuple_id in zip(batch["slide_id"], batch["tuple_id"])
            ])
            reconstruction = model.G(content, style)
            if uni is not None:
                uni_model, uni_size, uni_mean, uni_std = uni
                for name, image in (("raw_s60", source), ("standard_s60", standard),
                                    ("recon_s60", reconstruction), ("paired_at2", reference),
                                    ("standard_at2", standard_at2)):
                    embeddings[name].append(
                        embed_uni(uni_model, image, uni_size, uni_mean, uni_std, device).cpu()
                    )
            for i in range(len(source)):
                rows.append({
                    "slide_id": batch["slide_id"][i], "tuple_id": int(batch["tuple_id"][i]),
                    "q_reg_s60": float(batch["q_reg"][i]),
                    "raw_s60_distance_to_at2": float(robust_image_distance(source[i:i+1], reference[i:i+1]).cpu()),
                    "standard_distance_to_at2": float(robust_image_distance(standard[i:i+1], reference[i:i+1]).cpu()),
                    "standard_distance_to_raw_s60": float(robust_image_distance(standard[i:i+1], source[i:i+1]).cpu()),
                    "standard_at2_distance_to_at2": float(robust_image_distance(standard_at2[i:i+1], reference[i:i+1]).cpu()),
                    "recon_s60_distance_to_at2": float(robust_image_distance(reconstruction[i:i+1], reference[i:i+1]).cpu()),
                    "recon_s60_distance_to_raw_s60": float(robust_image_distance(reconstruction[i:i+1], source[i:i+1]).cpu()),
                    "raw_s60_ssim_to_at2": float(ssim(source[i:i+1], reference[i:i+1]).cpu()),
                    "standard_ssim_to_at2": float(ssim(standard[i:i+1], reference[i:i+1]).cpu()),
                    "standard_ssim_to_raw_s60": float(ssim(standard[i:i+1], source[i:i+1]).cpu()),
                    "standard_at2_ssim_to_at2": float(ssim(standard_at2[i:i+1], reference[i:i+1]).cpu()),
                    "recon_s60_ssim_to_at2": float(ssim(reconstruction[i:i+1], reference[i:i+1]).cpu()),
                    "recon_s60_ssim_to_raw_s60": float(ssim(reconstruction[i:i+1], source[i:i+1]).cpu()),
                    "raw_s60_focus": float(focus_score(source[i:i+1]).cpu()),
                    "standard_focus": float(focus_score(standard[i:i+1]).cpu()),
                    "recon_s60_focus": float(focus_score(reconstruction[i:i+1]).cpu()),
                })
                if len(sample_rows) < 40:
                    sample_rows.append((source[i].cpu(), standard[i].cpu(), reconstruction[i].cpu(), reference[i].cpu()))

    # Panels are intentionally limited and written after batching to avoid worker-side I/O.
    samples = output / "samples"; samples.mkdir(exist_ok=True)
    for i, images in enumerate(sample_rows):
        panel(images, ["S60 raw", "S60 canonical", "S60 context recon", "paired AT2"], samples / f"{i:03d}.png")
    frame = pd.DataFrame(rows); frame.to_csv(output / "per_tile.csv", index=False)
    metrics = {
        "n_tiles": int(len(frame)),
        "n_slides": int(frame["slide_id"].nunique()),
        "style_context_size": int(args.style_context_size),
        "style_seed": int(args.style_seed),
        "style_source": "64 random non-target same-slide S60 patches pooled by frozen E_set",
        **{key: float(frame[key].mean()) for key in frame.columns
           if key not in {"slide_id", "tuple_id", "q_reg_s60"}},
    }
    if not args.skip_uni:
        embeddings = {name: torch.cat(chunks) for name, chunks in embeddings.items()}
        for name in ("raw_s60", "standard_s60", "recon_s60", "paired_at2"):
            metrics[f"uni_cosine_{name}_to_at2"] = float(
                (embeddings[name] * embeddings["paired_at2"]).sum(1).mean()
            )
        metrics["uni_cosine_standard_s60_to_standard_at2"] = float(
            (embeddings["standard_s60"] * embeddings["standard_at2"]).sum(1).mean()
        )
        metrics["uni_cosine_standard_at2_to_at2"] = float(
            (embeddings["standard_at2"] * embeddings["paired_at2"]).sum(1).mean()
        )
        np.savez(output / "uni_embeddings.npz", **{name: value.numpy() for name, value in embeddings.items()})
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(f"[external-s60] {len(frame)} paired tiles -> {output}")


if __name__ == "__main__":
    main()

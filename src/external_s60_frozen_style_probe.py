"""Frozen S60 style-injection probe for a prototype-only phase-1 model.

This is deliberately an inference-only diagnostic.  It estimates one style code
per external S60 slide with ``E_set`` and injects that code directly into the
frozen generator.  No model parameter, including the prototype table, is
updated.  The result is *not* an adapted S60 renderer: E_set was trained only
to regress the observed scanner-prototype table.  The probe answers the more
basic question of whether an unseen scanner style estimate has any meaningful
effect on the decoder output.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from eval_report import embed_uni, load_uni
from external_s60_eval import ExternalPairs, panel
from prenorm.checkpoint import load_model_for_inference
from prenorm.losses import robust_image_distance
from prenorm.metrics import focus_score, ssim
from utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--style-context-size", type=int, default=32)
    parser.add_argument("--max-tiles", type=int)
    parser.add_argument("--skip-uni", action="store_true")
    return parser.parse_args()


def estimate_slide_styles(model, dataset, context_size, device):
    """Estimate one E_set code per S60 slide from deterministic dispersed tiles."""
    codes = {}
    for slide_id, positions in dataset.frame.groupby("slide_id", sort=True).groups.items():
        selected = list(positions)[:context_size]
        context = torch.stack([dataset[int(index)]["source"] for index in selected]).to(device)
        # E_set expects (..., K, C, H, W); all selected contexts are present.
        codes[str(slide_id)] = model.E_set(context.unsqueeze(0))[0]
    return codes


def metric_row(prefix, image, reference, raw):
    return {
        f"{prefix}_distance_to_at2": float(robust_image_distance(image, reference).cpu()),
        f"{prefix}_ssim_to_at2": float(ssim(image, reference).cpu()),
        f"{prefix}_focus": float(focus_score(image).cpu()),
        f"{prefix}_distance_to_raw_s60": float(robust_image_distance(image, raw).cpu()),
        f"{prefix}_ssim_to_raw_s60": float(ssim(image, raw).cpu()),
    }


def main():
    args = parse_args()
    if args.style_context_size < 1:
        raise ValueError("--style-context-size must be positive")
    cfg = load_config(args.model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ExternalPairs(args.index, args.store, "s60", "at2", args.max_tiles)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4,
                        pin_memory=device.type == "cuda")
    model = load_model_for_inference(args.checkpoint, cfg).to(device).eval()
    at2_index = model.scanners.index("at2")
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        slide_styles = estimate_slide_styles(model, dataset, args.style_context_size, device)
        prototype_norms = model.B.detach().norm(dim=1)
        style_summary = {
            "context_size": args.style_context_size,
            "n_slides": len(slide_styles),
            "estimated_style_norm_mean": float(torch.stack(list(slide_styles.values())).norm(dim=1).mean().cpu()),
            "prototype_norm_min": float(prototype_norms.min().cpu()),
            "prototype_norm_max": float(prototype_norms.max().cpu()),
        }

    uni = load_uni(device) if not args.skip_uni else None
    embeddings = {name: [] for name in ("raw_s60", "at2_render", "estimated_s60_render", "paired_at2")}
    rows, sample_rows = [], []
    with torch.no_grad():
        for batch in loader:
            source, reference = batch["source"].to(device), batch["reference"].to(device)
            content = model.encode(source)
            at2_render = model.decode(content, torch.full(
                (len(source),), at2_index, device=device, dtype=torch.long))
            style = torch.stack([slide_styles[str(slide_id)] for slide_id in batch["slide_id"]])
            estimated_s60_render = model.G(content, style)
            if uni is not None:
                uni_model, uni_size, uni_mean, uni_std = uni
                for name, image in (("raw_s60", source), ("at2_render", at2_render),
                                    ("estimated_s60_render", estimated_s60_render),
                                    ("paired_at2", reference)):
                    embeddings[name].append(embed_uni(
                        uni_model, image, uni_size, uni_mean, uni_std, device).cpu())
            for index in range(len(source)):
                row = {"slide_id": batch["slide_id"][index], "tuple_id": int(batch["tuple_id"][index]),
                       "q_reg_s60": float(batch["q_reg"][index])}
                row.update(metric_row("raw_s60", source[index:index + 1], reference[index:index + 1], source[index:index + 1]))
                row.update(metric_row("at2_render", at2_render[index:index + 1], reference[index:index + 1], source[index:index + 1]))
                row.update(metric_row("estimated_s60_render", estimated_s60_render[index:index + 1], reference[index:index + 1], source[index:index + 1]))
                row["estimated_s60_vs_at2_render_distance"] = float(robust_image_distance(
                    estimated_s60_render[index:index + 1], at2_render[index:index + 1]).cpu())
                rows.append(row)
                if len(sample_rows) < 40:
                    sample_rows.append((source[index].cpu(), at2_render[index].cpu(),
                                        estimated_s60_render[index].cpu(), reference[index].cpu()))

    samples = output / "samples"; samples.mkdir(exist_ok=True)
    for index, images in enumerate(sample_rows):
        panel(images, ["S60 raw", "AT2 prototype render", "E_set S60-style render", "paired AT2"],
              samples / f"{index:03d}.png")
    frame = pd.DataFrame(rows); frame.to_csv(output / "per_tile.csv", index=False)
    metrics = {"n_tiles": int(len(frame)), "n_slides": int(frame["slide_id"].nunique()),
               "frozen_model": True, **style_summary,
               **{key: float(frame[key].mean()) for key in frame.columns
                  if key not in {"slide_id", "tuple_id", "q_reg_s60"}}}
    if uni is not None:
        embeddings = {name: torch.cat(chunks) for name, chunks in embeddings.items()}
        for name, embedding in embeddings.items():
            metrics[f"uni_cosine_{name}_to_paired_at2"] = float((embedding * embeddings["paired_at2"]).sum(1).mean())
            metrics[f"uni_cosine_{name}_to_raw_s60"] = float((embedding * embeddings["raw_s60"]).sum(1).mean())
        np.savez(output / "uni_embeddings.npz", **{name: value.numpy() for name, value in embeddings.items()})
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(f"[frozen-style-probe] {len(frame)} paired tiles -> {output}")


if __name__ == "__main__":
    main()

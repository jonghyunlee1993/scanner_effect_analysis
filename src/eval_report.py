"""Sealed held-out evaluation for v3 canonical RGB and scanner rendering."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from torch.utils.data import DataLoader

from prenorm.checkpoint import load_model_for_inference
from prenorm.data.dataset import PairedTupleDataset
from prenorm.losses import robust_image_distance
from prenorm.metrics import focus_score, ssim
from utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.environ.get(
            "PRENORM_CONFIG",
            "archived/2026-07-14_v3/configs/main/"
            "phase_1_train_10slide_v3_ab16_nuclei_rgb_detail.yaml",
        ),
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument(
        "--snapshot-epoch", type=int, default=None,
        help="Sealed training epoch represented by the checkpoint (report metadata).",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--skip-uni", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Artifact directory. Defaults to outputs/<run.name>/eval.",
    )
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="Write numerical artifacts only; render after external evaluation completes.",
    )
    parser.add_argument(
        "--wandb-file",
        type=Path,
        default=None,
        help="Offline W&B .wandb file from which to recover learning curves.",
    )
    return parser.parse_args()


def load_uni(device):
    """Load the sealed UNI evaluator from the local Hugging Face cache."""
    import timm

    model = timm.create_model(
        "hf-hub:MahmoodLab/uni",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=True,
    ).eval().to(device)
    pretrained = model.pretrained_cfg
    size = pretrained["input_size"][-1]
    mean = torch.tensor(pretrained["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(pretrained["std"], device=device).view(1, 3, 1, 1)
    return model, size, mean, std


@torch.no_grad()
def embed_uni(model, images, size, mean, std, device, batch_size=32):
    """Return normalized UNI embeddings for RGB tensors in [-1, 1]."""
    embeddings = []
    for start in range(0, len(images), batch_size):
        image = images[start : start + batch_size].to(device)
        image = F.interpolate(
            (image.clamp(-1, 1) + 1) * 0.5,
            size=(size, size),
            mode="bicubic",
            align_corners=False,
        )
        embeddings.append(model((image - mean) / std).float().cpu())
    return F.normalize(torch.cat(embeddings), dim=1)


def scanner_probe(embedding, scanner_id, location_id):
    """Return location-grouped five-fold balanced scanner accuracy."""
    classifier = LogisticRegression(max_iter=2000, class_weight="balanced")
    classes = np.unique(scanner_id)
    group_count = len(np.unique(location_id))
    folds = min(5, group_count)
    if len(classes) < 2 or folds < 2:
        return float("nan")
    splitter = StratifiedGroupKFold(
        n_splits=folds, shuffle=True, random_state=1234
    )
    return float(cross_val_score(
        classifier, embedding, scanner_id, groups=location_id,
        cv=splitter, scoring="balanced_accuracy",
    ).mean())


def same_location_cosine(embedding, location_id):
    """Average cosine similarity across scanners at the same registered location."""
    values = []
    for location in np.unique(location_id):
        selected = np.flatnonzero(location_id == location)
        for i, left in enumerate(selected):
            for right in selected[i + 1 :]:
                values.append(float((embedding[left] * embedding[right]).sum()))
    return float(np.mean(values)) if values else float("nan")


def retrieval_metrics(embedding, location_id, scanner_id, k=5, chunk_size=512):
    """Compute same-location hit rate and scanner purity without an NxN allocation."""
    location = torch.as_tensor(location_id)
    scanner = torch.as_tensor(scanner_id)
    location_hits = []
    scanner_purity = []
    for start in range(0, len(embedding), chunk_size):
        stop = min(start + chunk_size, len(embedding))
        similarity = embedding[start:stop] @ embedding.t()
        row = torch.arange(stop - start)
        similarity[row, torch.arange(start, stop)] = -torch.inf
        neighbors = similarity.topk(min(k, len(embedding) - 1), dim=1).indices
        query_location = location[start:stop, None]
        query_scanner = scanner[start:stop, None]
        location_hits.append((location[neighbors] == query_location).any(dim=1).float())
        scanner_purity.append((scanner[neighbors] == query_scanner).float().mean(dim=1))
    return float(torch.cat(location_hits).mean()), float(torch.cat(scanner_purity).mean())


def scanner_purity_excluding_location(
    embedding, location_id, scanner_id, k=5, chunk_size=512
):
    """Scanner purity after removing every registered same-location counterpart."""
    location = torch.as_tensor(location_id)
    scanner = torch.as_tensor(scanner_id)
    values = []
    for start in range(0, len(embedding), chunk_size):
        stop = min(start + chunk_size, len(embedding))
        similarity = embedding[start:stop] @ embedding.t()
        same_location = location[start:stop, None] == location[None, :]
        similarity = similarity.masked_fill(same_location, -torch.inf)
        available = (~same_location).sum(dim=1).min().item()
        if available < 1:
            continue
        neighbors = similarity.topk(min(k, int(available)), dim=1).indices
        values.append(
            (scanner[neighbors] == scanner[start:stop, None]).float().mean(dim=1)
        )
    return float(torch.cat(values).mean()) if values else float("nan")


def training_history(wandb_file):
    """Recover epoch-level losses from an offline W&B transaction log."""
    if wandb_file is None or not wandb_file.exists():
        return pd.DataFrame(columns=["epoch", "train_loss", "val_loss"])
    from wandb.proto.v6 import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    rows = []
    store = DataStore()
    store.open_for_scan(str(wandb_file))
    while (data := store.scan_data()) is not None:
        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        if record.WhichOneof("record_type") != "history":
            continue
        values = {
            ".".join(item.nested_key) if item.nested_key else item.key:
            json.loads(item.value_json)
            for item in record.history.item
            if item.value_json
        }
        if "epoch" not in values:
            continue
        row = {"epoch": int(values["epoch"])}
        for key in ("train/loss", "train/l_total_epoch", "train/l_total"):
            if key in values:
                row["train_loss"] = float(values[key])
                break
        for key in ("val/loss", "val/l_total"):
            if key in values:
                row["val_loss"] = float(values[key])
                break
        if len(row) > 1:
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["epoch", "train_loss", "val_loss"])
    history = pd.DataFrame(rows)
    return history.groupby("epoch", as_index=False).mean(numeric_only=True)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scanners = list(cfg.scanners)
    reference_index = scanners.index(cfg.reference_scanner)
    dataset = PairedTupleDataset(
        cfg.paths.output_index,
        cfg.paths.output_store,
        scanners,
        cfg,
        args.split,
        train=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = load_model_for_inference(args.checkpoint, cfg, map_location="cpu").to(device)

    rows = []
    raw_images = []
    canonical_images = []
    scanner_ids = []
    location_ids = []
    location = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            rgb = batch["rgb"].to(device)
            present = batch["present"].to(device)
            batch_size, scanner_count = rgb.shape[:2]
            flat = rgb.flatten(0, 1)
            canonical = model.canonicalize(flat).view_as(rgb)
            model_targets = torch.tensor(
                [model.scanners.index(scanner) if scanner in model.scanners else -1
                 for scanner in scanners],
                device=device,
            ).repeat(batch_size)
            renderable = model_targets >= 0
            rendered = torch.zeros_like(flat)
            if bool(renderable.any()):
                rendered[renderable] = model.decode(
                    model.encode(flat[renderable]), model_targets[renderable]
                )
            rendered = rendered.view_as(rgb)

            reference = rgb[:, reference_index]
            for item in range(batch_size):
                for scanner_index, scanner in enumerate(scanners):
                    if not bool(present[item, scanner_index]):
                        continue
                    source = rgb[item, scanner_index]
                    standard = canonical[item, scanner_index]
                    standard_distance = float(
                        robust_image_distance(standard[None], reference[item][None]).cpu()
                    )
                    standard_ssim = float(ssim(standard[None], reference[item][None]).cpu())
                    renderer_distance = float("nan")
                    renderer_ssim = float("nan")
                    if scanner in model.scanners:
                        output = rendered[item, scanner_index]
                        renderer_distance = float(
                            robust_image_distance(output[None], source[None]).cpu()
                        )
                        renderer_ssim = float(ssim(output[None], source[None]).cpu())
                    rows.append({
                        "location": location,
                        "scanner": scanner,
                        "q_reg": float(batch["q_reg"][item, scanner_index]),
                        "raw_distance_to_at2": float(
                            robust_image_distance(source[None], reference[item][None]).cpu()
                        ),
                        "raw_ssim_to_at2": float(
                            ssim(source[None], reference[item][None]).cpu()
                        ),
                        "standard_distance": standard_distance,
                        "standard_ssim": standard_ssim,
                        "standard_distance_to_source": float(
                            robust_image_distance(standard[None], source[None]).cpu()
                        ),
                        "standard_ssim_to_source": float(
                            ssim(standard[None], source[None]).cpu()
                        ),
                        "renderer_distance": renderer_distance,
                        "renderer_ssim": renderer_ssim,
                        "raw_focus": float(focus_score(source[None]).cpu()),
                        "canonical_focus": float(focus_score(standard[None]).cpu()),
                    })
                    raw_images.append(source.cpu())
                    canonical_images.append(standard.cpu())
                    scanner_ids.append(scanner_index)
                    location_ids.append(location)
                location += 1

    frame = pd.DataFrame(rows)
    aggregate = {
        "n_images": float(len(frame)),
        "n_locations": float(frame["location"].nunique()),
        "raw_distance_to_at2": float(frame["raw_distance_to_at2"].mean()),
        "raw_ssim_to_at2": float(frame["raw_ssim_to_at2"].mean()),
        "standard_distance": float(frame["standard_distance"].mean()),
        "standard_ssim": float(frame["standard_ssim"].mean()),
        "standard_distance_to_source": float(
            frame["standard_distance_to_source"].mean()
        ),
        "standard_ssim_to_source": float(frame["standard_ssim_to_source"].mean()),
        "renderer_distance": float(frame["renderer_distance"].mean()),
        "renderer_ssim": float(frame["renderer_ssim"].mean()),
    }
    if args.snapshot_epoch is not None:
        aggregate["checkpoint_epoch"] = int(args.snapshot_epoch)

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else Path(cfg.paths.repo) / "outputs" / cfg.run.name / "eval"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "per_image.csv", index=False)
    if not args.skip_uni:
        uni, size, mean, std = load_uni(device)
        raw_embedding = embed_uni(uni, torch.stack(raw_images), size, mean, std, device)
        canonical_embedding = embed_uni(
            uni, torch.stack(canonical_images), size, mean, std, device
        )
        scanner_array = np.asarray(scanner_ids)
        location_array = np.asarray(location_ids)
        raw_location_hit, raw_scanner_purity = retrieval_metrics(
            raw_embedding, location_array, scanner_array
        )
        canonical_location_hit, canonical_scanner_purity = retrieval_metrics(
            canonical_embedding, location_array, scanner_array
        )
        aggregate.update({
            "raw_scanner_probe": scanner_probe(
                raw_embedding.numpy(), scanner_array, location_array
            ),
            "canonical_scanner_probe": scanner_probe(
                canonical_embedding.numpy(), scanner_array, location_array
            ),
            "scanner_probe_method": "5-fold StratifiedGroupKFold(location), balanced accuracy",
            "raw_same_location_cosine": same_location_cosine(raw_embedding, location_array),
            "canonical_same_location_cosine": same_location_cosine(
                canonical_embedding, location_array
            ),
            "raw_same_location_at5": raw_location_hit,
            "canonical_same_location_at5": canonical_location_hit,
            "raw_scanner_purity_at5": raw_scanner_purity,
            "canonical_scanner_purity_at5": canonical_scanner_purity,
        })
        np.savez(
            output_dir / "embeddings.npz",
            raw=raw_embedding.numpy(),
            canonical=canonical_embedding.numpy(),
            scanner_id=scanner_array,
            location_id=location_array,
        )
    (output_dir / "metrics.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n"
    )
    history = training_history(args.wandb_file)
    if args.snapshot_epoch is not None and not history.empty:
        history = history[history["epoch"] <= args.snapshot_epoch].copy()
    history.to_csv(output_dir / "training_history.csv", index=False)
    if not args.skip_render:
        from render_eval_report import render_report
        render_report(
            args.config, output_dir, args.checkpoint, args.split,
            workers=max(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")), 1),
        )
    print(f"[eval] {len(frame)} images across {frame['location'].nunique()} locations -> {output_dir}")


if __name__ == "__main__":
    main()

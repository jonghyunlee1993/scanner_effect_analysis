#!/usr/bin/env python3
"""Exploratory frozen-encoder review on the manuscript's fixed PanNormal locations.

Writes outside the manuscript. Each model sees the same 256-pixel physical field
of view. This is a review panel, not a replacement for the 20-location benchmark.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from contextlib import nullcontext
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from scripts.frequency.bandwise_uni_experiment import (
    BANDS,
    apply_reinhard,
    components,
    render_at_dose,
)
from scanner_batch_extensions import frequency_transform


ROOT = Path(__file__).resolve().parents[2]
VIRCHOW2_CHECKPOINT = Path(
    "/mnt/isilon/oldridge_lab/leej/PFM_alignment/datasets/image_only_pan_v1/"
    "hf_cache/models--paige-ai--Virchow2/snapshots/"
    "3158645804b69e3f3bc4439d4116edddf0840a72/pytorch_model.bin"
)
STUDY = ROOT / "outputs/scanner_batch_effect_analysis_2026-09-17"
COHORT = STUDY / "00_contract/cohort.csv"
PARAMETERS = STUDY / "02_correction/parameters.json"
UNI_PANEL = STUDY / "03_uni/shards"
TARGETS = ("versa", "akoya", "gt450", "s360", "s60")
SCANNERS = ("at2", *TARGETS)
DOSE = 0.25


def load_encoder(name: str, cache_dir: Path):
    if name in {"conch", "seal_conch"}:
        from conch.open_clip_custom import create_model_from_pretrained

        model, _ = create_model_from_pretrained(
            "conch_ViT-B-16",
            checkpoint_path="hf_hub:MahmoodLab/conch",
            cache_dir=str(cache_dir),
        )
        model = model.eval().cuda()
        if name == "seal_conch":
            merge_seal_conch_lora(model, cache_dir.parent)
        size = 448
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        variants = (("conch_pre", "conch_projected") if name == "conch"
                    else ("seal_conch_pre", "seal_conch_projected"))
    elif name == "uni_v1":
        from prenorm.embedding import load_uni

        model, size, mean_t, std_t = load_uni(torch.device("cuda"))
        mean = tuple(float(value) for value in mean_t.flatten().cpu())
        std = tuple(float(value) for value in std_t.flatten().cpu())
        variants = ("uni_v1",)
    elif name in {"uni2", "uni2_pair"}:
        import timm

        model = timm.create_model(
            "hf-hub:MahmoodLab/UNI2-h",
            pretrained=True,
            img_size=224,
            patch_size=14,
            depth=24,
            num_heads=24,
            init_values=1e-5,
            embed_dim=1536,
            mlp_ratio=2.66667 * 2,
            num_classes=0,
            no_embed_class=True,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            reg_tokens=8,
            dynamic_img_size=True,
        ).eval().cuda()
        if name == "uni2_pair":
            seal = copy.deepcopy(model)
            merge_seal_lora(seal, cache_dir.parent)
            model = (model, seal.eval())
        size = 224
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        variants = ("uni2", "seal_uni2") if name == "uni2_pair" else ("uni2",)
    elif name == "virchow2":
        import timm

        model = timm.create_model(
            "vit_huge_patch14_224", pretrained=False, img_size=224,
            init_values=1e-5, num_classes=0, reg_tokens=4,
            mlp_ratio=5.3375, global_pool="", dynamic_img_size=True,
            mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU,
        )
        if not VIRCHOW2_CHECKPOINT.is_file():
            raise FileNotFoundError(VIRCHOW2_CHECKPOINT)
        state = torch.load(VIRCHOW2_CHECKPOINT, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model = model.eval().cuda()
        size = 224
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        variants = ("virchow2",)
    elif name == "hoptimus1":
        import timm

        matches = sorted((cache_dir / "models--bioptimus--H-optimus-1/snapshots").glob(
            "*/pytorch_model.bin"
        ))
        if len(matches) != 1:
            raise FileNotFoundError(f"expected one H-optimus-1 checkpoint: {matches}")
        model = timm.create_model(
            "vit_giant_patch14_reg4_dinov2", pretrained=False,
            num_classes=0, img_size=224, global_pool="token",
            init_values=1e-5, dynamic_img_size=False,
        )
        state = torch.load(matches[0], map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model = model.eval().cuda()
        size = 224
        mean = (0.707223, 0.578729, 0.703617)
        std = (0.211883, 0.230117, 0.177517)
        variants = ("hoptimus1",)
    else:
        raise ValueError(name)
    return model, size, mean, std, variants


def merge_seal_lora(model, output: Path) -> None:
    """Merge the official SEAL UNI2 LoRA into its paired frozen UNI2 base.

    The released vision checkpoint contains only LoRA weights for blocks 21–23,
    plus a projection head and decoder. The official encoder's default
    inference path returns the adapted backbone output without either head.
    Its released config uses alpha=8, rank=8, dropout=0.15, rslora=False;
    dropout is inactive for this evaluation.
    """
    matches = sorted((output / "hf_cache/models--MahmoodLab--SEAL/snapshots").glob(
        "*/seal_univ2_vision.pth"
    ))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one pinned SEAL vision checkpoint: {matches}")
    checkpoint = torch.load(matches[0], map_location="cpu", weights_only=True)
    state = checkpoint["state_dict"]
    adapters: dict[str, dict[str, torch.Tensor]] = {}
    pattern = re.compile(
        r"module\.encoder\.base_model\.model\."
        r"(blocks\.(?:21|22|23)\.(?:attn\.(?:qkv|proj)|mlp\.(?:fc1|fc2)))\."
        r"lora_([AB])\.default\.weight"
    )
    for key, value in state.items():
        found = pattern.fullmatch(key)
        if found:
            adapters.setdefault(found.group(1), {})[found.group(2)] = value
    if len(adapters) != 12 or any(set(weights) != {"A", "B"} for weights in adapters.values()):
        raise ValueError(f"SEAL LoRA checkpoint does not have all 12 expected layers: {list(adapters)}")
    with torch.no_grad():
        for name, weights in adapters.items():
            linear = model.get_submodule(name)
            a, b = weights["A"], weights["B"]
            if a.shape[0] != 8 or b.shape[1] != 8 or linear.weight.shape != (b.shape[0], a.shape[1]):
                raise ValueError(f"SEAL LoRA shape mismatch at {name}")
            delta = (b.float() @ a.float()).to(linear.weight.device, linear.weight.dtype)
            linear.weight.add_(delta)  # released LoRA alpha/rank = 8/8 = 1
    print(f"merged SEAL-UNI2-h LoRA from {matches[0]} into 12 linear layers", flush=True)


def merge_seal_conch_lora(model, output: Path) -> None:
    """Merge the released SEAL-CONCH LoRA into CONCH's final visual block."""
    matches = sorted((output / "hf_cache/models--MahmoodLab--SEAL/snapshots").glob(
        "*/seal_conch_vision.pth"
    ))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one SEAL-CONCH vision checkpoint: {matches}")
    checkpoint = torch.load(matches[0], map_location="cpu", weights_only=True)
    state = checkpoint["state_dict"]
    adapters: dict[str, dict[str, torch.Tensor]] = {}
    pattern = re.compile(
        r"module\.encoder\.base_model\.model\."
        r"(trunk\.blocks\.11\.(?:attn\.(?:qkv|proj)|mlp\.(?:fc1|fc2)))\."
        r"lora_([AB])\.default\.weight"
    )
    for key, value in state.items():
        found = pattern.fullmatch(key)
        if found:
            adapters.setdefault(found.group(1), {})[found.group(2)] = value
    if len(adapters) != 4 or any(set(weights) != {"A", "B"} for weights in adapters.values()):
        raise ValueError(f"SEAL-CONCH checkpoint does not have all 4 LoRA layers: {list(adapters)}")
    with torch.no_grad():
        for name, weights in adapters.items():
            linear = model.get_submodule("visual." + name)
            a, b = weights["A"], weights["B"]
            if a.shape[0] != 8 or b.shape[1] != 8 or linear.weight.shape != (b.shape[0], a.shape[1]):
                raise ValueError(f"SEAL-CONCH LoRA shape mismatch at {name}")
            delta = (b.float() @ a.float()).to(linear.weight.device, linear.weight.dtype)
            linear.weight.add_(delta)
    print(f"merged SEAL-CONCH LoRA from {matches[0]} into 4 visual layers", flush=True)


@torch.inference_mode()
def embed(model, name: str, images: list[np.ndarray], size: int,
          mean: tuple[float, ...], std: tuple[float, ...], batch_size: int):
    collected: dict[str, list[np.ndarray]] = {}
    mean_t = torch.tensor(mean, device="cuda").view(1, 3, 1, 1)
    std_t = torch.tensor(std, device="cuda").view(1, 3, 1, 1)
    for start in range(0, len(images), batch_size):
        block = np.stack(images[start:start + batch_size]).astype(np.float32)
        x = torch.from_numpy(block).permute(0, 3, 1, 2).cuda()
        interpolation = "bilinear" if name == "hoptimus1" else "bicubic"
        x = F.interpolate(x, size=(size, size), mode=interpolation, align_corners=False)
        x = (x.clamp(0, 1) - mean_t) / std_t
        precision = (nullcontext() if name == "uni_v1" else
                     torch.autocast("cuda", dtype=torch.float16 if name in
                                    {"conch", "seal_conch", "hoptimus1"} else torch.bfloat16))
        with precision:
            if name in {"conch", "seal_conch"}:
                pre = model.visual.forward_no_head(x, normalize=False)
                prefix = "seal_conch" if name == "seal_conch" else "conch"
                out = {
                    f"{prefix}_pre": pre,
                    f"{prefix}_projected": pre @ model.visual.proj_contrast,
                }
            elif name == "uni2_pair":
                out = {"uni2": model[0](x), "seal_uni2": model[1](x)}
            elif name == "virchow2":
                tokens = model.forward_features(x)
                prefix = int(model.num_prefix_tokens)
                out = {"virchow2": torch.cat(
                    [tokens[:, 0], tokens[:, prefix:].mean(dim=1)], dim=-1
                )}
            elif name == "hoptimus1":
                out = {"hoptimus1": model(x)}
            elif name == "uni_v1":
                out = {"uni_v1": model(x)}
            else:
                out = {"uni2": model(x)}
        for variant, value in out.items():
            value = F.normalize(value.float(), dim=-1).cpu().numpy()
            collected.setdefault(variant, []).append(value)
    return {key: np.concatenate(value) for key, value in collected.items()}


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.clip(np.dot(a, b), -1.0, 1.0))


def slide_images(slide, fitted: dict) -> tuple[list[np.ndarray], list[dict]]:
    slide_id = str(slide.slide_id)
    with h5py.File(UNI_PANEL / f"{slide_id}.h5", "r") as panel:
        locked = np.asarray(panel["location_index"], dtype=np.int64)
    if len(locked) != 20:
        raise ValueError(f"{slide_id}: expected 20 locked UNI locations")
    locations = locked[[0, len(locked) // 2, -1]]
    with h5py.File(slide.cache_path, "r") as cache:
        names = tuple(x.decode().lower() for x in cache["scanner_names"][:])
        if names != SCANNERS:
            raise ValueError(f"{slide_id}: scanner order changed: {names}")
        paired = np.asarray(cache["images"][locations], dtype=np.uint8)
    images: list[np.ndarray] = []
    records: list[dict] = []
    for location_pos, location in enumerate(locations):
        source = paired[location_pos, 0]
        for target_pos, scanner in enumerate(TARGETS, start=1):
            parameter = fitted[scanner]
            reinhard = apply_reinhard(source, parameter)
            combined = frequency_transform(reinhard, parameter)
            target = paired[location_pos, target_pos]
            slot = {"location_index": int(location), "scanner": scanner,
                    "tissue_type": slide.tissue_type}
            for arm, image in (("source", source), ("target", target),
                               ("reinhard", reinhard), ("combined", combined)):
                slot[arm] = len(images)
                images.append(image.astype(np.float32) / 255.0)
            part = components(reinhard)
            base_rms = min(float(np.sqrt(np.mean(component * component)))
                           for component in part.values())
            slot["target_rms_od"] = DOSE * base_rms
            for band, component in part.items():
                for sign in (-1, 1):
                    changed, achieved, _ = render_at_dose(
                        reinhard, component, sign, slot["target_rms_od"]
                    )
                    slot[f"{band}:{sign}"] = len(images)
                    slot[f"{band}:{sign}:achieved_rms_od"] = achieved
                    images.append(changed)
            records.append(slot)
    return images, records


def run(model_name: str, fold: int, output: Path, batch_size: int,
        max_slides: int | None) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU allocation required")
    torch.set_num_threads(2)
    cache = output / "hf_cache"
    cache.mkdir(parents=True, exist_ok=True)
    model, size, mean, std, variants = load_encoder(model_name, cache)
    cohort = pd.read_csv(COHORT, dtype={"slide_id": str})
    subset = cohort[cohort.fold == fold]
    if max_slides is not None:
        subset = subset.head(max_slides)
    fitted_all = json.loads(PARAMETERS.read_text())["fold"][str(fold)]
    rows: list[dict] = []
    feature_root = output / "features" / model_name / f"fold_{fold}"
    feature_root.mkdir(parents=True, exist_ok=True)
    for slide in subset.itertuples(index=False):
        images, records = slide_images(slide, fitted_all)
        features = embed(model, model_name, images, size, mean, std, batch_size)
        for variant in variants:
            z = features[variant]
            compact: dict[str, list] = {arm: [] for arm in
                                        ("source", "target", "reinhard", "combined")}
            for slot in records:
                compact["source"].append(z[slot["source"]])
                compact["target"].append(z[slot["target"]])
                compact["reinhard"].append(z[slot["reinhard"]])
                compact["combined"].append(z[slot["combined"]])
                raw_dist = cosine_distance(z[slot["source"]], z[slot["target"]])
                color_dist = cosine_distance(z[slot["reinhard"]], z[slot["target"]])
                combined_dist = cosine_distance(z[slot["combined"]], z[slot["target"]])
                for band in BANDS:
                    for sign in (-1, 1):
                        key = f"{band}:{sign}"
                        rows.append({
                            "model": variant, "slide_id": str(slide.slide_id),
                            "tissue_type": slot["tissue_type"], "fold": fold,
                            "location_index": slot["location_index"],
                            "scanner": slot["scanner"], "band": band,
                            "sign": sign, "dose": DOSE,
                            "target_rms_od": slot["target_rms_od"],
                            "achieved_rms_od": slot[f"{key}:achieved_rms_od"],
                            "band_displacement": cosine_distance(z[slot["reinhard"]], z[slot[key]]),
                            "raw_target_distance": raw_dist,
                            "reinhard_target_distance": color_dist,
                            "combined_target_distance": combined_dist,
                            "frequency_incremental_gain": color_dist - combined_dist,
                        })
            np.savez_compressed(
                feature_root / f"{slide.slide_id}_{variant}.npz",
                slide_id=str(slide.slide_id), tissue_type=slide.tissue_type,
                scanner=np.array([slot["scanner"] for slot in records]),
                location_index=np.array([slot["location_index"] for slot in records]),
                **{arm: np.stack(values).astype(np.float32)
                   for arm, values in compact.items()},
            )
        print(f"{model_name} fold {fold}: {slide.slide_id} complete", flush=True)
    shard_root = output / "shards" / model_name
    shard_root.mkdir(parents=True, exist_ok=True)
    path = shard_root / f"fold_{fold}.csv.gz"
    pd.DataFrame(rows).to_csv(path, index=False, compression="gzip")
    print(json.dumps({"output": str(path), "rows": len(rows),
                      "slides": len(subset), "variants": variants}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=("conch", "seal_conch", "uni_v1", "uni2", "uni2_pair",
                                 "virchow2", "hoptimus1"))
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-slides", type=int)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/encoder_review_2026-09-25")
    args = parser.parse_args()
    run(args.model, args.fold, args.output, args.batch_size, args.max_slides)

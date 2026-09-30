#!/usr/bin/env python3
"""RV13 loader parity against the official code paths (GPU, one job, two stages).

``official`` (Python >= 3.10: the released EXAONEPath ``macenko.py`` uses ``match``; run in
the trident-uni2 env, same torch 2.5.1 / torchvision 0.20.1): imports ``macenko.py`` and
``vision_transformer.py`` from the pinned Hugging Face snapshot of LGAI-EXAONE/EXAONEPath
and torchstain from ``third_party/torchstain`` (vendored 1.4.1, MIT) and runs the model-card
quick start verbatim (``VisionTransformer.from_pretrained``, ``macenko_normalizer()`` from the
snapshot directory, Resize 256 bicubic / CenterCrop 224 / Normalize) on ``images/MHIST_aaa.png``
and on 36 raw `set20` patches (first two cohort slides, first three `set20` locations, six
scanners), with and without Macenko. Writes ``parity/exaonepath_official.npz``.

``compare`` (cpath env): runs ``anchor_frequency_diversity_models`` on the same inputs and
compares (i) EXAONEPath Macenko reference, normalized images and features (as released and
raw input) with the official outputs; (ii) PLIP and DINOv2 float-tensor preprocessing with the
released Hugging Face processors (``CLIPProcessor`` / ``AutoImageProcessor`` on uint8 PIL
images); (iii) for information, CONCH's released PIL preprocessing (``create_model_from_pretrained``
transform) against the review path used here. Writes ``parity/parity.csv`` and
``parity/parity.json``. Pass criterion: cosine >= 0.999 for every image.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import json
import os
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
REVISION = Path(__file__).resolve().parent
OUTPUT = REVISION / "results/anchor_frequency_diversity/parity"
SET20 = REVISION / "results/location_sets/set20.csv"
COHORT = PROJECT / "outputs/scanner_batch_effect_analysis_2026-09-17/00_contract/cohort.csv"
EXAONEPATH_REVISION = "383a1247a2c543d2a04787eb9a71292f74c9ce9f"
THRESHOLD = 0.999


def sample_patches() -> tuple[np.ndarray, list[dict]]:
    import h5py
    import pandas as pd

    cohort = pd.read_csv(COHORT, dtype={"slide_id": str}).sort_values("slide_id").head(2)
    set20 = pd.read_csv(SET20, dtype={"slide_id": str})
    images, meta = [], []
    for slide in cohort.itertuples(index=False):
        locations = set20[set20.slide_id.eq(slide.slide_id)].sort_values("location_index").location_index.to_numpy()[:3]
        with h5py.File(slide.cache_path, "r") as cache:
            names = [x.decode().lower() for x in cache["scanner_names"][:]]
            paired = np.asarray(cache["images"][np.sort(locations)], dtype=np.uint8)
        for i, location in enumerate(np.sort(locations)):
            for j, scanner in enumerate(names):
                images.append(paired[i, j])
                meta.append({"slide_id": slide.slide_id, "location_index": int(location), "scanner": scanner})
    return np.stack(images), meta


def official() -> None:
    import torch
    import torchvision.transforms as transforms
    from huggingface_hub import snapshot_download
    from PIL import Image

    root = Path(snapshot_download("LGAI-EXAONE/EXAONEPath", revision=EXAONEPATH_REVISION, local_files_only=True))
    sys.path.insert(0, str(REVISION / "third_party/torchstain"))
    sys.path.insert(0, str(root))
    from macenko import macenko_normalizer  # noqa: E402  (released file, Python >= 3.10)
    from vision_transformer import VisionTransformer  # noqa: E402

    model = VisionTransformer.from_pretrained("LGAI-EXAONE/EXAONEPath", revision=EXAONEPATH_REVISION)
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    cwd = os.getcwd()
    os.chdir(root)  # the model card runs from the repository root: default './macenko_target/...'
    try:
        normalizer = macenko_normalizer()
        mhist = Image.open("images/MHIST_aaa.png").convert("RGB")
    finally:
        os.chdir(cwd)
    model.cuda()
    model.eval()
    patches, meta = sample_patches()
    inputs = [np.asarray(mhist, dtype=np.uint8)] + list(patches)
    released, raw, normalized, fallback = [], [], [], []
    with torch.no_grad():
        for array in inputs:
            image = Image.fromarray(array)
            image_macenko = normalizer(image)
            normalized.append(image_macenko.permute(1, 2, 0).numpy())
            fallback.append(bool(torch.equal(image_macenko, transforms.ToTensor()(image))))
            released.append(model(transform(image_macenko).unsqueeze(0).cuda()).float().cpu().numpy()[0])
            raw.append(model(transform(transforms.ToTensor()(image)).unsqueeze(0).cuda()).float().cpu().numpy()[0])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT / "exaonepath_official.npz", mhist=inputs[0], patches=patches,
        meta=np.asarray([json.dumps(m) for m in meta]), normalized_mhist=normalized[0],
        normalized_patches=np.stack(normalized[1:]), released=np.stack(released), raw=np.stack(raw),
        fallback=np.asarray(fallback), HERef=normalizer.normalizer.HERef.numpy(),
        maxCRef=normalizer.normalizer.maxCRef.numpy(), python=sys.version, torch=torch.__version__)
    print(json.dumps({"stage": "official", "images": len(inputs), "fallback": int(np.sum(fallback)),
                      "HERef": normalizer.normalizer.HERef.tolist(), "maxCRef": normalizer.normalizer.maxCRef.tolist()}), flush=True)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.sum(a * b, axis=-1)


def compare() -> None:
    import pandas as pd
    import torch
    from PIL import Image

    sys.path.insert(0, str(REVISION))
    from anchor_frequency_diversity_models import (  # noqa: E402
        DINOV2_REPO, DINOV2_REVISION, PLIP_REPO, PLIP_REVISION, REVIEW_HF_CACHE, Encoder, ReleasedMacenko,
    )

    stored = np.load(OUTPUT / "exaonepath_official.npz")
    patches = stored["patches"]
    meta = [json.loads(m) for m in stored["meta"]]
    labels = [{"slide_id": "MHIST_aaa", "location_index": -1, "scanner": "model_card"}] + meta
    rows = []

    def add(model: str, check: str, values: np.ndarray, extra: dict | None = None) -> None:
        for label, value in zip(labels if len(values) == len(labels) else meta, values):
            rows.append({"model": model, "check": check, **label, "cosine": float(value), **(extra or {})})

    # (i) EXAONEPath
    macenko = ReleasedMacenko()
    reference_diff = {"HERef_max_abs_diff": float(np.abs(macenko.HERef.numpy() - stored["HERef"]).max()),
                      "maxCRef_max_abs_diff": float(np.abs(macenko.maxCRef.numpy() - stored["maxCRef"]).max()),
                      "HERef": macenko.HERef.tolist(), "maxCRef": macenko.maxCRef.tolist(),
                      "class_default_HERef": macenko.class_default_HERef.tolist(),
                      "class_default_maxCRef": macenko.class_default_maxCRef.tolist(),
                      "target_mode": macenko.target_mode, "target_size": list(macenko.target_size)}
    floats = [stored["mhist"].astype(np.float32) / 255.0] + [p.astype(np.float32) / 255.0 for p in patches]
    ours = [macenko(image) for image in floats]
    official_normalized = [stored["normalized_mhist"]] + list(stored["normalized_patches"])
    image_diff = [float(np.abs(a - b).max()) for (a, _), b in zip(ours, official_normalized)]
    status = [flag for _, flag in ours]
    encoder = Encoder("exaonepath")
    released = np.concatenate([encoder([f], [m])["exaonepath"] for f, (m, _) in zip(floats, ours)])
    raw = np.concatenate([encoder([f], [f])["exaonepath_raw"] for f in floats])
    add("exaonepath", "features_as_released_vs_official", cosine(released, stored["released"]))
    add("exaonepath", "features_raw_input_vs_official", cosine(raw, stored["raw"]))
    for row, diff, flag, fb in zip(rows[:len(labels)], image_diff, status, stored["fallback"]):
        row.update({"macenko_image_max_abs_diff": diff, "macenko_status": flag, "official_fallback": bool(fb)})
    # batched call as in the embedding job (256 x 256 patches only)
    batched = encoder(floats[1:], [m for m, _ in ours[1:]])
    add("exaonepath", "features_as_released_batched_vs_official", cosine(batched["exaonepath"], stored["released"][1:]))
    add("exaonepath", "features_raw_input_batched_vs_official", cosine(batched["exaonepath_raw"], stored["raw"][1:]))
    del encoder
    torch.cuda.empty_cache()

    # (ii) PLIP and DINOv2 released processors on uint8 PIL images
    from transformers import AutoImageProcessor, CLIPImageProcessor

    pil = [Image.fromarray(p) for p in patches]
    for name, repo, revision in (("plip", PLIP_REPO, PLIP_REVISION), ("dinov2", DINOV2_REPO, DINOV2_REVISION)):
        encoder = Encoder(name)
        processor = (CLIPImageProcessor.from_pretrained(repo, revision=revision) if name == "plip"
                     else AutoImageProcessor.from_pretrained(repo, revision=revision))
        pixels = processor(images=pil, return_tensors="pt")["pixel_values"].cuda()
        with torch.inference_mode():
            if name == "plip":
                official_features = encoder.model.get_image_features(pixel_values=pixels)
            else:
                official_features = encoder.model(pixel_values=pixels).pooler_output
        official_features = official_features.float().cpu().numpy()
        ours_features = encoder([p.astype(np.float32) / 255.0 for p in patches])[name]
        with torch.inference_mode():
            x = torch.from_numpy(np.stack([p.astype(np.float32) / 255.0 for p in patches])).permute(0, 3, 1, 2).cuda()
            ours_pixels = (encoder._plip_pixels(x) if name == "plip" else encoder._dinov2_pixels(x)).cpu().numpy()
        pixel_diff = np.abs(ours_pixels - pixels.cpu().numpy()).reshape(len(patches), -1).max(axis=1)
        before = len(rows)
        add(name, "features_float_path_vs_released_processor", cosine(ours_features, official_features))
        for row, diff in zip(rows[before:], pixel_diff):
            row["pixel_max_abs_diff"] = float(diff)
        del encoder
        torch.cuda.empty_cache()

    # (iii) CONCH: review path (used) vs CONCH's released PIL transform (information only)
    from conch.open_clip_custom import create_model_from_pretrained

    encoder = Encoder("conch")
    _, preprocess = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path="hf_hub:MahmoodLab/conch",
                                                 cache_dir=str(REVIEW_HF_CACHE))
    x = torch.stack([preprocess(image) for image in pil]).cuda()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        pre = encoder.model.visual.forward_no_head(x, normalize=False)
    ours_conch = encoder([p.astype(np.float32) / 255.0 for p in patches])["conch_pre"]
    add("conch_pre", "review_path_vs_released_pil_transform (information)", cosine(ours_conch, pre.float().cpu().numpy()))

    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "parity.csv", index=False)
    gated = frame[~frame.check.str.contains("information")]
    summary = {
        "threshold": THRESHOLD,
        "pass": bool((gated.cosine >= THRESHOLD).all()),
        "min_cosine_by_check": frame.groupby(["model", "check"]).cosine.min().round(7).reset_index().to_dict("records"),
        "exaonepath_macenko": {**reference_diff,
                               "normalized_image_max_abs_diff": float(np.max(image_diff)),
                               "port_status_counts": pd.Series(status).value_counts().to_dict(),
                               "official_fallback_count": int(np.sum(stored["fallback"])),
                               "official_python": str(stored["python"]), "official_torch": str(stored["torch"])},
    }
    summary["pixel_max_abs_diff"] = {m: float(frame[frame.model.eq(m)].pixel_max_abs_diff.max()) for m in ("plip", "dinov2")}
    (OUTPUT / "parity.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(json.dumps(summary, indent=2, default=float), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("official", "compare"))
    args = parser.parse_args()
    official() if args.stage == "official" else compare()

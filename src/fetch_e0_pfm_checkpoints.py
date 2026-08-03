"""Fetch and fingerprint the frozen four-encoder E0/E4 PFM checkpoint panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download


CONTRACT_VERSION = "e0_pfm_panel_v1"
TRIDENT_COMMIT = "a6305acfef68d4c6da65e837dde1e0d1870d60e1"
RUNTIME_DISTRIBUTIONS = {
    "torch": "2.5.1",
    "torchvision": "0.20.1",
    "timm": "0.9.8",
    "huggingface_hub": "0.29.1",
    "conch": "0.1.0",
}
CONCH_SOURCE_COMMIT = "02d6ac59cc20874bff0f581de258c2b257f69a84"
MODELS = (
    {
        "encoder_id": "resnet50",
        "repo_id": "timm/resnet50.tv_in1k",
        "revision": "78f3ecfdb38e06d9b8397f662e7ab8fee96026fa",
        "filename": "pytorch_model.bin",
        "feature_dim": 1024,
        "native_fov_px": 256,
        "encoder_input_px": 224,
        "precision": "float32",
        "feature_contract": "timm features_only out_indices=[3] + adaptive average pool",
    },
    {
        "encoder_id": "uni_v1",
        "repo_id": "MahmoodLab/uni",
        "revision": "b55a5ec6cade1a39edfe6534189a9b8ca7a022f0",
        "filename": "pytorch_model.bin",
        "feature_dim": 1024,
        "native_fov_px": 256,
        "encoder_input_px": 224,
        "precision": "float16",
        "feature_contract": "ViT-L/16 class token",
    },
    {
        "encoder_id": "conch_v1",
        "repo_id": "MahmoodLab/conch",
        "revision": "f9ca9f877171a28ade80228fb195ac5d79003357",
        "filename": "pytorch_model.bin",
        "feature_dim": 512,
        "native_fov_px": 512,
        "encoder_input_px": "official_CONCH_eval_transform",
        "precision": "float32",
        "feature_contract": "encode_image with_proj=False normalize=False",
    },
    {
        "encoder_id": "virchow2",
        "repo_id": "paige-ai/Virchow2",
        "revision": "3158645804b69e3f3bc4439d4116edddf0840a72",
        "filename": "pytorch_model.bin",
        "feature_dim": 2560,
        "native_fov_px": 224,
        "encoder_input_px": 224,
        "precision": "float16",
        "feature_contract": "class token concatenated with mean patch tokens after 4 registers",
    },
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/e0_pfm_contract")
    return parser.parse_args()


def sha256(path: Path, block_size=8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_contract(models=MODELS):
    ids = [model["encoder_id"] for model in models]
    if ids != ["resnet50", "uni_v1", "conch_v1", "virchow2"]:
        raise ValueError("PFM panel or order changed from the frozen contract")
    if len({(model["repo_id"], model["revision"]) for model in models}) != 4:
        raise ValueError("checkpoint repositories and revisions must be unique")
    for model in models:
        if len(model["revision"]) != 40:
            raise ValueError(f"checkpoint revision is not a commit SHA: {model}")


def main():
    args = parse_args()
    validate_contract()
    output = Path(args.output)
    cache = output / "hf_cache"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for frozen in MODELS:
        path = Path(
            hf_hub_download(
                repo_id=frozen["repo_id"],
                filename=frozen["filename"],
                revision=frozen["revision"],
                cache_dir=cache,
                token=True,
            )
        )
        record = dict(frozen)
        record.update(
            {
                "checkpoint_path": str(path.resolve()),
                "checkpoint_size_bytes": int(path.stat().st_size),
                "checkpoint_sha256": sha256(path),
            }
        )
        records.append(record)
        print(
            f"{record['encoder_id']}: {record['checkpoint_size_bytes']} bytes "
            f"sha256={record['checkpoint_sha256']}",
            flush=True,
        )
    manifest = {
        "analysis": "e0_pfm_checkpoint_contract",
        "contract_version": CONTRACT_VERSION,
        "trident_commit": TRIDENT_COMMIT,
        "runtime_distributions": RUNTIME_DISTRIBUTIONS,
        "conch_source_commit": CONCH_SOURCE_COMMIT,
        "panel_order": [model["encoder_id"] for model in MODELS],
        "models": records,
        "checkpoint_selection_uses_results": False,
        "trident_role": "encoder factory, official preprocessing, checkpoint loading only",
        "location_sampling_role": "prenorm frozen manifest/DataLoader",
    }
    (output / "checkpoint_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

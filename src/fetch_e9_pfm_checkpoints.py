"""Download and freeze the E9 patch-encoder panel.

Separate from the E0 panel on purpose.  E0 froze `resnet50, uni_v1, conch_v1,
virchow2` against timm 0.9.8, and two of the encoders wanted here cannot run
there: UNI2-h needs `timm.layers.SwiGLUPacked` and registration tokens, which
arrived in the 1.0 line, and TRIDENT's H-Optimus loader asserts timm is exactly
0.9.16.  One environment per encoder is the only arrangement that satisfies all
of them, so the runtime is recorded per encoder rather than once for the panel.

Geometry is *not* a free choice here.  TRIDENT publishes the extraction each
encoder was trained for, and all three are 20x:

    uni_v2      --patch_size 256 --mag 20     129.33 um at this study's 0.5052
    conch_v15   --patch_size 512 --mag 20     258.66 um
    hoptimus1   --patch_size 224 --mag 20     113.16 um

The study's shared render grid is already 512 px at 0.5052 um/px, so every one of
those is a centre crop out of one tile and no new geometry is introduced.  The
0.5052 grid is AT2's own 20x rather than a nominal 0.5; the 1.04% difference is
far below the 19% spread of native pixel sizes across the panel.

Checkpoints are fetched here, on a partition that has network access, and the
GPU jobs then load them from disk by path.  Each file's sha256 is recorded so a
feature set can be traced to the exact weights that produced it.

Contract: docs/e9_plism_native_ert_contract.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

# encoder_id -> everything the extraction needs to be reproducible.
#   native_fov_px  crop taken from the 512 px / 0.5052 um grid, per TRIDENT
#   encoder_input  what the model's own eval transform resizes that crop to
#   lattice_stride 1 keeps every grid location; 2 keeps every other one in each
#                  axis, which is what a tile wider than the 129.33 um lattice
#                  pitch needs if its tiles are not to overlap
PANEL = {
    "uni_v2": {
        "repo_id": "MahmoodLab/UNI2-h",
        "filename": "pytorch_model.bin",
        "feature_dim": 1536,
        "native_fov_px": 256,
        "encoder_input_px": 224,
        # TRIDENT runs UNI2-h in bfloat16, not fp16.  A ViT-giant in fp16 is the
        # classic overflow case, and the two formats are not interchangeable at
        # the same width -- bf16 trades mantissa for fp32's exponent range.
        "precision": "bfloat16",
        "batch_size": 32,
        "lattice_stride": 1,
        "conda_env": "trident-uni2",
        "timm": "1.0.12",
        "feature_contract": "ViT-giant/14 (UNI2-h) class token, 1536-d",
    },
    "conch_v15": {
        "repo_id": "MahmoodLab/conchv1_5",
        # CONCHv1.5 publishes only its vision tower, under its own name; TRIDENT's
        # own loader fetches exactly this file.
        "filename": "pytorch_model_vision.bin",
        "feature_dim": 768,
        "native_fov_px": 512,
        "encoder_input_px": 448,
        "precision": "float16",
        "batch_size": 16,
        "lattice_stride": 2,
        "conda_env": "trident-conch15",
        "timm": "0.9.16",
        "feature_contract": "CONCHv1.5 vision tower, 768-d",
    },
    "hoptimus1": {
        "repo_id": "bioptimus/H-optimus-1",
        "filename": "pytorch_model.bin",
        "feature_dim": 1536,
        "native_fov_px": 224,
        "encoder_input_px": 224,
        # TRIDENT's H-Optimus-1 loader runs fp16; the extraction asserts this
        # against the built encoder so the two cannot drift apart.
        "precision": "float16",
        "batch_size": 32,
        "lattice_stride": 1,
        "conda_env": "trident-hoptimus",
        "timm": "0.9.16",
        "feature_contract": "ViT-giant/14 reg4 (H-optimus-1) class token, 1536-d",
    },
}

TARGET_MPP = 0.5052
TILE_PX = 512
CONTRACT_VERSION = "e9_pfm_panel_v2"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoders", default="uni_v2,conch_v15",
                        help="comma separated; hoptimus1 is available but not default")
    parser.add_argument("--output", default="outputs/e9_pfm_contract")
    parser.add_argument("--trident-root",
                        default="/mnt/isilon/oldridge_lab/irinaz/github/env/trident")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    from huggingface_hub import hf_hub_download

    output = Path(args.output)
    cache = output / "hf_cache"
    cache.mkdir(parents=True, exist_ok=True)

    wanted = [e.strip() for e in args.encoders.split(",") if e.strip()]
    unknown = [e for e in wanted if e not in PANEL]
    if unknown:
        raise KeyError(f"not in the E9 panel: {unknown}")

    models = []
    for encoder_id in wanted:
        spec = dict(PANEL[encoder_id])
        print(f"{encoder_id}: {spec['repo_id']}")
        path = Path(hf_hub_download(repo_id=spec["repo_id"], filename=spec["filename"],
                                    cache_dir=str(cache)))
        spec.update({
            "encoder_id": encoder_id,
            "checkpoint_path": str(path.resolve()),
            "checkpoint_size_bytes": path.stat().st_size,
            "checkpoint_sha256": sha256(path),
            "native_fov_um": spec["native_fov_px"] * TARGET_MPP,
        })
        models.append(spec)
        print(f"  {path.stat().st_size / 1e9:.2f} GB  sha256 {spec['checkpoint_sha256'][:16]}…")
        print(f"  crop {spec['native_fov_px']} px = {spec['native_fov_um']:.2f} um"
              f" -> model input {spec['encoder_input_px']} px"
              f"  (lattice stride {spec['lattice_stride']})")

    import subprocess

    head = subprocess.run(["git", "-C", args.trident_root, "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()

    manifest = {
        "analysis": "e9_pfm_checkpoint_contract",
        "contract_version": CONTRACT_VERSION,
        "trident_commit": head,
        "trident_root": str(Path(args.trident_root).resolve()),
        "target_mpp": TARGET_MPP,
        "tile_px": TILE_PX,
        "grid_note": ("512 px at 0.5052 um/px; every encoder's TRIDENT-published "
                      "20x extraction is a centre crop of this one tile"),
        "panel_order": wanted,
        "models": models,
    }
    # Extraction jobs read this file while it is being rewritten to add an
    # encoder, so replace it atomically rather than truncating it in place.
    destination = output / "checkpoint_manifest.json"
    staging = destination.with_suffix(".json.tmp")
    staging.write_text(json.dumps(manifest, indent=2))
    import os

    os.replace(staging, destination)
    print(f"-> {destination}")


if __name__ == "__main__":
    main()

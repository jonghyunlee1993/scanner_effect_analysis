"""GPU smoke test for the frozen TRIDENT four-encoder checkpoint contract."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from fetch_e0_pfm_checkpoints import TRIDENT_COMMIT, sha256


RELEVANT_TRIDENT_PATHS = (
    "trident/IO.py",
    "trident/patch_encoder_models/load.py",
    "trident/patch_encoder_models/utils/constants.py",
    "trident/patch_encoder_models/utils/transform_utils.py",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract", default="outputs/e0_pfm_contract/checkpoint_manifest.json"
    )
    parser.add_argument(
        "--trident-root",
        default="/mnt/isilon/oldridge_lab/irinaz/github/env/trident",
    )
    parser.add_argument("--output", default="outputs/e0_pfm_contract/smoke")
    return parser.parse_args()


def git_output(root: Path, *args) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def verify_trident_source(root: Path):
    head = git_output(root, "rev-parse", "HEAD")
    if head != TRIDENT_COMMIT:
        raise RuntimeError(f"TRIDENT HEAD changed: expected {TRIDENT_COMMIT}, found {head}")
    result = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "diff",
            "--quiet",
            TRIDENT_COMMIT,
            "--",
            *RELEVANT_TRIDENT_PATHS,
        ],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("tracked TRIDENT encoder/transform source differs from frozen commit")
    files = []
    for relative in RELEVANT_TRIDENT_PATHS:
        committed = subprocess.check_output(
            ["git", "-C", str(root), "show", f"{TRIDENT_COMMIT}:{relative}"]
        )
        working = (root / relative).read_bytes()
        if committed != working:
            raise RuntimeError(f"working source differs from commit: {relative}")
        files.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(committed).hexdigest(),
            }
        )
    return {"commit": head, "relevant_source_files": files}


def synthetic_rgb(size: int) -> Image.Image:
    y, x = np.mgrid[:size, :size]
    value = np.stack(
        [
            (3 * x + 5 * y) % 256,
            (11 * x // 7 + 2 * y) % 256,
            ((x // 8 + y // 8) % 2) * 160 + 48,
        ],
        axis=2,
    ).astype(np.uint8)
    return Image.fromarray(value, mode="RGB")


def encoder_kwargs(encoder_id: str, checkpoint_path: str):
    kwargs = {"weights_path": checkpoint_path}
    if encoder_id == "conch_v1":
        kwargs.update({"with_proj": False, "normalize": False})
    elif encoder_id == "virchow2":
        kwargs.update({"return_cls": False})
    return kwargs


def package_version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main():
    import torch

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("PFM smoke test requires a visible CUDA device")
    contract_path = Path(args.contract)
    contract = json.loads(contract_path.read_text())
    if contract.get("trident_commit") != TRIDENT_COMMIT:
        raise RuntimeError("checkpoint and source contracts name different TRIDENT commits")
    source = verify_trident_source(Path(args.trident_root))
    sys.path.insert(0, str(Path(args.trident_root).resolve()))
    from trident.patch_encoder_models.load import encoder_factory

    results = []
    for model_contract in contract["models"]:
        checkpoint = Path(model_contract["checkpoint_path"])
        if sha256(checkpoint) != model_contract["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {model_contract['encoder_id']}")
        encoder_id = model_contract["encoder_id"]
        encoder = encoder_factory(
            encoder_id, **encoder_kwargs(encoder_id, str(checkpoint))
        )
        image = synthetic_rgb(int(model_contract["native_fov_px"]))
        transformed = encoder.eval_transforms(image)
        batch = transformed.unsqueeze(0).cuda(non_blocking=False)
        encoder = encoder.eval().cuda()
        if encoder.precision == torch.float16:
            encoder = encoder.half()
            batch = batch.half()
        else:
            encoder = encoder.float()
            batch = batch.float()
        with torch.inference_mode():
            first = encoder(batch)
            second = encoder(batch)
        first = first.float().cpu()
        second = second.float().cpu()
        expected_shape = (1, int(model_contract["feature_dim"]))
        if tuple(first.shape) != expected_shape:
            raise ValueError(
                f"{encoder_id}: feature shape {tuple(first.shape)} != {expected_shape}"
            )
        if not torch.isfinite(first).all():
            raise ValueError(f"{encoder_id}: non-finite features")
        repeat_max_abs = float(torch.max(torch.abs(first - second)))
        if repeat_max_abs != 0.0:
            raise ValueError(f"{encoder_id}: repeated eval differs by {repeat_max_abs}")
        results.append(
            {
                "encoder_id": encoder_id,
                "native_fov_px": int(model_contract["native_fov_px"]),
                "transformed_shape_chw": list(transformed.shape),
                "transform_output_dtype": str(transformed.dtype),
                "encoder_precision": str(encoder.precision),
                "feature_shape": list(first.shape),
                "feature_dtype_after_audit_cast": str(first.dtype),
                "feature_min": float(first.min()),
                "feature_max": float(first.max()),
                "feature_l2": float(torch.linalg.vector_norm(first)),
                "repeat_max_abs": repeat_max_abs,
                "checkpoint_sha256": model_contract["checkpoint_sha256"],
                "smoke_pass": True,
            }
        )
        del encoder, batch, first, second, transformed
        torch.cuda.empty_cache()
        print(json.dumps(results[-1], indent=2), flush=True)

    summary = {
        "analysis": "e0_pfm_encoder_gpu_smoke",
        "contract_version": contract["contract_version"],
        "trident_source": source,
        "models_expected": 4,
        "models_passing": len(results),
        "all_models_pass": len(results) == 4 and all(row["smoke_pass"] for row in results),
        "device": torch.cuda.get_device_name(0),
        "torch_version": str(torch.__version__),
        "torchvision_version": package_version("torchvision"),
        "timm_version": package_version("timm"),
        "conch_version": package_version("conch"),
        "results": results,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

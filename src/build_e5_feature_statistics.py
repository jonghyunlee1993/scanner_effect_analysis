"""Build all-slide feature sufficient statistics for exact LOSO CORAL/Procrustes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

from e5_comparator_population import E5_VERSION, feature_sufficient_statistics
from extract_e0_pfm_features import select_model
from fetch_e0_pfm_checkpoints import sha256


def parse_args():
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--encoder-id")
    selection.add_argument("--encoder-index", type=int)
    parser.add_argument("--contract", default="outputs/e0_pfm_contract/checkpoint_manifest.json")
    parser.add_argument("--raw-audit", default="outputs/e0_pfm_features/audit/summary.json")
    parser.add_argument("--raw", default="outputs/e0_pfm_features")
    parser.add_argument("--output", default="outputs/e5_feature_statistics")
    return parser.parse_args()


def existing_output_passes(path: Path, summary_path: Path, model: dict, audit_sha: str):
    if not path.exists() or not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text())
        return bool(
            summary.get("analysis") == "e5_feature_sufficient_statistics"
            and summary.get("e5_version") == E5_VERSION
            and summary.get("encoder_id") == model["encoder_id"]
            and summary.get("feature_dim") == int(model["feature_dim"])
            and summary.get("raw_audit_sha256") == audit_sha
            and summary.get("checkpoint_sha256") == model["checkpoint_sha256"]
            and summary.get("output_sha256") == sha256(path)
            and summary.get("statistics_gate_pass") is True
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def main():
    import torch

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("E5 feature sufficient statistics require a visible CUDA device")
    contract = json.loads(Path(args.contract).read_text())
    model = select_model(contract, args.encoder_id, args.encoder_index)
    audit_path = Path(args.raw_audit)
    audit = json.loads(audit_path.read_text())
    if audit.get("population_gate_pass") is not True or audit.get("features_observed") != 261_600:
        raise RuntimeError("raw four-PFM population has not passed its frozen audit")
    audit_sha = sha256(audit_path)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    model_id = model["encoder_id"]
    output_path = output_root / f"{model_id}.h5"
    summary_path = output_root / f"{model_id}.summary.json"
    if existing_output_passes(output_path, summary_path, model, audit_sha):
        print(summary_path.read_text(), end="")
        return

    paths = sorted((Path(args.raw) / model_id / "shards").glob("*.h5"))
    if len(paths) != 109:
        raise ValueError(f"{model_id}: expected 109 raw feature shards, got {len(paths)}")
    sums = grams = cross = None
    count = 0
    slide_ids = []
    for index, path in enumerate(paths):
        with h5py.File(path, "r") as source:
            slide_id = str(source.attrs["slide_id"])
            values = np.asarray(source["features"][:], dtype=np.float32)
        if slide_id != path.stem or values.shape != (6, 100, int(model["feature_dim"])):
            raise ValueError(f"{model_id}/{path.stem}: raw identity or shape mismatch")
        local = feature_sufficient_statistics(torch.from_numpy(values).cuda())
        if sums is None:
            sums = torch.zeros_like(local[0])
            grams = torch.zeros_like(local[1])
            cross = torch.zeros_like(local[2])
        sums += local[0]
        grams += local[1]
        cross += local[2]
        count += local[3]
        slide_ids.append(slide_id)
        del values, local
        print(f"[{index + 1}/109] {model_id} {slide_id}", flush=True)

    if count != 10_900 or sums is None or not all(
        torch.isfinite(value).all() for value in (sums, grams, cross)
    ):
        raise RuntimeError(f"{model_id}: incomplete/non-finite feature sufficient statistics")
    temporary = output_root / f".{model_id}.{os.getpid()}.tmp.h5"
    with h5py.File(temporary, "w") as target:
        target.create_dataset("sum", data=sums.cpu().numpy(), compression="lzf", shuffle=True)
        target.create_dataset("gram", data=grams.cpu().numpy(), compression="lzf", shuffle=True)
        target.create_dataset("cross_to_at2", data=cross.cpu().numpy(), compression="lzf", shuffle=True)
        target.create_dataset("slide_id", data=np.asarray(slide_ids, dtype=h5py.string_dtype("utf-8")))
        target.attrs["analysis"] = "e5_feature_sufficient_statistics"
        target.attrs["e5_version"] = E5_VERSION
        target.attrs["encoder_id"] = model_id
        target.attrs["feature_dim"] = int(model["feature_dim"])
        target.attrs["sample_count_per_scanner"] = count
        target.attrs["checkpoint_sha256"] = model["checkpoint_sha256"]
        target.attrs["raw_audit_sha256"] = audit_sha
        target.flush()
    os.replace(temporary, output_path)
    summary = {
        "analysis": "e5_feature_sufficient_statistics",
        "e5_version": E5_VERSION,
        "encoder_id": model_id,
        "feature_dim": int(model["feature_dim"]),
        "slides": len(slide_ids),
        "samples_per_scanner": count,
        "checkpoint_sha256": model["checkpoint_sha256"],
        "raw_audit": str(audit_path.resolve()),
        "raw_audit_sha256": audit_sha,
        "output": str(output_path.resolve()),
        "output_sha256": sha256(output_path),
        "statistics_gate_pass": True,
        "device": torch.cuda.get_device_name(0),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


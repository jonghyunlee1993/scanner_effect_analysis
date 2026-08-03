"""Extract the frozen Supplement-only LOSO Macenko population for four PFMs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

from e5_comparator_population import E5_VERSION, SCANNERS, centered_crop, rgb8_to_rgb01, uint8_from_rgb01
from e5_macenko_supplement import MACENKO_VERSION, aggregate_loso_reference, macenko_normalize_batch
from extract_e0_pfm_features import BATCH_SIZE, load_grid_contract, select_model, string_array
from extract_e5_image_features import infer
from fetch_e0_pfm_checkpoints import TRIDENT_COMMIT, sha256
from smoke_e0_pfm_encoders import encoder_kwargs, verify_runtime_contract, verify_trident_source


CONDITION = "macenko_supplement"


def parse_args():
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--encoder-id")
    selection.add_argument("--encoder-index", type=int)
    parser.add_argument("--slide-id")
    parser.add_argument("--slide-index", type=int)
    parser.add_argument("--contract", default="outputs/e0_pfm_contract/checkpoint_manifest.json")
    parser.add_argument("--grid", default="outputs/e0_native_aa_grid/shards")
    parser.add_argument("--raw", default="outputs/e0_pfm_features")
    parser.add_argument("--reference", default="outputs/e5_macenko_references")
    parser.add_argument("--supplement-contract", default="docs/e5_macenko_supplement_contract.md")
    parser.add_argument("--output", default="outputs/e5_macenko_features")
    parser.add_argument("--trident-root", default="/mnt/isilon/oldridge_lab/irinaz/github/env/trident")
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def load_reference(root: Path, fov: int):
    path = root / f"fov_{fov}.npz"
    summary_path = root / f"fov_{fov}.summary.json"
    summary = json.loads(summary_path.read_text())
    if not (
        summary.get("analysis") == "e5_macenko_reference_population"
        and summary.get("e5_version") == E5_VERSION
        and summary.get("macenko_version") == MACENKO_VERSION
        and summary.get("fov") == fov
        and summary.get("slides") == 109
        and summary.get("output_sha256") == sha256(path)
        and summary.get("reference_gate_pass") is True
    ):
        raise RuntimeError(f"invalid Macenko reference: {path}")
    with np.load(path) as source:
        values = {name: source[name] for name in source.files}
    return path, summary, values


def existing_output_passes(path, model, grid_summary, raw_summary, reference_summary, contract_sha):
    summary_path = path.with_suffix(".summary.json")
    if not path.exists() or not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text())
        return bool(
            summary.get("analysis") == "e5_macenko_feature_shard"
            and summary.get("macenko_version") == MACENKO_VERSION
            and summary.get("encoder_id") == model["encoder_id"]
            and summary.get("source_grid_sha256") == grid_summary["output_sha256"]
            and summary.get("raw_feature_sha256") == raw_summary["output_sha256"]
            and summary.get("reference_sha256") == reference_summary["output_sha256"]
            and summary.get("supplement_contract_sha256") == contract_sha
            and summary.get("output_sha256") == sha256(path)
            and summary.get("shard_gate_pass") is True
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def extract_shard(grid_path, raw_path, output_path, model, encoder, batch_size, reference_path, reference_summary, reference, contract_path):
    import torch

    grid_summary = load_grid_contract(grid_path)
    raw_summary = json.loads(raw_path.with_suffix(".summary.json").read_text())
    contract_sha = sha256(contract_path)
    if existing_output_passes(output_path, model, grid_summary, raw_summary, reference_summary, contract_sha):
        return json.loads(output_path.with_suffix(".summary.json").read_text()), True
    with h5py.File(grid_path, "r") as grid, h5py.File(raw_path, "r") as raw_source:
        slide_id = str(grid.attrs["slide_id"])
        raw = np.asarray(raw_source["features"][:], dtype=np.float32)
        slide_ids = [str(value) for value in reference["slide_ids"]]
        if slide_ids.count(slide_id) != 1:
            raise ValueError(f"{slide_id}: absent/duplicated Macenko reference")
        target_stains, target_maximum, train_slides = aggregate_loso_reference(
            reference["stains"], reference["maximum"], slide_ids.index(slide_id)
        )
        target_stains_tensor = torch.from_numpy(target_stains).cuda()
        target_maximum_tensor = torch.from_numpy(target_maximum).cuda()
        dimension = int(model["feature_dim"])
        fov = int(model["native_fov_px"])
        values = np.empty((1, 6, 100, dimension), dtype=np.float32)
        values[0, 0] = raw[0]
        clip_fraction = np.zeros((1, 6, 100), dtype=np.float32)
        clip_mae = np.zeros_like(clip_fraction)
        fallback = np.zeros((1, 6, 100), dtype=np.uint8)
        for scanner_index in range(1, 6):
            for start in range(0, 100, batch_size):
                stop = min(start + batch_size, 100)
                rgb8 = centered_crop(grid["rgb"][scanner_index, start:stop], fov)
                rgb = rgb8_to_rgb01(rgb8, device="cuda")
                report = macenko_normalize_batch(rgb, target_stains_tensor, target_maximum_tensor)
                rendered = uint8_from_rgb01(report["output"])
                values[0, scanner_index, start:stop] = infer(encoder, rendered, batch_size, dimension)
                clip_fraction[0, scanner_index, start:stop] = report["preclip_range_fraction"].cpu().numpy()
                clip_mae[0, scanner_index, start:stop] = report["clip_pixel_mae"].cpu().numpy()
                fallback[0, scanner_index, start:stop] = report["fallback"].byte().cpu().numpy()
            print(f"{model['encoder_id']} {slide_id} {SCANNERS[scanner_index]} Macenko", flush=True)
        metadata = {
            name: grid[name][:]
            for name in ("scanner", "location_id", "replicate_id", "canonical_center_x", "canonical_center_y")
        }
    norms = np.linalg.norm(values.astype(np.float64), axis=-1)
    if not np.isfinite(values).all() or np.any(norms <= 0) or not np.array_equal(values[:, 0], raw[None, 0]):
        raise ValueError(f"{model['encoder_id']}/{slide_id}: invalid Macenko features")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    with h5py.File(temporary, "w") as target:
        target.create_dataset("features", data=values, chunks=(1, 1, 100, dimension), compression="lzf", shuffle=True)
        target.create_dataset("preclip_range_fraction", data=clip_fraction)
        target.create_dataset("clip_pixel_mae", data=clip_mae)
        target.create_dataset("fallback", data=fallback)
        target.create_dataset("condition", data=string_array([CONDITION]))
        for name, local in metadata.items():
            target.create_dataset(name, data=local)
        target.attrs["analysis"] = "e5_macenko_feature_shard"
        target.attrs["e5_version"] = E5_VERSION
        target.attrs["macenko_version"] = MACENKO_VERSION
        target.attrs["slide_id"] = slide_id
        target.attrs["encoder_id"] = model["encoder_id"]
        target.attrs["feature_dim"] = dimension
        target.attrs["checkpoint_sha256"] = model["checkpoint_sha256"]
        target.attrs["source_grid_sha256"] = grid_summary["output_sha256"]
        target.attrs["raw_feature_sha256"] = raw_summary["output_sha256"]
        target.attrs["reference_sha256"] = reference_summary["output_sha256"]
        target.attrs["supplement_contract_sha256"] = contract_sha
        target.attrs["loso_target_train_slides"] = train_slides
        target.flush()
    os.replace(temporary, output_path)
    summary = {
        "analysis": "e5_macenko_feature_shard",
        "e5_version": E5_VERSION,
        "macenko_version": MACENKO_VERSION,
        "slide_id": slide_id,
        "encoder_id": model["encoder_id"],
        "feature_dim": dimension,
        "condition": CONDITION,
        "features": 600,
        "loso_target_train_slides": train_slides,
        "fallback_patches": int(fallback.sum()),
        "fallback_fraction": float(fallback[:, 1:].mean()),
        "preclip_fraction_max": float(clip_fraction.max()),
        "clip_pixel_mae_max": float(clip_mae.max()),
        "minimum_norm": float(norms.min()),
        "checkpoint_sha256": model["checkpoint_sha256"],
        "source_grid_sha256": grid_summary["output_sha256"],
        "raw_feature_sha256": raw_summary["output_sha256"],
        "reference": str(reference_path.resolve()),
        "reference_sha256": reference_summary["output_sha256"],
        "supplement_contract_sha256": contract_sha,
        "output": str(output_path.resolve()),
        "output_sha256": sha256(output_path),
        "shard_gate_pass": True,
    }
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary, False


def main():
    import torch

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Macenko feature extraction requires a visible CUDA device")
    contract = json.loads(Path(args.contract).read_text())
    if contract.get("trident_commit") != TRIDENT_COMMIT:
        raise RuntimeError("PFM contract does not use the frozen TRIDENT commit")
    verify_runtime_contract(contract)
    verify_trident_source(Path(args.trident_root))
    model = select_model(contract, args.encoder_id, args.encoder_index)
    if sha256(Path(model["checkpoint_path"])) != model["checkpoint_sha256"]:
        raise RuntimeError(f"checkpoint hash mismatch: {model['encoder_id']}")
    sys.path.insert(0, str(Path(args.trident_root).resolve()))
    from trident.patch_encoder_models.load import encoder_factory

    encoder = encoder_factory(model["encoder_id"], **encoder_kwargs(model["encoder_id"], model["checkpoint_path"])).eval().cuda()
    encoder = encoder.half() if encoder.precision == torch.float16 else encoder.float()
    batch_size = args.batch_size or BATCH_SIZE[model["encoder_id"]]
    reference_path, reference_summary, reference = load_reference(Path(args.reference), int(model["native_fov_px"]))
    paths = sorted(Path(args.grid).glob("*.h5"))
    if args.slide_id is not None:
        paths = [Path(args.grid) / f"{args.slide_id}.h5"]
    elif args.slide_index is not None:
        paths = [paths[args.slide_index]]
    expected = 1 if args.slide_id is not None or args.slide_index is not None else 109
    if len(paths) != expected:
        raise ValueError(f"expected {expected} grids, got {len(paths)}")
    summaries = []
    reused = 0
    for index, grid_path in enumerate(paths):
        raw_path = Path(args.raw) / model["encoder_id"] / "shards" / grid_path.name
        output_path = Path(args.output) / model["encoder_id"] / "shards" / grid_path.name
        summary, existing = extract_shard(
            grid_path, raw_path, output_path, model, encoder, batch_size,
            reference_path, reference_summary, reference, Path(args.supplement_contract)
        )
        summaries.append(summary)
        reused += int(existing)
        print(f"[{index + 1}/{len(paths)}] {model['encoder_id']} {summary['slide_id']} {'reused' if existing else 'written'}", flush=True)
    run_output = Path(args.output) / model["encoder_id"]
    run_output.mkdir(parents=True, exist_ok=True)
    run_summary = {
        "analysis": "e5_macenko_feature_extraction",
        "macenko_version": MACENKO_VERSION,
        "encoder_id": model["encoder_id"],
        "slides": len(summaries),
        "features": len(summaries) * 600,
        "fallback_patches": sum(item["fallback_patches"] for item in summaries),
        "new_shards": len(summaries) - reused,
        "reused_shards": reused,
        "all_shards_pass": all(item["shard_gate_pass"] for item in summaries),
        "device": torch.cuda.get_device_name(0),
    }
    (run_output / "summary.json").write_text(json.dumps(run_summary, indent=2) + "\n")
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()

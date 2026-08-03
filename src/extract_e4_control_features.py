"""Extract the frozen nine-condition E4 control population for four PFMs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

from e4_control_population import (
    CONTROL_SPECS,
    CONTROL_VERSION,
    PYRAMID_LEVELS,
    SCANNERS,
    analysis_tensor_to_uint8,
    centered_crop,
    condition_names,
    decompose_scanner_tensor,
    detail_rms,
    heldout_global_mean_bands,
    operational_detail_rms,
    registered_loo_bands,
    render_control,
    requested_bands,
    slide_band_sums,
    uint8_to_analysis_tensor,
)
from extract_e0_pfm_features import (
    BATCH_SIZE,
    load_grid_contract,
    select_model,
    string_array,
    transformed_batch,
)
from fetch_e0_pfm_checkpoints import TRIDENT_COMMIT, sha256
from prenorm.exp01.frequency import FixedLaplacianPyramid
from smoke_e0_pfm_encoders import (
    encoder_kwargs,
    verify_runtime_contract,
    verify_trident_source,
)


def parse_args():
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--encoder-id")
    selection.add_argument("--encoder-index", type=int)
    parser.add_argument("--slide-id")
    parser.add_argument("--slide-index", type=int)
    parser.add_argument(
        "--contract", default="outputs/e0_pfm_contract/checkpoint_manifest.json"
    )
    parser.add_argument(
        "--grid-audit", default="outputs/e0_native_aa_grid/audit/summary.json"
    )
    parser.add_argument("--grid", default="outputs/e0_native_aa_grid/shards")
    parser.add_argument("--reference", default="outputs/e4_control_references")
    parser.add_argument("--output", default="outputs/e4_control_features")
    parser.add_argument(
        "--trident-root",
        default="/mnt/isilon/oldridge_lab/irinaz/github/env/trident",
    )
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def load_reference(root: Path, fov: int, grid_audit_sha256: str):
    path = root / f"fov_{fov}.npz"
    summary_path = root / f"fov_{fov}.summary.json"
    summary = json.loads(summary_path.read_text())
    if not (
        summary.get("analysis") == "e4_global_hf_reference"
        and summary.get("control_version") == CONTROL_VERSION
        and summary.get("fov") == fov
        and summary.get("pyramid_levels") == PYRAMID_LEVELS
        and summary.get("grid_audit_sha256") == grid_audit_sha256
        and summary.get("output_sha256") == sha256(path)
        and summary.get("reference_gate_pass") is True
    ):
        raise RuntimeError(f"E4 global reference contract failed: {path}")
    with np.load(path) as source:
        total_band_sums = [
            np.asarray(source[f"band_sum_{index}"], dtype=np.float64)
            for index in range(PYRAMID_LEVELS)
        ]
        image_count = int(source["image_count"])
        slide_ids = [str(value) for value in source["slide_ids"]]
    if image_count != 65400 or len(slide_ids) != 109 or len(set(slide_ids)) != 109:
        raise RuntimeError(f"incomplete E4 global reference population: {path}")
    return path, summary, total_band_sums, image_count, set(slide_ids)


def existing_output_passes(
    path: Path,
    model: dict,
    source_summary: dict,
    reference_summary: dict,
):
    summary_path = path.with_suffix(".summary.json")
    if not path.exists() or not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text())
        return bool(
            summary.get("analysis") == "e4_control_feature_shard"
            and summary.get("control_version") == CONTROL_VERSION
            and summary.get("encoder_id") == model["encoder_id"]
            and summary.get("checkpoint_sha256") == model["checkpoint_sha256"]
            and summary.get("source_grid_sha256") == source_summary["output_sha256"]
            and summary.get("global_reference_sha256") == reference_summary["output_sha256"]
            and summary.get("conditions") == list(condition_names())
            and summary.get("output_sha256") == sha256(path)
            and summary.get("shard_gate_pass") is True
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def infer_condition(encoder, images: np.ndarray, batch_size: int, feature_dim: int):
    import torch

    flat = images.reshape(-1, *images.shape[2:])
    outputs = np.empty((len(flat), feature_dim), dtype=np.float32)
    for start in range(0, len(flat), batch_size):
        stop = min(start + batch_size, len(flat))
        batch = transformed_batch(flat[start:stop], encoder.eval_transforms)
        batch = batch.cuda(non_blocking=True)
        batch = batch.half() if encoder.precision == torch.float16 else batch.float()
        with torch.inference_mode():
            output = encoder(batch).float().cpu().numpy()
        if output.shape != (stop - start, feature_dim) or not np.isfinite(output).all():
            raise ValueError(f"invalid E4 feature batch: {output.shape}")
        outputs[start:stop] = output
    return outputs.reshape(6, 100, feature_dim)


def extract_shard(
    grid_path: Path,
    output_path: Path,
    model: dict,
    encoder,
    batch_size: int,
    reference_path: Path,
    reference_summary: dict,
    total_band_sums: list[np.ndarray],
    total_image_count: int,
    reference_slide_ids: set[str],
):
    import torch

    source_summary = load_grid_contract(grid_path)
    if existing_output_passes(output_path, model, source_summary, reference_summary):
        return json.loads(output_path.with_suffix(".summary.json").read_text()), True

    fov = int(model["native_fov_px"])
    feature_dim = int(model["feature_dim"])
    with h5py.File(grid_path, "r") as source:
        slide_id = str(source.attrs["slide_id"])
        scanner_values = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in source["scanner"][:]
        ]
        if source["rgb"].shape != (6, 100, 512, 512, 3) or scanner_values != list(SCANNERS):
            raise ValueError(f"{slide_id}: invalid native-AA source grid")
        if slide_id not in reference_slide_ids:
            raise ValueError(f"{slide_id}: absent from E4 global reference")
        rgb = centered_crop(source["rgb"][:], fov)
        metadata = {
            name: source[name][:]
            for name in (
                "scanner",
                "location_id",
                "replicate_id",
                "canonical_center_x",
                "canonical_center_y",
            )
        }

    pyramid = FixedLaplacianPyramid(PYRAMID_LEVELS).eval().cuda()
    images = uint8_to_analysis_tensor(rgb, device="cuda")
    coarse, source_bands = decompose_scanner_tensor(pyramid, images)
    registered_bands = registered_loo_bands(source_bands)
    heldout_sums = slide_band_sums(source_bands)
    global_bands = heldout_global_mean_bands(
        total_band_sums,
        heldout_sums,
        total_image_count,
        600,
        "cuda",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    clipping_max = 0.0
    clip_mae_max = 0.0
    feature_min = np.inf
    feature_max = -np.inf
    minimum_norm = np.inf
    with h5py.File(temporary, "w") as target:
        features = target.create_dataset(
            "features",
            shape=(len(CONTROL_SPECS), 6, 100, feature_dim),
            dtype=np.float32,
            chunks=(1, 1, 100, feature_dim),
            compression="lzf",
            shuffle=True,
        )
        preclip_fraction = target.create_dataset(
            "preclip_range_fraction", shape=(len(CONTROL_SPECS), 6, 100), dtype=np.float32
        )
        clip_pixel_mae = target.create_dataset(
            "clip_pixel_mae", shape=(len(CONTROL_SPECS), 6, 100), dtype=np.float32
        )
        requested_rms = target.create_dataset(
            "requested_hf_rms", shape=(len(CONTROL_SPECS), 6, 100), dtype=np.float32
        )
        operational_rms = target.create_dataset(
            "operational_hf_rms", shape=(len(CONTROL_SPECS), 6, 100), dtype=np.float32
        )

        for condition_index, spec in enumerate(CONTROL_SPECS):
            bands = requested_bands(spec, source_bands, registered_bands, global_bands)
            rendered = render_control(pyramid, coarse, bands)
            request_values = detail_rms(pyramid, coarse, bands)
            operational_values = operational_detail_rms(pyramid, rendered["image"])
            condition_rgb = analysis_tensor_to_uint8(rendered["image"])
            condition_features = infer_condition(
                encoder, condition_rgb, batch_size, feature_dim
            )
            norms = np.linalg.norm(condition_features, axis=-1)
            if not np.isfinite(norms).all() or np.any(norms <= 0):
                raise ValueError(f"{slide_id}/{model['encoder_id']}/{spec.name}: invalid norms")
            features[condition_index] = condition_features
            preclip_values = rendered["preclip_range_fraction"].float().cpu().numpy()
            clip_mae_values = rendered["clip_pixel_mae"].float().cpu().numpy()
            preclip_fraction[condition_index] = preclip_values
            clip_pixel_mae[condition_index] = clip_mae_values
            requested_rms[condition_index] = request_values.float().cpu().numpy()
            operational_rms[condition_index] = operational_values.float().cpu().numpy()
            clipping_max = max(clipping_max, float(preclip_values.max()))
            clip_mae_max = max(clip_mae_max, float(clip_mae_values.max()))
            feature_min = min(feature_min, float(condition_features.min()))
            feature_max = max(feature_max, float(condition_features.max()))
            minimum_norm = min(minimum_norm, float(norms.min()))
            print(
                f"{model['encoder_id']} {slide_id} {condition_index + 1}/"
                f"{len(CONTROL_SPECS)} {spec.name}",
                flush=True,
            )
            del bands, rendered, request_values, operational_values, condition_rgb, condition_features

        for name, values in metadata.items():
            target.create_dataset(name, data=values)
        target.create_dataset("condition", data=string_array(condition_names()))
        target.create_dataset(
            "condition_family", data=string_array(spec.family for spec in CONTROL_SPECS)
        )
        target.create_dataset(
            "condition_role", data=string_array(spec.role for spec in CONTROL_SPECS)
        )
        target.create_dataset(
            "condition_value",
            data=np.asarray([spec.value for spec in CONTROL_SPECS], dtype=np.float32),
        )
        target.attrs["analysis"] = "e4_control_feature_shard"
        target.attrs["control_version"] = CONTROL_VERSION
        target.attrs["slide_id"] = slide_id
        target.attrs["encoder_id"] = model["encoder_id"]
        target.attrs["feature_dim"] = feature_dim
        target.attrs["native_fov_px"] = fov
        target.attrs["pyramid_levels"] = PYRAMID_LEVELS
        target.attrs["checkpoint_sha256"] = model["checkpoint_sha256"]
        target.attrs["source_grid_sha256"] = source_summary["output_sha256"]
        target.attrs["global_reference_sha256"] = reference_summary["output_sha256"]
        target.attrs["global_reference_path"] = str(reference_path.resolve())
        target.attrs["global_reference_fit"] = "exact_leave_one_physical_slide_out"
        target.attrs["pixel_source"] = "native_wsi_only"
        target.attrs["trident_role"] = "encoder_factory_and_official_eval_transform_only"
        target.flush()
    os.replace(temporary, output_path)
    summary = {
        "analysis": "e4_control_feature_shard",
        "control_version": CONTROL_VERSION,
        "slide_id": slide_id,
        "encoder_id": model["encoder_id"],
        "feature_dim": feature_dim,
        "native_fov_px": fov,
        "pyramid_levels": PYRAMID_LEVELS,
        "conditions": list(condition_names()),
        "condition_count": len(CONTROL_SPECS),
        "scanners": 6,
        "locations_per_scanner": 100,
        "features": len(CONTROL_SPECS) * 600,
        "feature_min": feature_min,
        "feature_max": feature_max,
        "minimum_feature_norm": minimum_norm,
        "maximum_preclip_range_fraction": clipping_max,
        "maximum_clip_pixel_mae": clip_mae_max,
        "checkpoint_sha256": model["checkpoint_sha256"],
        "source_grid_sha256": source_summary["output_sha256"],
        "global_reference_sha256": reference_summary["output_sha256"],
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
        raise RuntimeError("E4 PFM control extraction requires a visible CUDA device")
    grid_audit_path = Path(args.grid_audit)
    grid_audit = json.loads(grid_audit_path.read_text())
    if grid_audit.get("grid_gate_pass") is not True:
        raise RuntimeError("native-AA grid audit has not passed")
    grid_audit_sha = sha256(grid_audit_path)
    contract = json.loads(Path(args.contract).read_text())
    if contract.get("trident_commit") != TRIDENT_COMMIT:
        raise RuntimeError("PFM contract does not use the frozen TRIDENT commit")
    verify_runtime_contract(contract)
    verify_trident_source(Path(args.trident_root))
    model = select_model(contract, args.encoder_id, args.encoder_index)
    if sha256(Path(model["checkpoint_path"])) != model["checkpoint_sha256"]:
        raise RuntimeError(f"checkpoint hash mismatch: {model['encoder_id']}")

    reference = load_reference(
        Path(args.reference), int(model["native_fov_px"]), grid_audit_sha
    )
    reference_path, reference_summary, total_sums, image_count, slide_ids = reference

    sys.path.insert(0, str(Path(args.trident_root).resolve()))
    from trident.patch_encoder_models.load import encoder_factory

    encoder = encoder_factory(
        model["encoder_id"],
        **encoder_kwargs(model["encoder_id"], model["checkpoint_path"]),
    ).eval().cuda()
    encoder = encoder.half() if encoder.precision == torch.float16 else encoder.float()
    batch_size = args.batch_size or BATCH_SIZE[model["encoder_id"]]
    if batch_size <= 0:
        raise ValueError("batch size must be positive")

    grid_root = Path(args.grid)
    paths = sorted(grid_root.glob("*.h5"))
    if args.slide_id is not None and args.slide_index is not None:
        raise ValueError("choose either --slide-id or --slide-index")
    if args.slide_id:
        paths = [grid_root / f"{args.slide_id}.h5"]
    elif args.slide_index is not None:
        if args.slide_index < 0 or args.slide_index >= len(paths):
            raise IndexError(f"slide index {args.slide_index} outside 0..{len(paths) - 1}")
        paths = [paths[args.slide_index]]
    expected = 1 if args.slide_id is not None or args.slide_index is not None else 109
    if len(paths) != expected:
        raise ValueError(f"expected {expected} source grids, got {len(paths)}")
    output_root = Path(args.output) / model["encoder_id"] / "shards"
    summaries = []
    reused = 0
    for index, grid_path in enumerate(paths):
        summary, existing = extract_shard(
            grid_path,
            output_root / grid_path.name,
            model,
            encoder,
            batch_size,
            reference_path,
            reference_summary,
            total_sums,
            image_count,
            slide_ids,
        )
        summaries.append(summary)
        reused += int(existing)
        print(
            f"[{index + 1}/{len(paths)}] {model['encoder_id']} {summary['slide_id']} "
            f"{'reused' if existing else 'written'}",
            flush=True,
        )

    run_summary = {
        "analysis": "e4_control_feature_extraction",
        "control_version": CONTROL_VERSION,
        "encoder_id": model["encoder_id"],
        "slides": len(summaries),
        "conditions": list(condition_names()),
        "features": len(summaries) * len(CONTROL_SPECS) * 600,
        "new_shards": len(summaries) - reused,
        "reused_shards": reused,
        "all_shards_pass": all(summary["shard_gate_pass"] for summary in summaries),
        "checkpoint_sha256": model["checkpoint_sha256"],
        "trident_commit": TRIDENT_COMMIT,
        "global_reference_sha256": reference_summary["output_sha256"],
        "device": torch.cuda.get_device_name(0),
    }
    run_output = Path(args.output) / model["encoder_id"]
    run_output.mkdir(parents=True, exist_ok=True)
    if expected == 1:
        run_summary_root = run_output / "runs"
        run_summary_root.mkdir(parents=True, exist_ok=True)
        run_summary_path = run_summary_root / f"{summaries[0]['slide_id']}.summary.json"
    else:
        run_summary_path = run_output / "summary.json"
    run_summary_path.write_text(json.dumps(run_summary, indent=2) + "\n")
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()


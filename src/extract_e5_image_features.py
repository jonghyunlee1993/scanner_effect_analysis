"""Extract the three frozen LOSO E5 image-comparator populations for four PFMs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

from e5_comparator_population import (
    E5_VERSION,
    IMAGE_CONDITIONS,
    SCANNERS,
    centered_crop,
    fitted_frequency_gain,
    frequency_calibration,
    mean_std,
    paired_od_affine,
    radial_geometry,
    reinhard_lab,
    rgb8_to_rgb01,
    solve_od_affine,
    uint8_from_rgb01,
)
from extract_e0_pfm_features import (
    BATCH_SIZE,
    load_grid_contract,
    select_model,
    string_array,
    transformed_batch,
)
from fetch_e0_pfm_checkpoints import TRIDENT_COMMIT, sha256
from smoke_e0_pfm_encoders import encoder_kwargs, verify_runtime_contract, verify_trident_source


def parse_args():
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--encoder-id")
    selection.add_argument("--encoder-index", type=int)
    parser.add_argument("--slide-id")
    parser.add_argument("--slide-index", type=int)
    parser.add_argument("--contract", default="outputs/e0_pfm_contract/checkpoint_manifest.json")
    parser.add_argument("--grid-audit", default="outputs/e0_native_aa_grid/audit/summary.json")
    parser.add_argument("--grid", default="outputs/e0_native_aa_grid/shards")
    parser.add_argument("--raw", default="outputs/e0_pfm_features")
    parser.add_argument("--image-statistics", default="outputs/e5_image_statistics")
    parser.add_argument("--stability", default="outputs/e5_input_stability/summary.json")
    parser.add_argument("--execution-contract", default="docs/e5_comparator_execution_contract.md")
    parser.add_argument("--output", default="outputs/e5_image_features")
    parser.add_argument(
        "--trident-root", default="/mnt/isilon/oldridge_lab/irinaz/github/env/trident"
    )
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def load_stability(path: Path, execution_path: Path):
    summary = json.loads(path.read_text())
    if not (
        summary.get("analysis") == "e5_input_only_stability"
        and summary.get("e5_version") == E5_VERSION
        and summary.get("outcome_access") is False
        and summary.get("stability_gate_pass") is True
        and summary.get("execution_contract_sha256") == sha256(execution_path)
        and summary.get("selected_frequency_gain_cap") in (
            1.01,
            1.02,
            1.03,
            1.04,
            1.05,
            1.1,
            1.15,
            1.2,
            1.25,
            1.5,
            2.0,
            3.0,
            4.0,
        )
    ):
        raise RuntimeError("E5 input-only stability manifest has not passed")
    return summary


def load_statistics(root: Path, fov: int, stability: dict):
    path = root / f"fov_{fov}.npz"
    summary_path = root / f"fov_{fov}.summary.json"
    summary = json.loads(summary_path.read_text())
    expected_hash = stability["image_statistics"][str(fov)]["sha256"]
    if not (
        summary.get("analysis") == "e5_image_sufficient_statistics"
        and summary.get("e5_version") == E5_VERSION
        and summary.get("fov") == fov
        and summary.get("output_sha256") == expected_hash == sha256(path)
        and summary.get("statistics_gate_pass") is True
    ):
        raise RuntimeError(f"invalid E5 image statistics: {path}")
    with np.load(path) as source:
        return path, summary, {name: source[name] for name in source.files}


def heldout_parameters(statistics: dict, slide_id: str, gain_cap: float):
    slide_ids = [str(value) for value in statistics["slide_ids"]]
    matches = [index for index, value in enumerate(slide_ids) if value == slide_id]
    if len(matches) != 1 or len(slide_ids) != 109:
        raise ValueError(f"{slide_id}: absent or duplicated in E5 image statistics")
    heldout = matches[0]
    lab_sum = statistics["lab_sum"].sum(axis=0) - statistics["lab_sum"][heldout]
    lab_square = (
        statistics["lab_square_sum"].sum(axis=0)
        - statistics["lab_square_sum"][heldout]
    )
    lab_count = statistics["lab_count"].sum(axis=0) - statistics["lab_count"][heldout]
    lab_stats = [
        mean_std(lab_sum[index], lab_square[index], int(lab_count[index]))
        for index in range(6)
    ]
    od_xtx = statistics["od_xtx"].sum(axis=0) - statistics["od_xtx"][heldout]
    od_xty = statistics["od_xty"].sum(axis=0) - statistics["od_xty"][heldout]
    od_coefficients = [solve_od_affine(od_xtx[index], od_xty[index]) for index in range(5)]
    radial = statistics["radial_power"].sum(axis=0) - statistics["radial_power"][heldout]
    frequency = statistics["radial_frequency"]
    frequency_gains = [
        fitted_frequency_gain(radial[index], radial[0], frequency, gain_cap)
        for index in range(1, 6)
    ]
    return {
        "lab": lab_stats,
        "od": od_coefficients,
        "frequency": frequency,
        "frequency_gain": frequency_gains,
        "train_slides": 108,
    }


def render(method: str, rgb, scanner_index: int, parameters: dict):
    import torch

    if scanner_index == 0:
        zeros = torch.zeros(len(rgb), dtype=rgb.dtype, device=rgb.device)
        return {"output": rgb, "preclip_range_fraction": zeros, "clip_pixel_mae": zeros}
    if method == "reinhard_lab":
        source_mean, source_std = parameters["lab"][scanner_index]
        target_mean, target_std = parameters["lab"][0]
        tensors = [
            torch.as_tensor(value, dtype=rgb.dtype, device=rgb.device).reshape(1, 1, 1, 3)
            for value in (source_mean, source_std, target_mean, target_std)
        ]
        return reinhard_lab(rgb, *tensors)
    if method == "paired_od_affine":
        coefficient = torch.as_tensor(
            parameters["od"][scanner_index - 1], dtype=rgb.dtype, device=rgb.device
        )
        return paired_od_affine(rgb, coefficient)
    if method == "frequency_calibration":
        return frequency_calibration(
            rgb,
            parameters["frequency_gain"][scanner_index - 1],
            parameters["frequency"],
        )
    raise ValueError(f"unknown E5 image method: {method}")


def infer(encoder, rgb8: np.ndarray, batch_size: int, feature_dim: int):
    import torch

    output = np.empty((len(rgb8), feature_dim), dtype=np.float32)
    for start in range(0, len(rgb8), batch_size):
        stop = min(start + batch_size, len(rgb8))
        batch = transformed_batch(rgb8[start:stop], encoder.eval_transforms).cuda(non_blocking=True)
        batch = batch.half() if encoder.precision == torch.float16 else batch.float()
        with torch.inference_mode():
            values = encoder(batch).float().cpu().numpy()
        if values.shape != (stop - start, feature_dim) or not np.isfinite(values).all():
            raise ValueError(f"invalid E5 image feature batch: {values.shape}")
        output[start:stop] = values
    return output


def existing_output_passes(
    path: Path,
    model: dict,
    source_summary: dict,
    raw_summary: dict,
    statistics_summary: dict,
    stability_sha: str,
):
    summary_path = path.with_suffix(".summary.json")
    if not path.exists() or not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text())
        return bool(
            summary.get("analysis") == "e5_image_feature_shard"
            and summary.get("e5_version") == E5_VERSION
            and summary.get("encoder_id") == model["encoder_id"]
            and summary.get("checkpoint_sha256") == model["checkpoint_sha256"]
            and summary.get("source_grid_sha256") == source_summary["output_sha256"]
            and summary.get("raw_feature_sha256") == raw_summary["output_sha256"]
            and summary.get("image_statistics_sha256") == statistics_summary["output_sha256"]
            and summary.get("stability_manifest_sha256") == stability_sha
            and summary.get("conditions") == list(IMAGE_CONDITIONS)
            and summary.get("output_sha256") == sha256(path)
            and summary.get("shard_gate_pass") is True
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def extract_shard(
    grid_path: Path,
    raw_path: Path,
    output_path: Path,
    model: dict,
    encoder,
    batch_size: int,
    statistics_path: Path,
    statistics_summary: dict,
    statistics: dict,
    stability_path: Path,
    stability: dict,
):
    source_summary = load_grid_contract(grid_path)
    raw_summary = json.loads(raw_path.with_suffix(".summary.json").read_text())
    stability_sha = sha256(stability_path)
    if existing_output_passes(
        output_path,
        model,
        source_summary,
        raw_summary,
        statistics_summary,
        stability_sha,
    ):
        return json.loads(output_path.with_suffix(".summary.json").read_text()), True
    fov = int(model["native_fov_px"])
    feature_dim = int(model["feature_dim"])
    with h5py.File(grid_path, "r") as grid, h5py.File(raw_path, "r") as raw_source:
        slide_id = str(grid.attrs["slide_id"])
        raw = np.asarray(raw_source["features"][:], dtype=np.float32)
        if raw.shape != (6, 100, feature_dim):
            raise ValueError(f"{model['encoder_id']}/{slide_id}: invalid raw feature shape")
        metadata = {
            name: grid[name][:]
            for name in (
                "scanner",
                "location_id",
                "replicate_id",
                "canonical_center_x",
                "canonical_center_y",
            )
        }
        if [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in metadata["scanner"]
        ] != list(SCANNERS):
            raise ValueError(f"{slide_id}: invalid scanner order")
        parameters = heldout_parameters(
            statistics, slide_id, float(stability["selected_frequency_gain_cap"])
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
        feature_min = np.inf
        feature_max = -np.inf
        minimum_norm = np.inf
        clip_max = 0.0
        clip_mae_max = 0.0
        with h5py.File(temporary, "w") as target:
            features = target.create_dataset(
                "features",
                shape=(len(IMAGE_CONDITIONS), 6, 100, feature_dim),
                dtype=np.float32,
                chunks=(1, 1, 100, feature_dim),
                compression="lzf",
                shuffle=True,
            )
            preclip_fraction = target.create_dataset(
                "preclip_range_fraction", shape=(len(IMAGE_CONDITIONS), 6, 100), dtype=np.float32
            )
            clip_mae = target.create_dataset(
                "clip_pixel_mae", shape=(len(IMAGE_CONDITIONS), 6, 100), dtype=np.float32
            )
            for condition_index, condition in enumerate(IMAGE_CONDITIONS):
                features[condition_index, 0] = raw[0]
                preclip_fraction[condition_index, 0] = 0.0
                clip_mae[condition_index, 0] = 0.0
                for scanner_index in range(1, 6):
                    scanner_features = np.empty((100, feature_dim), dtype=np.float32)
                    scanner_clip = np.empty(100, dtype=np.float32)
                    scanner_mae = np.empty(100, dtype=np.float32)
                    for start in range(0, 100, batch_size):
                        stop = min(start + batch_size, 100)
                        rgb8 = centered_crop(grid["rgb"][scanner_index, start:stop], fov)
                        rgb = rgb8_to_rgb01(rgb8, device="cuda")
                        report = render(condition, rgb, scanner_index, parameters)
                        rendered = uint8_from_rgb01(report["output"])
                        scanner_features[start:stop] = infer(
                            encoder, rendered, batch_size, feature_dim
                        )
                        scanner_clip[start:stop] = (
                            report["preclip_range_fraction"].float().cpu().numpy()
                        )
                        scanner_mae[start:stop] = report["clip_pixel_mae"].float().cpu().numpy()
                    features[condition_index, scanner_index] = scanner_features
                    preclip_fraction[condition_index, scanner_index] = scanner_clip
                    clip_mae[condition_index, scanner_index] = scanner_mae
                    norms = np.linalg.norm(scanner_features.astype(np.float64), axis=1)
                    feature_min = min(feature_min, float(scanner_features.min()))
                    feature_max = max(feature_max, float(scanner_features.max()))
                    minimum_norm = min(minimum_norm, float(norms.min()))
                    clip_max = max(clip_max, float(scanner_clip.max()))
                    clip_mae_max = max(clip_mae_max, float(scanner_mae.max()))
                print(f"{model['encoder_id']} {slide_id} {condition}", flush=True)
            for name, values in metadata.items():
                target.create_dataset(name, data=values)
            target.create_dataset("condition", data=string_array(IMAGE_CONDITIONS))
            target.attrs["analysis"] = "e5_image_feature_shard"
            target.attrs["e5_version"] = E5_VERSION
            target.attrs["slide_id"] = slide_id
            target.attrs["encoder_id"] = model["encoder_id"]
            target.attrs["feature_dim"] = feature_dim
            target.attrs["native_fov_px"] = fov
            target.attrs["checkpoint_sha256"] = model["checkpoint_sha256"]
            target.attrs["source_grid_sha256"] = source_summary["output_sha256"]
            target.attrs["raw_feature_sha256"] = raw_summary["output_sha256"]
            target.attrs["image_statistics_sha256"] = statistics_summary["output_sha256"]
            target.attrs["stability_manifest_sha256"] = stability_sha
            target.flush()
    os.replace(temporary, output_path)
    summary = {
        "analysis": "e5_image_feature_shard",
        "e5_version": E5_VERSION,
        "slide_id": slide_id,
        "encoder_id": model["encoder_id"],
        "feature_dim": feature_dim,
        "native_fov_px": fov,
        "conditions": list(IMAGE_CONDITIONS),
        "conditions_count": len(IMAGE_CONDITIONS),
        "features": len(IMAGE_CONDITIONS) * 600,
        "feature_min": feature_min,
        "feature_max": feature_max,
        "minimum_source_norm": minimum_norm,
        "preclip_fraction_max": clip_max,
        "clip_pixel_mae_max": clip_mae_max,
        "checkpoint_sha256": model["checkpoint_sha256"],
        "source_grid_sha256": source_summary["output_sha256"],
        "raw_feature_sha256": raw_summary["output_sha256"],
        "image_statistics": str(statistics_path.resolve()),
        "image_statistics_sha256": statistics_summary["output_sha256"],
        "stability_manifest": str(stability_path.resolve()),
        "stability_manifest_sha256": stability_sha,
        "output": str(output_path.resolve()),
        "output_sha256": sha256(output_path),
        "shard_gate_pass": bool(np.isfinite(minimum_norm) and minimum_norm > 0),
    }
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary, False


def main():
    import torch

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("E5 image feature extraction requires a visible CUDA device")
    grid_audit = json.loads(Path(args.grid_audit).read_text())
    if grid_audit.get("grid_gate_pass") is not True:
        raise RuntimeError("native-AA grid audit has not passed")
    contract = json.loads(Path(args.contract).read_text())
    if contract.get("trident_commit") != TRIDENT_COMMIT:
        raise RuntimeError("PFM contract does not use the frozen TRIDENT commit")
    verify_runtime_contract(contract)
    verify_trident_source(Path(args.trident_root))
    model = select_model(contract, args.encoder_id, args.encoder_index)
    if sha256(Path(model["checkpoint_path"])) != model["checkpoint_sha256"]:
        raise RuntimeError(f"checkpoint hash mismatch: {model['encoder_id']}")
    stability_path = Path(args.stability)
    stability = load_stability(stability_path, Path(args.execution_contract))
    statistics_path, statistics_summary, statistics = load_statistics(
        Path(args.image_statistics), int(model["native_fov_px"]), stability
    )

    sys.path.insert(0, str(Path(args.trident_root).resolve()))
    from trident.patch_encoder_models.load import encoder_factory

    encoder = encoder_factory(
        model["encoder_id"],
        **encoder_kwargs(model["encoder_id"], model["checkpoint_path"]),
    ).eval().cuda()
    encoder = encoder.half() if encoder.precision == torch.float16 else encoder.float()
    batch_size = args.batch_size or BATCH_SIZE[model["encoder_id"]]
    paths = sorted(Path(args.grid).glob("*.h5"))
    if args.slide_id is not None:
        paths = [Path(args.grid) / f"{args.slide_id}.h5"]
    elif args.slide_index is not None:
        paths = [paths[args.slide_index]]
    expected = 1 if args.slide_id is not None or args.slide_index is not None else 109
    if len(paths) != expected:
        raise ValueError(f"expected {expected} source grids, got {len(paths)}")
    summaries = []
    reused = 0
    for index, grid_path in enumerate(paths):
        raw_path = Path(args.raw) / model["encoder_id"] / "shards" / grid_path.name
        output_path = Path(args.output) / model["encoder_id"] / "shards" / grid_path.name
        summary, existing = extract_shard(
            grid_path,
            raw_path,
            output_path,
            model,
            encoder,
            batch_size,
            statistics_path,
            statistics_summary,
            statistics,
            stability_path,
            stability,
        )
        summaries.append(summary)
        reused += int(existing)
        print(
            f"[{index + 1}/{len(paths)}] {model['encoder_id']} {summary['slide_id']} "
            f"{'reused' if existing else 'written'}",
            flush=True,
        )
    run_summary = {
        "analysis": "e5_image_feature_extraction",
        "e5_version": E5_VERSION,
        "encoder_id": model["encoder_id"],
        "slides": len(summaries),
        "features": len(summaries) * len(IMAGE_CONDITIONS) * 600,
        "new_shards": len(summaries) - reused,
        "reused_shards": reused,
        "all_shards_pass": all(summary["shard_gate_pass"] for summary in summaries),
        "stability_manifest_sha256": sha256(stability_path),
        "device": torch.cuda.get_device_name(0),
    }
    run_output = Path(args.output) / model["encoder_id"]
    run_output.mkdir(parents=True, exist_ok=True)
    (run_output / "summary.json").write_text(json.dumps(run_summary, indent=2) + "\n")
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()

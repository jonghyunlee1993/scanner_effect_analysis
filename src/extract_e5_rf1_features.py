"""Extract cross-fitted E5-RF1 image features for the four frozen PFMs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

from e5_comparator_population import (
    SCANNERS,
    centered_crop,
    reinhard_lab,
    rgb8_to_rgb01,
    uint8_from_rgb01,
)
from e5_reinhard_residual_frequency import (
    RF1_CONDITION,
    RF1_VERSION,
    fitted_residual_gains,
    fold_assignments,
    fold_counts,
    shared_od_residual_frequency,
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


METRICS = (
    "base_preclip_range_fraction",
    "base_clip_pixel_mae",
    "preproject_range_fraction",
    "material_range_fraction",
    "projection_fraction",
    "projection_rgb_mae",
    "final_clamp_mae",
)


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
    parser.add_argument("--statistics", default="outputs/e5_rf1_fold_statistics")
    parser.add_argument("--stability", default="outputs/e5_rf1_input_stability/summary.json")
    parser.add_argument(
        "--execution-contract", default="docs/e5_reinhard_residual_frequency_contract.md"
    )
    parser.add_argument("--output", default="outputs/e5_rf1_features")
    parser.add_argument(
        "--trident-root", default="/mnt/isilon/oldridge_lab/irinaz/github/env/trident"
    )
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


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
            raise ValueError(f"invalid E5-RF1 feature batch: {values.shape}")
        output[start:stop] = values
    return output


def load_stability(path: Path, contract_path: Path):
    summary = json.loads(path.read_text())
    if not (
        summary.get("analysis") == "e5_rf1_input_stability"
        and summary.get("rf1_version") == RF1_VERSION
        and summary.get("outcome_access") is False
        and summary.get("selected_gain_cap") is not None
        and summary.get("execution_contract_sha256") == sha256(contract_path)
        and summary.get("stability_gate_pass") is True
    ):
        raise RuntimeError("E5-RF1 input-only stability gate has not passed")
    return summary


def load_fold_statistics(root: Path, fov: int, fold: int, stability: dict):
    path = root / f"fov_{fov}_fold_{fold}.npz"
    summary = json.loads(path.with_suffix(".summary.json").read_text())
    expected = stability["fold_statistics_sha256"][f"fov_{fov}_fold_{fold}"]
    if not (
        summary.get("analysis") == "e5_rf1_fold_statistics"
        and summary.get("rf1_version") == RF1_VERSION
        and summary.get("fov") == fov
        and summary.get("fold") == fold
        and summary.get("output_sha256") == expected == sha256(path)
        and summary.get("statistics_gate_pass") is True
    ):
        raise RuntimeError(f"invalid E5-RF1 fold statistics: {path}")
    with np.load(path) as source:
        values = {name: source[name] for name in source.files}
    return path, summary, values


def existing_output_passes(
    path: Path,
    model: dict,
    source_summary: dict,
    raw_summary: dict,
    fold_statistics_sha: str,
    stability_sha: str,
):
    summary_path = path.with_suffix(".summary.json")
    if not path.exists() or not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text())
        return bool(
            summary.get("analysis") == "e5_rf1_feature_shard"
            and summary.get("rf1_version") == RF1_VERSION
            and summary.get("encoder_id") == model["encoder_id"]
            and summary.get("checkpoint_sha256") == model["checkpoint_sha256"]
            and summary.get("source_grid_sha256") == source_summary["output_sha256"]
            and summary.get("raw_feature_sha256") == raw_summary["output_sha256"]
            and summary.get("fold_statistics_sha256") == fold_statistics_sha
            and summary.get("stability_manifest_sha256") == stability_sha
            and summary.get("condition") == RF1_CONDITION
            and summary.get("output_sha256") == sha256(path)
            and summary.get("shard_gate_pass") is True
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def extract_shard(
    grid_path: Path,
    raw_path: Path,
    output_path: Path,
    model: dict,
    encoder,
    batch_size: int,
    fold: int,
    statistics_path: Path,
    statistics: dict,
    stability_path: Path,
    stability: dict,
):
    import torch

    source_summary = load_grid_contract(grid_path)
    raw_summary = json.loads(raw_path.with_suffix(".summary.json").read_text())
    stability_sha = sha256(stability_path)
    statistics_sha = sha256(statistics_path)
    if existing_output_passes(
        output_path,
        model,
        source_summary,
        raw_summary,
        statistics_sha,
        stability_sha,
    ):
        return json.loads(output_path.with_suffix(".summary.json").read_text()), True

    fov = int(model["native_fov_px"])
    feature_dim = int(model["feature_dim"])
    slide_id = grid_path.stem
    if slide_id not in {str(value) for value in statistics["heldout_slide_ids"]}:
        raise ValueError(f"{slide_id}: not held out by RF1 fold {fold}")
    gain = fitted_residual_gains(
        statistics["post_reinhard_source_power"],
        statistics["raw_at2_target_power"],
        statistics["radial_frequency"],
        float(stability["selected_gain_cap"]),
    )
    lab_mean = torch.as_tensor(statistics["lab_mean"], dtype=torch.float32, device="cuda")
    lab_std = torch.as_tensor(statistics["lab_std"], dtype=torch.float32, device="cuda")

    with h5py.File(grid_path, "r") as grid, h5py.File(raw_path, "r") as raw_source:
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
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
        metric_values = {name: np.zeros((6, 100), dtype=np.float32) for name in METRICS}
        output_features = np.empty((6, 100, feature_dim), dtype=np.float32)
        output_features[0] = raw[0]
        with torch.inference_mode():
            for scanner_index in range(1, len(SCANNERS)):
                for start in range(0, 100, batch_size):
                    stop = min(start + batch_size, 100)
                    rgb = rgb8_to_rgb01(
                        centered_crop(grid["rgb"][scanner_index, start:stop], fov),
                        device="cuda",
                    )
                    base = reinhard_lab(
                        rgb,
                        lab_mean[scanner_index].reshape(1, 1, 1, 3),
                        lab_std[scanner_index].reshape(1, 1, 1, 3),
                        lab_mean[0].reshape(1, 1, 1, 3),
                        lab_std[0].reshape(1, 1, 1, 3),
                    )
                    report = shared_od_residual_frequency(
                        base["output"], gain[scanner_index - 1], statistics["radial_frequency"]
                    )
                    rendered = uint8_from_rgb01(report["output"])
                    output_features[scanner_index, start:stop] = infer(
                        encoder, rendered, batch_size, feature_dim
                    )
                    metric_values["base_preclip_range_fraction"][scanner_index, start:stop] = (
                        base["preclip_range_fraction"].cpu().numpy()
                    )
                    metric_values["base_clip_pixel_mae"][scanner_index, start:stop] = (
                        base["clip_pixel_mae"].cpu().numpy()
                    )
                    for name in METRICS[2:]:
                        metric_values[name][scanner_index, start:stop] = report[name].cpu().numpy()

        norms = np.linalg.norm(output_features.astype(np.float64), axis=-1)
        if not np.isfinite(output_features).all() or np.any(norms <= 0):
            raise RuntimeError(f"{model['encoder_id']}/{slide_id}: invalid RF1 features")
        with h5py.File(temporary, "w") as target:
            target.create_dataset(
                "features",
                data=output_features[None],
                dtype=np.float32,
                chunks=(1, 1, 100, feature_dim),
                compression="lzf",
                shuffle=True,
            )
            for name, value in metric_values.items():
                target.create_dataset(name, data=value[None], dtype=np.float32)
            for name, value in metadata.items():
                target.create_dataset(name, data=value)
            target.create_dataset("condition", data=string_array([RF1_CONDITION]))
            target.attrs["analysis"] = "e5_rf1_feature_shard"
            target.attrs["rf1_version"] = RF1_VERSION
            target.attrs["slide_id"] = slide_id
            target.attrs["encoder_id"] = model["encoder_id"]
            target.attrs["feature_dim"] = feature_dim
            target.attrs["native_fov_px"] = fov
            target.attrs["fold"] = fold
            target.attrs["selected_gain_cap"] = float(stability["selected_gain_cap"])
            target.attrs["checkpoint_sha256"] = model["checkpoint_sha256"]
            target.attrs["source_grid_sha256"] = source_summary["output_sha256"]
            target.attrs["raw_feature_sha256"] = raw_summary["output_sha256"]
            target.attrs["fold_statistics_sha256"] = statistics_sha
            target.attrs["stability_manifest_sha256"] = stability_sha
            target.flush()
    os.replace(temporary, output_path)
    summary = {
        "analysis": "e5_rf1_feature_shard",
        "rf1_version": RF1_VERSION,
        "condition": RF1_CONDITION,
        "slide_id": slide_id,
        "encoder_id": model["encoder_id"],
        "feature_dim": feature_dim,
        "native_fov_px": fov,
        "fold": fold,
        "features": 600,
        "minimum_norm": float(norms.min()),
        "maximum_norm": float(norms.max()),
        "selected_gain_cap": float(stability["selected_gain_cap"]),
        "checkpoint_sha256": model["checkpoint_sha256"],
        "source_grid_sha256": source_summary["output_sha256"],
        "raw_feature_sha256": raw_summary["output_sha256"],
        "fold_statistics_sha256": statistics_sha,
        "stability_manifest_sha256": stability_sha,
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
        raise RuntimeError("E5-RF1 feature extraction requires a visible CUDA device")
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
    execution_path = Path(args.execution_contract)
    stability_path = Path(args.stability)
    stability = load_stability(stability_path, execution_path)

    sys.path.insert(0, str(Path(args.trident_root).resolve()))
    from trident.patch_encoder_models.load import encoder_factory

    encoder = encoder_factory(
        model["encoder_id"], **encoder_kwargs(model["encoder_id"], model["checkpoint_path"])
    ).eval().cuda()
    encoder = encoder.half() if encoder.precision == torch.float16 else encoder.float()
    batch_size = args.batch_size or BATCH_SIZE[model["encoder_id"]]

    all_paths = sorted(Path(args.grid).glob("*.h5"))
    assignments = fold_assignments([path.stem for path in all_paths])
    if fold_counts(assignments) != (22, 22, 22, 22, 21):
        raise RuntimeError("RF1 fold assignment is invalid")
    paths = all_paths
    if args.slide_id is not None:
        paths = [Path(args.grid) / f"{args.slide_id}.h5"]
    elif args.slide_index is not None:
        paths = [all_paths[args.slide_index]]
    fov = int(model["native_fov_px"])
    statistics_cache = {
        fold: load_fold_statistics(Path(args.statistics), fov, fold, stability)
        for fold in range(5)
    }
    summaries = []
    reused = 0
    for index, grid_path in enumerate(paths):
        fold = assignments[grid_path.stem]
        statistics_path, _, statistics = statistics_cache[fold]
        raw_path = Path(args.raw) / model["encoder_id"] / "shards" / grid_path.name
        output_path = Path(args.output) / model["encoder_id"] / "shards" / grid_path.name
        summary, existing = extract_shard(
            grid_path,
            raw_path,
            output_path,
            model,
            encoder,
            batch_size,
            fold,
            statistics_path,
            statistics,
            stability_path,
            stability,
        )
        summaries.append(summary)
        reused += int(existing)
        print(
            f"[{index + 1}/{len(paths)}] {model['encoder_id']} {grid_path.stem} "
            f"{'reused' if existing else 'written'}",
            flush=True,
        )
    run_summary = {
        "analysis": "e5_rf1_feature_extraction",
        "rf1_version": RF1_VERSION,
        "encoder_id": model["encoder_id"],
        "slides": len(summaries),
        "features": len(summaries) * 600,
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


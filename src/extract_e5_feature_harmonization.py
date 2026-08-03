"""Materialize exact LOSO CORAL and orthogonal-Procrustes E5 embeddings."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

from e5_comparator_population import (
    E5_VERSION,
    FEATURE_CONDITIONS,
    SCANNERS,
    centered_covariance,
    feature_sufficient_statistics,
    procrustes_transform,
    regularized_covariance,
    symmetric_matrix_power,
)
from extract_e0_pfm_features import select_model, string_array
from fetch_e0_pfm_checkpoints import sha256


def parse_args():
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--encoder-id")
    selection.add_argument("--encoder-index", type=int)
    selection.add_argument("--task-index", type=int)
    parser.add_argument("--slide-id")
    parser.add_argument("--slide-index", type=int)
    parser.add_argument("--contract", default="outputs/e0_pfm_contract/checkpoint_manifest.json")
    parser.add_argument("--raw-audit", default="outputs/e0_pfm_features/audit/summary.json")
    parser.add_argument("--raw", default="outputs/e0_pfm_features")
    parser.add_argument("--statistics", default="outputs/e5_feature_statistics")
    parser.add_argument("--stability", default="outputs/e5_input_stability/summary.json")
    parser.add_argument("--execution-contract", default="docs/e5_comparator_execution_contract.md")
    parser.add_argument("--output", default="outputs/e5_feature_harmonization")
    return parser.parse_args()


def resolve_selection(args, contract: dict, raw_root: Path):
    if args.task_index is not None:
        if args.encoder_id is not None or args.encoder_index is not None:
            raise ValueError("--task-index cannot be combined with encoder selection")
        if args.task_index < 0 or args.task_index >= 436:
            raise IndexError(f"task index {args.task_index} outside 0..435")
        encoder_index, slide_index = divmod(args.task_index, 109)
        model = select_model(contract, None, encoder_index)
        paths = sorted((raw_root / model["encoder_id"] / "shards").glob("*.h5"))
        return model, paths[slide_index]
    model = select_model(contract, args.encoder_id, args.encoder_index)
    paths = sorted((raw_root / model["encoder_id"] / "shards").glob("*.h5"))
    if len(paths) != 109:
        raise ValueError(f"{model['encoder_id']}: expected 109 raw shards, got {len(paths)}")
    if args.slide_id is not None:
        return model, raw_root / model["encoder_id"] / "shards" / f"{args.slide_id}.h5"
    if args.slide_index is not None:
        return model, paths[args.slide_index]
    raise ValueError("feature harmonization requires --task-index, --slide-id or --slide-index")


def load_stability(path: Path, execution_path: Path):
    summary = json.loads(path.read_text())
    if not (
        summary.get("analysis") == "e5_input_only_stability"
        and summary.get("e5_version") == E5_VERSION
        and summary.get("outcome_access") is False
        and summary.get("stability_gate_pass") is True
        and summary.get("execution_contract_sha256") == sha256(execution_path)
        and summary.get("selected_coral_shrinkage") in (
            0.0001,
            0.001,
            0.01,
            0.05,
            0.1,
            0.2,
        )
    ):
        raise RuntimeError("E5 input-only stability manifest has not passed")
    return summary


def load_statistics(path: Path, summary_path: Path, model: dict, stability: dict):
    summary = json.loads(summary_path.read_text())
    expected = stability["feature_statistics"][model["encoder_id"]]["sha256"]
    if not (
        summary.get("analysis") == "e5_feature_sufficient_statistics"
        and summary.get("e5_version") == E5_VERSION
        and summary.get("encoder_id") == model["encoder_id"]
        and summary.get("checkpoint_sha256") == model["checkpoint_sha256"]
        and summary.get("output_sha256") == expected == sha256(path)
        and summary.get("statistics_gate_pass") is True
    ):
        raise RuntimeError(f"invalid E5 feature sufficient statistics: {path}")
    with h5py.File(path, "r") as source:
        values = {
            "sum": source["sum"][:],
            "gram": source["gram"][:],
            "cross": source["cross_to_at2"][:],
            "slide_ids": [
                item.decode() if isinstance(item, bytes) else str(item)
                for item in source["slide_id"][:]
            ],
            "count": int(source.attrs["sample_count_per_scanner"]),
        }
    return summary, values


def existing_output_passes(
    path: Path,
    model: dict,
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
            summary.get("analysis") == "e5_feature_harmonization_shard"
            and summary.get("e5_version") == E5_VERSION
            and summary.get("encoder_id") == model["encoder_id"]
            and summary.get("checkpoint_sha256") == model["checkpoint_sha256"]
            and summary.get("raw_feature_sha256") == raw_summary["output_sha256"]
            and summary.get("feature_statistics_sha256") == statistics_summary["output_sha256"]
            and summary.get("stability_manifest_sha256") == stability_sha
            and summary.get("conditions") == list(FEATURE_CONDITIONS)
            and summary.get("output_sha256") == sha256(path)
            and summary.get("shard_gate_pass") is True
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def transform_fold(raw: np.ndarray, statistics: dict, slide_id: str, shrinkage: float):
    import torch

    slide_ids = statistics["slide_ids"]
    if len(slide_ids) != 109 or slide_ids.count(slide_id) != 1:
        raise ValueError(f"{slide_id}: absent or duplicated in feature statistics")
    heldout = feature_sufficient_statistics(torch.from_numpy(raw).cuda())
    total_sum = torch.from_numpy(statistics["sum"]).cuda()
    total_gram = torch.from_numpy(statistics["gram"]).cuda()
    total_cross = torch.from_numpy(statistics["cross"]).cuda()
    train_sum = total_sum - heldout[0]
    train_gram = total_gram - heldout[1]
    train_cross = total_cross - heldout[2]
    train_count = int(statistics["count"] - heldout[3])
    if train_count != 10_800:
        raise ValueError(f"{slide_id}: LOSO train count is {train_count}, expected 10800")
    output = torch.empty(
        (len(FEATURE_CONDITIONS), 6, 100, raw.shape[-1]),
        dtype=torch.float32,
        device="cpu",
    )
    output[:, 0] = torch.from_numpy(raw[0]).float()

    target_mean = train_sum[0] / float(train_count)
    target_covariance = regularized_covariance(
        centered_covariance(train_sum[0], train_gram[0], train_count), shrinkage
    )
    target_root = symmetric_matrix_power(target_covariance, 0.5)
    for scanner_index in range(1, 6):
        source_mean = train_sum[scanner_index] / float(train_count)
        source_covariance = regularized_covariance(
            centered_covariance(
                train_sum[scanner_index], train_gram[scanner_index], train_count
            ),
            shrinkage,
        )
        source_inverse = symmetric_matrix_power(source_covariance, -0.5)
        heldout_source = torch.from_numpy(raw[scanner_index]).cuda().double()
        coral = (heldout_source - source_mean) @ source_inverse @ target_root + target_mean
        procrustes = procrustes_transform(
            heldout_source,
            train_sum[scanner_index],
            train_sum[0],
            train_cross[scanner_index - 1],
            train_count,
        )
        if not torch.isfinite(coral).all() or not torch.isfinite(procrustes).all():
            raise ValueError(f"{slide_id}/{SCANNERS[scanner_index]}: non-finite transform")
        output[0, scanner_index] = coral.float().cpu()
        output[1, scanner_index] = procrustes.float().cpu()
        print(f"{slide_id} {SCANNERS[scanner_index]} CORAL+Procrustes", flush=True)
    return output.numpy(), train_count


def main():
    import torch

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("E5 feature harmonization requires a visible CUDA device")
    raw_audit = json.loads(Path(args.raw_audit).read_text())
    if raw_audit.get("population_gate_pass") is not True or raw_audit.get("features_observed") != 261_600:
        raise RuntimeError("raw four-PFM population audit has not passed")
    contract = json.loads(Path(args.contract).read_text())
    model, raw_path = resolve_selection(args, contract, Path(args.raw))
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    stability_path = Path(args.stability)
    stability = load_stability(stability_path, Path(args.execution_contract))
    statistics_path = Path(args.statistics) / f"{model['encoder_id']}.h5"
    statistics_summary, statistics = load_statistics(
        statistics_path,
        Path(args.statistics) / f"{model['encoder_id']}.summary.json",
        model,
        stability,
    )
    raw_summary = json.loads(raw_path.with_suffix(".summary.json").read_text())
    output_path = Path(args.output) / model["encoder_id"] / "shards" / raw_path.name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stability_sha = sha256(stability_path)
    if existing_output_passes(
        output_path, model, raw_summary, statistics_summary, stability_sha
    ):
        print(output_path.with_suffix(".summary.json").read_text(), end="")
        return

    with h5py.File(raw_path, "r") as source:
        slide_id = str(source.attrs["slide_id"])
        raw = np.asarray(source["features"][:], dtype=np.float32)
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
    expected_shape = (6, 100, int(model["feature_dim"]))
    if raw.shape != expected_shape or slide_id != raw_path.stem:
        raise ValueError(f"{model['encoder_id']}/{raw_path.stem}: raw identity mismatch")
    values, train_count = transform_fold(
        raw, statistics, slide_id, float(stability["selected_coral_shrinkage"])
    )
    norms = np.linalg.norm(values.astype(np.float64), axis=-1)
    if values.shape != (2, *expected_shape) or not np.isfinite(values).all() or np.any(norms <= 0):
        raise ValueError(f"{model['encoder_id']}/{slide_id}: invalid transformed population")
    if not np.array_equal(values[:, 0], np.broadcast_to(raw[0], values[:, 0].shape)):
        raise ValueError(f"{model['encoder_id']}/{slide_id}: AT2 was altered")

    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    with h5py.File(temporary, "w") as target:
        target.create_dataset(
            "features", data=values, chunks=(1, 1, 100, int(model["feature_dim"])), compression="lzf", shuffle=True
        )
        for name, local in metadata.items():
            target.create_dataset(name, data=local)
        target.create_dataset("condition", data=string_array(FEATURE_CONDITIONS))
        target.attrs["analysis"] = "e5_feature_harmonization_shard"
        target.attrs["e5_version"] = E5_VERSION
        target.attrs["slide_id"] = slide_id
        target.attrs["encoder_id"] = model["encoder_id"]
        target.attrs["feature_dim"] = int(model["feature_dim"])
        target.attrs["checkpoint_sha256"] = model["checkpoint_sha256"]
        target.attrs["raw_feature_sha256"] = raw_summary["output_sha256"]
        target.attrs["feature_statistics_sha256"] = statistics_summary["output_sha256"]
        target.attrs["stability_manifest_sha256"] = stability_sha
        target.attrs["loso_train_count"] = train_count
        target.flush()
    os.replace(temporary, output_path)
    summary = {
        "analysis": "e5_feature_harmonization_shard",
        "e5_version": E5_VERSION,
        "slide_id": slide_id,
        "encoder_id": model["encoder_id"],
        "feature_dim": int(model["feature_dim"]),
        "conditions": list(FEATURE_CONDITIONS),
        "features": len(FEATURE_CONDITIONS) * 600,
        "loso_train_slides": 108,
        "loso_train_count_per_scanner": train_count,
        "minimum_norm": float(norms.min()),
        "maximum_norm": float(norms.max()),
        "checkpoint_sha256": model["checkpoint_sha256"],
        "raw_feature_sha256": raw_summary["output_sha256"],
        "feature_statistics": str(statistics_path.resolve()),
        "feature_statistics_sha256": statistics_summary["output_sha256"],
        "stability_manifest": str(stability_path.resolve()),
        "stability_manifest_sha256": stability_sha,
        "output": str(output_path.resolve()),
        "output_sha256": sha256(output_path),
        "shard_gate_pass": True,
        "device": torch.cuda.get_device_name(0),
    }
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Materialize exact 37-fold LOTO CORAL and Procrustes embeddings."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

from e5_comparator_population import (
    FEATURE_CONDITIONS,
    SCANNERS,
    centered_covariance,
    feature_sufficient_statistics,
    procrustes_transform,
    regularized_covariance,
    symmetric_matrix_power,
)
from e6_loto_population import E6_LOTO_VERSION, load_tissue_annotation
from extract_e0_pfm_features import select_model, string_array
from extract_e5_feature_harmonization import load_stability, load_statistics
from fetch_e0_pfm_checkpoints import sha256


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--contract", default="outputs/e0_pfm_contract/checkpoint_manifest.json")
    parser.add_argument("--raw-audit", default="outputs/e0_pfm_features/audit/summary.json")
    parser.add_argument("--raw", default="outputs/e0_pfm_features")
    parser.add_argument("--statistics", default="outputs/e5_feature_statistics")
    parser.add_argument("--stability", default="outputs/e5_input_stability/summary.json")
    parser.add_argument("--execution-contract", default="docs/e5_comparator_execution_contract.md")
    parser.add_argument("--e6-contract", default="docs/e6_heterogeneity_execution_contract.md")
    parser.add_argument("--geometry", default="outputs/e0_native_geometry_final/native_geometry_manifest.csv")
    parser.add_argument("--output", default="outputs/e6_loto_feature_harmonization")
    return parser.parse_args()


def tissue_fold_statistics(raw_paths, statistics, feature_dim: int):
    import torch

    heldout_sum = torch.zeros((6, feature_dim), dtype=torch.float64, device="cuda")
    heldout_gram = torch.zeros((6, feature_dim, feature_dim), dtype=torch.float64, device="cuda")
    heldout_cross = torch.zeros((5, feature_dim, feature_dim), dtype=torch.float64, device="cuda")
    heldout_count = 0
    raw_values = {}
    for raw_path in raw_paths:
        with h5py.File(raw_path, "r") as source:
            values = np.asarray(source["features"][:], dtype=np.float32)
        if values.shape != (6, 100, feature_dim):
            raise ValueError(f"{raw_path}: raw feature shape mismatch")
        local = feature_sufficient_statistics(torch.from_numpy(values).cuda())
        heldout_sum += local[0]
        heldout_gram += local[1]
        heldout_cross += local[2]
        heldout_count += local[3]
        raw_values[raw_path.stem] = values
    total_sum = torch.from_numpy(statistics["sum"]).cuda()
    total_gram = torch.from_numpy(statistics["gram"]).cuda()
    total_cross = torch.from_numpy(statistics["cross"]).cuda()
    train = {
        "sum": total_sum - heldout_sum,
        "gram": total_gram - heldout_gram,
        "cross": total_cross - heldout_cross,
        "count": int(statistics["count"] - heldout_count),
    }
    expected_count = 100 * (109 - len(raw_paths))
    if train["count"] != expected_count:
        raise ValueError(
            f"LOTO train count {train['count']} differs from {expected_count}"
        )
    return train, raw_values


def transform_tissue_fold(raw: np.ndarray, train: dict, shrinkage: float):
    import torch

    train_count = train["count"]
    target_mean = train["sum"][0] / float(train_count)
    target_covariance = regularized_covariance(
        centered_covariance(train["sum"][0], train["gram"][0], train_count),
        shrinkage,
    )
    target_root = symmetric_matrix_power(target_covariance, 0.5)
    output = torch.empty(
        (len(FEATURE_CONDITIONS), *raw.shape), dtype=torch.float32, device="cpu"
    )
    output[:, 0] = torch.from_numpy(raw[0]).float()
    for scanner_index in range(1, 6):
        source_mean = train["sum"][scanner_index] / float(train_count)
        source_covariance = regularized_covariance(
            centered_covariance(
                train["sum"][scanner_index],
                train["gram"][scanner_index],
                train_count,
            ),
            shrinkage,
        )
        source_inverse = symmetric_matrix_power(source_covariance, -0.5)
        heldout = torch.from_numpy(raw[scanner_index]).cuda().double()
        coral = (heldout - source_mean) @ source_inverse @ target_root + target_mean
        procrustes = procrustes_transform(
            heldout,
            train["sum"][scanner_index],
            train["sum"][0],
            train["cross"][scanner_index - 1],
            train_count,
        )
        if not torch.isfinite(coral).all() or not torch.isfinite(procrustes).all():
            raise ValueError(f"{SCANNERS[scanner_index]}: non-finite LOTO transform")
        output[0, scanner_index] = coral.float().cpu()
        output[1, scanner_index] = procrustes.float().cpu()
    return output.numpy()


def existing_output_passes(
    path: Path,
    model: dict,
    raw_summary: dict,
    statistics_summary: dict,
    stability_sha: str,
    e6_contract_sha: str,
    tissue: str,
    heldout_slides: int,
):
    summary_path = path.with_suffix(".summary.json")
    if not path.exists() or not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text())
        return bool(
            summary.get("analysis") == "e6_loto_feature_harmonization_shard"
            and summary.get("e6_loto_version") == E6_LOTO_VERSION
            and summary.get("encoder_id") == model["encoder_id"]
            and summary.get("checkpoint_sha256") == model["checkpoint_sha256"]
            and summary.get("raw_feature_sha256") == raw_summary["output_sha256"]
            and summary.get("feature_statistics_sha256") == statistics_summary["output_sha256"]
            and summary.get("stability_manifest_sha256") == stability_sha
            and summary.get("e6_contract_sha256") == e6_contract_sha
            and summary.get("heldout_tissue") == tissue
            and summary.get("heldout_tissue_slides") == heldout_slides
            and summary.get("conditions") == list(FEATURE_CONDITIONS)
            and summary.get("output_sha256") == sha256(path)
            and summary.get("shard_gate_pass") is True
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def main():
    import torch

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("E6 LOTO feature harmonization requires a visible CUDA device")
    if args.task_index < 0 or args.task_index >= 148:
        raise IndexError("LOTO feature task index must be in 0..147")
    raw_audit = json.loads(Path(args.raw_audit).read_text())
    if raw_audit.get("population_gate_pass") is not True or raw_audit.get("features_observed") != 261_600:
        raise RuntimeError("raw four-PFM population audit has not passed")
    e6_contract_path = Path(args.e6_contract)
    if "**Status:** FROZEN" not in e6_contract_path.read_text():
        raise RuntimeError("E6 execution contract is not frozen")
    e6_contract_sha = sha256(e6_contract_path)
    tissue_by_slide, tissues = load_tissue_annotation(Path(args.geometry))
    model_index, tissue_index = divmod(args.task_index, len(tissues))
    contract = json.loads(Path(args.contract).read_text())
    model = select_model(contract, None, model_index)
    tissue = tissues[tissue_index]
    heldout_ids = sorted(
        slide_id for slide_id, value in tissue_by_slide.items() if value == tissue
    )
    raw_paths = [Path(args.raw) / model["encoder_id"] / "shards" / f"{slide_id}.h5" for slide_id in heldout_ids]
    if not all(path.exists() for path in raw_paths):
        raise FileNotFoundError(f"{model['encoder_id']}/{tissue}: missing raw shard")
    stability_path = Path(args.stability)
    stability = load_stability(stability_path, Path(args.execution_contract))
    statistics_path = Path(args.statistics) / f"{model['encoder_id']}.h5"
    statistics_summary, statistics = load_statistics(
        statistics_path,
        Path(args.statistics) / f"{model['encoder_id']}.summary.json",
        model,
        stability,
    )
    if set(statistics["slide_ids"]) != set(tissue_by_slide):
        raise ValueError("feature-statistic and tissue-annotation slide sets differ")
    train, raw_values = tissue_fold_statistics(
        raw_paths, statistics, int(model["feature_dim"])
    )
    stability_sha = sha256(stability_path)
    written = 0
    reused = 0
    for raw_path in raw_paths:
        slide_id = raw_path.stem
        raw_summary = json.loads(raw_path.with_suffix(".summary.json").read_text())
        output_path = Path(args.output) / model["encoder_id"] / "shards" / raw_path.name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if existing_output_passes(
            output_path,
            model,
            raw_summary,
            statistics_summary,
            stability_sha,
            e6_contract_sha,
            tissue,
            len(raw_paths),
        ):
            reused += 1
            continue
        raw = raw_values[slide_id]
        values = transform_tissue_fold(
            raw, train, float(stability["selected_coral_shrinkage"])
        )
        norms = np.linalg.norm(values.astype(np.float64), axis=-1)
        if (
            values.shape != (2, 6, 100, int(model["feature_dim"]))
            or not np.isfinite(values).all()
            or np.any(norms <= 0)
            or not np.array_equal(values[:, 0], np.broadcast_to(raw[0], values[:, 0].shape))
        ):
            raise ValueError(f"{model['encoder_id']}/{slide_id}: invalid LOTO feature population")
        with h5py.File(raw_path, "r") as source:
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
        temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
        with h5py.File(temporary, "w") as target:
            target.create_dataset(
                "features",
                data=values,
                chunks=(1, 1, 100, int(model["feature_dim"])),
                compression="lzf",
                shuffle=True,
            )
            for name, local in metadata.items():
                target.create_dataset(name, data=local)
            target.create_dataset("condition", data=string_array(FEATURE_CONDITIONS))
            target.attrs["analysis"] = "e6_loto_feature_harmonization_shard"
            target.attrs["e6_loto_version"] = E6_LOTO_VERSION
            target.attrs["slide_id"] = slide_id
            target.attrs["encoder_id"] = model["encoder_id"]
            target.attrs["feature_dim"] = int(model["feature_dim"])
            target.attrs["checkpoint_sha256"] = model["checkpoint_sha256"]
            target.attrs["raw_feature_sha256"] = raw_summary["output_sha256"]
            target.attrs["feature_statistics_sha256"] = statistics_summary["output_sha256"]
            target.attrs["stability_manifest_sha256"] = stability_sha
            target.attrs["e6_contract_sha256"] = e6_contract_sha
            target.attrs["heldout_tissue"] = tissue
            target.attrs["heldout_tissue_slides"] = len(raw_paths)
            target.attrs["loto_train_count"] = train["count"]
            target.flush()
        os.replace(temporary, output_path)
        summary = {
            "analysis": "e6_loto_feature_harmonization_shard",
            "e6_loto_version": E6_LOTO_VERSION,
            "slide_id": slide_id,
            "encoder_id": model["encoder_id"],
            "feature_dim": int(model["feature_dim"]),
            "conditions": list(FEATURE_CONDITIONS),
            "features": len(FEATURE_CONDITIONS) * 600,
            "heldout_tissue": tissue,
            "heldout_tissue_slides": len(raw_paths),
            "loto_train_slides": 109 - len(raw_paths),
            "loto_train_count_per_scanner": train["count"],
            "minimum_norm": float(norms.min()),
            "maximum_norm": float(norms.max()),
            "checkpoint_sha256": model["checkpoint_sha256"],
            "raw_feature_sha256": raw_summary["output_sha256"],
            "feature_statistics_sha256": statistics_summary["output_sha256"],
            "stability_manifest_sha256": stability_sha,
            "e6_contract_sha256": e6_contract_sha,
            "output": str(output_path.resolve()),
            "output_sha256": sha256(output_path),
            "shard_gate_pass": True,
        }
        output_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        written += 1
        print(f"{model['encoder_id']} {tissue} {slide_id} written", flush=True)
    print(json.dumps({
        "analysis": "e6_loto_feature_harmonization_tissue_fold",
        "e6_loto_version": E6_LOTO_VERSION,
        "encoder_id": model["encoder_id"],
        "heldout_tissue": tissue,
        "heldout_slides": len(raw_paths),
        "train_slides": 109 - len(raw_paths),
        "train_count_per_scanner": train["count"],
        "written_shards": written,
        "reused_shards": reused,
        "fold_gate_pass": written + reused == len(raw_paths),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Materialize slide/scanner rows for the frozen E7 grouped tissue probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from e5_comparator_population import FEATURE_CONDITIONS, IMAGE_CONDITIONS, SCANNERS
from e6_loto_population import load_tissue_annotation
from extract_e0_pfm_features import select_model
from fetch_e0_pfm_checkpoints import sha256


CONDITIONS = ("raw", *IMAGE_CONDITIONS, *FEATURE_CONDITIONS)
POPULATIONS = {
    "loso": ("outputs/e5_image_features", "outputs/e5_feature_harmonization"),
    "loto": (
        "outputs/e6_loto_image_features",
        "outputs/e6_loto_feature_harmonization",
    ),
}


def parse_args():
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--encoder-id")
    selection.add_argument("--encoder-index", type=int)
    parser.add_argument("--contract", default="outputs/e0_pfm_contract/checkpoint_manifest.json")
    parser.add_argument("--raw", default="outputs/e0_pfm_features")
    parser.add_argument("--loso-image", default=POPULATIONS["loso"][0])
    parser.add_argument("--loso-feature", default=POPULATIONS["loso"][1])
    parser.add_argument("--loto-image", default=POPULATIONS["loto"][0])
    parser.add_argument("--loto-feature", default=POPULATIONS["loto"][1])
    parser.add_argument("--e5-audit", default="outputs/e5_comparator_features/audit/summary.json")
    parser.add_argument("--loto-audit", default="outputs/e6_loto_features/audit/summary.json")
    parser.add_argument("--geometry", default="outputs/e0_native_geometry_final/native_geometry_manifest.csv")
    parser.add_argument("--e7-contract", default="docs/e7_tissue_probe_execution_contract.md")
    parser.add_argument("--output", default="outputs/e7_tissue_probe_rows")
    return parser.parse_args()


def l2_normalize(values: np.ndarray):
    norms = np.linalg.norm(values.astype(np.float64), axis=-1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError("invalid raw AT2 norm in E7 reference bank")
    return (values / norms).astype(np.float32)


def load_raw_at2(paths, feature_dim: int):
    values = {}
    for path in paths:
        with h5py.File(path, "r") as source:
            local = np.asarray(source["features"][0], dtype=np.float32)
        if local.shape != (100, feature_dim):
            raise ValueError(f"{path}: invalid raw AT2 shape")
        values[path.stem] = l2_normalize(local)
    return values


def tissue_reference_sums(raw_at2: dict, tissue_by_slide: dict, eligible_tissues):
    sums = {}
    for tissue in eligible_tissues:
        local = [
            raw_at2[slide_id]
            for slide_id, value in tissue_by_slide.items()
            if value == tissue
        ]
        sums[tissue] = np.stack(local).sum(axis=(0, 1), dtype=np.float64)
    return sums


def heldout_centroids(
    slide_id: str,
    raw_at2: dict,
    tissue_by_slide: dict,
    eligible_tissues,
    reference_sums: dict,
):
    tissue = tissue_by_slide[slide_id]
    centroids = []
    for local_tissue in eligible_tissues:
        total = reference_sums[local_tissue].copy()
        if local_tissue == tissue:
            total -= raw_at2[slide_id].sum(axis=0, dtype=np.float64)
        norm = np.linalg.norm(total)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError(f"{slide_id}/{local_tissue}: invalid held-out centroid")
        centroids.append((total / norm).astype(np.float32))
    return np.stack(centroids), eligible_tissues.index(tissue)


def load_queries(raw_path: Path, image_path: Path, feature_path: Path, feature_dim: int):
    with h5py.File(raw_path, "r") as source:
        raw = np.asarray(source["features"][:], dtype=np.float32)
        raw_locations = source["location_id"][:]
    with h5py.File(image_path, "r") as source:
        image = np.asarray(source["features"][:], dtype=np.float32)
        image_locations = source["location_id"][:]
        image_conditions = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in source["condition"][:]
        ]
    with h5py.File(feature_path, "r") as source:
        feature = np.asarray(source["features"][:], dtype=np.float32)
        feature_locations = source["location_id"][:]
        feature_conditions = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in source["condition"][:]
        ]
    if (
        raw.shape != (6, 100, feature_dim)
        or image.shape != (3, 6, 100, feature_dim)
        or feature.shape != (2, 6, 100, feature_dim)
        or image_conditions != list(IMAGE_CONDITIONS)
        or feature_conditions != list(FEATURE_CONDITIONS)
        or not np.array_equal(raw_locations, image_locations)
        or not np.array_equal(raw_locations, feature_locations)
    ):
        raise ValueError(f"{raw_path}: E7 query identity/schema mismatch")
    return np.concatenate((raw[None], image, feature), axis=0)


def summarize_queries(
    queries: np.ndarray,
    centroids: np.ndarray,
    target_index: int,
    raw_at2: np.ndarray,
):
    import torch
    import torch.nn.functional as functional

    centroid_tensor = torch.from_numpy(centroids).cuda()
    query = torch.from_numpy(queries[:, 1:]).cuda()
    query = functional.normalize(query.float(), dim=-1)
    scores = torch.einsum("cslf,tf->cslt", query, centroid_tensor)
    reference = torch.from_numpy(raw_at2).cuda() @ centroid_tensor.T
    top1 = scores.argmax(dim=-1).eq(target_index).float()
    top5 = scores.topk(k=5, dim=-1).indices.eq(target_index).any(dim=-1).float()
    correct = scores[..., target_index]
    competitors = scores.clone()
    competitors[..., target_index] = -torch.inf
    margin = correct - competitors.max(dim=-1).values
    score_centered = scores - scores.mean(dim=-1, keepdim=True)
    reference_centered = reference - reference.mean(dim=-1, keepdim=True)
    agreement = (
        (score_centered * reference_centered[None, None]).sum(dim=-1)
        / (
            torch.linalg.vector_norm(score_centered, dim=-1)
            * torch.linalg.vector_norm(reference_centered, dim=-1)[None, None]
        ).clamp_min(1e-12)
    )
    return {
        "top1_accuracy": top1.mean(dim=-1).cpu().numpy(),
        "top5_accuracy": top5.mean(dim=-1).cpu().numpy(),
        "correct_tissue_margin": margin.mean(dim=-1).cpu().numpy(),
        "centroid_profile_agreement": agreement.mean(dim=-1).cpu().numpy(),
    }


def summarize_at2(raw_at2: np.ndarray, centroids: np.ndarray, target_index: int):
    scores = raw_at2 @ centroids.T
    top1 = (scores.argmax(axis=-1) == target_index).mean()
    top5 = np.any(
        np.argsort(scores, axis=-1)[:, -5:] == target_index,
        axis=-1,
    ).mean()
    competitors = scores.copy()
    competitors[:, target_index] = -np.inf
    margin = (scores[:, target_index] - competitors.max(axis=-1)).mean()
    return {
        "top1_accuracy": float(top1),
        "top5_accuracy": float(top5),
        "correct_tissue_margin": float(margin),
        "centroid_profile_agreement": 1.0,
    }


def main():
    import torch

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("E7 tissue probe row construction requires a visible CUDA device")
    e5_audit_path = Path(args.e5_audit)
    loto_audit_path = Path(args.loto_audit)
    e5_audit = json.loads(e5_audit_path.read_text())
    loto_audit = json.loads(loto_audit_path.read_text())
    if e5_audit.get("audit_pass") is not True or e5_audit.get("total_features_observed") != 1_308_000:
        raise RuntimeError("E5 LOSO comparator population has not passed")
    if loto_audit.get("audit_pass") is not True or loto_audit.get("total_features_observed") != 1_308_000:
        raise RuntimeError("E6 LOTO comparator population has not passed")
    e7_contract_path = Path(args.e7_contract)
    if "**Status:** FROZEN" not in e7_contract_path.read_text():
        raise RuntimeError("E7 tissue probe contract is not frozen")
    contract = json.loads(Path(args.contract).read_text())
    model = select_model(contract, args.encoder_id, args.encoder_index)
    model_id = model["encoder_id"]
    feature_dim = int(model["feature_dim"])
    tissue_by_slide, tissues = load_tissue_annotation(Path(args.geometry))
    tissue_sizes = {
        tissue: sum(value == tissue for value in tissue_by_slide.values())
        for tissue in tissues
    }
    eligible_tissues = sorted(
        tissue for tissue, size in tissue_sizes.items() if size >= 2
    )
    singleton_tissues = sorted(
        tissue for tissue, size in tissue_sizes.items() if size == 1
    )
    if len(eligible_tissues) != 36 or len(singleton_tissues) != 1:
        raise ValueError("E7 evaluable tissue population mismatch")
    raw_paths = sorted((Path(args.raw) / model_id / "shards").glob("*.h5"))
    if len(raw_paths) != 109 or {path.stem for path in raw_paths} != set(tissue_by_slide):
        raise ValueError(f"{model_id}: raw slide population mismatch")
    raw_at2 = load_raw_at2(raw_paths, feature_dim)
    reference_sums = tissue_reference_sums(
        raw_at2, tissue_by_slide, eligible_tissues
    )
    population_roots = {
        "loso": (Path(args.loso_image), Path(args.loso_feature)),
        "loto": (Path(args.loto_image), Path(args.loto_feature)),
    }
    source_rows = []
    at2_rows = []
    evaluable_ids = [
        path.stem
        for path in raw_paths
        if tissue_by_slide[path.stem] in eligible_tissues
    ]
    for slide_index, slide_id in enumerate(evaluable_ids):
        tissue = tissue_by_slide[slide_id]
        centroids, target_index = heldout_centroids(
            slide_id,
            raw_at2,
            tissue_by_slide,
            eligible_tissues,
            reference_sums,
        )
        baseline = summarize_at2(raw_at2[slide_id], centroids, target_index)
        at2_rows.append({
            "encoder_id": model_id,
            "slide_id": slide_id,
            "tissue_type": tissue,
            "slides_in_tissue": tissue_sizes[tissue],
            "centroid_classes": len(eligible_tissues),
            **baseline,
        })
        raw_path = Path(args.raw) / model_id / "shards" / f"{slide_id}.h5"
        for population, (image_root, feature_root) in population_roots.items():
            queries = load_queries(
                raw_path,
                image_root / model_id / "shards" / f"{slide_id}.h5",
                feature_root / model_id / "shards" / f"{slide_id}.h5",
                feature_dim,
            )
            metrics = summarize_queries(
                queries, centroids, target_index, raw_at2[slide_id]
            )
            for condition_index, condition in enumerate(CONDITIONS):
                for scanner_index, scanner in enumerate(SCANNERS[1:]):
                    source_rows.append({
                        "correction_population": population,
                        "encoder_id": model_id,
                        "condition": condition,
                        "scanner": scanner,
                        "slide_id": slide_id,
                        "tissue_type": tissue,
                        "slides_in_tissue": tissue_sizes[tissue],
                        "centroid_classes": len(eligible_tissues),
                        **{
                            name: float(values[condition_index, scanner_index])
                            for name, values in metrics.items()
                        },
                    })
        print(f"[{slide_index + 1}/{len(evaluable_ids)}] {model_id} {slide_id}", flush=True)
    source = pd.DataFrame(source_rows)
    at2 = pd.DataFrame(at2_rows)
    if (
        len(source) != 6_480
        or len(at2) != 108
        or source["tissue_type"].nunique() != 36
        or at2["tissue_type"].nunique() != 36
        or not np.isfinite(source.select_dtypes(include=[np.number])).all().all()
        or not np.isfinite(at2.select_dtypes(include=[np.number])).all().all()
    ):
        raise ValueError(f"{model_id}: E7 row population gate failed")
    raw_loso = source[
        (source["correction_population"] == "loso")
        & (source["condition"] == "raw")
    ].sort_values(["slide_id", "scanner"])
    raw_loto = source[
        (source["correction_population"] == "loto")
        & (source["condition"] == "raw")
    ].sort_values(["slide_id", "scanner"])
    metric_columns = [
        "top1_accuracy",
        "top5_accuracy",
        "correct_tissue_margin",
        "centroid_profile_agreement",
    ]
    if not np.array_equal(
        raw_loso[metric_columns].to_numpy(), raw_loto[metric_columns].to_numpy()
    ):
        raise ValueError(f"{model_id}: raw LOSO/LOTO tissue probe duplication differs")
    output = Path(args.output) / model_id
    output.mkdir(parents=True, exist_ok=True)
    source_path = output / "slide_source_probe.csv"
    at2_path = output / "slide_at2_probe.csv"
    source.to_csv(source_path, index=False)
    at2.to_csv(at2_path, index=False)
    summary = {
        "analysis": "e7_grouped_tissue_probe_rows",
        "encoder_id": model_id,
        "checkpoint_sha256": model["checkpoint_sha256"],
        "slides": 108,
        "evaluable_tissues": 36,
        "excluded_singleton_tissue": singleton_tissues[0],
        "centroid_reference": "raw AT2 training slides only",
        "conditions": list(CONDITIONS),
        "source_scanners": list(SCANNERS[1:]),
        "correction_populations": ["loso", "loto"],
        "source_rows": len(source),
        "at2_rows": len(at2),
        "source_sha256": sha256(source_path),
        "at2_sha256": sha256(at2_path),
        "e5_population_audit_sha256": sha256(e5_audit_path),
        "loto_population_audit_sha256": sha256(loto_audit_path),
        "e7_contract_sha256": sha256(e7_contract_path),
        "device": torch.cuda.get_device_name(0),
        "row_gate_pass": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

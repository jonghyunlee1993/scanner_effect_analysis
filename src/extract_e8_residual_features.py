"""Encode the E8 paired residual conditions for the four frozen PFMs.

One pass per target and encoder renders every gate-passing arm, so each grid
crop is read once.  The shard layout is the one every feature artifact in this
project shares, which is what lets `analyze_e8_condition_probe.py` read these
conditions without knowing anything about how they were produced:

    <root>/<encoder_id>/shards/<slide_id>.h5
        condition  (C,)            utf-8 names
        features   (C, 6, 100, D)  scanner-major, then location, then dimension

Each slide is corrected by the checkpoint of the outer fold that held it out, so
no slide is ever encoded by a network that saw it.  The RF1U base is rebuilt at
each encoder's native FOV with that FOV's frozen statistics; the network itself
is fully convolutional and is applied unchanged, which is the FOV approximation
declared in section 4 of the contract.

Encoding refuses to run unless the audit has unblocked the arm.

Frozen rules: `docs/e8_paired_residual_contract.md`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

from build_e8_cache import E8_CROP
from e5_comparator_population import SCANNERS, centered_crop, rgb8_to_rgb01, uint8_from_rgb01
from e5_reinhard_residual_frequency import RF1_FOLDS, fold_assignments, fold_counts
from e8_paired_residual import E8_ARMS, FrozenBase, ResidualCorrector, load_base_state
from extract_e0_pfm_features import (
    BATCH_SIZE,
    load_grid_contract,
    select_model,
    string_array,
)
from extract_rf1u_features import infer, verify_runtime_equivalence
from fetch_e0_pfm_checkpoints import TRIDENT_COMMIT, sha256
from rf1u_unpaired import RF1U_TARGETS, source_indices, target_index
from smoke_e0_pfm_encoders import encoder_kwargs, verify_trident_source


ANALYSIS = "e8_feature_shard"
E8_METRICS = ("projection_fraction", "material_range_fraction")


def parse_args():
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--encoder-id")
    selection.add_argument("--encoder-index", type=int)
    parser.add_argument("--target", choices=RF1U_TARGETS, default="gt450")
    parser.add_argument("--contract", default="outputs/e0_pfm_contract/checkpoint_manifest.json")
    parser.add_argument("--grid-audit", default="outputs/e0_native_aa_grid/audit/summary.json")
    parser.add_argument("--grid", default="outputs/e0_native_aa_grid/shards")
    parser.add_argument("--raw", default="outputs/e0_pfm_features")
    parser.add_argument("--statistics", default="outputs/e5_image_statistics")
    parser.add_argument("--audit", default="outputs/e8_residual/audit")
    parser.add_argument(
        "--runtime-drift", default="outputs/rf1u_multitarget/runtime_drift/summary.json"
    )
    parser.add_argument("--output", default="outputs/e8_residual/features")
    parser.add_argument(
        "--trident-root", default="/mnt/isilon/oldridge_lab/irinaz/github/env/trident"
    )
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def load_audit(root: Path, target: str) -> dict:
    path = Path(root) / target / "summary.json"
    report = json.loads(path.read_text())
    if report.get("analysis") != "e8_paired_residual_audit" or report.get("target") != target:
        raise RuntimeError(f"invalid E8 audit report: {path}")
    unblocked = [arm for arm, value in report["encoding_unblocked"].items() if value]
    if not unblocked:
        raise RuntimeError(f"no E8 arm has cleared the audit for {target}")
    return report, [arm for arm in E8_ARMS if arm in unblocked]


def passing_folds(report: dict, arm: str) -> dict:
    """Outer fold to checkpoint path, for the folds that cleared the gate.

    JSON has no integer keys, so the audit's `detail` map round-trips with string
    keys while `folds_passing` round-trips as integers.  Normalising here keeps
    that asymmetry from reaching the rest of the file.
    """
    gate = report["arms"][arm]["gate"]
    detail = {int(fold): value for fold, value in gate["detail"].items()}
    return {int(fold): detail[int(fold)]["checkpoint"] for fold in gate["folds_passing"]}


def build_correctors(report: dict, arm: str, target: str, fov: int, device: str):
    """One corrector per outer fold, with the base rebuilt at this encoder's FOV."""
    correctors = {}
    for fold, checkpoint in passing_folds(report, arm).items():
        model = ResidualCorrector.load_from_checkpoint(checkpoint, map_location=device)
        if fov != E8_CROP:
            model.base = FrozenBase(load_base_state(target, fold, fov=fov))
        correctors[fold] = model.eval().to(device)
    return correctors


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("E8 feature extraction requires a CUDA device")
    contract = json.loads(Path(args.contract).read_text())
    if contract.get("trident_commit") != TRIDENT_COMMIT:
        raise RuntimeError("PFM contract does not use the frozen TRIDENT commit")
    verify_trident_source(Path(args.trident_root))
    model_spec = select_model(contract, args.encoder_id, args.encoder_index)
    runtime_basis = verify_runtime_equivalence(
        Path(args.runtime_drift), contract, model_spec["encoder_id"]
    )
    if sha256(Path(model_spec["checkpoint_path"])) != model_spec["checkpoint_sha256"]:
        raise RuntimeError(f"checkpoint hash mismatch: {model_spec['encoder_id']}")

    audit = json.loads(Path(args.grid_audit).read_text())
    if audit.get("grid_gate_pass") is not True or audit.get("patches_observed") != 65_400:
        raise RuntimeError("native-AA grid audit has not passed")

    report, arms = load_audit(Path(args.audit), args.target)
    target = args.target
    reference = target_index(target)
    sources = source_indices(target)
    fov = int(model_spec["native_fov_px"])
    feature_dim = int(model_spec["feature_dim"])
    conditions = tuple(f"e8_{arm}_{target}" for arm in arms)

    with np.load(Path(args.statistics) / f"fov_{fov}.npz") as statistics:
        slide_ids = [str(value) for value in statistics["slide_ids"]]
    assignments = fold_assignments(slide_ids)
    if fold_counts(assignments) != (22, 22, 22, 22, 21):
        raise RuntimeError("invalid RF1 fold population")

    correctors = {
        arm: build_correctors(report, arm, target, fov, "cuda") for arm in arms
    }

    sys.path.insert(0, str(Path(args.trident_root).resolve()))
    from trident.patch_encoder_models.load import encoder_factory

    encoder = (
        encoder_factory(
            model_spec["encoder_id"],
            **encoder_kwargs(model_spec["encoder_id"], model_spec["checkpoint_path"]),
        )
        .eval()
        .cuda()
    )
    encoder = encoder.half() if encoder.precision == torch.float16 else encoder.float()
    batch_size = args.batch_size or BATCH_SIZE[model_spec["encoder_id"]]

    output_root = Path(args.output) / target / model_spec["encoder_id"] / "shards"
    output_root.mkdir(parents=True, exist_ok=True)
    skipped = []
    for slide_id in slide_ids:
        fold = assignments[slide_id]
        if any(fold not in correctors[arm] for arm in arms):
            skipped.append(slide_id)
            continue
        output_path = output_root / f"{slide_id}.h5"
        grid_path = Path(args.grid) / f"{slide_id}.h5"
        raw_path = Path(args.raw) / model_spec["encoder_id"] / "shards" / f"{slide_id}.h5"
        load_grid_contract(grid_path)

        with h5py.File(grid_path, "r") as grid, h5py.File(raw_path, "r") as raw_source:
            raw = np.asarray(raw_source["features"][:], dtype=np.float32)
            if raw.shape != (6, 100, feature_dim):
                raise ValueError(f"{slide_id}: invalid raw features")
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
            features = np.empty((len(conditions), 6, 100, feature_dim), dtype=np.float32)
            features[:, reference] = raw[reference]
            metrics = {
                name: np.zeros((len(conditions), 6, 100), dtype=np.float32)
                for name in E8_METRICS
            }
            with torch.inference_mode():
                for position, scanner_index in enumerate(sources):
                    for start in range(0, 100, batch_size):
                        stop = min(start + batch_size, 100)
                        rgb = rgb8_to_rgb01(
                            centered_crop(grid["rgb"][scanner_index, start:stop], fov),
                            device="cuda",
                        )
                        index = torch.full(
                            (stop - start,), position, dtype=torch.long, device="cuda"
                        )
                        for condition_index, arm in enumerate(arms):
                            result = correctors[arm][fold].correct(rgb, index)
                            features[condition_index, scanner_index, start:stop] = infer(
                                encoder,
                                uint8_from_rgb01(result["output"]),
                                batch_size,
                                feature_dim,
                            )
                            for name in E8_METRICS:
                                metrics[name][condition_index, scanner_index, start:stop] = (
                                    result[name].float().cpu().numpy()
                                )

        norms = np.linalg.norm(features.astype(np.float64), axis=-1)
        if not np.isfinite(features).all() or np.any(norms <= 0):
            raise RuntimeError(f"{slide_id}: invalid E8 features")
        temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
        with h5py.File(temporary, "w") as sink:
            sink.create_dataset(
                "features",
                data=features,
                dtype=np.float32,
                chunks=(1, 1, 100, feature_dim),
                compression="lzf",
                shuffle=True,
            )
            for name, value in metrics.items():
                sink.create_dataset(name, data=value, dtype=np.float32)
            for name, value in metadata.items():
                sink.create_dataset(name, data=value)
            sink.create_dataset("condition", data=string_array(list(conditions)))
            sink.create_dataset("outer_fold", data=np.int64(fold))
        temporary.replace(output_path)
        summary = {
            "analysis": ANALYSIS,
            "target": target,
            "slide_id": slide_id,
            "outer_fold": int(fold),
            "encoder_id": model_spec["encoder_id"],
            "conditions": list(conditions),
            "runtime_basis": runtime_basis,
            "output_sha256": sha256(output_path),
        }
        output_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
        print(f"{model_spec['encoder_id']}/{slide_id}: fold {fold}, {len(conditions)} conditions")

    manifest = {
        "analysis": ANALYSIS,
        "target": target,
        "encoder_id": model_spec["encoder_id"],
        "conditions": list(conditions),
        "arms": arms,
        "slides_encoded": len(slide_ids) - len(skipped),
        "slides_skipped": skipped,
        "native_fov_px": fov,
        "scanners": list(SCANNERS),
        "folds": RF1_FOLDS,
    }
    (output_root.parent / "manifest.json").write_text(json.dumps(manifest, indent=1))
    if skipped:
        print(f"skipped {len(skipped)} slides whose outer fold did not pass the gate")


if __name__ == "__main__":
    main()

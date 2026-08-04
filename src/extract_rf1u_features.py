"""Render and encode the RF1U multi-target conditions for the four frozen PFMs.

One pass per target and encoder produces both conditions of that target, the
Reinhard comparator and RF1U, so each grid crop is read once.  Gains are fitted
on the training folds from pooled population band energies only; the held-out
slide contributes nothing to its own transform.

Frozen rules: `docs/e5_rf1u_multitarget_contract.md`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

from analyze_rf1_multiscale import shared_od_multiscale
from build_rf1m_cell import load_e5_statistics
from build_rf1m_slide_band_energy import fold_lab_statistics
from e5_comparator_population import (
    SCANNERS,
    centered_crop,
    reinhard_lab,
    rgb8_to_rgb01,
    uint8_from_rgb01,
)
from e5_reinhard_residual_frequency import RF1_FOLDS, fold_assignments, fold_counts
from extract_e0_pfm_features import (
    BATCH_SIZE,
    load_grid_contract,
    select_model,
    string_array,
    transformed_batch,
)
from fetch_e0_pfm_checkpoints import TRIDENT_COMMIT, sha256
from rf1m_combined import RF1M_SIGMAS
from rf1u_unpaired import (
    RF1U_CONDITION,
    RF1U_SHRINKAGE_SE,
    RF1U_TARGETS,
    RF1U_VERSION,
    fitted_scanner_gains,
    source_indices,
    target_index,
)
from smoke_e0_pfm_encoders import (
    encoder_kwargs,
    package_version,
    verify_trident_source,
)


ANALYSIS = "rf1u_feature_shard"
RF1U_METRICS = (
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
    parser.add_argument("--target", choices=RF1U_TARGETS, required=True)
    parser.add_argument("--contract", default="outputs/e0_pfm_contract/checkpoint_manifest.json")
    parser.add_argument("--grid-audit", default="outputs/e0_native_aa_grid/audit/summary.json")
    parser.add_argument("--grid", default="outputs/e0_native_aa_grid/shards")
    parser.add_argument("--raw", default="outputs/e0_pfm_features")
    parser.add_argument("--e5-statistics", default="outputs/e5_image_statistics")
    parser.add_argument("--energy", default="outputs/rf1u_multitarget/energy")
    parser.add_argument(
        "--runtime-drift", default="outputs/rf1u_multitarget/runtime_drift/summary.json"
    )
    parser.add_argument(
        "--execution-contract", default="docs/e5_rf1u_multitarget_contract.md"
    )
    parser.add_argument("--output", default="outputs/rf1u_multitarget/features")
    parser.add_argument(
        "--trident-root", default="/mnt/isilon/oldridge_lab/irinaz/github/env/trident"
    )
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def verify_runtime_equivalence(path: Path, contract: dict, encoder_id: str) -> str:
    """Accept the runtime only if it matches the pin or provably reproduces it.

    The contract pins distribution versions so that new features stay comparable
    with the locked raw population.  A version match is one sufficient proof; a
    measured bit-identical re-encode of the audited crops is a stronger one.  Any
    other state is refused.
    """
    observed = {
        name: package_version(name) for name in contract["runtime_distributions"]
    }
    if observed == contract["runtime_distributions"]:
        return "version_match"
    audit = json.loads(path.read_text())
    rows = [
        row
        for row in audit["encoders"]
        if row.get("available") and row["encoder_id"] == encoder_id
    ]
    if not (
        audit.get("analysis") == "pfm_runtime_drift_audit"
        and audit.get("observed_runtime") == observed
        and audit.get("contract_runtime") == contract["runtime_distributions"]
        and rows
        and all(row["max_abs_difference"] == 0.0 for row in rows)
    ):
        raise RuntimeError(
            f"runtime differs from the contract and no bit-identical audit covers "
            f"{encoder_id}: {observed} != {contract['runtime_distributions']}"
        )
    return f"bit_identical_audit_over_{len(rows)}_slides"


def infer(encoder, rgb8: np.ndarray, batch_size: int, feature_dim: int):
    import torch

    values = np.empty((len(rgb8), feature_dim), dtype=np.float32)
    for start in range(0, len(rgb8), batch_size):
        stop = min(start + batch_size, len(rgb8))
        batch = transformed_batch(rgb8[start:stop], encoder.eval_transforms).cuda(
            non_blocking=True
        )
        batch = batch.half() if encoder.precision == torch.float16 else batch.float()
        with torch.inference_mode():
            values[start:stop] = encoder(batch).float().cpu().numpy()
    return values


def load_band_energy(root: Path, target: str, fov: int):
    path = root / f"{target}_fov_{fov}.npz"
    summary = json.loads(path.with_suffix(".summary.json").read_text())
    if not (
        summary.get("analysis") == "rf1m_slide_band_energy"
        and summary.get("target") == target
        and summary.get("fov") == fov
        and summary.get("pfm_feature_access") is False
        and summary.get("output_sha256") == sha256(path)
        and summary.get("energy_gate_pass") is True
    ):
        raise RuntimeError(f"invalid RF1U band-energy input: {path}")
    with np.load(path) as source:
        return path, summary, {name: source[name] for name in source.files}


def fold_gains(energy: dict, fold: int):
    """Shrunk band gains for one held-out fold, fitted on the other four."""
    train = energy["fold_of_slide"] != fold
    if train.sum() < 2:
        raise ValueError("a training fold must hold at least two slides")
    return fitted_scanner_gains(
        energy["target_band_energy"][train],
        energy["source_band_energy"][train, :, fold, :],
        RF1U_SHRINKAGE_SE,
    )


def existing_output_passes(path: Path, model: dict, target: str, energy_sha: str) -> bool:
    summary_path = path.with_suffix(".summary.json")
    if not path.exists() or not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return bool(
        summary.get("analysis") == ANALYSIS
        and summary.get("rf1u_version") == RF1U_VERSION
        and summary.get("target") == target
        and summary.get("encoder_id") == model["encoder_id"]
        and summary.get("band_energy_sha256") == energy_sha
        and summary.get("output_sha256") == sha256(path)
        and summary.get("shard_gate_pass") is True
    )


def main():
    import torch

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("RF1U feature extraction requires a CUDA device")
    contract = json.loads(Path(args.contract).read_text())
    if contract.get("trident_commit") != TRIDENT_COMMIT:
        raise RuntimeError("PFM contract does not use the frozen TRIDENT commit")
    verify_trident_source(Path(args.trident_root))
    model = select_model(contract, args.encoder_id, args.encoder_index)
    runtime_basis = verify_runtime_equivalence(
        Path(args.runtime_drift), contract, model["encoder_id"]
    )
    runtime_drift_sha = sha256(Path(args.runtime_drift))
    if sha256(Path(model["checkpoint_path"])) != model["checkpoint_sha256"]:
        raise RuntimeError(f"checkpoint hash mismatch: {model['encoder_id']}")

    audit = json.loads(Path(args.grid_audit).read_text())
    if audit.get("grid_gate_pass") is not True or audit.get("patches_observed") != 65_400:
        raise RuntimeError("native-AA grid audit has not passed")
    audit_sha = sha256(Path(args.grid_audit))

    fov = int(model["native_fov_px"])
    feature_dim = int(model["feature_dim"])
    target = args.target
    reference = target_index(target)
    sources = source_indices(target)
    conditions = (f"reinhard_{target}", f"{RF1U_CONDITION}_{target}")

    _, e5_summary, statistics = load_e5_statistics(
        Path(args.e5_statistics), fov, audit_sha
    )
    slide_ids = [str(value) for value in statistics["slide_ids"]]
    assignments = fold_assignments(slide_ids)
    if fold_counts(assignments) != (22, 22, 22, 22, 21):
        raise RuntimeError("invalid RF1 fold population")
    lab_mean, lab_std = fold_lab_statistics(statistics, slide_ids, assignments)
    energy_path, energy_summary, energy = load_band_energy(
        Path(args.energy), target, fov
    )
    energy_sha = energy_summary["output_sha256"]
    if [str(value) for value in energy["slide_ids"]] != slide_ids:
        raise RuntimeError("band-energy slide order differs from the E5 statistics")
    if list(energy["source_indices"]) != list(sources):
        raise RuntimeError("band-energy source order differs from the contract")
    gains = {fold: fold_gains(energy, fold) for fold in range(RF1_FOLDS)}
    contract_sha = sha256(Path(args.execution_contract))

    sys.path.insert(0, str(Path(args.trident_root).resolve()))
    from trident.patch_encoder_models.load import encoder_factory

    encoder = encoder_factory(
        model["encoder_id"], **encoder_kwargs(model["encoder_id"], model["checkpoint_path"])
    ).eval().cuda()
    encoder = encoder.half() if encoder.precision == torch.float16 else encoder.float()
    batch_size = args.batch_size or BATCH_SIZE[model["encoder_id"]]

    grid_root = Path(args.grid)
    output_root = Path(args.output) / target / model["encoder_id"] / "shards"
    output_root.mkdir(parents=True, exist_ok=True)
    emitted = 0
    for slide_number, slide_id in enumerate(slide_ids, start=1):
        grid_path = grid_root / f"{slide_id}.h5"
        raw_path = Path(args.raw) / model["encoder_id"] / "shards" / f"{slide_id}.h5"
        output_path = output_root / f"{slide_id}.h5"
        if existing_output_passes(output_path, model, target, energy_sha):
            emitted += 1
            continue
        fold = assignments[slide_id]
        gain = gains[fold]
        grid_summary = load_grid_contract(grid_path)
        raw_summary = json.loads(raw_path.with_suffix(".summary.json").read_text())
        mean_tensor = torch.as_tensor(lab_mean[fold], dtype=torch.float32, device="cuda")
        std_tensor = torch.as_tensor(lab_std[fold], dtype=torch.float32, device="cuda")

        with h5py.File(grid_path, "r") as grid, h5py.File(raw_path, "r") as raw_source:
            raw = np.asarray(raw_source["features"][:], dtype=np.float32)
            if raw.shape != (6, 100, feature_dim):
                raise ValueError(f"{model['encoder_id']}/{slide_id}: invalid raw features")
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
                name: np.zeros((6, 100), dtype=np.float32) for name in RF1U_METRICS
            }
            with torch.inference_mode():
                for position, scanner_index in enumerate(sources):
                    for start in range(0, 100, batch_size):
                        stop = min(start + batch_size, 100)
                        rgb = rgb8_to_rgb01(
                            centered_crop(grid["rgb"][scanner_index, start:stop], fov),
                            device="cuda",
                        )
                        base = reinhard_lab(
                            rgb,
                            mean_tensor[scanner_index].reshape(1, 1, 1, 3),
                            std_tensor[scanner_index].reshape(1, 1, 1, 3),
                            mean_tensor[reference].reshape(1, 1, 1, 3),
                            std_tensor[reference].reshape(1, 1, 1, 3),
                        )
                        report = shared_od_multiscale(
                            base["output"], gain["gain"][position], RF1M_SIGMAS
                        )
                        for condition_index, image in enumerate(
                            (base["output"], report["output"])
                        ):
                            features[condition_index, scanner_index, start:stop] = infer(
                                encoder, uint8_from_rgb01(image), batch_size, feature_dim
                            )
                        metrics["base_preclip_range_fraction"][scanner_index, start:stop] = (
                            base["preclip_range_fraction"].cpu().numpy()
                        )
                        metrics["base_clip_pixel_mae"][scanner_index, start:stop] = (
                            base["clip_pixel_mae"].cpu().numpy()
                        )
                        for name in RF1U_METRICS[2:]:
                            metrics[name][scanner_index, start:stop] = (
                                report[name].cpu().numpy()
                            )

        norms = np.linalg.norm(features.astype(np.float64), axis=-1)
        if not np.isfinite(features).all() or np.any(norms <= 0):
            raise RuntimeError(f"{model['encoder_id']}/{slide_id}: invalid RF1U features")
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
            sink.create_dataset("band_gain", data=gain["gain"], dtype=np.float64)
            sink.create_dataset("band_alpha", data=gain["alpha"], dtype=np.float64)
            sink.create_dataset("band_raw_log_gain", data=gain["raw_log_gain"], dtype=np.float64)
            sink.create_dataset("band_log_gain_se", data=gain["standard_error"], dtype=np.float64)
            sink.attrs["analysis"] = ANALYSIS
            sink.attrs["rf1u_version"] = RF1U_VERSION
            sink.attrs["target"] = target
            sink.attrs["slide_id"] = slide_id
            sink.attrs["encoder_id"] = model["encoder_id"]
            sink.attrs["feature_dim"] = feature_dim
            sink.attrs["native_fov_px"] = fov
            sink.attrs["fold"] = fold
            sink.attrs["checkpoint_sha256"] = model["checkpoint_sha256"]
            sink.attrs["source_grid_sha256"] = grid_summary["output_sha256"]
            sink.attrs["raw_feature_sha256"] = raw_summary["output_sha256"]
            sink.attrs["band_energy_sha256"] = energy_sha
            sink.attrs["execution_contract_sha256"] = contract_sha
            sink.attrs["runtime_basis"] = runtime_basis
            sink.attrs["runtime_drift_sha256"] = runtime_drift_sha
            sink.flush()
        os.replace(temporary, output_path)
        summary = {
            "analysis": ANALYSIS,
            "rf1u_version": RF1U_VERSION,
            "target": target,
            "conditions": list(conditions),
            "slide_id": slide_id,
            "encoder_id": model["encoder_id"],
            "feature_dim": feature_dim,
            "native_fov_px": fov,
            "fold": fold,
            "shrinkage_se": RF1U_SHRINKAGE_SE,
            "features": len(conditions) * 600,
            "minimum_norm": float(norms.min()),
            "maximum_norm": float(norms.max()),
            "checkpoint_sha256": model["checkpoint_sha256"],
            "source_grid_sha256": grid_summary["output_sha256"],
            "raw_feature_sha256": raw_summary["output_sha256"],
            "e5_statistics_sha256": e5_summary["output_sha256"],
            "band_energy_sha256": energy_sha,
            "execution_contract_sha256": contract_sha,
            "runtime_basis": runtime_basis,
            "runtime_drift_sha256": runtime_drift_sha,
            "output": str(output_path.resolve()),
            "output_sha256": sha256(output_path),
            "shard_gate_pass": True,
        }
        output_path.with_suffix(".summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        emitted += 1
        print(
            f"[{slide_number}/{len(slide_ids)}] {target} {model['encoder_id']} {slide_id}",
            flush=True,
        )

    print(
        json.dumps(
            {
                "analysis": "rf1u_feature_extraction_complete",
                "rf1u_version": RF1U_VERSION,
                "target": target,
                "encoder_id": model["encoder_id"],
                "shards": emitted,
                "conditions": list(conditions),
                "band_energy": str(energy_path.resolve()),
                "device": torch.cuda.get_device_name(0),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

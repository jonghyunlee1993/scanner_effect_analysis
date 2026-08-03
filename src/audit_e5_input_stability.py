"""Select frozen E5 numerical constants using input-domain diagnostics only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from e5_comparator_population import (
    CORAL_CANDIDATES,
    E5_VERSION,
    FOVS,
    GAIN_CAP_CANDIDATES,
    SCANNERS,
    centered_covariance,
    centered_crop,
    feature_sufficient_statistics,
    fitted_frequency_gain,
    frequency_calibration,
    procrustes_transform,
    radial_geometry,
    regularized_covariance,
    rgb8_to_rgb01,
    symmetric_matrix_power,
)
from fetch_e0_pfm_checkpoints import sha256


CONDITION_LIMIT = 10_000.0
NORM_RATIO_LIMITS = (0.1, 10.0)
CLIP_MEAN_LIMIT = 0.005
CLIP_Q99_LIMIT = 0.05
CLIP_MAE_LIMIT = 0.002
MATERIAL_EXCURSION = 1.0 / 255.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", default="outputs/e0_pfm_contract/checkpoint_manifest.json")
    parser.add_argument("--decision", default="docs/e4_e7_decision_record.md")
    parser.add_argument("--execution-contract", default="docs/e5_comparator_execution_contract.md")
    parser.add_argument("--grid", default="outputs/e0_native_aa_grid/shards")
    parser.add_argument("--raw", default="outputs/e0_pfm_features")
    parser.add_argument("--image-statistics", default="outputs/e5_image_statistics")
    parser.add_argument("--feature-statistics", default="outputs/e5_feature_statistics")
    parser.add_argument("--output", default="outputs/e5_input_stability")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def load_image_statistics(root: Path, fov: int):
    path = root / f"fov_{fov}.npz"
    summary_path = root / f"fov_{fov}.summary.json"
    summary = json.loads(summary_path.read_text())
    if not (
        summary.get("analysis") == "e5_image_sufficient_statistics"
        and summary.get("e5_version") == E5_VERSION
        and summary.get("fov") == fov
        and summary.get("output_sha256") == sha256(path)
        and summary.get("statistics_gate_pass") is True
    ):
        raise RuntimeError(f"invalid E5 image sufficient statistics: {path}")
    return path, summary, np.load(path)


def load_feature_statistics(root: Path, model: dict):
    path = root / f"{model['encoder_id']}.h5"
    summary_path = root / f"{model['encoder_id']}.summary.json"
    summary = json.loads(summary_path.read_text())
    if not (
        summary.get("analysis") == "e5_feature_sufficient_statistics"
        and summary.get("e5_version") == E5_VERSION
        and summary.get("encoder_id") == model["encoder_id"]
        and summary.get("checkpoint_sha256") == model["checkpoint_sha256"]
        and summary.get("output_sha256") == sha256(path)
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
    return path, summary, values


def frequency_audit(args, calibration_slide: str):
    import torch

    rows = []
    parameter_by_fov = {}
    source_hashes = {}
    for fov in FOVS:
        path, summary, loaded = load_image_statistics(Path(args.image_statistics), fov)
        source_hashes[str(fov)] = {
            "path": str(path.resolve()),
            "sha256": summary["output_sha256"],
        }
        slide_ids = [str(value) for value in loaded["slide_ids"]]
        if slide_ids[0] != calibration_slide or len(slide_ids) != 109:
            raise ValueError(f"FOV {fov}: deterministic calibration slide mismatch")
        heldout = 0
        total_power = loaded["radial_power"].sum(axis=0)
        train_power = total_power - loaded["radial_power"][heldout]
        frequency = loaded["radial_frequency"]
        parameter_by_fov[fov] = {
            cap: {
                scanner_index: fitted_frequency_gain(
                    train_power[scanner_index], train_power[0], frequency, cap
                )
                for scanner_index in range(1, 6)
            }
            for cap in GAIN_CAP_CANDIDATES
        }
        loaded.close()

    grid_path = Path(args.grid) / f"{calibration_slide}.h5"
    with h5py.File(grid_path, "r") as source:
        for fov in FOVS:
            geometry = radial_geometry(fov)
            for scanner_index, scanner in enumerate(SCANNERS[1:], start=1):
                by_cap = {
                    cap: {"clip": [], "material_clip": [], "mae": [], "finite": True}
                    for cap in GAIN_CAP_CANDIDATES
                }
                for start in range(0, 100, args.batch_size):
                    stop = min(start + args.batch_size, 100)
                    rgb8 = centered_crop(source["rgb"][scanner_index, start:stop], fov)
                    rgb = rgb8_to_rgb01(rgb8, device="cuda")
                    for cap in GAIN_CAP_CANDIDATES:
                        report = frequency_calibration(
                            rgb,
                            parameter_by_fov[fov][cap][scanner_index],
                            geometry.frequency,
                        )
                        by_cap[cap]["finite"] &= bool(
                            torch.isfinite(report["preclip"]).all()
                            and torch.isfinite(report["output"]).all()
                        )
                        by_cap[cap]["clip"].extend(
                            report["preclip_range_fraction"].cpu().tolist()
                        )
                        preclip = report["preclip"]
                        material = (
                            (preclip < -MATERIAL_EXCURSION)
                            | (preclip > 1.0 + MATERIAL_EXCURSION)
                        ).float().mean(dim=(1, 2, 3))
                        by_cap[cap]["material_clip"].extend(material.cpu().tolist())
                        by_cap[cap]["mae"].extend(report["clip_pixel_mae"].cpu().tolist())
                for cap in GAIN_CAP_CANDIDATES:
                    clip = np.asarray(by_cap[cap]["clip"], dtype=np.float64)
                    material = np.asarray(by_cap[cap]["material_clip"], dtype=np.float64)
                    mae = np.asarray(by_cap[cap]["mae"], dtype=np.float64)
                    passed = bool(
                        by_cap[cap]["finite"]
                        and material.mean() <= CLIP_MEAN_LIMIT
                        and np.quantile(material, 0.99) <= CLIP_Q99_LIMIT
                        and mae.mean() <= CLIP_MAE_LIMIT
                    )
                    rows.append(
                        {
                            "fov": fov,
                            "scanner": scanner,
                            "gain_cap": cap,
                            "finite": by_cap[cap]["finite"],
                            "clip_fraction_mean": float(clip.mean()),
                            "clip_fraction_q99": float(np.quantile(clip, 0.99)),
                            "material_clip_fraction_mean": float(material.mean()),
                            "material_clip_fraction_q99": float(np.quantile(material, 0.99)),
                            "clip_mae_mean": float(mae.mean()),
                            "cell_pass": passed,
                        }
                    )
                print(f"frequency stability FOV={fov} scanner={scanner}", flush=True)
    passing = [
        cap
        for cap in GAIN_CAP_CANDIDATES
        if all(row["cell_pass"] for row in rows if row["gain_cap"] == cap)
    ]
    return rows, (max(passing) if passing else None), source_hashes


def feature_audit(args, contract: dict, calibration_slide: str):
    import torch

    model_cache = []
    condition_rows = []
    source_hashes = {}
    for model in contract["models"]:
        path, summary, stats = load_feature_statistics(Path(args.feature_statistics), model)
        source_hashes[model["encoder_id"]] = {
            "path": str(path.resolve()),
            "sha256": summary["output_sha256"],
        }
        if stats["slide_ids"][0] != calibration_slide or len(stats["slide_ids"]) != 109:
            raise ValueError(f"{model['encoder_id']}: deterministic calibration slide mismatch")
        raw_path = Path(args.raw) / model["encoder_id"] / "shards" / f"{calibration_slide}.h5"
        with h5py.File(raw_path, "r") as source:
            raw = torch.from_numpy(np.asarray(source["features"][:], dtype=np.float32)).cuda()
        heldout = feature_sufficient_statistics(raw)
        total_sum = torch.from_numpy(stats["sum"]).cuda()
        total_gram = torch.from_numpy(stats["gram"]).cuda()
        total_cross = torch.from_numpy(stats["cross"]).cuda()
        train_sum = total_sum - heldout[0]
        train_gram = total_gram - heldout[1]
        train_cross = total_cross - heldout[2]
        train_count = int(stats["count"] - heldout[3])
        covariances = [
            centered_covariance(train_sum[index], train_gram[index], train_count)
            for index in range(6)
        ]
        eigenspectra = [torch.linalg.eigvalsh(value) for value in covariances]
        for scanner_index, scanner in enumerate(SCANNERS):
            eigenvalues = eigenspectra[scanner_index]
            scale = torch.trace(covariances[scanner_index]) / int(model["feature_dim"])
            for alpha in CORAL_CANDIDATES:
                regularized = (1.0 - alpha) * eigenvalues + alpha * scale
                condition = float(
                    (regularized[-1] / regularized[0].clamp_min(1e-30)).cpu()
                )
                condition_rows.append(
                    {
                        "encoder_id": model["encoder_id"],
                        "scanner": scanner,
                        "shrinkage": alpha,
                        "condition_number": condition,
                        "condition_pass": bool(
                            np.isfinite(condition)
                            and float(regularized[0].cpu()) > 0
                            and condition <= CONDITION_LIMIT
                        ),
                    }
                )
        model_cache.append(
            {
                "model": model,
                "raw": raw,
                "sum": train_sum,
                "gram": train_gram,
                "cross": train_cross,
                "count": train_count,
                "covariance": covariances,
            }
        )
        print(f"feature covariance stability {model['encoder_id']}", flush=True)

    candidate_pass = {
        alpha: all(
            row["condition_pass"]
            for row in condition_rows
            if row["shrinkage"] == alpha
        )
        for alpha in CORAL_CANDIDATES
    }
    transform_rows = []
    selected = None
    for alpha in CORAL_CANDIDATES:
        if not candidate_pass[alpha]:
            continue
        alpha_pass = True
        for cached in model_cache:
            model = cached["model"]
            target_cov = regularized_covariance(cached["covariance"][0], alpha)
            target_root = symmetric_matrix_power(target_cov, 0.5)
            target_mean = cached["sum"][0] / float(cached["count"])
            for scanner_index, scanner in enumerate(SCANNERS[1:], start=1):
                source_cov = regularized_covariance(
                    cached["covariance"][scanner_index], alpha
                )
                source_inverse = symmetric_matrix_power(source_cov, -0.5)
                source_mean = cached["sum"][scanner_index] / float(cached["count"])
                transformed = (
                    (cached["raw"][scanner_index].double() - source_mean)
                    @ source_inverse
                    @ target_root
                    + target_mean
                )
                raw_norm = torch.linalg.vector_norm(cached["raw"][scanner_index].double(), dim=1)
                new_norm = torch.linalg.vector_norm(transformed, dim=1)
                quantiles = torch.tensor(
                    [0.01, 0.99], dtype=raw_norm.dtype, device=raw_norm.device
                )
                raw_q = torch.quantile(raw_norm, quantiles)
                new_q = torch.quantile(new_norm, quantiles)
                ratios = new_q / raw_q
                passed = bool(
                    torch.isfinite(transformed).all()
                    and torch.all(ratios >= NORM_RATIO_LIMITS[0])
                    and torch.all(ratios <= NORM_RATIO_LIMITS[1])
                )
                alpha_pass &= passed
                transform_rows.append(
                    {
                        "encoder_id": model["encoder_id"],
                        "scanner": scanner,
                        "method": "coral",
                        "shrinkage": alpha,
                        "norm_q01_ratio": float(ratios[0].cpu()),
                        "norm_q99_ratio": float(ratios[1].cpu()),
                        "finite": bool(torch.isfinite(transformed).all()),
                        "transform_pass": passed,
                    }
                )
        if alpha_pass:
            selected = alpha
            break

    procrustes_pass = True
    for cached in model_cache:
        model = cached["model"]
        for scanner_index, scanner in enumerate(SCANNERS[1:], start=1):
            transformed = procrustes_transform(
                cached["raw"][scanner_index],
                cached["sum"][scanner_index],
                cached["sum"][0],
                cached["cross"][scanner_index - 1],
                cached["count"],
            )
            raw_norm = torch.linalg.vector_norm(cached["raw"][scanner_index].double(), dim=1)
            new_norm = torch.linalg.vector_norm(transformed, dim=1)
            quantiles = torch.tensor(
                [0.01, 0.99], dtype=raw_norm.dtype, device=raw_norm.device
            )
            raw_q = torch.quantile(raw_norm, quantiles)
            new_q = torch.quantile(new_norm, quantiles)
            ratios = new_q / raw_q
            passed = bool(
                torch.isfinite(transformed).all()
                and torch.all(ratios >= NORM_RATIO_LIMITS[0])
                and torch.all(ratios <= NORM_RATIO_LIMITS[1])
            )
            procrustes_pass &= passed
            transform_rows.append(
                {
                    "encoder_id": model["encoder_id"],
                    "scanner": scanner,
                    "method": "orthogonal_procrustes",
                    "shrinkage": None,
                    "norm_q01_ratio": float(ratios[0].cpu()),
                    "norm_q99_ratio": float(ratios[1].cpu()),
                    "finite": bool(torch.isfinite(transformed).all()),
                    "transform_pass": passed,
                }
            )
        print(f"feature Procrustes stability {model['encoder_id']}", flush=True)
    return condition_rows, transform_rows, selected, procrustes_pass, source_hashes


def main():
    import torch

    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("E5 input stability audit requires a visible CUDA device")
    paths = sorted(Path(args.grid).glob("*.h5"))
    if len(paths) != 109:
        raise ValueError(f"expected 109 native-AA shards, got {len(paths)}")
    calibration_slide = paths[0].stem
    contract = json.loads(Path(args.contract).read_text())
    frequency_rows, gain_cap, image_hashes = frequency_audit(args, calibration_slide)
    condition_rows, transform_rows, shrinkage, procrustes_pass, feature_hashes = feature_audit(
        args, contract, calibration_slide
    )
    gate = bool(gain_cap is not None and shrinkage is not None and procrustes_pass)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    pd.DataFrame(frequency_rows).to_csv(output / "frequency_gain_cap_candidates.csv", index=False)
    pd.DataFrame(condition_rows).to_csv(output / "coral_condition_candidates.csv", index=False)
    pd.DataFrame(transform_rows).to_csv(output / "feature_transform_stability.csv", index=False)
    decision_path = Path(args.decision)
    execution_path = Path(args.execution_contract)
    summary = {
        "analysis": "e5_input_only_stability",
        "e5_version": E5_VERSION,
        "outcome_access": False,
        "calibration_slide": calibration_slide,
        "coral_shrinkage_candidates": list(CORAL_CANDIDATES),
        "coral_condition_limit": CONDITION_LIMIT,
        "feature_norm_ratio_limits": list(NORM_RATIO_LIMITS),
        "selected_coral_shrinkage": shrinkage,
        "frequency_gain_cap_candidates": list(GAIN_CAP_CANDIDATES),
        "frequency_clip_mean_limit": CLIP_MEAN_LIMIT,
        "frequency_clip_q99_limit": CLIP_Q99_LIMIT,
        "frequency_clip_mae_limit": CLIP_MAE_LIMIT,
        "frequency_material_excursion": MATERIAL_EXCURSION,
        "selected_frequency_gain_cap": gain_cap,
        "procrustes_stability_pass": procrustes_pass,
        "decision_record": str(decision_path.resolve()),
        "decision_record_sha256": sha256(decision_path),
        "execution_contract": str(execution_path.resolve()),
        "execution_contract_sha256": sha256(execution_path),
        "image_statistics": image_hashes,
        "feature_statistics": feature_hashes,
        "stability_gate_pass": gate,
        "device": torch.cuda.get_device_name(0),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not gate:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

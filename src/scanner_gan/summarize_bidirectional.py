#!/usr/bin/env python3
"""Summarize the full bidirectional AT2/GT450 Pix2Pix experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DIRECTIONS = ("gt450_to_at2", "at2_to_gt450")
DISPLAY = {"gt450_to_at2": "GT450 → AT2", "at2_to_gt450": "AT2 → GT450"}
SUMMARY_VERSION = "pix2pix_bidirectional_full_summary_v1"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _training_summary(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    histories = []
    for direction in DIRECTIONS:
        for manifest_path in sorted(
            (root / "01_training" / "runs" / direction).glob(
                "fold_*/*/run_manifest.json"
            )
        ):
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("status") != "complete":
                raise ValueError(f"incomplete training run: {manifest_path}")
            config = manifest["config"]
            expected_contract = {
                "max_passes": 200,
                "constant_passes": 100,
                "selection_every": 5,
                "selection_policy": "target_l1",
            }
            observed_contract = {
                key: config.get(key) for key in expected_contract
            }
            if observed_contract != expected_contract:
                raise ValueError(
                    f"unexpected full-training contract in {manifest_path}: "
                    f"{observed_contract}"
                )
            if int(manifest["completed_passes"]) != 200:
                raise ValueError(
                    f"training did not complete 200 passes: {manifest_path}"
                )
            run_dir = manifest_path.parent
            history = pd.DataFrame(
                json.loads(line)
                for line in (run_dir / "metrics.jsonl").read_text().splitlines()
                if line.strip()
            )
            selection_every = int(manifest["config"]["selection_every"])
            candidates = history.loc[
                (history["completed_passes"] % selection_every == 0)
                | (history["completed_passes"] == int(manifest["completed_passes"]))
            ]
            best = candidates.sort_values(
                ["validation_l1_normalized", "completed_passes"], kind="stable"
            ).iloc[0]
            if not np.isclose(
                float(best["validation_l1_normalized"]),
                float(manifest["best_validation_l1"]),
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    f"best validation metric disagrees with history: {manifest_path}"
                )
            checkpoint_path = Path(manifest["best_image_only_checkpoint"]).resolve()
            if not checkpoint_path.is_file():
                raise FileNotFoundError(checkpoint_path)
            fold = int(manifest["config"]["test_fold"])
            rows.append(
                {
                    "direction": direction,
                    "test_fold": fold,
                    "best_pass": int(best["completed_passes"]),
                    "best_validation_l1": float(best["validation_l1_normalized"]),
                    "best_validation_psnr_db": float(best["validation_psnr_db"]),
                    "best_validation_target_gradient_ncc": float(
                        best["validation_target_gradient_ncc"]
                    ),
                    "best_validation_source_gradient_ncc": float(
                        best["validation_source_gradient_ncc"]
                    ),
                    "completed_passes": int(manifest["completed_passes"]),
                    "global_step": int(manifest["global_step"]),
                    "elapsed_seconds": float(manifest["elapsed_seconds"]),
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": sha256(checkpoint_path),
                }
            )
            history.insert(0, "test_fold", fold)
            history.insert(0, "direction", direction)
            histories.append(history)
    summary = pd.DataFrame(rows).sort_values(["direction", "test_fold"])
    if len(summary) != 10:
        raise ValueError(f"expected 10 completed training runs, found {len(summary)}")
    return summary, pd.concat(histories, ignore_index=True)


def _collect_endpoint_summaries(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    image_frames = []
    uni_frames = []
    for role, image_subdir, uni_subdir in (
        ("primary_translation", "03_image_evaluation", "04_uni"),
        ("target_identity_damage", "03_identity_image_evaluation", "04_identity_uni"),
    ):
        for direction in DIRECTIONS:
            source, target = direction.split("_to_", maxsplit=1)
            image_path = root / image_subdir / direction / "summary.csv"
            image_manifest_path = image_path.with_name("evaluation_manifest.json")
            image_manifest = json.loads(image_manifest_path.read_text())
            expected_input = target if role == "target_identity_damage" else source
            image_contract = (
                image_manifest.get("status") == "complete"
                and image_manifest.get("source_scanner") == source
                and image_manifest.get("target_scanner") == target
                and image_manifest.get("analysis_role") == role
                and image_manifest.get("inference_input_scanner") == expected_input
            )
            image_record = image_manifest.get("outputs", {}).get("summary", {})
            if (
                not image_contract
                or Path(image_record.get("path", "")).resolve() != image_path.resolve()
                or image_record.get("sha256") != sha256(image_path)
            ):
                raise ValueError(
                    f"image evaluation provenance mismatch: {image_manifest_path}"
                )
            image = pd.read_csv(image_path)
            image.insert(0, "direction", direction)
            image.insert(1, "analysis_role", role)
            image.insert(
                2,
                "all_available_safety_gates_pass",
                bool(image_manifest["all_available_safety_gates_pass"]),
            )
            image_frames.append(image)
            uni_path = root / uni_subdir / direction / "summary.csv"
            uni_manifest_path = uni_path.with_name("evaluation_manifest.json")
            uni_manifest = json.loads(uni_manifest_path.read_text())
            uni_contract = (
                uni_manifest.get("status") == "complete"
                and uni_manifest.get("source_scanner") == source
                and uni_manifest.get("target_scanner") == target
                and uni_manifest.get("analysis_role") == role
                and uni_manifest.get("inference_input_scanner") == expected_input
            )
            uni_record = uni_manifest.get("outputs", {}).get("summary", {})
            if (
                not uni_contract
                or Path(uni_record.get("path", "")).resolve() != uni_path.resolve()
                or uni_record.get("sha256") != sha256(uni_path)
            ):
                raise ValueError(
                    f"UNI evaluation provenance mismatch: {uni_manifest_path}"
                )
            uni = pd.read_csv(uni_path)
            uni.insert(0, "direction", direction)
            uni.insert(1, "analysis_role", role)
            uni_frames.append(uni)
    return pd.concat(image_frames, ignore_index=True), pd.concat(
        uni_frames, ignore_index=True
    )


def _metric(frame: pd.DataFrame, direction: str, role: str, metric: str) -> pd.Series:
    block = frame.loc[
        (frame["direction"] == direction)
        & (frame["analysis_role"] == role)
        & (frame["metric"] == metric)
    ]
    if len(block) != 1:
        raise ValueError(
            f"expected one {direction}/{role}/{metric} row, found {len(block)}"
        )
    return block.iloc[0]


def _bootstrap_paired_direction_gain(
    root: Path, replicates: int = 20_000
) -> dict[str, Any]:
    frames = []
    for direction in DIRECTIONS:
        path = root / "04_uni" / direction / "location_metrics.csv.gz"
        frame = pd.read_csv(path)[["slide_id", "location_index", "gain_to_target"]]
        frame = frame.rename(columns={"gain_to_target": direction})
        frames.append(frame)
    merged = frames[0].merge(frames[1], on=["slide_id", "location_index"], how="inner")
    if len(merged) != len(frames[0]) or len(merged) != len(frames[1]):
        raise ValueError("bidirectional UNI locations are not exactly paired")
    merged["gt450_target_minus_at2_target_gain"] = (
        merged["at2_to_gt450"] - merged["gt450_to_at2"]
    )
    slide = merged.groupby("slide_id", sort=True)[
        "gt450_target_minus_at2_target_gain"
    ].mean()
    rng = np.random.default_rng(20260917)
    values = slide.to_numpy(dtype=float)
    draws = rng.choice(values, size=(replicates, len(values)), replace=True).mean(
        axis=1
    )
    return {
        "definition": "gain(AT2->GT450) - gain(GT450->AT2); positive favours GT450 target",
        "estimate": float(values.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "physical_slides": int(len(values)),
        "locations": int(len(merged)),
        "bootstrap_replicates": replicates,
    }


def _save_training_plot(
    history: pd.DataFrame, training: pd.DataFrame, path: Path
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for column, direction in enumerate(DIRECTIONS):
        block = history.loc[history["direction"] == direction]
        for fold, fold_block in block.groupby("test_fold"):
            axes[0, column].plot(
                fold_block["completed_passes"],
                fold_block["validation_l1_normalized"],
                alpha=0.75,
                label=f"fold {fold}",
            )
            axes[1, column].plot(
                fold_block["completed_passes"],
                fold_block["validation_target_gradient_ncc"],
                alpha=0.75,
            )
        selected = training.loc[training["direction"] == direction]
        axes[0, column].scatter(
            selected["best_pass"],
            selected["best_validation_l1"],
            color="black",
            s=24,
            zorder=5,
        )
        axes[0, column].set_title(DISPLAY[direction])
        axes[0, column].set_ylabel("Validation target L1")
        axes[1, column].set_ylabel("Validation target gradient NCC")
        axes[1, column].set_xlabel("Training pass")
        axes[0, column].axvline(100, color="#999999", linestyle="--", linewidth=1)
        axes[1, column].axvline(100, color="#999999", linestyle="--", linewidth=1)
    axes[0, 0].legend(frameon=False, ncol=2)
    figure.suptitle("Full Pix2Pix training and UNI-blind checkpoint selection")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _errorbar_values(rows: list[pd.Series]) -> tuple[np.ndarray, np.ndarray]:
    estimates = np.asarray([float(row["estimate"]) for row in rows])
    lower = np.asarray([float(row["ci_low"]) for row in rows])
    upper = np.asarray([float(row["ci_high"]) for row in rows])
    return estimates, np.vstack((estimates - lower, upper - estimates))


def _save_image_endpoint_plot(image: pd.DataFrame, path: Path) -> None:
    labels = [DISPLAY[value] for value in DIRECTIONS]
    x = np.arange(2)
    figure, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    axes = axes.ravel()
    panels = (
        ("target_l1_unit", "Target L1 (lower is better)", 1.0),
        ("target_ssim", "Target SSIM (higher is better)", 1.0),
    )
    colors = ["#315a78", "#b75d3e"]
    for axis, (metric, title, scale) in zip(axes[:2], panels):
        rows = [
            _metric(image, direction, "primary_translation", metric)
            for direction in DIRECTIONS
        ]
        estimates, errors = _errorbar_values(rows)
        axis.bar(x, estimates * scale, color=colors)
        axis.errorbar(
            x,
            estimates * scale,
            yerr=errors * scale,
            color="black",
            capsize=4,
            linestyle="none",
        )
        axis.set(title=title, xticks=x, xticklabels=labels)

    raw = [
        _metric(
            image,
            direction,
            "primary_translation",
            "raw_source_target_gradient_ncc",
        )
        for direction in DIRECTIONS
    ]
    method = [
        _metric(image, direction, "primary_translation", "target_gradient_ncc")
        for direction in DIRECTIONS
    ]
    raw_estimate, raw_error = _errorbar_values(raw)
    method_estimate, method_error = _errorbar_values(method)
    width = 0.34
    axes[2].bar(x - width / 2, raw_estimate, width, label="Raw source")
    axes[2].bar(x + width / 2, method_estimate, width, label="Pix2Pix")
    axes[2].errorbar(
        x - width / 2,
        raw_estimate,
        yerr=raw_error,
        color="black",
        capsize=3,
        linestyle="none",
    )
    axes[2].errorbar(
        x + width / 2,
        method_estimate,
        yerr=method_error,
        color="black",
        capsize=3,
        linestyle="none",
    )
    axes[2].set(
        title="Gradient agreement with real target",
        xticks=x,
        xticklabels=labels,
    )
    axes[2].legend(frameon=False)

    for axis, metric, title in (
        (axes[3], "invented_edge_fraction", "Invented-edge fraction (%)"),
        (axes[4], "saturation_fraction", "Saturated-pixel fraction (%)"),
    ):
        rows = [
            _metric(image, direction, "primary_translation", metric)
            for direction in DIRECTIONS
        ]
        estimates, errors = _errorbar_values(rows)
        axis.bar(x, estimates * 100.0, color=colors)
        axis.errorbar(
            x,
            estimates * 100.0,
            yerr=errors * 100.0,
            color="black",
            capsize=4,
            linestyle="none",
        )
        axis.set(title=title, xticks=x, xticklabels=labels)
    axes[3].axhline(
        0.1,
        color="#a32020",
        linestyle="--",
        linewidth=1.25,
        label="Safety gate (0.1%)",
    )
    axes[3].legend(frameon=False, loc="upper right")
    axes[4].text(
        0.98,
        0.96,
        "Safety gate: 10% (off-scale)",
        transform=axes[4].transAxes,
        ha="right",
        va="top",
        color="#a32020",
        fontsize=9,
    )

    identity = [
        _metric(image, direction, "target_identity_damage", "target_l1_unit")
        for direction in DIRECTIONS
    ]
    identity_estimate, identity_error = _errorbar_values(identity)
    axes[5].bar(x, identity_estimate, color=colors)
    axes[5].errorbar(
        x,
        identity_estimate,
        yerr=identity_error,
        color="black",
        capsize=4,
        linestyle="none",
    )
    axes[5].set(
        title="Target-input identity L1 (lower is better)",
        xticks=x,
        xticklabels=labels,
    )
    for axis in axes:
        axis.tick_params(axis="x", rotation=12)
    figure.suptitle("Held-out image-level fidelity, safety, and identity")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_uni_endpoint_plot(uni: pd.DataFrame, path: Path) -> None:
    labels = [DISPLAY[value] for value in DIRECTIONS]
    x = np.arange(2)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    raw = [
        _metric(uni, direction, "primary_translation", "raw_to_target_distance")
        for direction in DIRECTIONS
    ]
    method = [
        _metric(
            uni, direction, "primary_translation", "method_to_target_distance"
        )
        for direction in DIRECTIONS
    ]
    width = 0.34
    axes[0].bar(x - width / 2, [v.estimate for v in raw], width, label="Raw source")
    axes[0].bar(x + width / 2, [v.estimate for v in method], width, label="Pix2Pix")
    axes[0].set(title="Frozen UNI distance to target", xticks=x, xticklabels=labels)
    axes[0].legend(frameon=False)
    gain = [
        _metric(uni, direction, "primary_translation", "gain_to_target")
        for direction in DIRECTIONS
    ]
    axes[1].bar(x, [v.estimate for v in gain], color=["#315a78", "#b75d3e"])
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set(
        title="UNI gain (positive is improvement)", xticks=x, xticklabels=labels
    )
    identity = [
        _metric(
            uni, direction, "target_identity_damage", "method_to_target_distance"
        )
        for direction in DIRECTIONS
    ]
    axes[2].bar(x, [v.estimate for v in identity], color=["#315a78", "#b75d3e"])
    axes[2].set(title="Target identity damage", xticks=x, xticklabels=labels)
    for axis in axes:
        axis.tick_params(axis="x", rotation=12)
    figure.suptitle("Frozen-UNI bridge analysis (secondary to image safety)")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _find_prediction_shard(
    root: Path, direction: str, fold: int, slide_id: str
) -> Path:
    manifest = json.loads(
        (
            root
            / "02_predictions"
            / direction
            / f"fold_{fold}"
            / "prediction_manifest.json"
        ).read_text()
    )
    matches = [
        Path(row["path"])
        for row in manifest["shards"]
        if str(row["slide_id"]) == slide_id
    ]
    if len(matches) != 1:
        raise ValueError(f"cannot resolve one shard for {direction}/{fold}/{slide_id}")
    return matches[0]


def _save_gallery(root: Path, path: Path) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(13, 7))
    for row, direction in enumerate(DIRECTIONS):
        metrics = pd.read_csv(
            root / "03_image_evaluation" / direction / "location_metrics.csv.gz"
        )
        median = metrics.iloc[
            (metrics["target_l1_unit"] - metrics["target_l1_unit"].median())
            .abs()
            .argmin()
        ]
        shard_path = _find_prediction_shard(
            root, direction, int(median["fold"]), str(median["slide_id"])
        )
        with h5py.File(shard_path, "r") as store:
            locations = np.asarray(store["metadata/location_index"], dtype=int)
            index = int(np.flatnonzero(locations == int(median["location_index"]))[0])
            source = np.asarray(store["aligned_valid/raw_source_valid"][index])
            generated = np.asarray(store["aligned_valid/generated_valid"][index])
            target = np.asarray(store["aligned_valid/real_target_valid"][index])
        residual = np.abs(generated.astype(np.int16) - target.astype(np.int16)).astype(
            np.uint8
        )
        for column, (value, title) in enumerate(
            zip(
                (source, generated, target, residual),
                ("Raw source", "Pix2Pix", "Real target", "|Error|"),
            )
        ):
            axes[row, column].imshow(value)
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(title)
        axes[row, 0].text(
            -0.055,
            0.5,
            DISPLAY[direction],
            transform=axes[row, 0].transAxes,
            rotation=90,
            ha="right",
            va="center",
            fontsize=11,
            fontweight="bold",
        )
    figure.suptitle("Representative held-out translations (median target L1)")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def summarize(input_root: Path, output_root: Path) -> dict[str, Any]:
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    training, history = _training_summary(input_root)
    image, uni = _collect_endpoint_summaries(input_root)
    comparison = _bootstrap_paired_direction_gain(input_root)
    training_path = output_root / "training_summary.csv"
    image_path = output_root / "image_summary.csv"
    uni_path = output_root / "uni_summary.csv"
    training.to_csv(training_path, index=False)
    image.to_csv(image_path, index=False)
    uni.to_csv(uni_path, index=False)
    training_plot = output_root / "training_convergence.png"
    endpoint_plot = output_root / "bidirectional_endpoints.png"
    uni_endpoint_plot = output_root / "bidirectional_uni_endpoints.png"
    gallery_plot = output_root / "bidirectional_gallery.png"
    _save_training_plot(history, training, training_plot)
    _save_image_endpoint_plot(image, endpoint_plot)
    _save_uni_endpoint_plot(uni, uni_endpoint_plot)
    _save_gallery(input_root, gallery_plot)

    lines = [
        "# Full Pix2Pix bidirectional experiment",
        "",
        "두 방향 모두 5-fold, 200-pass 학습을 완료했고 checkpoint는 UNI를 보지 않은 상태에서 validation target L1로 선택했다.",
        "",
    ]
    for direction in DIRECTIONS:
        gain = _metric(uni, direction, "primary_translation", "gain_to_target")
        raw = _metric(uni, direction, "primary_translation", "raw_to_target_distance")
        method = _metric(
            uni, direction, "primary_translation", "method_to_target_distance"
        )
        identity = _metric(
            uni, direction, "target_identity_damage", "method_to_target_distance"
        )
        image_l1 = _metric(
            image, direction, "primary_translation", "target_l1_unit"
        )
        image_ssim = _metric(
            image, direction, "primary_translation", "target_ssim"
        )
        target_gradient = _metric(
            image, direction, "primary_translation", "target_gradient_ncc"
        )
        target_gradient_gain = _metric(
            image, direction, "primary_translation", "target_gradient_gain"
        )
        identity_l1 = _metric(
            image, direction, "target_identity_damage", "target_l1_unit"
        )
        saturation = _metric(
            image, direction, "primary_translation", "saturation_fraction"
        )
        invented = _metric(
            image, direction, "primary_translation", "invented_edge_fraction"
        )
        safety_pass = bool(invented["all_available_safety_gates_pass"])
        lines.extend(
            [
                f"## {DISPLAY[direction]}",
                "",
                f"- image target L1 / SSIM: {image_l1.estimate:.4f} / {image_ssim.estimate:.4f}",
                f"- target-gradient NCC: {target_gradient.estimate:.4f} (raw 대비 {target_gradient_gain.estimate:+.4f})",
                f"- invented-edge / saturation: {invented.estimate:.6f} / {saturation.estimate:.6f}",
                f"- available image-safety gates: {'PASS' if safety_pass else 'FAIL'}",
                f"- target-input identity image L1: {identity_l1.estimate:.4f}",
                f"- UNI target distance: {raw.estimate:.4f} → {method.estimate:.4f}",
                f"- UNI gain: {gain.estimate:.4f} (95% CI {gain.ci_low:.4f}, {gain.ci_high:.4f})",
                f"- UNI target-identity displacement: {identity.estimate:.4f}",
                "",
            ]
        )
    if all(
        _metric(uni, direction, "primary_translation", "gain_to_target").estimate
        < 0
        for direction in DIRECTIONS
    ):
        direction_interpretation = (
            "두 방향 모두 UNI 공간에서 paired target으로부터 멀어졌다. 양의 contrast는 "
            "GT450 target 방향의 악화가 상대적으로 작았다는 뜻이지 성공을 뜻하지 않는다."
        )
    else:
        direction_interpretation = (
            "양수면 GT450 target, 음수면 AT2 target 방향이 UNI 공간을 더 잘 닫았다는 뜻이다."
        )
    lines.extend(
        [
            "## 방향 비교",
            "",
            f"GT450-target gain minus AT2-target gain은 {comparison['estimate']:.4f} "
            f"(95% CI {comparison['ci_low']:.4f}, {comparison['ci_high']:.4f})이다.",
            direction_interpretation,
            "",
            "좋은 UNI 결과도 image-safety 또는 identity-damage 실패를 상쇄하지 않는다.",
        ]
    )
    narrative_path = output_root / "narrative_ko.md"
    narrative_path.write_text("\n".join(lines) + "\n")
    outputs = {
        name: {"path": str(path), "sha256": sha256(path)}
        for name, path in {
            "training_summary": training_path,
            "image_summary": image_path,
            "uni_summary": uni_path,
            "training_convergence": training_plot,
            "bidirectional_endpoints": endpoint_plot,
            "bidirectional_uni_endpoints": uni_endpoint_plot,
            "bidirectional_gallery": gallery_plot,
            "narrative_ko": narrative_path,
        }.items()
    }
    payload = {
        "summary_version": SUMMARY_VERSION,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "directions": list(DIRECTIONS),
        "training_runs": int(len(training)),
        "checkpoint_selection": "minimum inner-validation target L1 every five passes; UNI blind",
        "paired_direction_comparison": comparison,
        "outputs": outputs,
    }
    write_json(output_root / "summary.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            summarize(args.input_root, args.output_root), indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()

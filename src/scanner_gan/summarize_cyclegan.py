#!/usr/bin/env python3
"""Summarize the locked bidirectional CycleGAN experiment without reanalysis."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


SUMMARY_VERSION = "cyclegan_bidirectional_full_summary_v1"
TRAINING_VERSION = "cyclegan_unpaired_bidirectional_v1"
DIRECTIONS = ("gt450_to_at2", "at2_to_gt450")
DISPLAY = {"gt450_to_at2": "GT450 → AT2", "at2_to_gt450": "AT2 → GT450"}


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


def _training_summary(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in range(5):
        manifests = sorted(
            (root / "01_training/runs/gt450_at2" / f"fold_{fold}").glob(
                "seed_*/run_manifest.json"
            )
        )
        if len(manifests) != 1:
            raise ValueError(f"fold {fold}: expected one CycleGAN manifest")
        path = manifests[0]
        manifest = json.loads(path.read_text())
        config = manifest.get("config", {})
        expected = {
            "domain_a": "gt450",
            "domain_b": "at2",
            "test_fold": fold,
            "validation_fold": (fold + 1) % 5,
            "max_passes": 200,
            "constant_passes": 100,
            "selection_every": 5,
            "batch_size": 16,
        }
        observed = {key: config.get(key) for key in expected}
        if (
            manifest.get("status") != "complete"
            or manifest.get("training_version") != TRAINING_VERSION
            or observed != expected
            or int(manifest.get("completed_passes", -1)) != 200
        ):
            raise ValueError(f"fold {fold}: full CycleGAN contract differs: {observed}")
        history = [
            json.loads(line)
            for line in (path.parent / "metrics.jsonl").read_text().splitlines()
            if line.strip()
        ]
        candidates = [item for item in history if isinstance(item.get("validation"), Mapping)]
        if not candidates:
            raise ValueError(f"fold {fold}: no pair-blind validation checkpoints")
        best = min(
            candidates,
            key=lambda item: (
                float(item["validation"]["marginal_score"]),
                int(item["completed_passes"]),
            ),
        )
        best_score = float(best["validation"]["marginal_score"])
        if not np.isclose(
            best_score,
            float(manifest["best_validation_marginal_score"]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"fold {fold}: selected marginal score differs")
        accidental_slide = sum(
            int(item.get("accidental_same_slide_pairs", 0)) for item in history
        )
        accidental_location = sum(
            int(item.get("accidental_same_location_pairs", 0)) for item in history
        )
        if accidental_slide or accidental_location:
            raise ValueError(f"fold {fold}: paired information entered unpaired training")
        checkpoint = Path(manifest["best_image_only_checkpoint"]).resolve()
        rows.append(
            {
                "test_fold": fold,
                "best_pass": int(best["completed_passes"]),
                "best_validation_marginal_score": best_score,
                "best_validation_gt450_to_at2_marginal": float(
                    best["validation"]["a_to_b_marginal"]["distance"]
                ),
                "best_validation_at2_to_gt450_marginal": float(
                    best["validation"]["b_to_a_marginal"]["distance"]
                ),
                "accidental_same_slide_pairs": accidental_slide,
                "accidental_same_location_pairs": accidental_location,
                "completed_passes": int(manifest["completed_passes"]),
                "global_step": int(manifest["global_step"]),
                "elapsed_seconds": float(manifest["elapsed_seconds"]),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
            }
        )
    return pd.DataFrame(rows).sort_values("test_fold", kind="stable")


def _validate_evaluation_manifest(
    manifest: Mapping[str, Any], *, direction: str, role: str
) -> None:
    source, target = direction.split("_to_", maxsplit=1)
    expected_input = target if role == "target_identity_damage" else source
    if not (
        manifest.get("status") == "complete"
        and manifest.get("method") == "cyclegan"
        and manifest.get("source_scanner") == source
        and manifest.get("target_scanner") == target
        and manifest.get("analysis_role") == role
        and manifest.get("inference_input_scanner") == expected_input
    ):
        raise ValueError(f"evaluation provenance differs for {direction}/{role}")


def _collect_internal(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    image_frames: list[pd.DataFrame] = []
    uni_frames: list[pd.DataFrame] = []
    for role, image_subdir, uni_subdir in (
        ("primary_translation", "03_image_evaluation", "04_uni"),
        (
            "target_identity_damage",
            "03_identity_image_evaluation",
            "04_identity_uni",
        ),
    ):
        for direction in DIRECTIONS:
            for subdir, destination in (
                (image_subdir, image_frames),
                (uni_subdir, uni_frames),
            ):
                result = root / subdir / direction
                manifest = json.loads((result / "evaluation_manifest.json").read_text())
                _validate_evaluation_manifest(manifest, direction=direction, role=role)
                summary_path = result / "summary.csv"
                record = manifest.get("outputs", {}).get("summary", {})
                if (
                    Path(record.get("path", "")).resolve() != summary_path.resolve()
                    or record.get("sha256") != sha256(summary_path)
                ):
                    raise ValueError(f"summary hash differs for {result}")
                frame = pd.read_csv(summary_path)
                if set(frame["method"].astype(str)) != {"cyclegan"}:
                    raise ValueError(f"method label differs for {result}")
                frame.insert(0, "direction", direction)
                frame.insert(1, "analysis_role", role)
                if "all_available_safety_gates_pass" in manifest:
                    frame.insert(
                        2,
                        "all_available_safety_gates_pass",
                        bool(manifest["all_available_safety_gates_pass"]),
                    )
                destination.append(frame)
    return pd.concat(image_frames, ignore_index=True), pd.concat(
        uni_frames, ignore_index=True
    )


def _metric(
    frame: pd.DataFrame, direction: str, role: str, metric: str
) -> pd.Series:
    selected = frame.loc[
        frame["direction"].eq(direction)
        & frame["analysis_role"].eq(role)
        & frame["metric"].eq(metric)
    ]
    if len(selected) != 1:
        raise ValueError(f"expected one {direction}/{role}/{metric} row")
    return selected.iloc[0]


def _external_metric(
    frame: pd.DataFrame,
    direction: str,
    metric: str,
    *,
    quadrant: str | None = None,
) -> pd.Series:
    selected = frame.loc[
        frame["direction"].eq(direction) & frame["metric"].eq(metric)
    ]
    if quadrant is not None:
        selected = selected.loc[selected["quadrant"].eq(quadrant)]
    if len(selected) != 1:
        raise ValueError(f"expected one external {direction}/{quadrant}/{metric} row")
    return selected.iloc[0]


def _compact_results(
    image: pd.DataFrame,
    uni: pd.DataFrame,
    external_translation: pd.DataFrame,
    external_identity: pd.DataFrame,
    external_gates: pd.DataFrame,
) -> pd.DataFrame:
    def as_bool(value: object) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        return str(value).strip().casefold() in {"true", "1", "yes"}

    rows = []
    for direction in DIRECTIONS:
        internal_gate = as_bool(
            _metric(
                image, direction, "primary_translation", "target_l1_unit"
            )["all_available_safety_gates_pass"]
        )
        external_gate = bool(
            external_gates.loc[external_gates["direction"].eq(direction), "pass"]
            .map(as_bool)
            .all()
        )
        internal_gain = _metric(
            uni, direction, "primary_translation", "gain_to_target"
        )
        external_gain = _external_metric(
            external_translation,
            direction,
            "gain_to_target",
            quadrant="overall",
        )
        rows.append(
            {
                "direction": direction,
                "internal_target_l1": _metric(
                    image, direction, "primary_translation", "target_l1_unit"
                ).estimate,
                "internal_target_ssim": _metric(
                    image, direction, "primary_translation", "target_ssim"
                ).estimate,
                "internal_raw_gradient_ncc": _metric(
                    image,
                    direction,
                    "primary_translation",
                    "raw_source_target_gradient_ncc",
                ).estimate,
                "internal_method_gradient_ncc": _metric(
                    image, direction, "primary_translation", "target_gradient_ncc"
                ).estimate,
                "internal_identity_l1": _metric(
                    image, direction, "target_identity_damage", "target_l1_unit"
                ).estimate,
                "internal_raw_uni_distance": _metric(
                    uni, direction, "primary_translation", "raw_to_target_distance"
                ).estimate,
                "internal_method_uni_distance": _metric(
                    uni,
                    direction,
                    "primary_translation",
                    "method_to_target_distance",
                ).estimate,
                "internal_uni_gain": internal_gain.estimate,
                "internal_uni_gain_ci_low": internal_gain.ci_low,
                "internal_uni_gain_ci_high": internal_gain.ci_high,
                "internal_safety_gates_pass": internal_gate,
                "external_target_l1": _external_metric(
                    external_translation,
                    direction,
                    "target_l1_unit",
                    quadrant="overall",
                ).estimate,
                "external_raw_uni_distance": _external_metric(
                    external_translation,
                    direction,
                    "raw_to_target_distance",
                    quadrant="overall",
                ).estimate,
                "external_method_uni_distance": _external_metric(
                    external_translation,
                    direction,
                    "method_to_target_distance",
                    quadrant="overall",
                ).estimate,
                "external_uni_gain": external_gain.estimate,
                "external_uni_gain_ci_low": external_gain.ci_low,
                "external_uni_gain_ci_high": external_gain.ci_high,
                "external_identity_l1": _external_metric(
                    external_identity, direction, "target_l1_unit"
                ).estimate,
                "external_safety_gates_pass": external_gate,
            }
        )
    return pd.DataFrame(rows)


def summarize(input_root: Path, output_root: Path) -> dict[str, Any]:
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    external_root = input_root / "10_plism_external_bidirectional/02_aggregate"
    external_summary = json.loads((external_root / "summary.json").read_text())
    if external_summary.get("method") != "cyclegan":
        raise ValueError("external summary is not CycleGAN")
    training = _training_summary(input_root)
    image, uni = _collect_internal(input_root)
    external_translation = pd.read_csv(
        external_root / "translation_ensemble_summary.csv"
    )
    external_identity = pd.read_csv(external_root / "identity_ensemble_summary.csv")
    external_gates = pd.read_csv(external_root / "external_image_safety_gates.csv")
    compact = _compact_results(
        image, uni, external_translation, external_identity, external_gates
    )
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "training_summary": output_root / "training_summary.csv",
        "image_summary": output_root / "image_summary.csv",
        "uni_summary": output_root / "uni_summary.csv",
        "compact_results": output_root / "compact_results.csv",
    }
    training.to_csv(paths["training_summary"], index=False)
    image.to_csv(paths["image_summary"], index=False)
    uni.to_csv(paths["uni_summary"], index=False)
    compact.to_csv(paths["compact_results"], index=False)

    lines = [
        "# Full unpaired CycleGAN bidirectional experiment",
        "",
        "두 scanner domain은 slide split 안에서 독립적으로 표본추출했고, 같은 slide와 같은 위치의 우연한 pairing은 0건이었다. 다섯 모델은 200 pass를 완료했으며 checkpoint는 paired target과 UNI를 보지 않은 marginal image score로 선택했다.",
        "",
    ]
    for row in compact.itertuples(index=False):
        lines.extend(
            [
                f"## {DISPLAY[row.direction]}",
                "",
                f"- internal image L1 / SSIM: {row.internal_target_l1:.4f} / {row.internal_target_ssim:.4f}",
                f"- internal target-gradient NCC: {row.internal_raw_gradient_ncc:.4f} → {row.internal_method_gradient_ncc:.4f}",
                f"- internal frozen-UNI distance: {row.internal_raw_uni_distance:.4f} → {row.internal_method_uni_distance:.4f}",
                f"- internal UNI gain: {row.internal_uni_gain:+.4f} (95% CI {row.internal_uni_gain_ci_low:+.4f}, {row.internal_uni_gain_ci_high:+.4f})",
                f"- target-input identity L1: {row.internal_identity_l1:.4f}",
                f"- PLISM frozen-UNI distance: {row.external_raw_uni_distance:.4f} → {row.external_method_uni_distance:.4f}",
                f"- PLISM UNI gain: {row.external_uni_gain:+.4f} (95% CI {row.external_uni_gain_ci_low:+.4f}, {row.external_uni_gain_ci_high:+.4f})",
                "",
            ]
        )
    narrative_path = output_root / "narrative_ko.md"
    narrative_path.write_text("\n".join(lines) + "\n")
    paths["narrative_ko"] = narrative_path
    payload = {
        "summary_version": SUMMARY_VERSION,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_runs": int(len(training)),
        "completed_passes": int(training["completed_passes"].min()),
        "accidental_same_slide_pairs": int(
            training["accidental_same_slide_pairs"].sum()
        ),
        "accidental_same_location_pairs": int(
            training["accidental_same_location_pairs"].sum()
        ),
        "checkpoint_selection": (
            "minimum pair-blind bidirectional marginal image score every five passes; "
            "same-location target and UNI blind"
        ),
        "all_internal_uni_gains_negative": bool(
            (compact["internal_uni_gain"] < 0).all()
        ),
        "all_external_uni_gains_negative": bool(
            (compact["external_uni_gain"] < 0).all()
        ),
        "outputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in paths.items()
        },
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

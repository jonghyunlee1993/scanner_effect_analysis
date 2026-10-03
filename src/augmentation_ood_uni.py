#!/usr/bin/env python3
"""Representative frozen-UNI bridge for the augmentation reachability study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from skimage.color import rgb2hed
import torch
import torch.nn.functional as F

from augmentation_ood_study import (
    ANALYSIS_VERSION,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    OUTPUT_ROOT,
    REPLICATE_GROUPS,
    SOURCE_ROOT,
    TARGET_SCANNERS,
    augmentation_seed,
    load_libraries,
    render_augmentation,
    sha256,
    stable_seed,
    write_frame,
    write_json,
)
from prenorm.embedding import load_uni


PFM_VERSION = "augmentation_ood_uni_v1"
PFM_LOCATIONS_PER_SLIDE = 20
PFM_ARMS = (
    "identity",
    "default_oracle",
    "strong_global",
    "strong_unconstrained_oracle",
    "strong_oracle",
)
FEATURE_DIM = 1024


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def choose_pfm_locations(
    slide_id: str,
    selected: pd.DataFrame,
    source_root: Path,
) -> list[int]:
    available = sorted(
        selected[
            (selected["slide_id"] == slide_id)
            & (selected["scanner"] == TARGET_SCANNERS[0])
            & (selected["arm"] == "strong_oracle")
        ]["location_index"].astype(int).unique()
    )
    metrics_path = source_root / "02_image_phenotypes/shards" / slide_id / "location_metrics.h5"
    with h5py.File(metrics_path, "r") as store:
        replicate = np.asarray(store["replicate_id"], dtype=int)
    chosen = []
    for group in range(REPLICATE_GROUPS):
        group_locations = [index for index in available if replicate[index] == group]
        if len(group_locations) < PFM_LOCATIONS_PER_SLIDE // REPLICATE_GROUPS:
            raise ValueError(f"{slide_id}: too few PFM locations in replicate {group}")
        chosen.extend(group_locations[: PFM_LOCATIONS_PER_SLIDE // REPLICATE_GROUPS])
    return sorted(chosen)


@torch.inference_mode()
def embed_images(
    model,
    images: np.ndarray,
    size: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    outputs = []
    for start in range(0, len(images), batch_size):
        stop = min(start + batch_size, len(images))
        tensor = (
            torch.from_numpy(images[start:stop].astype(np.float32))
            .permute(0, 3, 1, 2)
            .div_(255.0)
            .to(device, non_blocking=True)
        )
        tensor = F.interpolate(
            tensor,
            size=(size, size),
            mode="bicubic",
            align_corners=False,
        )
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model((tensor - mean) / std)
        outputs.append(F.normalize(output.float(), dim=1).cpu().numpy())
    result = np.concatenate(outputs, axis=0).astype(np.float32)
    if result.shape != (len(images), FEATURE_DIM) or not np.isfinite(result).all():
        raise ValueError(f"invalid UNI embedding output {result.shape}")
    return result


def condition_names() -> list[str]:
    values = ["source"]
    values.extend(f"target:{scanner}" for scanner in TARGET_SCANNERS)
    for arm in PFM_ARMS[1:]:
        values.extend(f"{arm}:{scanner}" for scanner in TARGET_SCANNERS)
    return values


def extract_one(
    row,
    selected: pd.DataFrame,
    libraries: dict[str, np.ndarray],
    source_root: Path,
    output_root: Path,
    model,
    size: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> None:
    slide_id = str(row.slide_id)
    output = output_root / "03_pfm_uni/shards" / f"{slide_id}.h5"
    summary_path = output.with_suffix(".summary.json")
    selected_path = output_root / "02_aggregate/selected_candidates.csv"
    if output.exists() and summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if (
            summary.get("pfm_version") == PFM_VERSION
            and summary.get("status") == "pass"
            and summary.get("selected_candidates_sha256") == sha256(selected_path)
            and summary.get("output_sha256") == sha256(output)
        ):
            print(json.dumps({"status": "already_complete", "slide_id": slide_id}))
            return

    locations = choose_pfm_locations(slide_id, selected, source_root)
    cache_path = Path(row.cache_path)
    with h5py.File(cache_path, "r") as store:
        rgb = np.asarray(store["images"][locations], dtype=np.uint8)
        source_index = np.asarray(store["source_index"][locations], dtype=np.int64)
    names = condition_names()
    images = np.empty(
        (len(locations), len(names), rgb.shape[2], rgb.shape[3], 3), dtype=np.uint8
    )
    images[:, 0] = rgb[:, 0]
    for scanner_index, scanner in enumerate(TARGET_SCANNERS, start=1):
        images[:, scanner_index] = rgb[:, scanner_index]

    slide_selection = selected[selected["slide_id"] == slide_id].copy()
    name_to_index = {name: index for index, name in enumerate(names)}
    for location_offset, location_index in enumerate(locations):
        source = rgb[location_offset, 0]
        source_hed = rgb2hed(source.astype(np.float32) / 255.0).astype(np.float32)
        for scanner in TARGET_SCANNERS:
            for arm in PFM_ARMS[1:]:
                match = slide_selection[
                    (slide_selection["location_index"] == location_index)
                    & (slide_selection["scanner"] == scanner)
                    & (slide_selection["arm"] == arm)
                ]
                if len(match) != 1:
                    raise ValueError(
                        f"{slide_id}/{location_index}/{scanner}/{arm}: {len(match)} selections"
                    )
                match = match.iloc[0]
                library_name = str(match["library"])
                candidate_index = int(match["candidate_index"])
                images[location_offset, name_to_index[f"{arm}:{scanner}"]] = render_augmentation(
                    source,
                    source_hed,
                    libraries[library_name][candidate_index],
                    augmentation_seed(
                        slide_id,
                        location_index,
                        libraries[library_name][candidate_index],
                    ),
                )

    flat = images.reshape(-1, *images.shape[2:])
    features = embed_images(model, flat, size, mean, std, device, batch_size)
    features = features.reshape(len(locations), len(names), FEATURE_DIM)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    with h5py.File(temporary, "w") as store:
        store.attrs["pfm_version"] = PFM_VERSION
        store.attrs["slide_id"] = slide_id
        store.attrs["tissue_type"] = str(row.tissue_type)
        store.attrs["encoder_id"] = "uni_v1"
        store.create_dataset("features", data=features, compression="lzf")
        store.create_dataset("condition_names", data=np.asarray(names, dtype="S40"))
        store.create_dataset("location_index", data=np.asarray(locations, dtype=np.int64))
        store.create_dataset("source_index", data=source_index)
    temporary.replace(output)
    summary = {
        "pfm_version": PFM_VERSION,
        "status": "pass",
        "completed_utc": utc_now(),
        "slide_id": slide_id,
        "tissue_type": str(row.tissue_type),
        "encoder_id": "uni_v1",
        "locations": len(locations),
        "conditions": names,
        "features": int(features.shape[0] * features.shape[1]),
        "feature_dim": FEATURE_DIM,
        "minimum_norm": float(np.linalg.norm(features, axis=2).min()),
        "maximum_norm": float(np.linalg.norm(features, axis=2).max()),
        "selected_candidates_sha256": sha256(selected_path),
        "output": str(output.resolve()),
        "output_sha256": sha256(output),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)


def extract_task(
    task_index: int,
    task_count: int,
    max_slides: int,
    source_root: Path,
    output_root: Path,
    batch_size: int,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("UNI extraction requires CUDA")
    cohort = pd.read_csv(output_root / "00_contract/cohort.csv", dtype={"slide_id": str})
    assigned = cohort.iloc[task_index::task_count]
    if max_slides > 0:
        assigned = assigned.head(max_slides)
    selected = pd.read_csv(
        output_root / "02_aggregate/selected_candidates.csv", dtype={"slide_id": str}
    )
    libraries = load_libraries(output_root)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    model, size, mean, std = load_uni(device)
    for row in assigned.itertuples(index=False):
        extract_one(
            row,
            selected,
            libraries,
            source_root,
            output_root,
            model,
            size,
            mean,
            std,
            device,
            batch_size,
        )


def bootstrap_mean(frame: pd.DataFrame, column: str, seed: int) -> tuple[float, float, float]:
    values = frame.groupby("slide_id")[column].mean().to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(BOOTSTRAP_REPLICATES, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def aggregate(output_root: Path) -> None:
    cohort = pd.read_csv(output_root / "00_contract/cohort.csv", dtype={"slide_id": str})
    folds = pd.read_csv(output_root / "00_contract/slide_folds.csv", dtype={"slide_id": str})
    fold_map = dict(zip(folds["slide_id"], folds["fold"]))
    records = []
    names = condition_names()
    for row in cohort.itertuples(index=False):
        path = output_root / "03_pfm_uni/shards" / f"{row.slide_id}.h5"
        summary_path = path.with_suffix(".summary.json")
        if not path.exists() or not summary_path.exists():
            raise FileNotFoundError(path)
        summary = json.loads(summary_path.read_text())
        if summary.get("status") != "pass" or summary.get("output_sha256") != sha256(path):
            raise ValueError(f"invalid UNI shard {path}")
        with h5py.File(path, "r") as store:
            features = np.asarray(store["features"], dtype=np.float32)
            locations = np.asarray(store["location_index"], dtype=int)
            observed_names = [value.decode() for value in store["condition_names"][:]]
        if observed_names != names:
            raise ValueError(f"condition order mismatch: {row.slide_id}")
        index = {name: position for position, name in enumerate(names)}
        source = features[:, index["source"]]
        for scanner in TARGET_SCANNERS:
            target = features[:, index[f"target:{scanner}"]]
            for arm in PFM_ARMS:
                generated = source if arm == "identity" else features[:, index[f"{arm}:{scanner}"]]
                for location_offset, location_index in enumerate(locations):
                    records.append(
                        {
                            "slide_id": str(row.slide_id),
                            "tissue_type": str(row.tissue_type),
                            "fold": int(fold_map[str(row.slide_id)]),
                            "location_index": int(location_index),
                            "scanner": scanner,
                            "arm": arm,
                            "source_target_cosine": float(source[location_offset] @ target[location_offset]),
                            "generated_target_cosine": float(generated[location_offset] @ target[location_offset]),
                            "generated_source_cosine": float(generated[location_offset] @ source[location_offset]),
                            "target_gain": float(
                                generated[location_offset] @ target[location_offset]
                                - source[location_offset] @ target[location_offset]
                            ),
                            **{
                                f"generated_f{feature}": float(generated[location_offset, feature])
                                for feature in range(FEATURE_DIM)
                            },
                            **{
                                f"target_f{feature}": float(target[location_offset, feature])
                                for feature in range(FEATURE_DIM)
                            },
                        }
                    )
    frame = pd.DataFrame(records)
    feature_generated = [f"generated_f{index}" for index in range(FEATURE_DIM)]
    feature_target = [f"target_f{index}" for index in range(FEATURE_DIM)]
    predictions = []
    for scanner in TARGET_SCANNERS:
        for arm in PFM_ARMS:
            subset = frame[(frame["scanner"] == scanner) & (frame["arm"] == arm)]
            for fold in range(5):
                train = subset[subset["fold"] != fold]
                test = subset[subset["fold"] == fold]
                x_train = np.concatenate(
                    [train[feature_generated].to_numpy(np.float32), train[feature_target].to_numpy(np.float32)]
                )
                y_train = np.concatenate([np.zeros(len(train), int), np.ones(len(train), int)])
                x_test = np.concatenate(
                    [test[feature_generated].to_numpy(np.float32), test[feature_target].to_numpy(np.float32)]
                )
                y_test = np.concatenate([np.zeros(len(test), int), np.ones(len(test), int)])
                classifier = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        C=0.02,
                        max_iter=3000,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=BOOTSTRAP_SEED,
                    ),
                )
                classifier.fit(x_train, y_train)
                predicted = classifier.predict(x_test)
                meta = pd.concat(
                    [test[["slide_id", "location_index"]], test[["slide_id", "location_index"]]],
                    ignore_index=True,
                )
                for position, item in meta.reset_index(drop=True).iterrows():
                    predictions.append(
                        {
                            "scanner": scanner,
                            "arm": arm,
                            "fold": fold,
                            "slide_id": str(item.slide_id),
                            "location_index": int(item.location_index),
                            "truth": int(y_test[position]),
                            "prediction": int(predicted[position]),
                            "correct": int(predicted[position] == y_test[position]),
                        }
                    )
    prediction_frame = pd.DataFrame(predictions)

    retrieval_rows = []
    for (slide_id, scanner, arm), subset in frame.groupby(["slide_id", "scanner", "arm"]):
        generated = subset[feature_generated].to_numpy(np.float32)
        target = subset[feature_target].to_numpy(np.float32)
        similarity = generated @ target.T
        top1 = np.argmax(similarity, axis=1) == np.arange(len(subset))
        retrieval_rows.append(
            {
                "slide_id": slide_id,
                "scanner": scanner,
                "arm": arm,
                "same_location_top1": float(top1.mean()),
            }
        )
    retrieval = pd.DataFrame(retrieval_rows)

    summaries = []
    for scanner in (*TARGET_SCANNERS, "all"):
        for arm in PFM_ARMS:
            subset = frame[frame["arm"] == arm]
            prediction = prediction_frame[prediction_frame["arm"] == arm]
            retrieval_subset = retrieval[retrieval["arm"] == arm]
            if scanner != "all":
                subset = subset[subset["scanner"] == scanner]
                prediction = prediction[prediction["scanner"] == scanner]
                retrieval_subset = retrieval_subset[retrieval_subset["scanner"] == scanner]
            cosine = bootstrap_mean(
                subset, "generated_target_cosine", stable_seed("uni", scanner, arm, "cosine")
            )
            fidelity = bootstrap_mean(
                subset, "generated_source_cosine", stable_seed("uni", scanner, arm, "fidelity")
            )
            gain = bootstrap_mean(
                subset, "target_gain", stable_seed("uni", scanner, arm, "gain")
            )
            accuracy_by_slide = prediction.groupby("slide_id")["correct"].mean().reset_index()
            accuracy = bootstrap_mean(
                accuracy_by_slide, "correct", stable_seed("uni", scanner, arm, "accuracy")
            )
            top1 = bootstrap_mean(
                retrieval_subset,
                "same_location_top1",
                stable_seed("uni", scanner, arm, "top1"),
            )
            summaries.append(
                {
                    "encoder_id": "uni_v1",
                    "scanner": scanner,
                    "arm": arm,
                    "generated_target_cosine": cosine[0],
                    "generated_target_ci_low": cosine[1],
                    "generated_target_ci_high": cosine[2],
                    "generated_source_cosine": fidelity[0],
                    "generated_source_ci_low": fidelity[1],
                    "generated_source_ci_high": fidelity[2],
                    "target_gain": gain[0],
                    "target_gain_ci_low": gain[1],
                    "target_gain_ci_high": gain[2],
                    "real_vs_augmented_balanced_accuracy": accuracy[0],
                    "classifier_ci_low": accuracy[1],
                    "classifier_ci_high": accuracy[2],
                    "same_location_top1": top1[0],
                    "top1_ci_low": top1[1],
                    "top1_ci_high": top1[2],
                }
            )
    summary_frame = pd.DataFrame(summaries)
    compact = frame.drop(columns=feature_generated + feature_target)
    write_frame(output_root / "03_pfm_uni/location_metrics.csv", compact)
    write_frame(output_root / "03_pfm_uni/classifier_predictions.csv", prediction_frame)
    write_frame(output_root / "03_pfm_uni/retrieval_by_slide.csv", retrieval)
    write_frame(output_root / "03_pfm_uni/summary.csv", summary_frame)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    colors = {
        "identity": "#8c9698",
        "default_oracle": "#175d72",
        "strong_global": "#d19a52",
        "strong_unconstrained_oracle": "#7c3aed",
        "strong_oracle": "#a63d40",
    }
    x = np.arange(len(TARGET_SCANNERS))
    width = 0.17
    for offset, arm in enumerate(PFM_ARMS):
        part = summary_frame[
            (summary_frame["scanner"] != "all") & (summary_frame["arm"] == arm)
        ].set_index("scanner").reindex(TARGET_SCANNERS)
        position = x + (offset - 2.0) * width
        axes[0].errorbar(
            position,
            part["target_gain"],
            yerr=np.vstack(
                [
                    part["target_gain"] - part["target_gain_ci_low"],
                    part["target_gain_ci_high"] - part["target_gain"],
                ]
            ),
            fmt="o",
            capsize=3,
            color=colors[arm],
            label=arm.replace("_", " "),
        )
        axes[1].errorbar(
            position,
            part["real_vs_augmented_balanced_accuracy"],
            yerr=np.vstack(
                [
                    part["real_vs_augmented_balanced_accuracy"] - part["classifier_ci_low"],
                    part["classifier_ci_high"] - part["real_vs_augmented_balanced_accuracy"],
                ]
            ),
            fmt="o",
            capsize=3,
            color=colors[arm],
        )
    axes[0].axhline(0, color="#5a6b70", ls="--", lw=1)
    axes[0].set_ylabel("Gain in cosine similarity to real target")
    axes[0].set_title("A  Does image-space mimicry move UNI toward target?", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].axhline(0.5, color="#5a6b70", ls="--", lw=1)
    axes[1].set_ylabel("Held-out balanced accuracy")
    axes[1].set_title("B  Real target vs augmented AT2", loc="left", fontweight="bold")
    for axis in axes:
        axis.set_xticks(x, [item.upper() for item in TARGET_SCANNERS])
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Frozen UNI v1 augmentation-support bridge", fontsize=15, fontweight="bold")
    figure.savefig(output_root / "03_pfm_uni/figure01_uni_bridge.png", dpi=180)
    plt.close(figure)

    pooled = summary_frame[summary_frame["scanner"] == "all"].set_index("arm")
    result = {
        "pfm_version": PFM_VERSION,
        "status": "pass",
        "completed_utc": utc_now(),
        "encoder_id": "uni_v1",
        "slides": len(cohort),
        "locations_per_slide": PFM_LOCATIONS_PER_SLIDE,
        "strong_oracle": {
            key: float(pooled.loc["strong_oracle", key])
            for key in (
                "generated_target_cosine",
                "generated_source_cosine",
                "target_gain",
                "real_vs_augmented_balanced_accuracy",
                "same_location_top1",
            )
        },
        "claim_boundary": (
            "Frozen UNI is a representative feature-space consequence test.  It does not "
            "identify the causal pretraining recipe or generalize automatically to every PFM."
        ),
    }
    write_json(output_root / "03_pfm_uni/summary.json", result)
    print(json.dumps(result, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract-task")
    extract.add_argument("--task-index", type=int, required=True)
    extract.add_argument("--task-count", type=int, default=8)
    extract.add_argument("--max-slides", type=int, default=0)
    extract.add_argument("--batch-size", type=int, default=32)
    commands.add_parser("aggregate")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "extract-task":
        extract_task(
            args.task_index,
            args.task_count,
            args.max_slides,
            args.source_root.resolve(),
            args.output_root.resolve(),
            args.batch_size,
        )
    elif args.command == "aggregate":
        aggregate(args.output_root.resolve())
    else:
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

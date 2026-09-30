#!/usr/bin/env python3
"""RV-P0d (iii): restore and re-run the GT450 edge-constraint (structure) ablation chain.

Frozen output: `outputs/discussion_followup_2026-09-25/structure_ablation/` (01 training,
02 predictions, 03 image evaluation, 04 UNI, 05 comparison). The discussion-specific
producers were moved to `.Trash/2026-09-29_paper_code_cleanup/`; they are restored verbatim
as `restore_structure_ablation_{train,compare,highband}.py`. The generic stages use the
unchanged `src/scanner_gan/` modules with the arguments of the original sbatch files
(`.Trash/2026-09-29_paper_code_cleanup/scripts/discussion_structure_*.sbatch`).

Stages (all outputs under `results/provenance_restoration/structure_ablation/`):

  predict --fold K     GPU. Regenerate fold-K predictions from the frozen checkpoint.
  evaluate_images      CPU. `scanner_gan.evaluate_images` on the regenerated predictions.
  evaluate_uni         GPU. `scanner_gan.evaluate_uni` on the regenerated predictions.
  compare              CPU. Run the recovered comparison and high-band audit twice:
                       (a) on the frozen 02-04 intermediates (read through symlinks in
                       `from_frozen_intermediates/`), expected byte-identical to frozen 05;
                       (b) on the regenerated 02-04 chain in `regenerated/`.
                       Then compare every stage with the frozen files.
  train --fold K       GPU, about 1 h per fold. Recovered training code; NOT part of the
                       submitted verification (GAN training with TF32/bfloat16 and cuDNN is
                       not bit-reproducible, so a retrain is a replicate, not a restoration).

Nothing under `outputs/` is written. Tolerances for the regenerated chain were fixed before
any regenerated value was seen (see `TOLERANCE`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
FROZEN = PROJECT / "outputs/discussion_followup_2026-09-25/structure_ablation"
OUTPUT = Path(__file__).resolve().parent / "results/provenance_restoration/structure_ablation"
REGENERATED = OUTPUT / "regenerated"
FROM_FROZEN = OUTPUT / "from_frozen_intermediates"
CONTRACT = PROJECT / "outputs/scanner_batch_effect_analysis_2026-09-17/06_learned_baselines/00_contract"
DIRECTION = "at2_to_gt450"
COMPARISON_FILES = ("paired_slide_contrasts.csv", "tissue_retrieval.csv", "highband_per_slide.csv",
                    "highband_summary.csv", "highband_contrast.csv")
# Absolute tolerance on every 05 estimate and CI bound of the regenerated chain: agreement
# at the rounding used when these contrasts are reported (three decimals).
TOLERANCE = 5e-4


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prediction_dirs(root: Path) -> list[Path]:
    return [root / "02_predictions" / DIRECTION / f"fold_{fold}" for fold in range(5)]


def stage_predict(fold: int, workers: int) -> None:
    from scanner_gan.predict import predict

    run_dir = FROZEN / "01_training" / DIRECTION / f"fold_{fold}"
    result = predict(
        checkpoint_path=run_dir / "checkpoints/best_image_only.pt",
        run_manifest_path=run_dir / "run_manifest.json",
        sample_index_path=CONTRACT / "sample_index.csv.gz",
        locked_index_path=CONTRACT / "locked_image_evaluation_index.csv.gz",
        source_scanner="at2", target_scanner="gt450", test_fold=fold,
        output_dir=prediction_dirs(REGENERATED)[fold], batch_size=32,
        num_workers=workers, device="cuda", amp="bfloat16",
    )
    print(json.dumps({"fold": fold, "status": result.get("status"),
                      "slides": result.get("slides")}), flush=True)


def stage_evaluate_images() -> None:
    from scanner_gan.evaluate_images import evaluate

    evaluate(prediction_dirs(REGENERATED), REGENERATED / "03_image_evaluation" / DIRECTION,
             bootstrap_replicates=20_000)


def stage_evaluate_uni() -> None:
    from scanner_gan.evaluate_uni import evaluate

    evaluate(prediction_dirs(REGENERATED), REGENERATED / "04_uni" / DIRECTION,
             batch_size=64, bootstrap_replicates=20_000)


def stage_train(fold: int) -> None:
    import restore_structure_ablation_train as recovered

    recovered.train(fold, REGENERATED / "01_training" / DIRECTION / f"fold_{fold}", 100.0)


def run_recovered_comparison(root: Path) -> None:
    import restore_structure_ablation_compare as compare
    import restore_structure_ablation_highband as highband

    for module in (compare, highband):
        saved = sys.argv
        sys.argv = [module.__name__, "--ablation-root", str(root)]
        try:
            module.main()
        finally:
            sys.argv = saved


def link_frozen_intermediates() -> None:
    FROM_FROZEN.mkdir(parents=True, exist_ok=True)
    for name in ("02_predictions", "03_image_evaluation", "04_uni"):
        link = FROM_FROZEN / name
        if not link.exists():
            os.symlink(FROZEN / name, link, target_is_directory=True)
        if link.resolve() != (FROZEN / name).resolve():
            raise ValueError(f"unexpected link target: {link}")


def compare_frames(new: pd.DataFrame, old: pd.DataFrame, keys: list[str]) -> dict:
    merged = new.merge(old, on=keys, how="outer", suffixes=("_new", "_frozen"),
                       indicator=True, validate="one_to_one")
    result = {"rows_regenerated": int(len(new)), "rows_frozen": int(len(old)),
              "rows_unmatched": int((merged["_merge"] != "both").sum())}
    both = merged[merged["_merge"] == "both"]
    worst, worst_column = 0.0, ""
    for column in new.columns:
        if column in keys or not pd.api.types.is_numeric_dtype(new[column]):
            continue
        delta = float((both[f"{column}_new"] - both[f"{column}_frozen"]).abs().max())
        if np.isfinite(delta) and delta > worst:
            worst, worst_column = delta, column
    result.update({"max_abs_diff": worst, "max_abs_diff_column": worst_column})
    return result


def verify_predictions() -> pd.DataFrame:
    rows = []
    datasets = ("images/generated", "images/raw_source", "images/real_target",
                "aligned_valid/generated_valid", "aligned_valid/raw_source_valid",
                "aligned_valid/real_target_valid", "metadata/location_index")
    for fold, (new_dir, old_dir) in enumerate(zip(prediction_dirs(REGENERATED),
                                                  prediction_dirs(FROZEN))):
        old_files = sorted((old_dir / "slides").glob("*.h5"))
        for old_path in old_files:
            new_path = new_dir / "slides" / old_path.name
            row = {"fold": fold, "shard": old_path.name, "regenerated_exists": new_path.is_file()}
            if not new_path.is_file():
                rows.append(row)
                continue
            row["shard_byte_identical"] = sha256(new_path) == sha256(old_path)
            with h5py.File(new_path, "r") as new, h5py.File(old_path, "r") as old:
                for name in datasets:
                    a = np.asarray(new[name]).astype(np.int32)
                    b = np.asarray(old[name]).astype(np.int32)
                    label = name.split("/")[-1]
                    if a.shape != b.shape:
                        row[f"{label}_shape_equal"] = False
                        continue
                    delta = np.abs(a - b)
                    row[f"{label}_equal"] = bool((delta == 0).all())
                    if label.startswith("generated"):
                        row[f"{label}_max_abs_diff"] = int(delta.max())
                        row[f"{label}_mean_abs_diff"] = float(delta.mean())
                        row[f"{label}_exact_pixel_fraction"] = float((delta == 0).mean())
            rows.append(row)
    return pd.DataFrame(rows)


def verify() -> dict:
    records = []
    # 05 from frozen intermediates: must be byte-identical.
    for name in COMPARISON_FILES:
        new, old = FROM_FROZEN / "05_comparison" / name, FROZEN / "05_comparison" / name
        records.append({"stage": "05_from_frozen_intermediates", "file": name,
                        "byte_identical": sha256(new) == sha256(old)})
    # 03 and 04 regenerated vs frozen.
    for stage in ("03_image_evaluation", "04_uni"):
        for name, keys in (("slide_metrics.csv", ["slide_id"]), ("summary.csv", ["metric"])):
            new = pd.read_csv(REGENERATED / stage / DIRECTION / name)
            old = pd.read_csv(FROZEN / stage / DIRECTION / name)
            records.append({"stage": stage, "file": name,
                            "byte_identical": sha256(REGENERATED / stage / DIRECTION / name)
                            == sha256(FROZEN / stage / DIRECTION / name),
                            **compare_frames(new, old, keys)})
    # 05 regenerated chain vs frozen.
    keys = {"paired_slide_contrasts.csv": ["space", "metric"],
            "tissue_retrieval.csv": [], "highband_per_slide.csv": ["slide_id", "method"],
            "highband_summary.csv": ["method"], "highband_contrast.csv": ["metric"]}
    for name in COMPARISON_FILES:
        new = pd.read_csv(REGENERATED / "05_comparison" / name)
        old = pd.read_csv(FROZEN / "05_comparison" / name)
        if not keys[name]:
            new, old = new.assign(row=0), old.assign(row=0)
        records.append({"stage": "05_regenerated_chain", "file": name,
                        "byte_identical": sha256(REGENERATED / "05_comparison" / name)
                        == sha256(FROZEN / "05_comparison" / name),
                        **compare_frames(new, old, keys[name] or ["row"])})
    table = pd.DataFrame(records)
    table.to_csv(OUTPUT / "verification_stages.csv", index=False)

    predictions = verify_predictions()
    predictions.to_csv(OUTPUT / "verification_predictions.csv", index=False)

    contrasts_new = pd.read_csv(REGENERATED / "05_comparison/paired_slide_contrasts.csv")
    contrasts_old = pd.read_csv(FROZEN / "05_comparison/paired_slide_contrasts.csv")
    paired = contrasts_new.merge(contrasts_old, on=["space", "metric"],
                                 suffixes=("_new", "_frozen"))
    headline = paired[paired.metric.isin(["gain_to_target", "target_ssim", "source_gradient_ncc",
                                          "target_gradient_ncc", "invented_edge_fraction"])]
    headline.to_csv(OUTPUT / "verification_headline_contrasts.csv", index=False)
    columns = ["regularized_minus_baseline", "ci_low", "ci_high"]
    regenerated_max = float(max(
        (paired[f"{c}_new"] - paired[f"{c}_frozen"]).abs().max() for c in columns))
    frozen_rows = table[table.stage == "05_from_frozen_intermediates"]
    result = {
        "recovered_code": ["restore_structure_ablation_compare.py",
                           "restore_structure_ablation_highband.py",
                           "restore_structure_ablation_train.py (not run)"],
        "from_frozen_intermediates_05_byte_identical": bool(frozen_rows.byte_identical.all()),
        "prediction_shards_found": int(predictions.regenerated_exists.sum()),
        "prediction_shards_expected": int(len(predictions)),
        "prediction_shards_byte_identical": int(predictions.get("shard_byte_identical",
                                                                pd.Series(dtype=bool)).fillna(False).sum()),
        "generated_images_identical_shards": int(predictions.get("generated_equal",
                                                                 pd.Series(dtype=bool)).fillna(False).sum()),
        "generated_max_abs_diff_uint8": int(predictions.get("generated_max_abs_diff",
                                                            pd.Series([0])).max()),
        "tolerance_05_abs": TOLERANCE,
        "regenerated_05_paired_contrast_max_abs_diff": regenerated_max,
        "regenerated_05_within_tolerance": bool(regenerated_max <= TOLERANCE),
        "training_rerun": "not run (see module docstring); training stage verified by code identity only",
    }
    (OUTPUT / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(table.to_string(index=False))
    print(json.dumps(result, indent=2))
    return result


def stage_compare() -> None:
    link_frozen_intermediates()
    run_recovered_comparison(FROM_FROZEN)
    run_recovered_comparison(REGENERATED)
    verify()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=(
        "predict", "evaluate_images", "evaluate_uni", "compare", "train"))
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.stage in {"predict", "train"} and args.fold is None:
        parser.error("--fold is required for predict and train")
    if args.stage == "predict":
        stage_predict(args.fold, args.workers)
    elif args.stage == "evaluate_images":
        stage_evaluate_images()
    elif args.stage == "evaluate_uni":
        stage_evaluate_uni()
    elif args.stage == "train":
        stage_train(args.fold)
    else:
        stage_compare()


if __name__ == "__main__":
    main()

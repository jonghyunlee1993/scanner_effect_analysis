#!/usr/bin/env python3
"""RV-P0c: parity of the two UNI v1 extraction paths for the raw six-scanner images.

Path A (evaluation): ``03_uni/shards`` written by
``scanner_batch_extensions.uni_extract_task`` (x/255, bicubic resize to 224,
fp16 autocast, TF32 matmul), 20 locations per slide (set20).

Path B (feature-correction fitting): the raw conditions ``raw_source`` and
``real_target`` of the 40-location Pix2Pix UNI evaluation, read by
``manuscript_completion.feature_panel40.load_panel40`` (the ``--train-panel 40``
input of ``feature_uni_v1.py``): ``scanner_gan.evaluate_uni`` via
``embed_frozen_uni(value_range='uint8')`` (x/127.5-1, clamp, (x+1)/2, bicubic,
fp32 with TF32 matmul). GT450 comes from ``06_learned_baselines/09_bidirectional_full_training``.

For every set20 location the script compares the AT2 embedding (against each of
the five direction shards of path B) and each target-scanner embedding. It also
checks that the images fed to the two paths are pixel-identical and, as a
secondary check, compares per-location raw target distances with the third
extraction used for the published Pix2Pix/CycleGAN UNI distances
(``outputs/gan_encoder_review_2026-09-25``, path C: x/255, bicubic, clamp to
[0, 1] after resizing, fp32).

Output: ``analysis/revision/results/uni_extraction_parity/``. Submit from the repository root with
``sbatch --export=ALL,JOB_CONDA_PREFIX=$CONDA_PREFIX,JOB_WORKDIR=$PWD analysis/revision/uni_extraction_parity.sbatch``.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
BATCH = PROJECT / "outputs/scanner_batch_effect_analysis_2026-09-17"
PATH_A = BATCH / "03_uni/shards"
GAN_REVIEW = PROJECT / "outputs/gan_encoder_review_2026-09-25"
LOCATIONS = Path(__file__).resolve().parent / "results/location_sets"
OUTPUT = Path(__file__).resolve().parent / "results/uni_extraction_parity"
SCANNERS = ("versa", "akoya", "gt450", "s360", "s60")
THRESHOLD = 0.999


def path_b_root(scanner: str) -> Path:
    """Same directories as manuscript_completion.feature_panel40.feature_dir."""
    if scanner == "gt450":
        return BATCH / "06_learned_baselines/09_bidirectional_full_training/04_uni/at2_to_gt450"
    return BATCH / f"12_manuscript_completion/02_baseline_benchmark/03_pix2pix/04_uni/at2_to_{scanner}"


def cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left.astype(np.float64)
    right = right.astype(np.float64)
    return (left * right).sum(1) / (np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1))


def read_rows(path: Path, dataset: str, index: np.ndarray) -> np.ndarray:
    order = np.argsort(index)
    with h5py.File(path, "r") as store:
        values = np.asarray(store[dataset][index[order]])
    return values[np.argsort(order)]


def slide_task(payload: tuple[str, str, list[int], dict[str, str]]) -> dict:
    slide_id, cache_path, locations, prediction_paths = payload
    locations = np.asarray(locations, dtype=int)
    with h5py.File(PATH_A / f"{slide_id}.h5", "r") as store:
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in store["condition_names"][:]]
        a_locations = np.asarray(store["location_index"], dtype=int)
        wanted = [names.index("source")] + [names.index(f"target:{s}") for s in SCANNERS]
        a_features = np.asarray(store["features"][:, wanted, :], dtype=np.float32)
    if not np.array_equal(np.sort(a_locations), np.sort(locations)):
        raise ValueError(f"{slide_id}: path A locations differ from set20")
    a_lookup = {int(v): i for i, v in enumerate(a_locations)}
    a_features = a_features[[a_lookup[int(v)] for v in locations]]
    with h5py.File(cache_path, "r") as store:
        cache = np.asarray(store["images"][np.sort(locations)], dtype=np.uint8)
    cache = cache[np.argsort(np.argsort(locations))]

    rows, image_rows = [], []
    b_sources = []
    for scanner_index, scanner in enumerate(SCANNERS, start=1):
        feature_path = path_b_root(scanner) / "feature_shards" / f"{slide_id}.h5"
        with h5py.File(feature_path, "r") as store:
            if str(store.attrs["slide_id"]) != slide_id or str(store.attrs["target_scanner"]) != scanner:
                raise ValueError(f"path B identity mismatch: {feature_path}")
            b_locations = np.asarray(store["location_index"], dtype=int)
            b_lookup = {int(v): i for i, v in enumerate(b_locations)}
            index = np.asarray([b_lookup[int(v)] for v in locations])
            b_source = np.asarray(store["raw_source"][:], dtype=np.float32)[index]
            b_target = np.asarray(store["real_target"][:], dtype=np.float32)[index]
        b_sources.append(b_source)
        source_cos = cosine(a_features[:, 0], b_source)
        target_cos = cosine(a_features[:, scanner_index], b_target)
        raw_a = 1.0 - cosine(a_features[:, 0], a_features[:, scanner_index])
        raw_b = 1.0 - cosine(b_source, b_target)
        prediction = Path(prediction_paths[scanner])
        with h5py.File(prediction, "r") as store:
            p_locations = np.asarray(store["metadata/location_index"], dtype=int)
            p_lookup = {int(v): i for i, v in enumerate(p_locations)}
            p_index = np.asarray([p_lookup[int(v)] for v in locations])
            p_source = read_rows(prediction, "images/raw_source", p_index)
            p_target = read_rows(prediction, "images/real_target", p_index)
        for offset, location in enumerate(locations):
            common = {"slide_id": slide_id, "location_index": int(location), "direction": f"at2_to_{scanner}"}
            rows.append({**common, "image": "at2_source", "cosine_a_vs_b": float(source_cos[offset])})
            rows.append({**common, "image": f"{scanner}_target", "cosine_a_vs_b": float(target_cos[offset])})
            image_rows.append({
                **common,
                "source_pixels_equal": bool(np.array_equal(cache[offset, 0], p_source[offset])),
                "target_pixels_equal": bool(np.array_equal(cache[offset, scanner_index], p_target[offset])),
                "raw_distance_path_a": float(raw_a[offset]),
                "raw_distance_path_b": float(raw_b[offset]),
            })
    b_sources = np.stack(b_sources)
    spread = float(np.abs(b_sources - b_sources[:1]).max())
    return {"rows": rows, "image_rows": image_rows, "slide_id": slide_id, "path_b_source_spread": spread}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument("--max-slides", type=int, default=0)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(BATCH / "00_contract/cohort.csv", dtype={"slide_id": str})
    set20 = pd.read_csv(LOCATIONS / "set20.csv", dtype={"slide_id": str})
    predictions: dict[tuple[str, str], str] = {}
    for scanner in SCANNERS:
        manifest = json.loads((path_b_root(scanner) / "evaluation_manifest.json").read_text())
        for shard in manifest["shards"]:
            predictions[(str(shard["slide_id"]), scanner)] = str(shard["prediction_path"])
    tasks = []
    for row in cohort.itertuples(index=False):
        slide_id = str(row.slide_id)
        locations = sorted(int(v) for v in set20.loc[set20.slide_id.eq(slide_id), "location_index"])
        if len(locations) != 20:
            raise ValueError(f"{slide_id}: set20 has {len(locations)} locations")
        tasks.append((slide_id, str(row.cache_path), locations,
                      {s: predictions[(slide_id, s)] for s in SCANNERS}))
    if args.max_slides:
        tasks = tasks[: args.max_slides]
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        results = list(pool.map(slide_task, tasks))

    per_location = pd.DataFrame([r for result in results for r in result["rows"]])
    images = pd.DataFrame([r for result in results for r in result["image_rows"]])
    spread = {result["slide_id"]: result["path_b_source_spread"] for result in results}

    review = pd.concat([pd.read_csv(p, dtype={"slide_id": str}) for p in
                        sorted((GAN_REVIEW / "shards/uni_v1").glob("fold_*.csv.gz"))], ignore_index=True)
    review["direction"] = "at2_to_" + review["scanner"]
    images = images.merge(review[["slide_id", "location_index", "direction", "raw_distance"]].rename(
        columns={"raw_distance": "raw_distance_path_c"}), on=["slide_id", "location_index", "direction"],
        how="left", validate="one_to_one")
    per_location["below_threshold"] = per_location.cosine_a_vs_b < THRESHOLD
    per_location.to_csv(OUTPUT / "per_location_cosine.csv", index=False)
    images.to_csv(OUTPUT / "per_location_inputs_and_raw_distance.csv", index=False)
    flagged = per_location[per_location.below_threshold]
    flagged.to_csv(OUTPUT / "flagged_below_0.999.csv", index=False)

    cos = per_location.cosine_a_vs_b
    unique_images = per_location.groupby(["slide_id", "location_index", "image"]).cosine_a_vs_b.min()
    quantiles = {str(q): float(cos.quantile(q)) for q in (0.0, 0.001, 0.01, 0.05, 0.5, 0.95, 1.0)}
    by_image = per_location.assign(kind=np.where(per_location.image.eq("at2_source"), "at2_source",
                                                 per_location.image)).groupby("kind").cosine_a_vs_b.agg(
        ["count", "min", "median", "mean"]).reset_index()
    by_image.to_csv(OUTPUT / "summary_by_image.csv", index=False)
    distance_diff = {
        "a_minus_b": (images.raw_distance_path_a - images.raw_distance_path_b),
        "a_minus_c": (images.raw_distance_path_a - images.raw_distance_path_c),
    }
    summary = {
        "slides": int(per_location.slide_id.nunique()),
        "locations": int(per_location[["slide_id", "location_index"]].drop_duplicates().shape[0]),
        "comparisons": int(len(per_location)),
        "distinct_images_compared": int(len(unique_images)),
        "threshold": THRESHOLD,
        "below_threshold": int(per_location.below_threshold.sum()),
        "min_cosine": float(cos.min()),
        "median_cosine": float(cos.median()),
        "mean_cosine": float(cos.mean()),
        "quantiles": quantiles,
        "path_b_at2_features_max_abs_spread_across_direction_shards": float(max(spread.values())),
        "input_pixels_equal": {
            "source": int(images.source_pixels_equal.sum()),
            "target": int(images.target_pixels_equal.sum()),
            "of": int(len(images)),
        },
        "raw_target_distance_difference": {
            name: {"max_abs": float(values.abs().max()), "mean": float(values.mean()),
                   "sd": float(values.std(ddof=1)), "n_missing": int(values.isna().sum())}
            for name, values in distance_diff.items()
        },
        "slide_mean_raw_distance_all_scanners": {
            name: float(images.groupby("slide_id")[f"raw_distance_path_{name}"].mean().mean())
            for name in ("a", "b", "c")
        },
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    passed = summary["below_threshold"] == 0
    lines = [
        "# RV-P0c: UNI v1 extraction parity (raw six-scanner images, set20)",
        "",
        "Generated by `analysis/revision/uni_extraction_parity.py`. Protocol: `analysis/revision/README.md` (RV-P0c).",
        "",
        "## What ran",
        "",
        "- Path A (evaluation embeddings): `outputs/scanner_batch_effect_analysis_2026-09-17/03_uni/shards`, "
        "conditions `source` and `target:<scanner>` (x/255, bicubic to 224, fp16 autocast).",
        "- Path B (40-location embeddings used to fit feature correction; confirmed by reading "
        "`src/manuscript_completion/feature_uni_v1.py` -> `feature_panel40.load_panel40`): `raw_source` and "
        "`real_target` in the Pix2Pix UNI `feature_shards` (02_baseline_benchmark/03_pix2pix/04_uni for VERSA, "
        "AKOYA, S360, S60; 06_learned_baselines/09_bidirectional_full_training/04_uni for GT450); "
        "x/127.5-1, clamp, (x+1)/2, bicubic to 224, fp32.",
        "- Cosine similarity per location at the 2,060 shared set20 locations: the AT2 image against each of the "
        "five direction shards of path B (5 comparisons per location) and each target-scanner image (5).",
        "- Input check: the patch-cache images used by path A vs the prediction-file images used by path B.",
        "- Secondary: per-location raw AT2-target distance from path A, path B and path C (the "
        "`gan_encoder_review_2026-09-25` re-embedding behind the published Pix2Pix/CycleGAN UNI distances; "
        "x/255, bicubic, clamp to [0, 1] after resizing, fp32).",
        "",
        "## Result (pre-specified: cosine >= 0.999 at every location)",
        "",
        f"- Comparisons: {summary['comparisons']} ({summary['locations']} locations x 10; "
        f"{summary['distinct_images_compared']} distinct images).",
        f"- Minimum cosine {summary['min_cosine']:.7f}; median {summary['median_cosine']:.7f}; "
        f"0.1% quantile {quantiles['0.001']:.7f}.",
        f"- Locations below 0.999: {summary['below_threshold']} -> "
        + ("**pass**." if passed else "**fail**; see `flagged_below_0.999.csv`."),
        f"- Path B AT2 features are identical across its five direction shards up to "
        f"{summary['path_b_at2_features_max_abs_spread_across_direction_shards']:.2e} (max abs element difference).",
        f"- Input pixels identical: source {summary['input_pixels_equal']['source']}/{summary['input_pixels_equal']['of']}, "
        f"target {summary['input_pixels_equal']['target']}/{summary['input_pixels_equal']['of']}.",
        "",
        "## Secondary: raw AT2-target distance by extraction path",
        "",
        "| comparison | max abs difference | mean difference | SD |",
        "| --- | --- | --- | --- |",
    ]
    for name, values in summary["raw_target_distance_difference"].items():
        lines.append(f"| path {name.replace('_minus_', ' minus path ')} | {values['max_abs']:.5f} | "
                     f"{values['mean']:+.6f} | {values['sd']:.6f} |")
    means = summary["slide_mean_raw_distance_all_scanners"]
    lines += [
        "",
        f"Mean raw distance over 103 slides (five scanners equally weighted): path A {means['a']:.5f}, "
        f"path B {means['b']:.5f}, path C {means['c']:.5f}.",
        "",
        "Interpretation boundary: paths A and B differ only in arithmetic precision and the equivalent "
        "value scaling; path C additionally clamps after bicubic resizing. The protocol's >=0.999 criterion "
        "applies to A vs B; the path-C row is reported because the published Pix2Pix/CycleGAN UNI distances "
        "(Table 2) come from path C while their percentage change is computed against a path-A raw distance.",
        "",
        "## Files",
        "",
        "- `per_location_cosine.csv`, `flagged_below_0.999.csv`, `summary_by_image.csv`",
        "- `per_location_inputs_and_raw_distance.csv` (pixel equality and raw distances, paths A/B/C)",
        "- `summary.json`",
    ]
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

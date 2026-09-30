#!/usr/bin/env python3
"""RV-P0d (ii): restore the PLISM Pix2Pix/CycleGAN cross-PFM evaluation.

Recovered producers (moved to `.Trash/2026-09-29_paper_code_cleanup/scripts/`):
`review_plism_gan_crossencoder.py` (GPU array 23670688, tasks 0-17, 2026-09-25 23:05-23:17)
and `review_plism_gan_crossencoder_aggregate.py` (job 23670935). Their run-time byte-code
caches survive in `.Trash/2026-09-26_paper_refactor/generated/scripts/__pycache__/`; the
trashed sources differ from them only in one relocated path constant each
(`00_manuscript/analysis/...` -> `analysis/paper/...`), see `restore_code_identity.py`.

Functions `load_inputs`, `checkpoint_rows`, `run`, `seed`, `load`, `section_values`,
`verify_baseline`, `summarize` and `aggregate_main` (the aggregate's `main`) are copied
verbatim. Changes: `OUTPUT` points to `results/provenance_restoration/plism_gan_crossencoder/`,
`ROOT` is the repository root, and the encoder helpers are imported from their current
location `scripts/features/review_multiencoder_scanner.py` (function byte code identical to
the run-time cache). Nothing under `outputs/` is written.

Modes:
  --task-index K   regenerate one (PFM, scanner, method) shard on a GPU (as the original);
  --aggregate      aggregate the regenerated shards with the recovered code, re-aggregate the
                   frozen shards with the same code, and compare both with the frozen files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.features.review_multiencoder_scanner import cosine_distance, embed, load_encoder
from scanner_gan.plism import _generate_uint8
from scanner_gan.plism_bidirectional import _load_render
from scanner_gan.predict import load_validated_generator
from scanner_gan.train import sha256
from scanner_gan.train_cyclegan import load_validated_cyclegan_generator


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "outputs/scanner_batch_effect_analysis_2026-09-17"
PLISM = STUDY / "12_manuscript_completion/06_plism_learned_external"
GT450 = ROOT / "analysis/paper/results/plism_gt450_extension"
FROZEN = ROOT / "outputs/plism_gan_crossencoder_2026-09-25"
OUTPUT = Path(__file__).resolve().parent / "results/provenance_restoration/plism_gan_crossencoder"
CACHE = ROOT / "outputs/encoder_review_2026-09-25/hf_cache"
MODELS = ("uni2", "virchow2", "hoptimus1")
SCANNERS = ("gt450", "s360", "s60")
METHODS = ("pix2pix", "cyclegan")
TASKS = [(model, scanner, method) for model in MODELS
         for scanner in SCANNERS for method in METHODS]
GT450_CHECKPOINTS = {
    "pix2pix": STUDY / "06_learned_baselines/09_bidirectional_full_training"
    / "10_plism_external_bidirectional/00_contract/checkpoint_candidates.csv",
    "cyclegan": STUDY / "06_learned_baselines/11_cyclegan_full_training"
    / "10_plism_external_bidirectional/00_contract/checkpoint_candidates.csv",
}
TABLE4 = ROOT / "outputs/table4_color_frequency_crossencoder_2026-09-25"
TABLE3 = ROOT / "analysis/paper/table2_baseline_summary.csv"


def load_inputs(scanner: str):
    contract = PLISM / "pix2pix/00_contract"
    selected = pd.read_csv(contract / "selected_locations.csv")
    if len(selected) != 2387 or selected.stain.nunique() != 13:
        raise ValueError("PLISM evaluation index changed")
    renders = pd.read_csv(contract / "read_only_renders.csv")
    renders = pd.concat([renders, pd.read_csv(GT450 / "gt450_renders.csv")],
                        ignore_index=True)
    sections = {}
    for section, group in selected.groupby("stain", sort=True):
        group = group.sort_values("location").reset_index(drop=True)
        source, source_locations, _ = _load_render(renders, "AT2", section)
        target, target_locations, _ = _load_render(renders, scanner, section)
        locations = group.location.to_numpy(np.int64)
        if (not np.array_equal(source_locations, target_locations) or
                not np.array_equal(source_locations, locations)):
            raise ValueError(f"PLISM paired locations differ: {scanner}/{section}")
        sections[section] = (group, source, target)
    return sections


def checkpoint_rows(scanner: str, method: str) -> pd.DataFrame:
    if scanner == "gt450":
        path = GT450_CHECKPOINTS[method]
    else:
        path = PLISM / method / "00_contract/checkpoint_candidates.csv"
    candidates = pd.read_csv(path)
    chosen = candidates.loc[candidates.direction.eq(f"at2_to_{scanner}")].copy()
    chosen = chosen.sort_values("fold")
    if (chosen.fold.astype(int).tolist() != list(range(5)) or
            not chosen.source_scanner.astype(str).str.lower().eq("at2").all() or
            not chosen.target_scanner.astype(str).str.lower().eq(scanner).all()):
        raise ValueError(f"unexpected {method}/{scanner} checkpoint panel")
    return chosen


def run(task_index: int, batch_size: int, generator_batch_size: int,
        max_sections: int | None, max_locations: int | None,
        max_folds: int | None, output_dir: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU allocation required")
    torch.set_num_threads(2)
    model_name, scanner, method = TASKS[task_index]
    model, size, mean, std, variants = load_encoder(model_name, CACHE)
    if len(variants) != 1:
        raise ValueError(f"expected one embedding for {model_name}")
    variant = variants[0]
    sections = load_inputs(scanner)
    if max_sections is not None:
        sections = dict(list(sections.items())[:max_sections])
    if max_locations is not None:
        sections = {name: (meta.head(max_locations), source[:max_locations],
                           target[:max_locations])
                    for name, (meta, source, target) in sections.items()}

    # Source and target are identical across the five frozen generators.
    reference = {}
    for section, (meta, source, target) in sections.items():
        images = [image.astype(np.float32) / 255.0
                  for pair in zip(source, target) for image in pair]
        vectors = embed(model, model_name, images, size, mean, std,
                        batch_size)[variant]
        source_vectors, target_vectors = vectors[0::2], vectors[1::2]
        raw_distance = [cosine_distance(a, b) for a, b in
                        zip(source_vectors, target_vectors)]
        reference[section] = (target_vectors, raw_distance)
        print(f"{variant} {scanner} {method}: references {section}", flush=True)

    rows = []
    candidates = checkpoint_rows(scanner, method)
    if max_folds is not None:
        candidates = candidates.head(max_folds)
    loader = (load_validated_generator if method == "pix2pix"
              else load_validated_cyclegan_generator)
    for candidate in candidates.itertuples(index=False):
        fold = int(candidate.fold)
        checkpoint = Path(str(candidate.checkpoint_path))
        if sha256(checkpoint) != str(candidate.checkpoint_sha256):
            raise ValueError(f"checkpoint hash changed: {checkpoint}")
        generator, _, _ = loader(
            checkpoint, source_scanner="at2", target_scanner=scanner,
            test_fold=fold, device=torch.device("cuda"),
        )
        for section, (meta, source, _) in sections.items():
            generated = _generate_uint8(
                generator, source, torch.device("cuda"),
                batch_size=generator_batch_size, amp="bfloat16",
            )
            vectors = embed(model, model_name,
                            [image.astype(np.float32) / 255.0
                             for image in generated],
                            size, mean, std, batch_size)[variant]
            target_vectors, raw_distances = reference[section]
            for item, vector, target_vector, raw in zip(
                    meta.itertuples(index=False), vectors, target_vectors,
                    raw_distances):
                corrected = cosine_distance(vector, target_vector)
                rows.append({
                    "model": variant, "scanner": scanner, "method": method,
                    "fold": fold, "section": section, "core": int(item.core),
                    "location_index": int(item.location),
                    "raw_distance": raw, "corrected_distance": corrected,
                    "gain": raw - corrected,
                })
            print(f"{variant} {scanner} {method} fold {fold}: {section}",
                  flush=True)
        del generator
        torch.cuda.empty_cache()

    expected = sum(len(meta) for meta, _, _ in sections.values()) * len(candidates)
    if len(rows) != expected:
        raise ValueError(f"expected {expected} rows, found {len(rows)}")
    frame = pd.DataFrame(rows)
    if (frame.duplicated(["model", "scanner", "method", "fold", "section",
                         "core", "location_index"]).any() or
            not np.isfinite(frame[["raw_distance", "corrected_distance",
                                   "gain"]].to_numpy()).all()):
        raise ValueError("duplicate or nonfinite PLISM evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"part_{task_index:02d}.csv.gz"
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False, compression="gzip")
    temporary.replace(path)
    print(json.dumps({"task_index": task_index, "model": variant,
                      "scanner": scanner, "method": method,
                      "sections": len(sections), "folds": len(candidates),
                      "rows": len(frame), "path": str(path)}), flush=True)


def seed(*items: str) -> int:
    value = "|".join(items).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "little")


def load() -> pd.DataFrame:
    frames = []
    for index, task in enumerate(TASKS):
        path = OUTPUT / "shards" / f"part_{index:02d}.csv.gz"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        if (len(frame) != 2387 * 5 or frame.section.nunique() != 13 or
                frame.fold.nunique() != 5 or
                set(zip(frame.model.unique(), frame.scanner.unique(),
                        frame.method.unique())) != {task}):
            raise ValueError(f"incomplete task {index}: {task}")
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    keys = ["model", "scanner", "method", "fold", "section", "core",
            "location_index"]
    if data.duplicated(keys).any():
        raise ValueError("duplicate PLISM GAN evaluation key")
    if not np.isfinite(data[["raw_distance", "corrected_distance",
                              "gain"]].to_numpy()).all():
        raise ValueError("nonfinite PLISM GAN distance")
    if not np.allclose(data.raw_distance - data.corrected_distance,
                       data.gain, atol=1e-8):
        raise ValueError("incorrect gain")
    return data


def section_values(data: pd.DataFrame) -> pd.DataFrame:
    metrics = ["raw_distance", "corrected_distance", "gain"]
    core = data.groupby(["model", "section", "scanner", "method", "fold",
                         "core"], as_index=False)[metrics].mean()
    section_fold = core.groupby(["model", "section", "scanner", "method",
                                 "fold"], as_index=False)[metrics].mean()
    section_scanner = section_fold.groupby(
        ["model", "section", "scanner", "method"], as_index=False,
    )[metrics].mean()
    pooled = section_scanner.groupby(
        ["model", "section", "method"], as_index=False,
    )[metrics].mean()
    pooled["scanner"] = "all"
    result = pd.concat([section_scanner, pooled], ignore_index=True)
    if (result.groupby(["model", "scanner", "method"]).size().ne(13).any() or
            result.groupby(["model", "section", "method"]).scanner.nunique().ne(4).any()):
        raise ValueError("incomplete section/scanner aggregation")
    return result


def verify_baseline(section: pd.DataFrame) -> dict:
    existing = pd.read_csv(TABLE4 / "unit_distances_and_gains.csv.gz")
    existing = existing[(existing.dataset == "PLISM") &
                        (existing.model.isin(MODELS))]
    compare = section.merge(existing[["model", "unit", "scanner", "raw"]],
                            left_on=["model", "section", "scanner"],
                            right_on=["model", "unit", "scanner"],
                            validate="many_to_one")
    error = (compare.raw_distance - compare.raw).abs()
    if len(compare) != len(section) or error.max() > 0.001:
        raise ValueError(f"raw distance mismatch with Table 4: {error.max()}")
    return {"max_abs_raw_difference_from_table4": float(error.max()),
            "mean_abs_raw_difference_from_table4": float(error.mean())}


def summarize(section: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, scanner, method), group in section.groupby(
            ["model", "scanner", "method"]):
        group = group.sort_values("section")
        raw = group.raw_distance.to_numpy(float)
        corrected = group.corrected_distance.to_numpy(float)
        gain = raw - corrected
        draws = np.random.default_rng(seed(model, scanner, method)).integers(
            0, 13, (4000, 13))
        draw_means = gain[draws].mean(axis=1)
        low, high = np.quantile(draw_means, [0.025, 0.975])
        rows.append({"model": model, "scanner": scanner,
                     "method": method, "raw_distance": raw.mean(),
                     "corrected_distance": corrected.mean(),
                     "gain": gain.mean(), "ci_low": low, "ci_high": high,
                     "sections": len(group)})
    return pd.DataFrame(rows).sort_values(["model", "scanner", "method"])


def aggregate_main() -> None:
    data = load()
    section = section_values(data)
    audit = verify_baseline(section)
    summary = summarize(section)

    # The original UNI-v1 benchmark provides the already evaluated S5 values.
    baseline = pd.read_csv(TABLE3)
    baseline = baseline[baseline.dataset.eq("PLISM")].set_index("method")
    uni = {method: {
        "raw_distance": float(baseline.loc["raw", "uni_distance_mean"]),
        "corrected_distance": float(baseline.loc[method, "uni_distance_mean"]),
    } for method in METHODS}
    for values in uni.values():
        values["gain"] = values["raw_distance"] - values["corrected_distance"]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    section.to_csv(OUTPUT / "gan_section_distances.csv", index=False)
    summary.to_csv(OUTPUT / "gan_gain_summary.csv", index=False)
    compact = pd.DataFrame([
        {"model": model, **{
            f"{method}_gain": (uni[method]["gain"] if model == "uni_v1" else
                               float(summary[(summary.model == model) &
                                             (summary.scanner == "all") &
                                             (summary.method == method)].gain.iloc[0]))
            for method in METHODS}}
        for model in ("uni_v1", *MODELS)
    ])
    compact.to_csv(OUTPUT / "proposed_s5_gan_gains.csv", index=False)
    audit.update({"models": list(MODELS), "scanners": list(SCANNERS),
                  "methods": list(METHODS), "sections": 13,
                  "locations": 2387, "folds": 5,
                  "location_rows": len(data),
                  "uni_v1_source": str(TABLE3)})
    (OUTPUT / "aggregate_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(compact.to_string(index=False))
    print(json.dumps(audit, indent=2))




# ---------------------------------------------------------------------------
# Verification (added for RV-P0d; not part of the recovered producer)
# ---------------------------------------------------------------------------

KEYS = ["model", "scanner", "method", "fold", "section", "core", "location_index"]
# Declared before any regenerated value was seen: a summary-level gain within 1e-3 of the
# frozen value agrees at the three decimals used when these gains are reported.
SUMMARY_TOLERANCE = 1e-3


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compare_tables(new: pd.DataFrame, old: pd.DataFrame, keys: list[str],
                    values: list[str], label: str) -> dict:
    merged = new.merge(old, on=keys, suffixes=("_new", "_frozen"), how="outer",
                       indicator=True, validate="one_to_one")
    result = {"table": label, "rows_regenerated": int(len(new)), "rows_frozen": int(len(old)),
              "rows_unmatched": int((merged["_merge"] != "both").sum())}
    both = merged[merged["_merge"] == "both"]
    for value in values:
        delta = (both[f"{value}_new"] - both[f"{value}_frozen"]).abs()
        result[f"{value}_max_abs_diff"] = float(delta.max())
        result[f"{value}_p99_abs_diff"] = float(delta.quantile(0.99))
        result[f"{value}_exact_fraction"] = float((delta == 0).mean())
    return result


def verify() -> None:
    """Compare regenerated shards and summaries with the frozen outputs."""
    global OUTPUT
    frames = []
    for index in range(len(TASKS)):
        new = pd.read_csv(OUTPUT / "shards" / f"part_{index:02d}.csv.gz")
        old = pd.read_csv(FROZEN / "shards" / f"part_{index:02d}.csv.gz")
        row = _compare_tables(new, old, KEYS, ["raw_distance", "corrected_distance", "gain"],
                              f"shards/part_{index:02d}.csv.gz")
        row.update(dict(zip(("model", "scanner", "method"), TASKS[index])))
        frames.append(row)
    location = pd.DataFrame(frames)
    location.to_csv(OUTPUT / "verification_location_level.csv", index=False)

    # The recovered aggregation applied to the frozen shards must reproduce the frozen
    # summaries exactly; this isolates aggregation-code provenance from GPU numerics.
    # `load()` reads the module-level OUTPUT, so it is pointed at the frozen shards briefly.
    regenerated_root = OUTPUT
    OUTPUT = FROZEN
    try:
        frozen_data = load()
    finally:
        OUTPUT = regenerated_root
    frozen_section = section_values(frozen_data)
    frozen_summary = summarize(frozen_section)
    frozen_section.to_csv(OUTPUT / "reaggregated_frozen_shards_section_distances.csv", index=False)
    frozen_summary.to_csv(OUTPUT / "reaggregated_frozen_shards_gain_summary.csv", index=False)

    rows = []
    summary_keys = ["model", "scanner", "method"]
    metrics = ["raw_distance", "corrected_distance", "gain", "ci_low", "ci_high"]
    frozen_files = {
        "gan_section_distances.csv": (["model", "section", "scanner", "method"],
                                      ["raw_distance", "corrected_distance", "gain"]),
        "gan_gain_summary.csv": (summary_keys, metrics),
        "proposed_s5_gan_gains.csv": (["model"], ["pix2pix_gain", "cyclegan_gain"]),
    }
    for name, (keys, values) in frozen_files.items():
        old = pd.read_csv(FROZEN / name)
        new = pd.read_csv(OUTPUT / name)
        row = _compare_tables(new, old, keys, values, f"regenerated vs frozen {name}")
        row["byte_identical"] = _file_sha256(OUTPUT / name) == _file_sha256(FROZEN / name)
        rows.append(row)
    for name, frame in (("gan_section_distances.csv", frozen_section),
                        ("gan_gain_summary.csv", frozen_summary)):
        keys, values = frozen_files[name]
        old = pd.read_csv(FROZEN / name)
        rederived = pd.read_csv(OUTPUT / f"reaggregated_frozen_shards_{name.replace('gan_', '')}")
        row = _compare_tables(rederived, old, keys, values,
                              f"recovered aggregation of frozen shards vs frozen {name}")
        row["byte_identical"] = _file_sha256(
            OUTPUT / f"reaggregated_frozen_shards_{name.replace('gan_', '')}") == _file_sha256(FROZEN / name)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT / "verification_summary_level.csv", index=False)

    new = pd.read_csv(OUTPUT / "gan_gain_summary.csv").set_index(summary_keys)
    old = pd.read_csv(FROZEN / "gan_gain_summary.csv").set_index(summary_keys)
    new = new.loc[old.index]
    gain_delta = float((new.gain - old.gain).abs().max())
    result = {
        "producer": "recovered review_plism_gan_crossencoder.py + _aggregate.py",
        "original_jobs": {"extraction": "23670688 (array 0-17)", "aggregate": "23670935"},
        "summary_tolerance_abs_gain": SUMMARY_TOLERANCE,
        "max_abs_gain_difference_summary": gain_delta,
        "within_tolerance": bool(gain_delta <= SUMMARY_TOLERANCE),
        "all_gain_signs_agree": bool((np.sign(new.gain) == np.sign(old.gain)).all()),
        "ci_excludes_zero_agrees": bool((((new.ci_high < 0) | (new.ci_low > 0)) ==
                                         ((old.ci_high < 0) | (old.ci_low > 0))).all()),
        "max_abs_corrected_distance_difference_location": float(
            location.corrected_distance_max_abs_diff.max()),
        "max_abs_raw_distance_difference_location": float(location.raw_distance_max_abs_diff.max()),
        "frozen_shards_reaggregated_byte_identical": {
            row["table"]: bool(row["byte_identical"]) for row in rows
            if row["table"].startswith("recovered aggregation")},
    }
    (OUTPUT / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(location.to_string(index=False))
    print(summary.to_string(index=False))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--task-index", type=int, choices=range(len(TASKS)))
    mode.add_argument("--aggregate", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--generator-batch-size", type=int, default=16)
    parser.add_argument("--max-sections", type=int)
    parser.add_argument("--max-locations", type=int)
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT / "shards")
    args = parser.parse_args()
    if args.aggregate:
        aggregate_main()
        verify()
    else:
        run(args.task_index, args.batch_size, args.generator_batch_size,
            args.max_sections, args.max_locations, args.max_folds, args.output_dir)

#!/usr/bin/env python3
"""Add PLISM AT2-to-GT450 to the frozen common-metric benchmark.

The S360/S60 and PanNormal shards are read from the original benchmark. New
GT450 shards are written under the manuscript tree. Model checkpoints,
locations, renders, and prior GT450 UNI-v1 measurements remain read-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import table2_benchmark as base


HERE = Path(__file__).resolve().parent
OUT = HERE / "results/plism_gt450_extension"
ORIGINAL = base.COMPLETION / "09_table2_benchmark"
GT450_RENDER = base.EVIDENCE / "08_plism_external/pix2pix_gt450_to_at2/01_rendered_inputs/GT450"
GT450_EXTERNAL = {
    "pix2pix": base.EVIDENCE / "06_learned_baselines/09_bidirectional_full_training/10_plism_external_bidirectional",
    "cyclegan": base.EVIDENCE / "06_learned_baselines/11_cyclegan_full_training/10_plism_external_bidirectional",
}
METRICS = ["target_ssim", "lpips_vgg", "uni_distance"]
KEYS = ["section", "core", "location", "scanner", "fold", "method"]
TABLE_FILES = [
    "table2_main_summary.csv",
    "table2_scanner_summary.csv",
    "table2_baseline_summary.csv",
    "table_frequency_summary.csv",
    "table2_main.tex",
    "table_frequency_hypothesis.tex",
    "table2_scanner.tex",
    "table2_manifest.json",
]


def inputs() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    contract, selected, renders = base._plism_inputs("pix2pix")
    rows = []
    for stain, group in selected.groupby("stain", sort=True):
        path = GT450_RENDER / f"{stain}.h5"
        summary_path = path.with_suffix(".summary.json")
        summary = json.loads(summary_path.read_text())
        if summary["output_path"] != str(path) or summary["status"] != "complete":
            raise ValueError(f"unfrozen GT450 render: {path}")
        if int(summary["locations"]) != len(group):
            raise ValueError(f"GT450 location count differs for {stain}")
        rows.append({
            "path": str(path), "locations": len(group),
            "sha256": summary["output_sha256"], "stain": stain,
            "scanner": "GT450", "summary_path": str(summary_path),
            "summary_sha256": base.sha256(summary_path),
        })
    if len(rows) != 13 or len(selected) != 2387:
        raise ValueError("unexpected PLISM section/location count")
    renders = pd.concat([renders, pd.DataFrame(rows)], ignore_index=True)
    return contract, selected, renders


def candidates(method: str) -> pd.DataFrame:
    path = GT450_EXTERNAL[method] / "00_contract/checkpoint_candidates.csv"
    frame = pd.read_csv(path)
    frame = frame.loc[frame["direction"].eq("at2_to_gt450")].sort_values("fold")
    if frame["fold"].astype(int).tolist() != list(range(5)):
        raise ValueError(f"expected five frozen {method} GT450 folds")
    if not frame["source_scanner"].astype(str).str.lower().eq("at2").all():
        raise ValueError("unexpected model source")
    if not frame["target_scanner"].astype(str).str.lower().eq("gt450").all():
        raise ValueError("unexpected model target")
    return frame.reset_index(drop=True)


def prepare() -> None:
    _, selected, renders = inputs()
    for method in GT450_EXTERNAL:
        candidates(method)
    OUT.mkdir(parents=True, exist_ok=True)
    base.write_frame(OUT / "gt450_renders.csv", renders.loc[renders["scanner"].eq("GT450")])
    base.write_json(OUT / "input_manifest.json", {
        "extension": "PLISM reference scanner to GT450; frozen PanNormal folds",
        "selected_locations_sha256": base.sha256(
            base.PLISM_LEARNED / "pix2pix/00_contract/selected_locations.csv"
        ),
        "selected_locations": len(selected),
        "sections": 13,
        "checkpoint_candidate_sha256": {
            method: base.sha256(root / "00_contract/checkpoint_candidates.csv")
            for method, root in GT450_EXTERNAL.items()
        },
        "render_manifest_sha256": base.sha256(OUT / "gt450_renders.csv"),
        "existing_table_sha256": {
            name: base.sha256(base.TABLE_PATHS.get(name, HERE / name)) for name in TABLE_FILES
        },
        "original_benchmark": str(ORIGINAL),
    })


def conventional_task(index: int, batch_size: int, uni_batch_size: int) -> None:
    contract, selected, renders = inputs()
    stains = sorted(selected["stain"].astype(str).unique())
    stain = stains[index]
    metadata, source, target = base._plism_pair(selected, renders, "GT450", stain)
    parameter_folds = json.loads((base.EVIDENCE / "02_correction/parameters.json").read_text())["fold"]
    selection = base.load_vahadane_selection(
        base.PANEL_A / "00_contract/vahadane_selection_manifest_v4.json"
    )
    vahadane_configuration = selection["selected_configuration"]
    source_stain = [
        base._stain_source_parameters(
            image, base.stable_seed("table2", stain, int(location)),
            vahadane_configuration,
        )
        for image, location in zip(source, metadata["location"].astype(int))
    ]
    candidate_rows: list[tuple[str, int, np.ndarray]] = [("raw", -1, source)]
    for fold in range(5):
        parameter = parameter_folds[str(fold)]["gt450"]
        payload = json.loads(
            (base.PANEL_A / f"01_stain_v4_mu_convergence/references/fold_{fold}/gt450.json").read_text()
        )
        references = {
            method: base.parameters_from_json(payload["parameters"][method])
            for method in ("macenko", "vahadane")
        }
        generated: dict[str, list[np.ndarray]] = {
            method: [] for method in ("reinhard", "frequency", "combined", "macenko", "vahadane")
        }
        for image, fitted in zip(source, source_stain):
            reinhard = base.reinhard_transform(image, parameter)
            generated["reinhard"].append(reinhard)
            generated["frequency"].append(base.frequency_transform(image, parameter))
            generated["combined"].append(base.frequency_transform(reinhard, parameter))
            for method in ("macenko", "vahadane"):
                corrected, _ = base._apply_stain_from_source(image, fitted[method], references[method])
                generated[method].append(corrected)
        candidate_rows.extend((method, fold, np.stack(images)) for method, images in generated.items())

    device = torch.device("cuda")
    lpips = base.lpips_model(device)
    frames = []
    for method, fold, images in candidate_rows:
        metrics = base.image_metrics(source, images, target)
        metrics["lpips_vgg"] = base.lpips_scores(lpips, images, target, device, batch_size)
        frame = pd.concat([metadata.reset_index(drop=True), metrics], axis=1)
        frame["dataset"] = "PLISM"
        frame["section"] = stain
        frame["scanner"] = "GT450"
        frame["method"] = method
        frame["method_class"] = base.CLASSES[method]
        frame["fold"] = fold
        frames.append(frame)
    del lpips
    torch.cuda.empty_cache()

    encoder = contract["primary_encoder"]
    uni, size, mean, std, _ = base.load_frozen_uni1(
        device, checkpoint_path=encoder["checkpoint_path"],
        expected_sha256=encoder["checkpoint_sha256"],
    )
    target_features = base.embed_frozen_uni(
        uni, target, size, mean, std, device,
        batch_size=uni_batch_size, value_range="uint8",
    )
    for frame, (_, _, images) in zip(frames, candidate_rows):
        features = base.embed_frozen_uni(
            uni, images, size, mean, std, device,
            batch_size=uni_batch_size, value_range="uint8",
        )
        frame["uni_distance"] = 1.0 - np.einsum("ij,ij->i", features, target_features)
    base.write_frame(
        OUT / f"plism_conventional/shards/GT450_{stain}.csv.gz",
        pd.concat(frames, ignore_index=True),
    )


def learned_task(index: int, batch_size: int, generator_batch_size: int) -> None:
    method = "pix2pix" if index < 5 else "cyclegan"
    fold = index if index < 5 else index - 5
    candidate = candidates(method).iloc[fold]
    checkpoint = Path(str(candidate["checkpoint_path"]))
    if base.sha256(checkpoint) != str(candidate["checkpoint_sha256"]):
        raise ValueError(f"frozen {method} checkpoint changed: {checkpoint}")
    _, selected, renders = inputs()
    device = torch.device("cuda")
    loader = base.load_validated_generator if method == "pix2pix" else base.load_validated_cyclegan_generator
    generator, _, _ = loader(
        checkpoint, source_scanner="at2", target_scanner="gt450",
        test_fold=fold, device=device,
    )
    lpips = base.lpips_model(device)
    frames = []
    for stain in sorted(selected["stain"].astype(str).unique()):
        metadata, source, target = base._plism_pair(selected, renders, "GT450", stain)
        generated = base._generate_uint8(
            generator, source, device, batch_size=generator_batch_size, amp="bfloat16"
        )
        metrics = base.image_metrics(source, generated, target)
        metrics["lpips_vgg"] = base.lpips_scores(lpips, generated, target, device, batch_size)
        frame = pd.concat([metadata.reset_index(drop=True), metrics], axis=1)
        frame["dataset"] = "PLISM"
        frame["section"] = stain
        frame["scanner"] = "GT450"
        frame["method"] = method
        frame["method_class"] = base.CLASSES[method]
        frame["fold"] = fold
        frame["checkpoint_sha256"] = str(candidate["checkpoint_sha256"])
        frames.append(frame)
    base.write_frame(
        OUT / f"plism_learned/shards/{method}_gt450_fold_{fold}.csv.gz",
        pd.concat(frames, ignore_index=True),
    )


def gt450_existing_measurements() -> pd.DataFrame:
    parts = []
    for method, root in GT450_EXTERNAL.items():
        data = pd.read_csv(root / "02_aggregate/translation_location_metrics.csv.gz")
        data = data.loc[data["direction"].eq("at2_to_gt450")].copy()
        data["method"] = method
        data["scanner"] = "GT450"
        data["section"] = data["stain"].astype(str)
        data["fold"] = data["checkpoint_fold"].astype(int)
        data["uni_distance"] = data["method_to_target_distance"]
        parts.append(data[KEYS + ["checkpoint_sha256", "uni_distance", "target_ssim", "raw_to_target_distance"]])
    return pd.concat(parts, ignore_index=True)


def _verified_gt450_learned(new: pd.DataFrame, prior: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    merged = new.merge(
        prior, on=KEYS + ["checkpoint_sha256"], validate="one_to_one",
        suffixes=("", "_prior"),
    )
    if len(merged) != len(new) or len(merged) != len(prior):
        raise ValueError("GT450 learned locations do not match prior frozen evaluation")
    difference = float(np.max(np.abs(merged["target_ssim"] - merged["target_ssim_prior"])))
    if difference > 1e-5:
        raise ValueError(f"GT450 image metrics disagree with frozen evaluation: {difference}")
    return merged.drop(columns=["target_ssim_prior", "raw_to_target_distance"]), difference


def aggregate() -> None:
    manifest = json.loads((OUT / "input_manifest.json").read_text())
    for name, expected in manifest["existing_table_sha256"].items():
        if base.sha256(base.TABLE_PATHS.get(name, HERE / name)) != expected:
            raise ValueError(f"existing manuscript table changed during GT450 analysis: {name}")

    conventional_old = base._read_shards(ORIGINAL / "plism_conventional/shards", 26)
    conventional_gt = base._read_shards(OUT / "plism_conventional/shards", 13)
    learned_old = base._read_shards(ORIGINAL / "plism_learned/shards", 20)
    learned_gt = base._read_shards(OUT / "plism_learned/shards", 10)
    gt_prior = gt450_existing_measurements()
    learned_gt, ssim_difference = _verified_gt450_learned(learned_gt, gt_prior)
    learned_old = learned_old.merge(base._plism_learned_uni(), on=KEYS, validate="one_to_one")
    raw_gt = conventional_gt.loc[conventional_gt["method"].eq("raw")]
    prior_raw = gt_prior.loc[gt_prior["method"].eq("pix2pix")].drop_duplicates(
        ["section", "core", "location"]
    )
    raw_check = raw_gt.merge(
        prior_raw[["section", "core", "location", "raw_to_target_distance"]],
        on=["section", "core", "location"], validate="one_to_one",
    )
    if len(raw_check) != len(raw_gt):
        raise ValueError("GT450 raw locations do not match prior frozen evaluation")
    raw_difference = float(np.max(np.abs(
        raw_check["uni_distance"] - raw_check["raw_to_target_distance"]
    )))
    if raw_difference > 1e-5:
        raise ValueError(f"GT450 raw UNI differs from prior evaluation: {raw_difference}")
    old_physical = base._plism_section(pd.concat([
        conventional_old[KEYS + METRICS], learned_old[KEYS + METRICS]
    ], ignore_index=True))
    old_main, _ = base._summaries(old_physical, "section", 2)
    published_old = pd.read_csv(HERE / "table2_main_summary.csv").query("dataset == 'PLISM'")
    old_check = old_main.merge(published_old, on="method", validate="one_to_one", suffixes=("", "_published"))
    old_difference = max(
        float(np.max(np.abs(old_check[f"{metric}_mean"] - old_check[f"{metric}_mean_published"])))
        for metric in METRICS
    )
    if len(old_check) != len(base.METHODS) or old_difference > 1e-10:
        raise ValueError(f"two-scanner benchmark was not reproduced: {old_difference}")
    conventional = pd.concat([conventional_old, conventional_gt], ignore_index=True)
    learned = pd.concat([learned_old, learned_gt], ignore_index=True)
    common = KEYS + METRICS
    physical = base._plism_section(pd.concat([conventional[common], learned[common]], ignore_index=True))
    expected_methods = set(base.METHODS)
    if set(physical["scanner"]) != {"GT450", "S360", "S60"}:
        raise ValueError("three-scanner PLISM benchmark is incomplete")
    for scanner, block in physical.groupby("scanner"):
        if set(block["method"]) != expected_methods:
            raise ValueError(f"missing method for {scanner}")
        if not block.groupby("method")["section"].nunique().eq(13).all():
            raise ValueError(f"missing PLISM section for {scanner}")
    plism_main, plism_scanner = base._summaries(physical, "section", 3)
    plism_main.insert(0, "dataset", "PLISM")
    plism_scanner.insert(0, "dataset", "PLISM")
    pan_main = pd.read_csv(HERE / "table2_main_summary.csv").query("dataset == 'PanNormal'")
    pan_scanner = pd.read_csv(HERE / "table2_scanner_summary.csv").query("dataset == 'PanNormal'")
    main = pd.concat([pan_main, plism_main], ignore_index=True)
    scanner = pd.concat([pan_scanner, plism_scanner], ignore_index=True)
    method_order = {method: index for index, method in enumerate(base.METHODS)}
    dataset_order = {"PanNormal": 0, "PLISM": 1}
    for table in (main, scanner):
        table["method_order"] = table["method"].map(method_order)
        table["dataset_order"] = table["dataset"].map(dataset_order)
    main = main.sort_values(["dataset_order", "method_order"]).drop(columns=["dataset_order", "method_order"])
    scanner = scanner.sort_values(["dataset_order", "scanner", "method_order"]).drop(columns=["dataset_order", "method_order"])
    baseline = base._baseline_table(main)
    frequency = main.loc[main["method"].isin(base.FREQUENCY_METHODS)].copy()
    frames = {
        "table2_main_summary.csv": main,
        "table2_scanner_summary.csv": scanner,
        "table2_baseline_summary.csv": baseline,
        "table_frequency_summary.csv": frequency,
    }
    for name, frame in frames.items():
        base.write_frame(OUT / name, frame)
    latex = {
        "table2_main.tex": base._main_latex(baseline),
        "table_frequency_hypothesis.tex": base._frequency_latex(main),
        "table2_scanner.tex": base._scanner_latex(scanner),
    }
    for name, content in latex.items():
        (OUT / name).write_text(content)
    output_hashes = {name: base.sha256(OUT / name) for name in [*frames, *latex]}
    base.write_json(OUT / "table2_manifest.json", {
        "aggregation": "location -> core -> section x scanner; five frozen folds averaged; equal scanner weight; 13 sections",
        "plism_scanners": ["GT450", "S360", "S60"],
        "pan_scanners": ["VERSA", "AKOYA", "GT450", "S360", "S60"],
        "metric_contract": "SSIM, LPIPS-VGG16, UNI-v1 cosine distance, identical to prior benchmark",
        "old_benchmark": str(ORIGINAL),
        "gt450_output": str(OUT),
        "gt450_learned_ssim_max_abs_difference_from_prior": ssim_difference,
        "gt450_raw_uni_max_abs_difference_from_prior": raw_difference,
        "prior_two_scanner_mean_max_abs_difference": old_difference,
        "input_manifest_sha256": base.sha256(OUT / "input_manifest.json"),
        "output_sha256": output_hashes,
    })


def publish() -> None:
    old_hashes = json.loads((OUT / "input_manifest.json").read_text())["existing_table_sha256"]
    new_hashes = json.loads((OUT / "table2_manifest.json").read_text())["output_sha256"]
    new_hashes["table2_manifest.json"] = base.sha256(OUT / "table2_manifest.json")
    for name in TABLE_FILES:
        current = base.sha256(base.TABLE_PATHS.get(name, HERE / name))
        if current not in {old_hashes[name], new_hashes[name]}:
            raise ValueError(f"manuscript analysis table changed independently: {name}")
        if base.sha256(OUT / name) != new_hashes[name]:
            raise ValueError(f"staged GT450 table changed: {name}")
    for name in TABLE_FILES:
        source = OUT / name
        target = base.TABLE_PATHS.get(name, HERE / name)
        temporary = target.with_name(f".{target.name}.gt450.tmp")
        temporary.write_bytes(source.read_bytes())
        temporary.replace(target)


def bootstrap() -> None:
    conventional_old = base._read_shards(ORIGINAL / "plism_conventional/shards", 26)
    conventional_gt = base._read_shards(OUT / "plism_conventional/shards", 13)
    learned_old = base._read_shards(ORIGINAL / "plism_learned/shards", 20)
    learned_old = learned_old.merge(base._plism_learned_uni(), on=KEYS, validate="one_to_one")
    learned_gt = base._read_shards(OUT / "plism_learned/shards", 10)
    learned_gt, _ = _verified_gt450_learned(learned_gt, gt450_existing_measurements())
    all_rows = pd.concat([
        conventional_old[KEYS + METRICS], conventional_gt[KEYS + METRICS],
        learned_old[KEYS + METRICS], learned_gt[KEYS + METRICS],
    ], ignore_index=True)
    physical = base._plism_section(all_rows)
    sections = sorted(physical["section"].unique())
    scanners = ("GT450", "S360", "S60")
    methods = base.METHODS
    columns = pd.MultiIndex.from_product([scanners, methods], names=["scanner", "method"])
    metric_arrays = []
    for metric in METRICS:
        wide = physical.pivot(index="section", columns=["scanner", "method"], values=metric)
        wide = wide.reindex(index=sections, columns=columns)
        if wide.isna().any().any():
            raise ValueError(f"incomplete section/scanner/method grid for {metric}")
        metric_arrays.append(wide.to_numpy().reshape(len(sections), len(scanners), len(methods)))
    if len(sections) != 13:
        raise ValueError("expected 13 independent physical sections")
    values = np.stack(metric_arrays, axis=-1)
    rng = np.random.default_rng(20260924)
    resamples = 20_000
    weights = rng.multinomial(len(sections), [1 / len(sections)] * len(sections), size=resamples)
    draws = (weights @ values.reshape(len(sections), -1)).reshape(
        resamples, len(scanners), len(methods), len(METRICS)
    ) / len(sections)
    point = values.mean(axis=0)
    raw_index = methods.index("raw")
    labels = {"target_ssim": "SSIM", "lpips_vgg": "LPIPS-VGG16", "uni_distance": "UNI distance"}
    rows = []
    for scope in ("pooled", *scanners):
        if scope == "pooled":
            scope_point = point.mean(axis=0)
            scope_draws = draws.mean(axis=1)
        else:
            index = scanners.index(scope)
            scope_point = point[index]
            scope_draws = draws[:, index]
        for method_index, method in enumerate(methods):
            for metric_index, metric in enumerate(METRICS):
                raw = scope_point[raw_index, metric_index]
                candidate = scope_point[method_index, metric_index]
                raw_draw = scope_draws[:, raw_index, metric_index]
                candidate_draw = scope_draws[:, method_index, metric_index]
                sign = 1.0 if metric == "target_ssim" else -1.0
                estimate = 100.0 * sign * (candidate - raw) / raw
                bootstrap_values = 100.0 * sign * (candidate_draw - raw_draw) / raw_draw
                lower, upper = np.quantile(bootstrap_values, [0.025, 0.975])
                rows.append({
                    "scanner_scope": scope, "method": method, "metric": labels[metric],
                    "improvement_pct": estimate, "ci_low": lower, "ci_high": upper,
                    "sections": len(sections), "bootstrap_replicates": resamples,
                    "bootstrap_seed": 20260924,
                })
    result = pd.DataFrame(rows)
    published = pd.read_csv(HERE / "table2_main_summary.csv").query("dataset == 'PLISM'").set_index("method")
    for method in methods:
        for metric in METRICS:
            label = labels[metric]
            observed = result.loc[
                result["scanner_scope"].eq("pooled") & result["method"].eq(method)
                & result["metric"].eq(label), "improvement_pct"
            ].iloc[0]
            expected = published.loc[method, f"{metric}_improvement_pct"]
            if abs(observed - expected) > 1e-10:
                raise ValueError(f"bootstrap point estimate differs from published {method}/{metric}")
    base.write_frame(OUT / "plism_paired_section_bootstrap.csv", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("prepare")
    conventional = sub.add_parser("conventional")
    conventional.add_argument("--task-index", type=int, required=True)
    conventional.add_argument("--batch-size", type=int, default=8)
    conventional.add_argument("--uni-batch-size", type=int, default=32)
    learned = sub.add_parser("learned")
    learned.add_argument("--task-index", type=int, required=True)
    learned.add_argument("--batch-size", type=int, default=8)
    learned.add_argument("--generator-batch-size", type=int, default=16)
    sub.add_parser("aggregate")
    sub.add_parser("publish")
    sub.add_parser("bootstrap")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "conventional":
        conventional_task(args.task_index, args.batch_size, args.uni_batch_size)
    elif args.stage == "learned":
        learned_task(args.task_index, args.batch_size, args.generator_batch_size)
    elif args.stage == "aggregate":
        aggregate()
    elif args.stage == "bootstrap":
        bootstrap()
    else:
        publish()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Uniform image/PFM benchmark used by the manuscript correction tables.

The script never trains or selects a model.  It evaluates frozen PanNormal
predictions/parameters and frozen PLISM renders/checkpoints with one common
metric contract: SSIM, LPIPS-VGG16, and UNI-v1 cosine distance.  Aggregation
is performed at the physical slide (PanNormal) or section (PLISM) level.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from manuscript_completion.panel_a import (
    load_vahadane_selection,
    selected_vahadane_kwargs,
    stable_seed,
)
from manuscript_completion.plism_external import (
    _apply_stain_from_source,
    _stain_source_parameters,
)
from manuscript_completion.stain import normalize, parameters_from_json
from scanner_batch_extensions import frequency_transform, reinhard_transform
from scanner_gan.evaluate_images import image_metrics
from scanner_gan.plism import _generate_uint8, load_frozen_uni1
from scanner_gan.plism_bidirectional import _load_render
from scanner_gan.predict import load_validated_generator
from scanner_gan.train import sha256
from scanner_gan.train_cyclegan import load_validated_cyclegan_generator
from scanner_gan.uni import embed_frozen_uni


REPOSITORY = Path(__file__).resolve().parents[2]
EVIDENCE = REPOSITORY / "outputs/scanner_batch_effect_analysis_2026-09-17"
COMPLETION = EVIDENCE / "12_manuscript_completion"
PANEL_A = COMPLETION / "02_baseline_benchmark"
PLISM_CONVENTIONAL = COMPLETION / "05_plism_external_correction"
PLISM_LEARNED = COMPLETION / "06_plism_learned_external"
OUTPUT = COMPLETION / "09_table2_benchmark"
TABLE_PATHS = {
    "table2_main.tex": REPOSITORY / "analysis/paper/results/table2_benchmark/table2_main.tex",
    "table_frequency_hypothesis.tex": REPOSITORY / "analysis/paper/results/table2_benchmark/table_frequency_hypothesis.tex",
    "table2_scanner.tex": REPOSITORY / "analysis/paper/results/table2_benchmark/table_scanner_correction_benchmark.tex",
}

TARGETS = ("versa", "akoya", "gt450", "s360", "s60")
METHODS = ("raw", "reinhard", "macenko", "vahadane", "frequency", "combined", "pix2pix", "cyclegan")
BASELINE_METHODS = ("raw", "reinhard", "macenko", "vahadane", "pix2pix", "cyclegan")
FREQUENCY_METHODS = ("raw", "reinhard", "frequency", "combined")
BASELINE_CLASSES = {
    "raw": "Baseline",
    "reinhard": "Stain Norm",
    "macenko": "Stain Norm",
    "vahadane": "Stain Norm",
    "pix2pix": "Style Transfer",
    "cyclegan": "Style Transfer",
}
CLASSES = {
    "raw": "Baseline",
    "reinhard": "Statistical colour transfer",
    "macenko": "Stain normalization",
    "vahadane": "Stain normalization",
    "frequency": "Radial frequency-response correction",
    "combined": "Colour + radial frequency-response correction",
    "pix2pix": "Paired neural translation",
    "cyclegan": "Unpaired neural translation",
}
METHOD_LABELS = {
    "raw": "Raw",
    "reinhard": "Reinhard",
    "macenko": "Macenko",
    "vahadane": "Vahadane",
    "frequency": "Radial frequency-response matching",
    "combined": "Colour + radial frequency-response matching",
    "pix2pix": "Pix2Pix",
    "cyclegan": "CycleGAN",
}
CLASS_LABELS = {
    "Baseline": "Baseline",
    "Statistical colour transfer": "Statistical colour transfer",
    "Stain normalization": "Stain normalization",
    "Radial frequency-response correction": "Radial frequency response",
    "Colour + radial frequency-response correction": "Colour + radial frequency",
    "Paired neural translation": "Paired neural translation",
    "Unpaired neural translation": "Unpaired neural translation",
}


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, compression="gzip" if path.suffix == ".gz" else None)
    temporary.replace(path)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def lpips_model(device: torch.device) -> LearnedPerceptualImagePatchSimilarity:
    # normalize=True documents that callers provide [0, 1], not [-1, 1].
    model = LearnedPerceptualImagePatchSimilarity(
        net_type="vgg", reduction="none", normalize=True
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def lpips_scores(
    model: LearnedPerceptualImagePatchSimilarity,
    first: np.ndarray,
    second: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if first.shape != second.shape or first.ndim != 4 or first.shape[-1] != 3:
        raise ValueError(f"invalid LPIPS pair shapes: {first.shape}, {second.shape}")
    values: list[np.ndarray] = []
    for start in range(0, len(first), batch_size):
        left = torch.from_numpy(first[start : start + batch_size]).permute(0, 3, 1, 2)
        right = torch.from_numpy(second[start : start + batch_size]).permute(0, 3, 1, 2)
        left = left.to(device=device, dtype=torch.float32).div(255.0)
        right = right.to(device=device, dtype=torch.float32).div(255.0)
        with torch.inference_mode():
            score = model(left, right)
        values.append(score.detach().float().cpu().numpy().reshape(-1))
    return np.concatenate(values)


def prepare_lpips() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = lpips_model(device)
    first = np.zeros((1, 64, 64, 3), dtype=np.uint8)
    score = float(lpips_scores(model, first, first, device, 1)[0])
    torch_home = Path(os.environ.get("TORCH_HOME", ""))
    weights = sorted(
        str(path.resolve())
        for path in torch_home.rglob("*")
        if path.is_file() and path.suffix in {".pth", ".pt"}
    ) if torch_home.is_dir() else []
    write_json(
        OUTPUT / "lpips_contract.json",
        {
            "metric": "LearnedPerceptualImagePatchSimilarity",
            "network": "vgg",
            "network_interpretation": "LPIPS-VGG16",
            "normalize": True,
            "reduction": "none",
            "identical_image_smoke_score": score,
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("torch", "torchvision", "torchmetrics")
            },
            "torch_home": str(torch_home.resolve()) if torch_home else "",
            "weight_files": [
                {"path": name, "sha256": sha256(name)} for name in weights
            ],
        },
    )


def _prediction_root(method: str, target: str) -> Path:
    if target != "gt450":
        return PANEL_A / f"{'03_pix2pix' if method == 'pix2pix' else '04_cyclegan'}/02_predictions"
    subdir = "09_bidirectional_full_training" if method == "pix2pix" else "11_cyclegan_full_training"
    return EVIDENCE / f"06_learned_baselines/{subdir}/02_predictions"


def _slide_prediction(method: str, target: str, fold: int, slide_id: str) -> Path:
    matches = sorted(
        (_prediction_root(method, target) / f"at2_to_{target}/fold_{fold}/slides").glob(
            f"{slide_id}__*.h5"
        )
    )
    if len(matches) != 1:
        raise ValueError(f"expected one prediction for {method}/{target}/{fold}/{slide_id}, got {matches}")
    return matches[0]


def _load_prediction(path: Path, locations: np.ndarray) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as store:
        observed = np.asarray(store["metadata/location_index"], dtype=int)
        lookup = {int(value): index for index, value in enumerate(observed)}
        indices = np.asarray([lookup[int(value)] for value in locations], dtype=int)
        return {
            key: np.asarray(store[key][indices], dtype=np.uint8)
            for key in (
                "images/raw_source", "images/real_target", "images/generated",
                "aligned_valid/raw_source_valid", "aligned_valid/real_target_valid",
                "aligned_valid/generated_valid",
            )
        }


def _crop_valid(images: np.ndarray) -> np.ndarray:
    if images.shape[1:3] != (256, 256):
        raise ValueError(f"expected 256-pixel full images, got {images.shape}")
    return images[:, 2:-2, 2:-2]


def pannormal_task(task_index: int, batch_size: int) -> None:
    uni_locations = pd.read_csv(EVIDENCE / "03_uni/location_metrics.csv", dtype={"slide_id": str})
    locked = pd.read_csv(
        EVIDENCE / "06_learned_baselines/00_contract/locked_image_evaluation_index.csv.gz",
        dtype={"slide_id": str},
    )
    slides = sorted(locked["slide_id"].unique())
    if not 0 <= task_index < len(slides):
        raise IndexError(task_index)
    slide = slides[task_index]
    selected = (
        uni_locations.loc[uni_locations["slide_id"].eq(slide), "location_index"]
        .astype(int).drop_duplicates().sort_values().to_numpy()
    )
    if len(selected) != 20:
        raise ValueError(f"{slide}: expected 20 UNI locations, found {len(selected)}")
    fold = int(locked.loc[locked["slide_id"].eq(slide), "fold"].iloc[0])
    tissue = str(locked.loc[locked["slide_id"].eq(slide), "tissue_type"].iloc[0])
    parameters = json.loads((EVIDENCE / "02_correction/parameters.json").read_text())["fold"][str(fold)]
    selection = load_vahadane_selection(PANEL_A / "00_contract/vahadane_selection_manifest_v4.json")
    vahadane_config = selected_vahadane_kwargs(selection, reference=False)

    device = torch.device("cuda")
    lpips = lpips_model(device)
    rows: list[pd.DataFrame] = []
    for target in TARGETS:
        pix = _load_prediction(_slide_prediction("pix2pix", target, fold, slide), selected)
        cyc = _load_prediction(_slide_prediction("cyclegan", target, fold, slide), selected)
        if not np.array_equal(pix["aligned_valid/raw_source_valid"], cyc["aligned_valid/raw_source_valid"]):
            raise ValueError(f"source mismatch between learned methods for {slide}/{target}")
        source_full = pix["images/raw_source"]
        target_full = pix["images/real_target"]
        source_valid = pix["aligned_valid/raw_source_valid"]
        target_valid = pix["aligned_valid/real_target_valid"]
        reference_payload = json.loads(
            (PANEL_A / f"01_stain_v4_mu_convergence/references/fold_{fold}/{target}.json").read_text()
        )
        references = {
            method: parameters_from_json(reference_payload["parameters"][method])
            for method in ("macenko", "vahadane")
        }
        candidates: dict[str, np.ndarray] = {
            "raw": source_valid,
            "pix2pix": pix["aligned_valid/generated_valid"],
            "cyclegan": cyc["aligned_valid/generated_valid"],
        }
        conventional_full: dict[str, list[np.ndarray]] = {
            method: [] for method in ("reinhard", "frequency", "combined", "macenko", "vahadane")
        }
        for index, image in enumerate(source_full):
            reinhard = reinhard_transform(image, parameters[target])
            conventional_full["reinhard"].append(reinhard)
            conventional_full["frequency"].append(frequency_transform(image, parameters[target]))
            conventional_full["combined"].append(frequency_transform(reinhard, parameters[target]))
            for method in ("macenko", "vahadane"):
                result = normalize(
                    image, references[method], method=method,
                    seed=stable_seed(slide, int(selected[index]), target, fold, method),
                    vahadane_config=vahadane_config if method == "vahadane" else None,
                )
                conventional_full[method].append(result.image)
        for method, images in conventional_full.items():
            candidates[method] = _crop_valid(np.stack(images))
        for method in METHODS:
            generated = candidates[method]
            metrics = image_metrics(source_valid, generated, target_valid)
            metrics["lpips_vgg"] = lpips_scores(lpips, generated, target_valid, device, batch_size)
            metrics["dataset"] = "PanNormal"
            metrics["slide_id"] = slide
            metrics["tissue_type"] = tissue
            metrics["scanner"] = target.upper()
            metrics["method"] = method
            metrics["method_class"] = CLASSES[method]
            metrics["fold"] = fold
            metrics["location_index"] = selected
            rows.append(metrics)
    output = pd.concat(rows, ignore_index=True)
    write_frame(OUTPUT / f"pannormal/shards/{slide}.csv.gz", output)


def _plism_inputs(method: str = "pix2pix") -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    contract = json.loads((PLISM_LEARNED / f"{method}/00_contract/analysis_contract.json").read_text())
    selected = pd.read_csv(contract["outputs"]["selected_locations"]["path"])
    renders = pd.read_csv(contract["outputs"]["read_only_renders"]["path"])
    return contract, selected, renders


def _plism_pair(
    selected: pd.DataFrame, renders: pd.DataFrame, scanner: str, stain: str
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    source, source_locations, _ = _load_render(renders, "AT2", stain)
    target, target_locations, _ = _load_render(renders, scanner, stain)
    if not np.array_equal(source_locations, target_locations):
        raise ValueError(f"PLISM paired locations differ for {scanner}/{stain}")
    metadata = selected.loc[selected["stain"].astype(str).eq(stain)].set_index("location")
    metadata = metadata.reindex(source_locations).reset_index()
    if metadata["core"].isna().any():
        raise ValueError(f"missing PLISM metadata for {scanner}/{stain}")
    return metadata, source, target


def plism_conventional_task(task_index: int, batch_size: int, uni_batch_size: int) -> None:
    contract, selected, renders = _plism_inputs()
    stains = sorted(selected["stain"].astype(str).unique())
    tasks = [(scanner, stain) for scanner in ("S360", "S60") for stain in stains]
    if not 0 <= task_index < len(tasks):
        raise IndexError(task_index)
    scanner, stain = tasks[task_index]
    metadata, source, target = _plism_pair(selected, renders, scanner, stain)
    parameter_folds = json.loads((EVIDENCE / "02_correction/parameters.json").read_text())["fold"]
    selection = load_vahadane_selection(PANEL_A / "00_contract/vahadane_selection_manifest_v4.json")
    vahadane_configuration = selection["selected_configuration"]
    source_stain = [
        _stain_source_parameters(
            image, stable_seed("table2", stain, int(location)), vahadane_configuration
        )
        for image, location in zip(source, metadata["location"].astype(int))
    ]

    candidate_rows: list[tuple[str, int, np.ndarray]] = [("raw", -1, source)]
    for fold in range(5):
        parameter = parameter_folds[str(fold)][scanner.casefold()]
        reference_payload = json.loads(
            (PANEL_A / f"01_stain_v4_mu_convergence/references/fold_{fold}/{scanner.casefold()}.json").read_text()
        )
        references = {
            method: parameters_from_json(reference_payload["parameters"][method])
            for method in ("macenko", "vahadane")
        }
        generated: dict[str, list[np.ndarray]] = {
            method: [] for method in ("reinhard", "frequency", "combined", "macenko", "vahadane")
        }
        for image, fitted in zip(source, source_stain):
            reinhard = reinhard_transform(image, parameter)
            generated["reinhard"].append(reinhard)
            generated["frequency"].append(frequency_transform(image, parameter))
            generated["combined"].append(frequency_transform(reinhard, parameter))
            for method in ("macenko", "vahadane"):
                corrected, _ = _apply_stain_from_source(image, fitted[method], references[method])
                generated[method].append(corrected)
        candidate_rows.extend((method, fold, np.stack(images)) for method, images in generated.items())

    device = torch.device("cuda")
    lpips = lpips_model(device)
    frames: list[pd.DataFrame] = []
    for method, fold, images in candidate_rows:
        metrics = image_metrics(source, images, target)
        metrics["lpips_vgg"] = lpips_scores(lpips, images, target, device, batch_size)
        frame = pd.concat([metadata.reset_index(drop=True), metrics], axis=1)
        frame["dataset"] = "PLISM"
        frame["section"] = stain
        frame["scanner"] = scanner
        frame["method"] = method
        frame["method_class"] = CLASSES[method]
        frame["fold"] = fold
        frames.append(frame)
    del lpips
    torch.cuda.empty_cache()

    encoder = contract["primary_encoder"]
    uni, size, mean, std, _ = load_frozen_uni1(
        device,
        checkpoint_path=encoder["checkpoint_path"],
        expected_sha256=encoder["checkpoint_sha256"],
    )
    target_features = embed_frozen_uni(
        uni, target, size, mean, std, device, batch_size=uni_batch_size, value_range="uint8"
    )
    for frame, (_, _, images) in zip(frames, candidate_rows):
        generated_features = embed_frozen_uni(
            uni, images, size, mean, std, device,
            batch_size=uni_batch_size, value_range="uint8",
        )
        frame["uni_distance"] = 1.0 - np.einsum("ij,ij->i", generated_features, target_features)
    write_frame(OUTPUT / f"plism_conventional/shards/{scanner}_{stain}.csv.gz", pd.concat(frames, ignore_index=True))


def plism_learned_task(task_index: int, batch_size: int, generator_batch_size: int) -> None:
    tasks = [(method, index) for method in ("pix2pix", "cyclegan") for index in range(10)]
    if not 0 <= task_index < len(tasks):
        raise IndexError(task_index)
    method, candidate_index = tasks[task_index]
    contract, selected, renders = _plism_inputs(method)
    candidates = pd.read_csv(contract["outputs"]["checkpoint_candidates"]["path"])
    candidate = candidates.loc[candidates["task_index"].astype(int).eq(candidate_index)].iloc[0]
    source_scanner = str(candidate["source_scanner"]).casefold()
    target_scanner = str(candidate["target_scanner"]).casefold()
    fold = int(candidate["fold"])
    checkpoint = Path(str(candidate["checkpoint_path"])).resolve()
    if sha256(checkpoint) != str(candidate["checkpoint_sha256"]):
        raise ValueError("frozen checkpoint hash mismatch")
    device = torch.device("cuda")
    loader = load_validated_generator if method == "pix2pix" else load_validated_cyclegan_generator
    generator, _, _ = loader(
        checkpoint, source_scanner=source_scanner, target_scanner=target_scanner,
        test_fold=fold, device=device,
    )
    lpips = lpips_model(device)
    frames: list[pd.DataFrame] = []
    for stain in sorted(selected["stain"].astype(str).unique()):
        metadata, source, target = _plism_pair(selected, renders, target_scanner.upper(), stain)
        generated = _generate_uint8(
            generator, source, device, batch_size=generator_batch_size, amp="bfloat16"
        )
        metrics = image_metrics(source, generated, target)
        metrics["lpips_vgg"] = lpips_scores(lpips, generated, target, device, batch_size)
        frame = pd.concat([metadata.reset_index(drop=True), metrics], axis=1)
        frame["dataset"] = "PLISM"
        frame["section"] = stain
        frame["scanner"] = target_scanner.upper()
        frame["method"] = method
        frame["method_class"] = CLASSES[method]
        frame["fold"] = fold
        frames.append(frame)
    write_frame(
        OUTPUT / f"plism_learned/shards/{method}_{target_scanner}_fold_{fold}.csv.gz",
        pd.concat(frames, ignore_index=True),
    )


def _read_shards(path: Path, expected: int) -> pd.DataFrame:
    files = sorted(path.glob("*.csv.gz"))
    if len(files) != expected:
        raise RuntimeError(f"expected {expected} shards in {path}, found {len(files)}")
    return pd.concat([pd.read_csv(file, dtype={"slide_id": str}) for file in files], ignore_index=True)


def _pan_uni() -> pd.DataFrame:
    data = pd.read_csv(PANEL_A / "baseline_location_or_slide_metrics.csv.gz", dtype={"slide_id": str})
    data = data.loc[data["space"].eq("uni")].copy()
    data["scanner"] = data["scanner"].astype(str).str.upper()
    learned = data["method"].isin(["pix2pix", "cyclegan"])
    data["uni_distance"] = 1.0 - data["generated_target_cosine"]
    data.loc[learned, "uni_distance"] = data.loc[learned, "method_to_target_distance"]
    if data["uni_distance"].isna().any():
        raise ValueError("PanNormal UNI distances contain missing values")
    return data[["slide_id", "scanner", "method", "uni_distance"]]


def _plism_learned_uni() -> pd.DataFrame:
    parts = []
    for method in ("pix2pix", "cyclegan"):
        data = pd.read_csv(
            PLISM_LEARNED / f"{method}/02_aggregate/translation_location_metrics.csv.gz"
        )
        data = data.loc[data["direction"].isin(["at2_to_s360", "at2_to_s60"])].copy()
        data["method"] = method
        data["scanner"] = data["target_scanner"].str.upper()
        data["section"] = data["stain"].astype(str)
        data["fold"] = data["checkpoint_fold"].astype(int)
        data["uni_distance"] = data["method_to_target_distance"]
        parts.append(data[["section", "core", "location", "scanner", "fold", "method", "uni_distance"]])
    return pd.concat(parts, ignore_index=True)


def _plism_section(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = ["target_ssim", "lpips_vgg", "uni_distance"]
    location_keys = ["section", "core", "scanner", "method", "fold"]
    core = frame.groupby(location_keys, as_index=False)[metrics].mean()
    section = core.groupby(["section", "scanner", "method", "fold"], as_index=False)[metrics].mean()
    raw = section.loc[section["method"].eq("raw")].copy()
    nonraw = section.loc[~section["method"].eq("raw")]
    nonraw = nonraw.groupby(["section", "scanner", "method"], as_index=False)[metrics].mean()
    raw = raw.drop(columns="fold")
    return pd.concat([raw, nonraw], ignore_index=True)


def _summaries(physical: pd.DataFrame, unit: str, scanners: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = ["target_ssim", "lpips_vgg", "uni_distance"]
    scanner = (
        physical.groupby(["scanner", "method"])[metrics]
        .agg(["mean", "std"]).reset_index()
    )
    scanner.columns = ["scanner", "method"] + [f"{metric}_{stat}" for metric, stat in scanner.columns[2:]]
    scanner["n_physical_units"] = physical[unit].nunique()
    scanner["method_class"] = scanner["method"].map(CLASSES)
    macro = physical.groupby([unit, "method"], as_index=False)[metrics].mean()
    main = macro.groupby("method")[metrics].agg(["mean", "std"]).reset_index()
    main.columns = ["method"] + [f"{metric}_{stat}" for metric, stat in main.columns[1:]]
    main["n_physical_units"] = physical[unit].nunique()
    main["n_scanners"] = scanners
    main["method_class"] = main["method"].map(CLASSES)
    for table, grouping in ((scanner, "scanner"), (main, None)):
        for metric in metrics:
            raw = table.loc[table["method"].eq("raw"), [metric + "_mean"] + ([grouping] if grouping else [])]
            if grouping:
                raw_lookup = raw.set_index(grouping)[metric + "_mean"]
                reference = table[grouping].map(raw_lookup)
            else:
                reference = pd.Series(float(raw[metric + "_mean"].iloc[0]), index=table.index)
            if metric == "target_ssim":
                improvement = 100.0 * (table[metric + "_mean"] - reference) / reference
            else:
                improvement = 100.0 * (reference - table[metric + "_mean"]) / reference
            table[metric + "_improvement_pct"] = improvement.where(~table["method"].eq("raw"), 0.0)
    return main, scanner


def _metric_cells(row: pd.Series) -> list[str]:
    cells: list[str] = []
    for metric in ("target_ssim", "lpips_vgg", "uni_distance"):
        decimals = 4 if metric == "uni_distance" else 3
        cells.append(
            f"{row[metric + '_mean']:.{decimals}f} $\\pm$ "
            f"{row[metric + '_std']:.{decimals}f}"
        )
        cells.append("Ref." if row["method"] == "raw" else f"{row[metric + '_improvement_pct']:+.1f}")
    return cells


def _baseline_table(main: pd.DataFrame) -> pd.DataFrame:
    baseline = main.loc[main["method"].isin(BASELINE_METHODS)].copy()
    baseline["method_class"] = baseline["method"].map(BASELINE_CLASSES)
    rank_contract = {
        "target_ssim": False,
        "lpips_vgg": True,
        "uni_distance": True,
    }
    for metric, ascending in rank_contract.items():
        baseline[metric + "_rank"] = baseline.groupby("dataset")[metric + "_mean"].rank(
            method="average", ascending=ascending
        )
    baseline["average_rank"] = baseline[
        [metric + "_rank" for metric in rank_contract]
    ].mean(axis=1)
    return baseline


def _rank_text(value: float) -> str:
    return f"{int(value)}" if float(value).is_integer() else f"{value:.1f}"


def _ranked_metric_cells(row: pd.Series) -> list[str]:
    cells: list[str] = []
    for metric in ("target_ssim", "lpips_vgg", "uni_distance"):
        decimals = 4 if metric == "uni_distance" else 3
        cells.extend(
            [
                f"{row[metric + '_mean']:.{decimals}f} $\\pm$ "
                f"{row[metric + '_std']:.{decimals}f}",
                "Ref."
                if row["method"] == "raw"
                else f"{row[metric + '_improvement_pct']:+.1f}",
                _rank_text(float(row[metric + "_rank"])),
            ]
        )
    return cells


def _main_latex(baseline: pd.DataFrame) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"    \caption{PanNormal과 PLISM의 영상 교정법별 paired-target 성능.}",
        r"    \label{table_image_correction_benchmark}",
        r"    \centering",
        r"    \scriptsize",
        r"    \setlength{\tabcolsep}{2.5pt}",
        r"    \begin{tabularx}{\textwidth}{@{}CCC*{10}{c}@{}}",
        r"        \toprule",
        "        \\multirow[c]{2}{=}{Dataset} & \\multirow[c]{2}{=}{Model class} & \\multirow[c]{2}{=}{Methods} & \\multicolumn{3}{c}{SSIM $\\uparrow$} & \\multicolumn{3}{c}{LPIPS--VGG16 $\\downarrow$} & \\multicolumn{3}{c}{UNI distance $\\downarrow$} & \\multirow[c]{2}{*}{Avg. rank $\\downarrow$} \\\\",
        r"        \cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(lr){10-12}",
        "        & & & Mean $\\pm$ SD & $\\Delta$ raw (\\%) & Rank & Mean $\\pm$ SD & $\\Delta$ raw (\\%) & Rank & Mean $\\pm$ SD & $\\Delta$ raw (\\%) & Rank & \\\\",
        r"        \midrule",
    ]
    dataset_labels = {"PanNormal": "Pan Normal", "PLISM": "PLISM"}
    for dataset_index, dataset in enumerate(("PanNormal", "PLISM")):
        block = (
            baseline.loc[baseline["dataset"].eq(dataset)]
            .set_index("method")
            .reindex(BASELINE_METHODS)
            .reset_index()
        )
        class_counts = block["method_class"].value_counts().to_dict()
        for row_index, row in block.iterrows():
            dataset_cell = (
                rf"\multirow[c]{{-{len(BASELINE_METHODS)}}}{{=}}{{{dataset_labels[dataset]}}}"
                if row_index == len(BASELINE_METHODS) - 1
                else ""
            )
            method_class = str(row["method_class"])
            is_class_end = (
                row_index == len(block) - 1
                or block.iloc[row_index + 1]["method_class"] != method_class
            )
            count = int(class_counts[method_class])
            class_cell = (
                (rf"\multirow[c]{{-{count}}}{{=}}{{{method_class}}}" if count > 1 else method_class)
                if is_class_end
                else ""
            )
            shade = r"\rowcolor{black!5} " if method_class == "Stain Norm" else ""
            cells = " & ".join(_ranked_metric_cells(row))
            lines.append(
                f"        {shade}{dataset_cell} & {class_cell} & {METHOD_LABELS[row['method']]} & "
                f"{cells} & {_rank_text(float(row['average_rank']))} \\\\"
            )
        if dataset_index == 0:
            lines.append(r"        \midrule")
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabularx}",
            r"    \vspace{2pt}\parbox{\textwidth}{\scriptsize 화살표는 좋은 방향을 나타낸다. $\Delta$는 Raw 대비 상대 개선율이며 양수가 개선이다. Rank는 각 dataset의 여섯 방법 안에서 계산했으며 1이 가장 좋고, Avg. rank는 세 metric rank의 산술평균이다. Pan Normal은 103개 physical slide, PLISM은 13개 physical section의 mean $\pm$ sample SD이다. PLISM에는 재학습 또는 target 기반 재적합을 하지 않았다.}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def _frequency_latex(main: pd.DataFrame) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"    \caption{PanNormal과 PLISM의 색 및 주파수 교정에 따른 paired-target 성능.}",
        r"    \label{table_color_frequency_correction}",
        r"    \centering",
        r"    \scriptsize",
        r"    \begin{tabularx}{\textwidth}{@{}CCCrrrrrr@{}}",
        r"        \toprule",
        "        \\multirow[c]{2}{=}{Dataset} & \\multirow[c]{2}{=}{Color} & \\multirow[c]{2}{=}{Frequency} & \\multicolumn{2}{c}{SSIM $\\uparrow$} & \\multicolumn{2}{c}{LPIPS--VGG16 $\\downarrow$} & \\multicolumn{2}{c}{UNI distance $\\downarrow$} \\\\",
        r"        \cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        "        & & & Mean $\\pm$ SD & $\\Delta$ raw (\\%) & Mean $\\pm$ SD & $\\Delta$ raw (\\%) & Mean $\\pm$ SD & $\\Delta$ raw (\\%) \\\\",
        r"        \midrule",
    ]
    indicators = {
        "raw": (r"$\times$", r"$\times$"),
        "reinhard": (r"$\circ$", r"$\times$"),
        "frequency": (r"$\times$", r"$\circ$"),
        "combined": (r"$\circ$", r"$\circ$"),
    }
    dataset_labels = {"PanNormal": "Pan Normal", "PLISM": "PLISM"}
    for dataset_index, dataset in enumerate(("PanNormal", "PLISM")):
        block = (
            main.loc[main["dataset"].eq(dataset)]
            .set_index("method").reindex(FREQUENCY_METHODS).reset_index()
        )
        for row_index, row in block.iterrows():
            dataset_cell = (
                rf"\multirow[c]{{{len(FREQUENCY_METHODS)}}}{{=}}{{{dataset_labels[dataset]}}}"
                if row_index == 0
                else ""
            )
            colour, frequency = indicators[str(row["method"])]
            lines.append(
                f"        {dataset_cell} & {colour} & {frequency} & "
                + " & ".join(_metric_cells(row))
                + " \\\\"
            )
        if dataset_index == 0:
            lines.append(r"        \midrule")
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabularx}",
            r"    \vspace{2pt}\parbox{\textwidth}{\scriptsize $\circ$는 적용, $\times$는 미적용이다. Color는 Reinhard, Frequency는 radial frequency-response matching이며, $\Delta$는 Raw 대비 상대 개선율이다.}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def _scanner_latex(scanner: pd.DataFrame) -> str:
    lines = [
        r"\begin{landscape}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{longtable}{@{}lllrrrrrrr@{}}",
        r"\caption{PanNormal과 PLISM의 scanner별 영상 교정 성능.}\label{table_scanner_correction_benchmark}\\",
        r"\toprule",
        r"& & & \multicolumn{2}{c}{SSIM $\uparrow$} & \multicolumn{2}{c}{LPIPS--VGG16 $\downarrow$} & \multicolumn{2}{c}{UNI cosine distance $\downarrow$} & Biology ratio\\",
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        r"Scanner & Model class & Method & Mean $\pm$ SD & $\Delta$ (\%) & Mean $\pm$ SD & $\Delta$ (\%) & Mean $\pm$ SD & $\Delta$ (\%) & (\%)\\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{10}{l}{\tablename\ \thetable\ continued}\\",
        r"\toprule",
        r"& & & \multicolumn{2}{c}{SSIM $\uparrow$} & \multicolumn{2}{c}{LPIPS--VGG16 $\downarrow$} & \multicolumn{2}{c}{UNI cosine distance $\downarrow$} & Biology ratio\\",
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        r"Scanner & Model class & Method & Mean $\pm$ SD & $\Delta$ (\%) & Mean $\pm$ SD & $\Delta$ (\%) & Mean $\pm$ SD & $\Delta$ (\%) & (\%)\\",
        r"\midrule",
        r"\endhead",
        r"\midrule\multicolumn{10}{r}{Continued on next page}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    plism_present = set(scanner.loc[scanner["dataset"].eq("PLISM"), "scanner"])
    scanner_order = {
        "PanNormal": ("VERSA", "AKOYA", "GT450", "S360", "S60"),
        "PLISM": tuple(name for name in ("GT450", "S360", "S60") if name in plism_present),
    }
    for dataset in ("PanNormal", "PLISM"):
        lines.append(rf"\multicolumn{{10}}{{@{{}}l}}{{\textbf{{{dataset}}}}}\\")
        for target in scanner_order[dataset]:
            block = scanner.loc[scanner["dataset"].eq(dataset) & scanner["scanner"].eq(target)]
            block = block.set_index("method").reindex(METHODS).reset_index()
            class_counts = block["method_class"].value_counts().to_dict()
            seen_classes: set[str] = set()
            for row_index, row in block.iterrows():
                scanner_cell = rf"\multirow{{8}}{{*}}{{{target}}}" if row_index == 0 else ""
                method_class = str(row["method_class"])
                if method_class in seen_classes:
                    class_cell = ""
                else:
                    seen_classes.add(method_class)
                    count = int(class_counts[method_class])
                    label = CLASS_LABELS[method_class]
                    class_cell = rf"\multirow{{{count}}}{{*}}{{{label}}}" if count > 1 else label
                cells = _metric_cells(row)
                ratio = "--"
                if dataset == "PanNormal" and row["method"] == "raw":
                    ratio = f"{100.0 * row['uni_distance_mean'] / 0.877784779:.1f}"
                lines.append(
                    f"{scanner_cell} & {class_cell} & {METHOD_LABELS[row['method']]} & "
                    + " & ".join(cells)
                    + f" & {ratio} \\\\"
                )
            lines.append(r"\addlinespace")
    lines.extend(
        [
            r"\end{longtable}",
            r"\noindent\footnotesize 값은 각 scanner에서 physical slide (PanNormal, $n=103$) 또는 section (PLISM, $n=13$) 사이의 mean $\pm$ sample SD이다. $\Delta$는 같은 scanner의 Raw 대비 상대 개선율이며 양수가 개선이다. Biology ratio는 기존 Table 1의 정의를 이어받아 PanNormal Raw UNI distance를 AT2 between-tissue distance 0.8778로 나눈 값이다. PLISM의 target parameter와 checkpoint는 PanNormal에서 고정했으며 PLISM에 재적합하지 않았다.",
            r"\end{landscape}",
        ]
    )
    return "\n".join(lines) + "\n"


def aggregate() -> None:
    published_manifest = Path(__file__).parent / "table2_manifest.json"
    if published_manifest.exists():
        published = json.loads(published_manifest.read_text())
        if published.get("plism_scanners") == ["GT450", "S360", "S60"]:
            raise RuntimeError(
                "the published benchmark includes PLISM GT450; regenerate it "
                "with table2_gt450_extension.py aggregate and publish"
            )
    pan = _read_shards(OUTPUT / "pannormal/shards", 103)
    pan_image = pan.groupby(["slide_id", "scanner", "method"], as_index=False)[["target_ssim", "lpips_vgg"]].mean()
    pan_physical = pan_image.merge(_pan_uni(), on=["slide_id", "scanner", "method"], validate="one_to_one")
    pan_main, pan_scanner = _summaries(pan_physical, "slide_id", 5)
    pan_main.insert(0, "dataset", "PanNormal")
    pan_scanner.insert(0, "dataset", "PanNormal")

    conventional = _read_shards(OUTPUT / "plism_conventional/shards", 26)
    learned = _read_shards(OUTPUT / "plism_learned/shards", 20)
    learned_uni = _plism_learned_uni()
    learned = learned.merge(
        learned_uni,
        on=["section", "core", "location", "scanner", "fold", "method"],
        validate="one_to_one",
    )
    common = ["section", "core", "location", "scanner", "fold", "method", "target_ssim", "lpips_vgg", "uni_distance"]
    plism_physical = _plism_section(pd.concat([conventional[common], learned[common]], ignore_index=True))
    plism_main, plism_scanner = _summaries(plism_physical, "section", 2)
    plism_main.insert(0, "dataset", "PLISM")
    plism_scanner.insert(0, "dataset", "PLISM")

    order = {method: index for index, method in enumerate(METHODS)}
    main = pd.concat([pan_main, plism_main], ignore_index=True)
    scanner = pd.concat([pan_scanner, plism_scanner], ignore_index=True)
    main["method_order"] = main["method"].map(order)
    scanner["method_order"] = scanner["method"].map(order)
    dataset_order = {"PanNormal": 0, "PLISM": 1}
    main["dataset_order"] = main["dataset"].map(dataset_order)
    scanner["dataset_order"] = scanner["dataset"].map(dataset_order)
    main = main.sort_values(["dataset_order", "method_order"]).drop(columns=["dataset_order", "method_order"])
    scanner = scanner.sort_values(["dataset_order", "scanner", "method_order"]).drop(columns=["dataset_order", "method_order"])
    write_frame(OUTPUT / "table2_main_summary.csv", main)
    write_frame(OUTPUT / "table2_scanner_summary.csv", scanner)
    write_frame(Path(__file__).parent / "table2_main_summary.csv", main)
    write_frame(Path(__file__).parent / "table2_scanner_summary.csv", scanner)
    baseline = _baseline_table(main)
    write_frame(Path(__file__).parent / "table2_baseline_summary.csv", baseline)
    write_frame(
        Path(__file__).parent / "table_frequency_summary.csv",
        main.loc[main["method"].isin(FREQUENCY_METHODS)],
    )
    for path in TABLE_PATHS.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATHS["table2_main.tex"].write_text(_main_latex(baseline))
    TABLE_PATHS["table_frequency_hypothesis.tex"].write_text(_frequency_latex(main))
    TABLE_PATHS["table2_scanner.tex"].write_text(_scanner_latex(scanner))
    write_json(
        Path(__file__).parent / "table2_manifest.json",
        {
            "metric_contract": {
                "SSIM": {"column": "target_ssim", "direction": "higher"},
                "LPIPS": {"column": "lpips_vgg", "network": "VGG16", "direction": "lower"},
                "UNI cosine distance": {"column": "uni_distance", "direction": "lower"},
            },
            "aggregation": {
                "PanNormal": "location -> slide x scanner; scanner macro-average within slide; mean and sample SD across 103 slides",
                "PLISM": "location -> core -> section x scanner; five frozen folds averaged; scanner macro-average within section; mean and sample SD across 13 sections",
            },
            "improvement": {
                "higher_better": "100 * (method - raw) / raw",
                "lower_better": "100 * (raw - method) / raw",
            },
            "method_sets": {
                "baseline_table": list(BASELINE_METHODS),
                "frequency_hypothesis_table": list(FREQUENCY_METHODS),
                "scanner_supplement": list(METHODS),
            },
            "lpips_contract": str((OUTPUT / "lpips_contract.json").resolve()),
            "main_summary_sha256": sha256(Path(__file__).parent / "table2_main_summary.csv"),
            "baseline_summary_sha256": sha256(Path(__file__).parent / "table2_baseline_summary.csv"),
            "frequency_summary_sha256": sha256(Path(__file__).parent / "table_frequency_summary.csv"),
            "scanner_summary_sha256": sha256(Path(__file__).parent / "table2_scanner_summary.csv"),
            "baseline_latex_sha256": sha256(TABLE_PATHS["table2_main.tex"]),
            "frequency_latex_sha256": sha256(TABLE_PATHS["table_frequency_hypothesis.tex"]),
            "scanner_latex_sha256": sha256(TABLE_PATHS["table2_scanner.tex"]),
        },
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare-lpips")
    for command in ("pannormal", "plism-conventional", "plism-learned"):
        child = sub.add_parser(command)
        child.add_argument("--task-index", type=int, required=True)
        child.add_argument("--batch-size", type=int, default=8)
        child.add_argument("--uni-batch-size", type=int, default=32)
        child.add_argument("--generator-batch-size", type=int, default=16)
    sub.add_parser("aggregate")
    return result


def main() -> None:
    arguments = parser().parse_args()
    if arguments.command == "prepare-lpips":
        prepare_lpips()
    elif arguments.command == "pannormal":
        pannormal_task(arguments.task_index, arguments.batch_size)
    elif arguments.command == "plism-conventional":
        plism_conventional_task(arguments.task_index, arguments.batch_size, arguments.uni_batch_size)
    elif arguments.command == "plism-learned":
        plism_learned_task(arguments.task_index, arguments.batch_size, arguments.generator_batch_size)
    elif arguments.command == "aggregate":
        aggregate()


if __name__ == "__main__":
    main()

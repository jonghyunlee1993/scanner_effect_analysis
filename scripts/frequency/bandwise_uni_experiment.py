#!/usr/bin/env python3
"""Band-isolated, equal-OD-RMS UNI experiment on held-out PanNormal slides.

The source is AT2 after training-fold Reinhard colour mapping to each target.
Each source receives positive and negative amplitude changes in one Fourier
band at a time. The dose is fixed from the source image, before UNI or target
outcomes are read. Existing train-fold scanner parameters determine which sign
is the target-like direction. Outputs are exploratory and kept separate from
the locked manuscript results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from skimage.color import lab2rgb, rgb2lab


OLD_ROOT = Path("/mnt/isilon/oldridge_lab/leej/prenorm")
ANALYSIS = OLD_ROOT / "outputs/scanner_batch_effect_analysis_2026-09-17"
COHORT = ANALYSIS / "00_contract/cohort.csv"
PARAMETERS = ANALYSIS / "02_correction/parameters.json"
UNI_PANEL = ANALYSIS / "03_uni/shards"
BANDS = {
    "low_mid": (0.05052, 0.15156),
    "mid": (0.15156, 0.30312),
    "high": (0.30312, 0.45468),
}
TARGETS = ("versa", "akoya", "gt450", "s360", "s60")
DOSES = (0.25, 0.50)
PAD = 32


def band_mask(shape: tuple[int, int], bounds: tuple[float, float]) -> np.ndarray:
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.fftfreq(shape[1])[None, :]
    radius = np.sqrt(fy * fy + fx * fx)
    lo, hi = bounds
    taper = 0.015
    rise = np.clip((radius - lo + taper) / (2 * taper), 0, 1)
    fall = np.clip((hi + taper - radius) / (2 * taper), 0, 1)
    rise = rise * rise * (3 - 2 * rise)
    fall = fall * fall * (3 - 2 * fall)
    return np.minimum(rise, fall).astype(np.float32)


def apply_reinhard(image: np.ndarray, fitted: dict) -> np.ndarray:
    """Apply the existing AT2→scanner train-fold colour mapping."""
    lab = rgb2lab(image.astype(np.float32) / 255.0).astype(np.float32)
    source_mean = np.asarray(fitted["source_mean"], dtype=np.float32)
    source_std = np.asarray(fitted["source_std"], dtype=np.float32)
    target_mean = np.asarray(fitted["target_mean"], dtype=np.float32)
    target_std = np.asarray(fitted["target_std"], dtype=np.float32)
    corrected = (lab - source_mean) / source_std * target_std + target_mean
    corrected[..., 0] = np.clip(corrected[..., 0], 0, 100)
    corrected[..., 1:] = np.clip(corrected[..., 1:], -127, 127)
    return np.clip(np.rint(lab2rgb(corrected) * 255), 0, 255).astype(np.uint8)


def mean_od(image: np.ndarray) -> np.ndarray:
    return -np.log((image.astype(np.float32) + 1.0) / 256.0).mean(axis=-1)


def components(image: np.ndarray) -> dict[str, np.ndarray]:
    od = mean_od(image)
    padded = np.pad(od, PAD, mode="reflect")
    spectrum = np.fft.fft2(padded - padded.mean())
    return {
        name: np.fft.ifft2(spectrum * band_mask(padded.shape, bounds))
        .real[PAD:-PAD, PAD:-PAD]
        .astype(np.float32)
        for name, bounds in BANDS.items()
    }


def render(image: np.ndarray, component: np.ndarray, coefficient: float) -> np.ndarray:
    od = -np.log((image.astype(np.float32) + 1.0) / 256.0)
    changed = od + coefficient * component[..., None]
    return np.clip((np.exp(-changed) * 256.0 - 1.0) / 255.0, 0, 1).astype(np.float32)


def achieved_rms(image: np.ndarray, rendered: np.ndarray) -> float:
    changed_od = -np.log((rendered.astype(np.float32) * 255.0 + 1.0) / 256.0).mean(axis=-1)
    return float(np.sqrt(np.mean((changed_od - mean_od(image)) ** 2)))


def render_at_dose(
    image: np.ndarray, component: np.ndarray, sign: int, target_rms: float
) -> tuple[np.ndarray, float, float]:
    """Match the dose after RGB gamut clipping, using image information only."""
    upper = 1.0
    for _ in range(8):
        trial = render(image, component, sign * upper)
        if achieved_rms(image, trial) >= target_rms:
            break
        upper *= 2
    low = 0.0
    for _ in range(20):
        middle = (low + upper) / 2
        trial = render(image, component, sign * middle)
        if achieved_rms(image, trial) < target_rms:
            low = middle
        else:
            upper = middle
    trial = render(image, component, sign * upper)
    return trial, achieved_rms(image, trial), sign * upper


def target_like_sign(fitted: dict, bounds: tuple[float, float]) -> int:
    frequency = np.asarray(fitted["frequency_cyc_per_pixel"], dtype=float)
    # Stored train-fold curve maps AT2 to the target scanner.
    log_gain = np.asarray(fitted["log2_amplitude_gain"], dtype=float)
    selected = (frequency >= bounds[0]) & (frequency < bounds[1])
    if not selected.any():
        raise ValueError("empty fitted frequency band")
    return 1 if float(np.mean(log_gain[selected])) >= 0 else -1


@torch.inference_mode()
def embed(model, images: list[np.ndarray], size, mean, std, device, batch_size: int) -> np.ndarray:
    output = []
    for start in range(0, len(images), batch_size):
        block = np.stack(images[start : start + batch_size]).astype(np.float32)
        tensor = torch.from_numpy(block).permute(0, 3, 1, 2).to(device)
        tensor = F.interpolate(tensor, size=(size, size), mode="bicubic", align_corners=False)
        with torch.autocast("cuda", dtype=torch.float16):
            value = model((tensor - mean) / std)
        output.append(F.normalize(value.float(), dim=1).cpu().numpy())
    return np.concatenate(output)


def run_fold(fold: int, output_dir: Path, batch_size: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("run_fold requires a GPU allocation")
    from prenorm.embedding import load_uni

    cohort = pd.read_csv(COHORT, dtype={"slide_id": str})
    parameters = json.loads(PARAMETERS.read_text())["fold"][str(fold)]
    slides = cohort[cohort.fold == fold]
    device = torch.device("cuda")
    model, size, mean, std = load_uni(device)
    rows = []
    for _, slide in slides.iterrows():
        slide_id = str(slide.slide_id)
        with h5py.File(UNI_PANEL / f"{slide_id}.h5", "r") as panel:
            locked = np.asarray(panel["location_index"], dtype=int)
            prior_features = np.asarray(panel["features"][[0, len(locked) // 2, -1]], dtype=np.float32)
            prior_names = [item.decode() for item in panel["condition_names"][:]]
        locations = locked[[0, len(locked) // 2, -1]]
        with h5py.File(slide.cache_path, "r") as cache:
            scanner_names = tuple(v.decode().lower() for v in cache["scanner_names"][:])
            if scanner_names != ("at2", "versa", "akoya", "gt450", "s360", "s60"):
                raise ValueError(f"unexpected scanner order: {slide_id}")
            images = np.asarray(cache["images"][locations], dtype=np.uint8)

        image_batch: list[np.ndarray] = []
        index: dict[tuple, int] = {}
        pending = []
        for loc_pos, loc in enumerate(locations):
            for scanner in TARGETS:
                source = apply_reinhard(images[loc_pos, 0], parameters[scanner])
                target = images[loc_pos, scanner_names.index(scanner)]
                key = (int(loc), scanner)
                index[key, "base"] = len(image_batch)
                image_batch.append(source.astype(np.float32) / 255.0)
                index[key, "target"] = len(image_batch)
                image_batch.append(target.astype(np.float32) / 255.0)
                part = components(source)
                minimum_rms = min(float(np.sqrt(np.mean(x * x))) for x in part.values())
                for dose in DOSES:
                    target_rms = dose * minimum_rms
                    for band, component in part.items():
                        predicted_sign = target_like_sign(parameters[scanner], BANDS[band])
                        for sign in (-1, 1):
                            changed, actual_rms, coefficient = render_at_dose(
                                source, component, sign, target_rms
                            )
                            idx = len(image_batch)
                            image_batch.append(changed)
                            pending.append((key, idx, dose, band, sign, predicted_sign,
                                            target_rms, actual_rms, coefficient))
        features = embed(model, image_batch, size, mean, std, device, batch_size)
        for loc_pos, loc in enumerate(locations):
            for scanner in TARGETS:
                key = (int(loc), scanner)
                for condition, old_name in (("base", f"reinhard:{scanner}"),
                                            ("target", f"target:{scanner}")):
                    agreement = float(np.dot(features[index[key, condition]],
                                             prior_features[loc_pos, prior_names.index(old_name)]))
                    if agreement < 0.995:
                        raise ValueError(
                            f"{slide_id} {loc} {old_name}: frozen baseline agreement {agreement:.5f}"
                        )
        for key, idx, dose, band, sign, predicted_sign, target_rms, actual_rms, coefficient in pending:
            baseline = features[index[key, "base"]]
            target = features[index[key, "target"]]
            changed = features[idx]
            rows.append({
                "slide_id": slide_id, "tissue_type": slide.tissue_type, "fold": fold,
                "location_index": key[0], "scanner": key[1], "band": band,
                "dose_fraction": dose, "sign": sign,
                "target_like": sign == predicted_sign,
                "target_rms_od": target_rms, "achieved_rms_od": actual_rms,
                "coefficient": coefficient,
                "embedding_displacement": float(1 - np.dot(baseline, changed)),
                "baseline_target_distance": float(1 - np.dot(baseline, target)),
                "changed_target_distance": float(1 - np.dot(changed, target)),
                "target_gain": float(np.dot(changed, target) - np.dot(baseline, target)),
            })
        print(f"fold {fold}: {slide_id}, {len(rows)} rows", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"fold_{fold}.csv.gz"
    pd.DataFrame(rows).to_csv(path, index=False, compression="gzip")
    print(json.dumps({"path": str(path), "slides": len(slides), "rows": len(rows)}), flush=True)


def _bootstrap(values: np.ndarray, seed: int = 20260924) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    draws = rng.integers(0, n, size=(2000, n))
    means = values[draws].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def aggregate(output_dir: Path) -> None:
    paths = [output_dir / f"fold_{fold}.csv.gz" for fold in range(5)]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError("all five fold outputs are required")
    frame = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    expected_rows = 103 * 3 * len(TARGETS) * len(BANDS) * len(DOSES) * 2
    per_slide = frame.groupby("slide_id").location_index.nunique()
    if (len(frame) != expected_rows or frame.slide_id.nunique() != 103
            or not per_slide.eq(3).all()):
        raise ValueError("incomplete held-out panel")
    key = ["slide_id", "location_index", "scanner", "band", "dose_fraction", "sign"]
    if frame.duplicated(key).any():
        raise ValueError("duplicate perturbation rows")
    frame["dose_relative_error"] = (frame.achieved_rms_od / frame.target_rms_od - 1).abs()
    dose_audit = frame.groupby(["band", "dose_fraction", "sign"]).dose_relative_error.agg(
        ["median", "max"]
    ).reset_index()
    dose_audit.to_csv(output_dir / "dose_audit.csv", index=False)
    if (dose_audit["median"] > 0.02).any():
        raise ValueError("achieved dose differs by more than 2% from target")

    summary = []
    for metric in ("embedding_displacement", "target_gain"):
        for (scanner, band, dose, direction), group in frame.groupby(
            ["scanner", "band", "dose_fraction", "target_like"]
        ):
            slide_mean = group.groupby("slide_id")[metric].mean().to_numpy()
            mean, low, high = _bootstrap(slide_mean)
            summary.append({"metric": metric, "scanner": scanner, "band": band,
                            "dose_fraction": dose, "target_like": direction,
                            "mean": mean, "ci_low": low, "ci_high": high,
                            "slides": len(slide_mean)})
    pd.DataFrame(summary).to_csv(output_dir / "summary.csv", index=False)

    # Each contrast pairs the same source image, dose, and perturbation sign.
    wide = frame.pivot_table(index=["slide_id", "location_index", "scanner", "dose_fraction", "sign"],
                             columns="band", values="embedding_displacement")
    contrasts = []
    for comparator in ("low_mid", "mid"):
        for dose in DOSES:
            subset = wide.xs(dose, level="dose_fraction")
            paired = (subset["high"] - subset[comparator]).groupby(level="slide_id").mean()
            mean, low, high = _bootstrap(paired.to_numpy())
            contrasts.append({"contrast": f"high_minus_{comparator}", "dose_fraction": dose,
                              "mean": mean, "ci_low": low, "ci_high": high,
                              "slides": len(paired)})
    pd.DataFrame(contrasts).to_csv(output_dir / "sensitivity_contrasts.csv", index=False)
    print(json.dumps({"rows": len(frame), "slides": frame.slide_id.nunique(),
                      "summary": str(output_dir / "summary.csv")}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run-fold", "aggregate"))
    parser.add_argument("--fold", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bandwise_uni_experiment"))
    args = parser.parse_args()
    if args.command == "run-fold":
        if args.fold is None or not 0 <= args.fold < 5:
            parser.error("--fold must be between 0 and 4")
        run_fold(args.fold, args.output_dir, args.batch_size)
    else:
        aggregate(args.output_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""RV-P0d (iii): compare generated high-band amplitude with paired source and target images.

Recovered verbatim from `.Trash/2026-09-29_paper_code_cleanup/src/manuscript_completion/discussion_highband_audit.py`
(RV-P0d iii). Run-time byte-code cache: `.Trash/2026-09-26_paper_refactor/generated/src/
manuscript_completion/__pycache__/discussion_highband_audit.cpython-39.pyc` (same source size and mtime).
Paths are relative to the repository root. The caller passes an output location under
`analysis/revision/results/provenance_restoration/`; see `restore_structure_ablation.py`.
Only the module docstring differs. Output goes to `<ablation-root>/05_comparison`.
"""


from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


BASE = Path(
    "outputs/scanner_batch_effect_analysis_2026-09-17/06_learned_baselines/"
    "09_bidirectional_full_training/02_predictions/at2_to_gt450"
)


def band_amplitudes(images: np.ndarray) -> np.ndarray:
    mean_od = -np.log((images.astype(np.float32) + 1.0) / 256.0).mean(axis=-1)
    mean_od -= mean_od.mean(axis=(-2, -1), keepdims=True)
    height, width = mean_od.shape[-2:]
    window = np.hanning(height)[:, None] * np.hanning(width)[None, :]
    spectrum = np.abs(np.fft.fft2(mean_od * window))
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    radial = np.sqrt(fx**2 + fy**2)
    bands = ((0.015156, 0.05052), (0.05052, 0.15156),
             (0.15156, 0.30312), (0.30312, 0.45468))
    return np.stack([np.sqrt(np.mean(spectrum[:, (radial >= lo) & (radial < hi)] ** 2, axis=1))
                     for lo, hi in bands], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.ablation_root / "05_comparison"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    new_root = args.ablation_root / "02_predictions/at2_to_gt450"
    for fold in range(5):
        old = {p.name: p for p in (BASE / f"fold_{fold}/slides").glob("*.h5")}
        new = {p.name: p for p in (new_root / f"fold_{fold}/slides").glob("*.h5")}
        if set(old) != set(new):
            raise ValueError(f"fold {fold}: baseline and regularized slide sets differ")
        for filename in sorted(old):
            with h5py.File(old[filename], "r") as baseline, h5py.File(new[filename], "r") as regularized:
                old_locations = np.asarray(baseline["metadata/location_index"][:3])
                new_locations = np.asarray(regularized["metadata/location_index"][:3])
                if not np.array_equal(old_locations, new_locations):
                    raise ValueError(f"{filename}: evaluation locations differ")
                source = band_amplitudes(np.asarray(baseline["aligned_valid/raw_source_valid"][:3]))
                target = band_amplitudes(np.asarray(baseline["aligned_valid/real_target_valid"][:3]))
                outputs = {
                    "baseline": band_amplitudes(np.asarray(baseline["aligned_valid/generated_valid"][:3])),
                    "regularized": band_amplitudes(np.asarray(regularized["aligned_valid/generated_valid"][:3])),
                }
                for method, measured in outputs.items():
                    source_relative = source[:, 3] / source[:, 0]
                    target_relative = target[:, 3] / target[:, 0]
                    generated_relative = measured[:, 3] / measured[:, 0]
                    rows.append({
                        "slide_id": str(baseline.attrs["slide_id"]), "fold": fold,
                        "method": method, "locations": 3,
                        "source_high": float(source[:, 3].mean()),
                        "target_high": float(target[:, 3].mean()),
                        "generated_high": float(measured[:, 3].mean()),
                        "generated_to_source_high": float(np.mean(measured[:, 3] / source[:, 3])),
                        "generated_to_target_high": float(np.mean(measured[:, 3] / target[:, 3])),
                        "source_to_target_high": float(np.mean(source[:, 3] / target[:, 3])),
                        "source_high_over_anchor": float(source_relative.mean()),
                        "target_high_over_anchor": float(target_relative.mean()),
                        "generated_high_over_anchor": float(generated_relative.mean()),
                        "generated_to_source_relative_high": float(np.mean(generated_relative / source_relative)),
                        "generated_to_target_relative_high": float(np.mean(generated_relative / target_relative)),
                        "source_to_target_relative_high": float(np.mean(source_relative / target_relative)),
                    })
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "highband_per_slide.csv", index=False)
    frame.groupby("method", sort=True)[
        ["source_high", "target_high", "generated_high",
         "generated_to_source_high", "generated_to_target_high", "source_to_target_high",
         "source_high_over_anchor", "target_high_over_anchor", "generated_high_over_anchor",
         "generated_to_source_relative_high", "generated_to_target_relative_high",
         "source_to_target_relative_high"]
    ].mean().to_csv(output / "highband_summary.csv")
    pair = frame.pivot(index="slide_id", columns="method", values="generated_to_target_relative_high")
    delta = (pair["regularized"] - pair["baseline"]).to_numpy()
    rng = np.random.default_rng(20260925)
    samples = rng.integers(0, len(delta), size=(5000, len(delta)))
    boot = delta[samples].mean(axis=1)
    pd.DataFrame([{
        "metric": "regularized_minus_baseline_generated_to_target_relative_high",
        "estimate": float(delta.mean()), "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
    }]).to_csv(output / "highband_contrast.csv", index=False)


if __name__ == "__main__":
    main()

"""Synthetic destinations along the band-power axis, and what they cost.

The destination result rests on three points -- AT2, S60, GT450 -- and those
three differ in colour *and* in post-Reinhard band power at once, so they cannot
separate "aim where there is more detail" from "aim where the foundation model
was pretrained". Three points also cannot show a shape.

This builds a one-axis sweep instead. A synthetic destination is GT450's colour
with GT450's post-Reinhard band power scaled by lambda, so colour is held fixed
and only detail power moves. lambda = 1 reproduces the real GT450 destination
exactly, which anchors the sweep to a point already measured.

The algebra is why this is cheap. The fitted gain is half the log ratio of
pooled band energies, so scaling the target energy by lambda shifts every log
gain by 0.5*log(lambda) and leaves the bootstrap standard error untouched -- a
constant inside the log cancels in a standard deviation. Only the reliability
shrinkage moves, because it divides that unchanged error by a now-different gain
magnitude.

Two things are recorded here rather than discovered later on a GPU. The gains
themselves are the image-domain half of the sweep and are a result on their own.
And the frozen hard cap of 4.0 will bind at the top of the sweep for the
scanners that already need the most amplification, which is a real saturation of
the method and is reported as a fraction of capped bands, not hidden.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from e5_reinhard_residual_frequency import RF1_FOLDS
from fetch_e0_pfm_checkpoints import sha256
from rf1m_combined import RF1M_SIGMAS
from rf1u_unpaired import (
    RF1U_HARD_CAP,
    RF1U_SHRINKAGE_SE,
    RF1U_VERSION,
    fitted_scanner_gains,
)


BASE_TARGET = "gt450"
LAMBDAS = (0.5, 1.5, 2.0, 3.0)
FOVS = (224, 256, 512)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--energy", default="outputs/rf1u_multitarget/energy")
    parser.add_argument("--base-target", default=BASE_TARGET)
    parser.add_argument(
        "--lambdas", default=",".join(str(value) for value in LAMBDAS)
    )
    parser.add_argument("--output", default="outputs/e8_lambda_energy")
    return parser.parse_args()


def label_for(base: str, value: float) -> str:
    """A filesystem-safe destination name, e.g. gt450_l150 for lambda = 1.5."""
    return f"{base}_l{round(value * 100):03d}"


def fold_gains(energy: dict, fold: int):
    train = energy["fold_of_slide"] != fold
    if train.sum() < 2:
        raise ValueError("a training fold must hold at least two slides")
    return fitted_scanner_gains(
        energy["target_band_energy"][train],
        energy["source_band_energy"][train, :, fold, :],
        RF1U_SHRINKAGE_SE,
    )


def main():
    args = parse_args()
    lambdas = [float(value) for value in args.lambdas.split(",") if value.strip()]
    if any(value <= 0 for value in lambdas):
        raise ValueError("lambda must be positive")
    energy_root = Path(args.energy)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    limit = float(np.log(RF1U_HARD_CAP))
    rows, artifacts = [], {}
    for fov in FOVS:
        source_path = energy_root / f"{args.base_target}_fov_{fov}.npz"
        base_summary = json.loads(source_path.with_suffix(".summary.json").read_text())
        if not (
            base_summary.get("analysis") == "rf1m_slide_band_energy"
            and base_summary.get("target") == args.base_target
            and base_summary.get("fov") == fov
            and base_summary.get("energy_gate_pass") is True
            and base_summary.get("output_sha256") == sha256(source_path)
        ):
            raise RuntimeError(f"invalid base band-energy input: {source_path}")
        with np.load(source_path, allow_pickle=True) as handle:
            base = {name: handle[name] for name in handle.files}
        scanners = [str(value) for value in base["scanners"]]

        for value in lambdas:
            label = label_for(args.base_target, value)
            payload = dict(base)
            # Colour is untouched: only the destination's band energy is scaled,
            # so lambda moves detail power and nothing else.
            payload["target_band_energy"] = base["target_band_energy"] * value
            payload["target"] = np.asarray(label)
            payload["lambda_scale"] = np.asarray(value, dtype=np.float64)
            payload["base_target"] = np.asarray(args.base_target)

            path = output_root / f"{label}_fov_{fov}.npz"
            temporary = output_root / f".{label}_fov_{fov}.{os.getpid()}.tmp.npz"
            np.savez_compressed(temporary, **payload)
            os.replace(temporary, path)
            summary = dict(base_summary)
            summary.update(
                {
                    "target": label,
                    "base_target": args.base_target,
                    "lambda_scale": value,
                    "status": "POST_CORE_EXPLORATORY_EXTENSION",
                    "note": (
                        "Synthetic destination: base target colour with its "
                        "post-Reinhard band energy scaled by lambda. Derived "
                        "from the locked band-energy file, no pixels re-read."
                    ),
                    "derived_from": str(source_path),
                    "derived_from_sha256": base_summary["output_sha256"],
                    "output_sha256": sha256(path),
                }
            )
            path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            artifacts[path.name] = summary["output_sha256"]

            for fold in range(RF1_FOLDS):
                fitted = fold_gains(payload, fold)
                for index, scanner in enumerate(scanners):
                    for band, sigma in enumerate(RF1M_SIGMAS):
                        rows.append(
                            {
                                "fov": fov,
                                "destination": label,
                                "lambda_scale": value,
                                "fold": fold,
                                "scanner": scanner,
                                "sigma": float(sigma),
                                "raw_log_gain": float(fitted["raw_log_gain"][index, band]),
                                "standard_error": float(fitted["standard_error"][index, band]),
                                "alpha": float(fitted["alpha"][index, band]),
                                "log_gain": float(fitted["log_gain"][index, band]),
                                "gain": float(fitted["gain"][index, band]),
                                "at_hard_cap": bool(
                                    abs(abs(fitted["log_gain"][index, band]) - limit) < 1e-9
                                ),
                            }
                        )
            print(f"fov {fov} lambda {value:.2f} -> {label}", flush=True)

    table = output_root / "lambda_band_gains.csv"
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    capped = sum(row["at_hard_cap"] for row in rows)
    by_lambda = {}
    for value in lambdas:
        block = [row for row in rows if row["lambda_scale"] == value]
        by_lambda[str(value)] = {
            "median_gain": float(np.median([row["gain"] for row in block])),
            "min_gain": float(np.min([row["gain"] for row in block])),
            "max_gain": float(np.max([row["gain"] for row in block])),
            "mean_alpha": float(np.mean([row["alpha"] for row in block])),
            "fraction_at_hard_cap": float(
                np.mean([row["at_hard_cap"] for row in block])
            ),
        }

    summary = {
        "analysis": "e8_lambda_energy",
        "rf1u_version": RF1U_VERSION,
        "status": "POST_CORE_EXPLORATORY_EXTENSION",
        "note": (
            "One-axis destination sweep. Colour is held at the base target and "
            "only post-Reinhard band power is scaled, so lambda separates the "
            "detail-power axis from the colour axis that the three real "
            "destinations confound. lambda = 1 reproduces the base destination."
        ),
        "base_target": args.base_target,
        "lambdas": lambdas,
        "fovs": list(FOVS),
        "shrinkage_se": RF1U_SHRINKAGE_SE,
        "hard_cap": RF1U_HARD_CAP,
        "rows": len(rows),
        "bands_at_hard_cap": int(capped),
        "per_lambda": by_lambda,
        "artifacts": {**artifacts, table.name: sha256(table)},
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

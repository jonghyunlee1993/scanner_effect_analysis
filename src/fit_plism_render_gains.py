"""Fit RF1U band gains on the 0.5052 um/px grid the correction is deployed on.

Arm A fitted gains from native-resolution energies, which answered a question
about optics.  Deployment is different: the tiles a foundation model sees have
been reduced onto the study's target grid, and that reduction removes exactly the
high-frequency content the gains are estimated from.  Fitting on native energies
and applying on the reduced grid would over-sharpen.  So the gains are refitted
here, from the rendered Reinhard tiles themselves.

Everything else is the frozen RF1U rule: pooled population energies, a gain of
sqrt(E_target / E_source), shrinkage toward identity by the ratio's own bootstrap
standard error, and the hard cap.  The bootstrap resamples stain sections.

Contract: docs/e9_plism_native_ert_contract.md
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from rf1m_combined import RF1M_SIGMAS
from rf1u_unpaired import (
    RF1U_BOOTSTRAP_REPLICATES,
    RF1U_BOOTSTRAP_SEED,
    RF1U_HARD_CAP,
    RF1U_SHRINKAGE_SE,
)

BAND_COUNT = len(RF1M_SIGMAS)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", default="outputs/plism_render")
    parser.add_argument("--band-energy", default="",
                        help="directory of per-slide band energies; skips reading tiles")
    parser.add_argument("--destinations", default="at2,gt450")
    parser.add_argument("--output", default="outputs/plism_render_gains")
    parser.add_argument("--batch", type=int, default=64)
    return parser.parse_args()


def tile_energies(path: Path, batch: int) -> tuple[np.ndarray, dict]:
    import torch

    from rf1m_combined import band_energy
    from e5_comparator_population import rgb01_to_od

    with h5py.File(path, "r") as handle:
        meta = json.loads(handle.attrs["meta"])
        total = handle["rgb"].shape[0]
        collected = []
        for start in range(0, total, batch):
            block = handle["rgb"][start:start + batch]
            rgb01 = torch.from_numpy(np.ascontiguousarray(block)).float() / 255.0
            mean_od = rgb01_to_od(rgb01).mean(dim=-1)
            collected.append(band_energy(mean_od, RF1M_SIGMAS).numpy())
    return np.concatenate(collected), meta


def main() -> None:
    args = parse_args()
    render_root = Path(args.render)
    destinations = [d.strip() for d in args.destinations.split(",") if d.strip()]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RF1U_BOOTSTRAP_SEED)
    all_gains, all_energy = [], []

    for destination in destinations:
        if args.band_energy:
            # Measured in one pass by measure_plism_core_render_bands.py rather
            # than read back off rendered tiles; the schema is the same.
            measured = pd.concat(
                [pd.read_csv(p) for p in sorted(Path(args.band_energy).glob("*.csv"))],
                ignore_index=True)
            energy = measured.loc[measured["destination"] == destination].copy()
            if energy.empty:
                raise RuntimeError(f"no band energies for destination {destination}")
        else:
            rows = []
            for path in sorted(glob.glob(str(render_root / f"reinhard_{destination}" / "*.h5"))):
                energies, meta = tile_energies(Path(path), args.batch)
                entry = {"stain": meta["stain"], "scanner": meta["scanner"],
                         "tiles": len(energies), "destination": destination}
                for index in range(BAND_COUNT):
                    entry[f"b{index + 1}"] = float(energies[:, index].mean())
                rows.append(entry)
            energy = pd.DataFrame(rows)
        all_energy.append(energy)

        target_scanner = destination.upper() if destination.upper() != "AT2" else "AT2"
        names = {s.upper(): s for s in energy["scanner"].unique()}
        target_scanner = names.get(destination.upper(), destination.upper())

        sections = sorted(energy["stain"].unique())
        for source in sorted(energy["scanner"].unique()):
            if source == target_scanner:
                continue
            for index in range(BAND_COUNT):
                band = f"b{index + 1}"
                pivot = energy.pivot(index="stain", columns="scanner", values=band).loc[sections]
                source_energy = pivot[source].to_numpy()
                target_energy = pivot[target_scanner].to_numpy()
                log_gain = 0.5 * np.log(target_energy.mean() / source_energy.mean())
                draws = rng.integers(0, len(sections),
                                     size=(RF1U_BOOTSTRAP_REPLICATES, len(sections)))
                samples = 0.5 * np.log(target_energy[draws].mean(axis=1)
                                       / source_energy[draws].mean(axis=1))
                standard_error = float(samples.std(ddof=1))
                alpha = 0.0
                if abs(log_gain) > 0:
                    alpha = max(0.0, 1.0 - RF1U_SHRINKAGE_SE * standard_error / abs(log_gain))
                gain = float(np.clip(np.exp(alpha * log_gain),
                                     1.0 / RF1U_HARD_CAP, RF1U_HARD_CAP))
                all_gains.append({"destination": destination, "source": source,
                                  "band": band, "sigma_px": RF1M_SIGMAS[index],
                                  "log_gain": float(log_gain), "se": standard_error,
                                  "alpha": float(alpha), "gain": gain,
                                  "sections": len(sections)})

    gains = pd.DataFrame(all_gains)
    gains.to_csv(output_dir / "fitted_gains.csv", index=False)
    pd.concat(all_energy, ignore_index=True).to_csv(output_dir / "band_energy.csv", index=False)

    pd.set_option("display.width", 200)
    print("=== RF1U gains fitted on the 0.5052 um/px render ===")
    wide = gains.pivot_table(index=["destination", "source"], columns="band", values="gain")
    print(wide.round(3).to_string())
    print("\n=== regime by destination ===")
    summary = gains.groupby("destination").agg(
        cells=("gain", "size"),
        amplify=("gain", lambda s: int((s > 1).sum())),
        attenuate=("gain", lambda s: int((s < 1).sum())),
        mean_log_gain=("log_gain", "mean"),
        mean_alpha=("alpha", "mean"))
    print(summary.round(3).to_string())
    (output_dir / "summary.json").write_text(json.dumps({
        "sigmas_px": list(RF1M_SIGMAS), "target_mpp": 0.5052,
        "bootstrap": RF1U_BOOTSTRAP_REPLICATES, "seed": RF1U_BOOTSTRAP_SEED,
        "shrinkage_se": RF1U_SHRINKAGE_SE, "hard_cap": RF1U_HARD_CAP,
        "destinations": destinations}, indent=2))
    print(f"\nwrote {output_dir}")


if __name__ == "__main__":
    main()

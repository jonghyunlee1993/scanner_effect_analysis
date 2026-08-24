"""RF1U band gains on PLISM, and whether the destination rule replicates.

Section 9 says the destination decides the outcome, and gives the reason: what
governs it is the target's **post-Reinhard** detail power, because that sets
whether the fitted gains attenuate or amplify.  Aiming at a detail-poor target
blurs every source, which costs content and buys no invariance; aiming at a
detail-rich one sharpens.  In PanNormal, AT2 sits near the bottom on that measure
and 12 of 15 scanner-band gains toward it attenuate; GT450 and S60 sit high and 12
of 15 amplify.

PLISM lets that rule be tested out of sample, on seven destinations instead of
three, with sampling density equalised across the panel.  The gain formula, the
shrinkage rule, the bootstrap settings and the hard cap are the frozen RF1U ones;
only the cohort changes.  The replicate unit for the bootstrap is the section.

Contract: docs/e9_plism_native_ert_contract.md
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from plism_dataset_corrections import drop_blocks, parse_exclusions
from rf1u_unpaired import (
    RF1U_BOOTSTRAP_REPLICATES,
    RF1U_BOOTSTRAP_SEED,
    RF1U_HARD_CAP,
    RF1U_SHRINKAGE_SE,
)

BANDS = ("b1", "b2", "b3")
# locked PanNormal reference: post-Reinhard fine-band power against AT2 (report section 9)
PANNORMAL_POST_REINHARD = {"GT450": 4.14, "S60": 2.94, "AT2": 1.00}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/plism_reinhard_bands")
    parser.add_argument("--exclude", default="",
                        help="stain:scanner blocks to drop, or 'default' for the\n                              known-defective table in plism_dataset_corrections")
    parser.add_argument("--output", default="outputs/plism_rf1u_destinations")
    return parser.parse_args()


def section_energies(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean post-Reinhard band energy per (section, scanner) — the bootstrap unit."""
    columns = {f"rein_{band}": band for band in BANDS}
    block = frame.groupby(["stain", "scanner"])[list(columns)].mean()
    return block.rename(columns=columns).reset_index()


def fitted_gain(energies: pd.DataFrame, source: str, target: str, band: str,
                rng: np.random.Generator) -> dict:
    """Frozen RF1U rule: pooled-population gain, shrunk by its own section bootstrap."""
    sections = sorted(set(energies.loc[energies["scanner"] == source, "stain"])
                      & set(energies.loc[energies["scanner"] == target, "stain"]))
    pivot = energies.pivot(index="stain", columns="scanner", values=band).loc[sections]
    source_energy = pivot[source].to_numpy()
    target_energy = pivot[target].to_numpy()

    log_gain = 0.5 * np.log(target_energy.mean() / source_energy.mean())
    draws = rng.integers(0, len(sections), size=(RF1U_BOOTSTRAP_REPLICATES, len(sections)))
    samples = 0.5 * np.log(target_energy[draws].mean(axis=1) / source_energy[draws].mean(axis=1))
    standard_error = float(samples.std(ddof=1))

    alpha = 0.0
    if abs(log_gain) > 0:
        alpha = max(0.0, 1.0 - RF1U_SHRINKAGE_SE * standard_error / abs(log_gain))
    gain = float(np.clip(np.exp(alpha * log_gain), 1.0 / RF1U_HARD_CAP, RF1U_HARD_CAP))
    return {"log_gain": float(log_gain), "se": standard_error, "alpha": float(alpha),
            "gain": gain, "sections": len(sections)}


def main() -> None:
    args = parse_args()
    frame = pd.concat([pd.read_csv(path) for path in sorted(glob.glob(f"{args.input}/*.csv"))],
                      ignore_index=True)
    frame, removed = drop_blocks(frame, parse_exclusions(args.exclude))
    for entry in removed:
        print(f"excluded {entry['stain']}/{entry['scanner']}: "
              f"{entry['rows']:,} patches — {entry['reason']}")
    energies = section_energies(frame)
    scanners = sorted(energies["scanner"].unique())
    rng = np.random.default_rng(RF1U_BOOTSTRAP_SEED)

    pooled = energies.groupby("scanner")[list(BANDS)].mean()
    reference = pooled.loc["AT2"]
    detail = (pooled / reference).rename(columns={b: f"{b}_vs_AT2" for b in BANDS})
    detail["pannormal_b1"] = [PANNORMAL_POST_REINHARD.get(s, np.nan) for s in detail.index]

    rows = []
    for target in scanners:
        for source in scanners:
            if source == target:
                continue
            for band in BANDS:
                rows.append({"target": target, "source": source, "band": band,
                             **fitted_gain(energies, source, target, band, rng)})
    gains = pd.DataFrame(rows)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    gains.to_csv(output_dir / "fitted_gains.csv", index=False)
    detail.to_csv(output_dir / "post_reinhard_detail.csv")

    summary = gains.groupby("target").agg(
        cells=("gain", "size"),
        amplify=("gain", lambda s: int((s > 1).sum())),
        attenuate=("gain", lambda s: int((s < 1).sum())),
        mean_log_gain=("log_gain", "mean"),
        mean_alpha=("alpha", "mean"),
        capped=("gain", lambda s: int(((s >= RF1U_HARD_CAP - 1e-9) |
                                       (s <= 1 / RF1U_HARD_CAP + 1e-9)).sum())),
    )
    summary["detail_b1_vs_AT2"] = detail["b1_vs_AT2"]
    summary["regime"] = np.where(summary["mean_log_gain"] > 0, "amplify", "attenuate")
    summary = summary.sort_values("detail_b1_vs_AT2", ascending=False)
    summary.to_csv(output_dir / "destination_summary.csv")

    pd.set_option("display.width", 220)
    print("=== post-Reinhard detail power, AT2 = 1 ===")
    print(detail.round(3).to_string(), "\n")
    print("=== fitted RF1U gains by destination (18 source-band cells each) ===")
    print(summary.round(3).to_string(), "\n")

    order = summary["detail_b1_vs_AT2"]
    rho = order.corr(summary["mean_log_gain"], method="spearman")
    print(f"destination detail power vs mean fitted log gain: spearman {rho:+.3f}")
    worst = summary.index[-1]
    best = summary.index[0]
    print(f"lowest-detail destination {worst}: {summary.loc[worst, 'attenuate']}/"
          f"{summary.loc[worst, 'cells']} cells attenuate")
    print(f"highest-detail destination {best}: {summary.loc[best, 'amplify']}/"
          f"{summary.loc[best, 'cells']} cells amplify")

    (output_dir / "summary.json").write_text(json.dumps({
        "bands": list(BANDS),
        "bootstrap": RF1U_BOOTSTRAP_REPLICATES,
        "seed": RF1U_BOOTSTRAP_SEED,
        "shrinkage_se": RF1U_SHRINKAGE_SE,
        "hard_cap": RF1U_HARD_CAP,
        "replicate_unit": "stain section",
        "detail_vs_gain_spearman": float(rho),
        "pannormal_reference": PANNORMAL_POST_REINHARD,
    }, indent=2))
    print(f"\nwrote {output_dir}")


if __name__ == "__main__":
    main()

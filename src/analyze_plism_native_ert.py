"""PLISM native-resolution effective relative transfer, against the frozen predictions.

Reads the per-WSI spectra written by ``extract_plism_native_psd.py`` and evaluates
the endpoints registered in ``docs/e9_plism_native_ert_contract.md``.

The replicate unit is the stain section, n = 13, because within one stain the
seven scanners image the same physical section.  Bootstraps resample sections;
patches are observations inside a section and are never treated as replicates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REFERENCE = "AT2"

# Locked high-band ERT against AT2, report section 3.1.  Frozen before PLISM was opened.
LOCKED_ERT = {"GT450": 1.478, "S60": 1.044, "AT2": 1.000, "VERSA": 0.938, "S360": 0.842, "AKOYA": 0.335}
SHARED = ["GT450", "S60", "AT2", "S360"]

PREDICTIONS = {
    "P1": ("GT450", "greater"),
    "P2": ("S360", "less"),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/plism_native_psd")
    parser.add_argument("--output", default="outputs/plism_native_ert")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--figure", action="store_true")
    return parser.parse_args()


def load(input_dir: Path) -> tuple[pd.DataFrame, dict, np.ndarray]:
    rows, curves = [], {}
    edges = None
    for path in sorted(input_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            names = [str(x) for x in data["band_names"]]
            values = dict(zip(names, data["band_means"].tolist()))
            rows.append({**meta, **values})
            curves[meta["slide"]] = data["curve_mean"]
            edges = data["curve_edges"]
    if not rows:
        raise RuntimeError(f"no spectra found in {input_dir}")
    return pd.DataFrame(rows), curves, edges


def ratio_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Per stain section, log2 band-power ratio of each scanner against AT2."""
    records = []
    for stain, block in frame.groupby("stain"):
        reference = block.loc[block["scanner"] == REFERENCE]
        if len(reference) != 1:
            raise RuntimeError(f"stain {stain}: expected exactly one {REFERENCE} slide")
        reference = reference.iloc[0]
        for _, row in block.iterrows():
            records.append(
                {
                    "stain": stain,
                    "scanner": row["scanner"],
                    "log2_ert": np.log2(row["high"] / reference["high"]),
                    "log2_ert_normalised": np.log2(
                        (row["high"] / row["low"]) / (reference["high"] / reference["low"])
                    ),
                    "log2_extended": np.log2(row["extended"] / reference["extended"]),
                }
            )
    return pd.DataFrame(records)


def bootstrap_ci(ratios: pd.DataFrame, column: str, replicates: int, seed: int) -> pd.DataFrame:
    """Section-level bootstrap: resample the 13 stain sections, not the patches."""
    stains = sorted(ratios["stain"].unique())
    wide = ratios.pivot(index="stain", columns="scanner", values=column).loc[stains]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(stains), size=(replicates, len(stains)))
    samples = np.stack([wide.to_numpy()[draw].mean(axis=0) for draw in draws])

    return pd.DataFrame(
        {
            "scanner": wide.columns,
            "estimate": wide.mean(axis=0).to_numpy(),
            "ci_low": np.percentile(samples, 2.5, axis=0),
            "ci_high": np.percentile(samples, 97.5, axis=0),
            "n_sections": len(stains),
        }
    ).set_index("scanner")


def variance_components(frame: pd.DataFrame) -> dict:
    """Share of log high-band power explained by scanner, stain, and their interaction.

    The design has one observation per (scanner, stain) cell, so interaction and
    error are confounded; the residual term is reported as the interaction.
    """
    wide = frame.pivot(index="stain", columns="scanner", values="high").apply(np.log10)
    grand = wide.to_numpy().mean()
    scanner_effect = wide.mean(axis=0) - grand
    stain_effect = wide.mean(axis=1) - grand
    fitted = grand + stain_effect.to_numpy()[:, None] + scanner_effect.to_numpy()[None, :]
    residual = wide.to_numpy() - fitted
    total = ((wide.to_numpy() - grand) ** 2).sum()
    return {
        "scanner": float((scanner_effect**2).sum() * wide.shape[0] / total),
        "stain": float((stain_effect**2).sum() * wide.shape[1] / total),
        "interaction_residual": float((residual**2).sum() / total),
    }


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame, curves, edges = load(input_dir)
    expected = frame["stain"].nunique() * frame["scanner"].nunique()
    if len(frame) != expected:
        print(f"WARNING: {len(frame)} slides for a {expected}-cell design; results are partial")

    frame.to_csv(output_dir / "band_power.csv", index=False)
    ratios = ratio_table(frame)
    ratios.to_csv(output_dir / "log2_ert_by_section.csv", index=False)

    ert = bootstrap_ci(ratios, "log2_ert", args.bootstrap, args.seed)
    normalised = bootstrap_ci(ratios, "log2_ert_normalised", args.bootstrap, args.seed)

    ert["fold"] = 2 ** ert["estimate"]
    ert["locked_fold"] = [LOCKED_ERT.get(s, np.nan) for s in ert.index]
    ert["locked_log2"] = np.log2(ert["locked_fold"])
    ert["normalised_log2"] = normalised["estimate"]
    ert["normalised_ci_low"] = normalised["ci_low"]
    ert["normalised_ci_high"] = normalised["ci_high"]
    ert = ert.sort_values("estimate", ascending=False)
    ert.to_csv(output_dir / "ert_summary.csv")

    verdicts = {}
    for key, (scanner, direction) in PREDICTIONS.items():
        row = ert.loc[scanner]
        passed = row["ci_low"] > 0 if direction == "greater" else row["ci_high"] < 0
        verdicts[key] = {
            "scanner": scanner,
            "direction": direction,
            "log2_ert": float(row["estimate"]),
            "ci": [float(row["ci_low"]), float(row["ci_high"])],
            "locked_log2": float(row["locked_log2"]),
            "pass": bool(passed),
            "normalised_pass": bool(
                row["normalised_ci_low"] > 0 if direction == "greater" else row["normalised_ci_high"] < 0
            ),
        }

    order = [s for s in ["GT450", "S60", "S360"] if s in ert.index]
    values = [float(ert.loc[s, "estimate"]) for s in order]
    verdicts["P3"] = {
        "requested_order": ["GT450", "S60", "S360"],
        "observed": dict(zip(order, values)),
        "pass": bool(all(a > b for a, b in zip(values, values[1:]))),
    }

    shared = [s for s in SHARED if s in ert.index]
    verdicts["S1"] = {
        "scanners": shared,
        "spearman": float(
            pd.Series([ert.loc[s, "estimate"] for s in shared]).corr(
                pd.Series([np.log2(LOCKED_ERT[s]) for s in shared]), method="spearman"
            )
        ),
        "pearson": float(
            pd.Series([ert.loc[s, "estimate"] for s in shared]).corr(
                pd.Series([np.log2(LOCKED_ERT[s]) for s in shared])
            )
        ),
    }
    verdicts["S3"] = variance_components(frame)

    # Content-matching check: the seven scanners of a section now read the same
    # locations, so their patch optical density should agree.  A scanner that
    # drifts here is not seeing the same tissue and its ratio is not comparable.
    od = frame.pivot(index="stain", columns="scanner", values="od_mean")
    spread = (od.max(axis=1) - od.min(axis=1)) / od.mean(axis=1)
    verdicts["content_check"] = {
        "od_mean_by_scanner": od.mean(axis=0).round(4).to_dict(),
        "worst_within_section_relative_spread": float(spread.max()),
        "median_within_section_relative_spread": float(spread.median()),
    }

    summary = {
        "sections": int(frame["stain"].nunique()),
        "scanners": int(frame["scanner"].nunique()),
        "slides": int(len(frame)),
        "patches_per_slide": int(frame["patches"].iloc[0]),
        "band_high_cyc_um": [0.10, 0.99],
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "verdicts": verdicts,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    pd.set_option("display.width", 200)
    print(f"\n{frame['stain'].nunique()} sections x {frame['scanner'].nunique()} scanners "
          f"= {len(frame)} slides, {frame['patches'].iloc[0]} patches each\n")
    display = ert[["estimate", "ci_low", "ci_high", "fold", "locked_fold", "normalised_log2"]]
    print(display.rename(columns={"estimate": "log2_ERT", "fold": "fold_vs_AT2",
                                  "locked_fold": "locked_fold"}).round(4).to_string())
    print("\nverdicts")
    for key in ("P1", "P2", "P3"):
        print(f"  {key}: {'PASS' if verdicts[key]['pass'] else 'FAIL'}  {verdicts[key]}")
    print(f"  S1 rank correlation with locked: spearman {verdicts['S1']['spearman']:.3f}, "
          f"pearson {verdicts['S1']['pearson']:.3f}")
    print(f"  S3 variance shares: {verdicts['S3']}")
    print(f"\nwrote {output_dir}")

    if args.figure:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        centres = 0.5 * (edges[:-1] + edges[1:])
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for scanner in sorted(frame["scanner"].unique()):
            slides = frame.loc[frame["scanner"] == scanner, "slide"]
            stack = np.nanmean(np.stack([curves[s] for s in slides]), axis=0)
            axes[0].loglog(centres, stack, label=scanner, lw=1.4)
        axes[0].set_xlabel("cycles / µm"); axes[0].set_ylabel("mean mode power")
        axes[0].axvspan(0.10, 0.99, color="0.9", zorder=0)
        axes[0].legend(fontsize=7); axes[0].set_title("Native radial spectra, pooled over stains")

        reference_curves = {}
        for scanner in sorted(frame["scanner"].unique()):
            slides = frame.loc[frame["scanner"] == scanner, "slide"]
            reference_curves[scanner] = np.nanmean(np.stack([curves[s] for s in slides]), axis=0)
        for scanner, curve in reference_curves.items():
            axes[1].semilogx(centres, np.log2(curve / reference_curves[REFERENCE]), label=scanner, lw=1.4)
        axes[1].axhline(0, color="k", lw=0.8)
        axes[1].axvspan(0.10, 0.99, color="0.9", zorder=0)
        axes[1].set_xlabel("cycles / µm"); axes[1].set_ylabel(f"log2 transfer vs {REFERENCE}")
        axes[1].set_title("Effective relative transfer, native grid")
        fig.tight_layout()
        fig.savefig(output_dir / "native_ert.png", dpi=180)
        print(f"wrote {output_dir / 'native_ert.png'}")


if __name__ == "__main__":
    main()

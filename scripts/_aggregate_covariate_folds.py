"""Merge the leave-one-section-out folds into the covariate table.

The folds and the full-data table are separate jobs -- the folds because they are
independent and belong in an array, the table because it was already fitted in
the Arm A pass on the identical rows.  This joins them and applies the stability
rule, without refitting anything.

A coefficient is stable only if it keeps its sign and its significance in every
fold.  That is the check the sparse cohort's conclusion rested on, run by hand on
one scanner and never written down; the point of doing it here is that a reader
can see which fold breaks which coefficient.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BAND = "high"


def aggregate(directory: Path) -> pd.DataFrame:
    table = pd.read_csv(directory / f"stain_covariate_{BAND}.csv")
    table = table[[c for c in table.columns
                   if c not in ("folds", "folds_significant", "sign_stable", "stable")]]
    parts = sorted((directory / f"loo_{BAND}").glob("*.csv"))
    if not parts:
        raise FileNotFoundError(f"no folds under {directory / f'loo_{BAND}'}")
    folds = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    folds.to_csv(directory / f"stain_covariate_{BAND}_loo.csv", index=False)

    rows = []
    for (scanner, covariate), block in folds.groupby(["scanner", "covariate"]):
        full = table.loc[(table["scanner"] == scanner)
                         & (table["covariate"] == covariate)].iloc[0]
        same_sign = bool((np.sign(block["beta"]) == np.sign(full["beta"])).all())
        significant = int((block["p"] < 0.05).sum())
        rows.append({
            "scanner": scanner, "covariate": covariate,
            "beta_full": full["beta"], "p_full": full["p"],
            "beta_min": float(block["beta"].min()),
            "beta_max": float(block["beta"].max()),
            "folds": len(block), "folds_significant": significant,
            "sign_stable": same_sign,
            "stable": bool(same_sign and significant == len(block) and full["p"] < 0.05)})
    stability = pd.DataFrame(rows)
    stability.to_csv(directory / f"stain_covariate_{BAND}_stability.csv", index=False)

    merged = table.merge(
        stability[["scanner", "covariate", "folds", "folds_significant",
                   "sign_stable", "stable"]], on=["scanner", "covariate"], how="left")
    merged.to_csv(directory / f"stain_covariate_{BAND}.csv", index=False)

    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary.update({"leave_one_out": True,
                    "folds": int(folds["held_out"].nunique()),
                    "stable_coefficients": int(stability["stable"].sum()),
                    "coefficients": int(len(stability))})
    summary_path.write_text(json.dumps(summary, indent=2))
    return stability


if __name__ == "__main__":
    for name in sys.argv[1:] or ["all_blocks", "focus_ok"]:
        directory = Path("outputs/plism_core_stain_covariate") / name
        stability = aggregate(directory)
        block = stability.loc[stability["covariate"] == "stain_od"]
        print(f"=== {name}: {int(stability['stable'].sum())} of {len(stability)} "
              f"coefficients stable over {block['folds'].iloc[0]} folds ===")
        print(block.round(4).to_string(index=False), "\n")

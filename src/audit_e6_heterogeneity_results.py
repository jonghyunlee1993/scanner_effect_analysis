"""Independently audit and lock the frozen E6 LOSO heterogeneity results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_exp05_tissue_random_slopes import benjamini_hochberg
from e5_comparator_population import FEATURE_CONDITIONS, IMAGE_CONDITIONS, SCANNERS
from fetch_e0_pfm_checkpoints import sha256


METHODS = (*IMAGE_CONDITIONS, *FEATURE_CONDITIONS)
TABLE_ROWS = {
    "variance_components.csv": 240,
    "tissue_blups.csv": 8_160,
    "slide_within_tissue_blups.csv": 24_840,
    "tissue_class_sizes.csv": 37,
    "tissue_winner_ranks.csv": 1_224,
    "winner_summary.csv": 8,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="outputs/e6_loso_heterogeneity")
    parser.add_argument(
        "--contrasts", default="outputs/e6_loso_replicate_contrasts/summary.json"
    )
    parser.add_argument(
        "--e5-lock", default="outputs/e5_comparator_results_lock/summary.json"
    )
    parser.add_argument(
        "--contract", default="docs/e6_heterogeneity_execution_contract.md"
    )
    parser.add_argument("--output", default="outputs/e6_heterogeneity_results_lock")
    return parser.parse_args()


def main():
    args = parse_args()
    results = Path(args.results)
    analysis_summary_path = results / "summary.json"
    analysis_summary = json.loads(analysis_summary_path.read_text())
    contrast_summary_path = Path(args.contrasts)
    e5_lock_path = Path(args.e5_lock)
    contract_path = Path(args.contract)
    if analysis_summary.get("analysis_gate_pass") is not True:
        raise RuntimeError("E6 heterogeneity analysis gate has not passed")
    if analysis_summary.get("contrast_summary_sha256") != sha256(contrast_summary_path):
        raise ValueError("E6 contrast-summary hash drift")
    if analysis_summary.get("e5_result_lock_sha256") != sha256(e5_lock_path):
        raise ValueError("E5 result-lock hash drift")
    if analysis_summary.get("e6_contract_sha256") != sha256(contract_path):
        raise ValueError("E6 execution-contract hash drift")

    manifest_rows = []
    frames = {}
    for filename, expected_rows in TABLE_ROWS.items():
        path = results / filename
        frame = pd.read_csv(path, dtype={"slide_id": str})
        frames[filename] = frame
        numeric = frame.select_dtypes(include=[np.number])
        if len(frame) != expected_rows or not np.isfinite(numeric).all().all():
            raise ValueError(f"{filename}: count/finiteness audit failed")
        manifest_rows.append({
            "artifact": str(path),
            "rows": len(frame),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    for filename in (
        "figure_e6_correction_heterogeneity.png",
        "figure_e6_correction_heterogeneity.pdf",
    ):
        path = results / filename
        if path.stat().st_size <= 10_000:
            raise ValueError(f"{filename}: implausibly small figure")
        manifest_rows.append({
            "artifact": str(path),
            "rows": 1,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })

    components = frames["variance_components.csv"]
    expected_family_sizes = {
        ("radius_benefit", "full_37_tissues"): 20,
        ("content_change", "full_37_tissues"): 100,
        ("radius_benefit", "min3_sensitivity"): 20,
        ("content_change", "min3_sensitivity"): 100,
    }
    observed_family_sizes = components.groupby(
        ["endpoint", "analysis_set"]
    ).size().to_dict()
    if observed_family_sizes != expected_family_sizes:
        raise ValueError(f"BH family-size mismatch: {observed_family_sizes}")
    if not (
        components["optimizer_converged"].astype(bool).all()
        and components["reduced_optimizer_converged"].astype(bool).all()
    ):
        raise ValueError("one or more nested REML fits did not converge")
    for keys, frame in components.groupby(["endpoint", "analysis_set"]):
        expected_q = benjamini_hochberg(frame["tissue_variance_p_mixture"])
        if not np.allclose(expected_q, frame["tissue_variance_q_bh"], atol=1e-12):
            raise ValueError(f"BH audit failed for {keys}")
        if not np.array_equal(
            frame["tissue_heterogeneity_claim"].astype(bool).to_numpy(),
            expected_q < 0.05,
        ):
            raise ValueError(f"heterogeneity decision audit failed for {keys}")

    tissue_classes = frames["tissue_class_sizes.csv"]
    if (
        tissue_classes["slides"].sum() != 109
        or (tissue_classes["slides"] == 1).sum() != 1
        or (tissue_classes["slides"] == 2).sum() != 5
        or tissue_classes.loc[
            tissue_classes["included_min3_sensitivity"].astype(bool), "slides"
        ].sum()
        != 98
    ):
        raise ValueError("tissue class-size/minimum-three sensitivity audit failed")
    tissue_blups = frames["tissue_blups.csv"]
    if set(tissue_blups["method"]) != set(METHODS):
        raise ValueError("tissue BLUP method population mismatch")
    full_tissue_counts = tissue_blups[
        tissue_blups["analysis_set"] == "full_37_tissues"
    ].groupby(["endpoint", "encoder_id", "method"])["tissue_type"].nunique()
    if not (full_tissue_counts == 37).all():
        raise ValueError("full tissue BLUP population mismatch")

    winners = frames["winner_summary.csv"]
    if (
        set(winners["frozen_global_winner"]) != {"orthogonal_procrustes"}
        or not (winners.groupby("analysis_set")["encoder_id"].nunique() == 4).all()
    ):
        raise ValueError("frozen global-winner diagnostic mismatch")
    ranks = frames["tissue_winner_ranks.csv"]
    top = ranks[ranks["predicted_rank"] == 1]
    recalculated = (
        top.groupby(["analysis_set", "encoder_id"])["global_winner_remains_first"]
        .agg(["sum", "count", "mean"])
        .reset_index()
        .merge(winners, on=["analysis_set", "encoder_id"], validate="one_to_one")
    )
    if not (
        np.array_equal(recalculated["sum"], recalculated["global_winner_first_tissues"])
        and np.array_equal(recalculated["count"], recalculated["tissues"])
        and np.allclose(
            recalculated["mean"], recalculated["global_winner_first_fraction"]
        )
    ):
        raise ValueError("winner fraction audit failed")

    manifest_rows.extend([
        {
            "artifact": str(analysis_summary_path),
            "rows": 1,
            "bytes": analysis_summary_path.stat().st_size,
            "sha256": sha256(analysis_summary_path),
        },
        {
            "artifact": str(contrast_summary_path),
            "rows": 1,
            "bytes": contrast_summary_path.stat().st_size,
            "sha256": sha256(contrast_summary_path),
        },
    ])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = output / "artifact_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    claims = (
        components.groupby(["analysis_set", "endpoint"])[
            "tissue_heterogeneity_claim"
        ]
        .agg(["sum", "count"])
        .reset_index()
        .to_dict("records")
    )
    non_global_winners = top[
        (top["analysis_set"] == "full_37_tissues")
        & (top["predicted_tissue_winner"] != "orthogonal_procrustes")
    ][
        [
            "encoder_id",
            "tissue_type",
            "slides_in_tissue",
            "predicted_tissue_winner",
            "predicted_radius_benefit",
        ]
    ].to_dict("records")
    summary = {
        "analysis": "e6_loso_heterogeneity_result_lock",
        "model": "nested profiled REML with tissue and slide-within-tissue random slopes",
        "full_population": {"slides": 109, "tissues": 37},
        "min3_sensitivity_population": {"slides": 98, "tissues": 31},
        "fits": 240,
        "all_full_and_reduced_fits_converged": True,
        "bh_claim_counts": claims,
        "winner_summary": winners.to_dict("records"),
        "non_global_winner_tissues_full": non_global_winners,
        "contrast_summary_sha256": sha256(contrast_summary_path),
        "e5_result_lock_sha256": sha256(e5_lock_path),
        "e6_contract_sha256": sha256(contract_path),
        "analysis_summary_sha256": sha256(analysis_summary_path),
        "artifacts": len(manifest),
        "artifact_bytes": int(manifest["bytes"].sum()),
        "artifact_manifest": str(manifest_path.resolve()),
        "result_lock_pass": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

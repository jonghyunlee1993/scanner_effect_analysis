"""Aggregate strict nested-training-exclusion RF1.1 image-only selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_rf1_noharm_gain import (
    IDENTITY_CAP,
    SPECTRUM_TOLERANCE,
    candidate_metrics,
    candidate_power,
    choose_candidate_one_se,
    format_selection,
    gamut_gate,
    load_fold,
    render_figure,
    sha256,
)
from build_rf1_noharm_nested_cell import ANALYSIS as INNER_ANALYSIS
from e5_comparator_population import FOVS, SCANNERS
from e5_reinhard_residual_frequency import (
    RF1_FOLDS,
    RF1_GAIN_CAP_CANDIDATES,
    RF1_VERSION,
    log_spectrum_rmse,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inner", default="outputs/rf1_improvement_pilot/noharm_nested/inner"
    )
    parser.add_argument("--outer", default="outputs/e5_rf1_input_fold_audit")
    parser.add_argument(
        "--output", default="outputs/rf1_improvement_pilot/noharm_nested/result"
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def load_inner(path: Path, fov: int, outer_fold: int, inner_fold: int):
    summary_path = path.with_suffix(".summary.json")
    if not path.exists() or not summary_path.exists():
        raise FileNotFoundError(path)
    summary = json.loads(summary_path.read_text())
    training_folds = set(summary.get("training_folds", []))
    if not (
        summary.get("analysis") == INNER_ANALYSIS
        and summary.get("parent_rf1_version") == RF1_VERSION
        and summary.get("outcome_access") is False
        and summary.get("pfm_feature_access") is False
        and summary.get("strict_nested_training_exclusion") is True
        and summary.get("fov") == fov
        and summary.get("outer_fold") == outer_fold
        and summary.get("inner_validation_fold") == inner_fold
        and outer_fold not in training_folds
        and inner_fold not in training_folds
        and training_folds == set(range(RF1_FOLDS)) - {outer_fold, inner_fold}
        and summary.get("output_sha256") == sha256(path)
        and summary.get("inner_audit_gate_pass") is True
    ):
        raise RuntimeError(f"invalid strict-nested inner audit: {path}")
    with np.load(path) as source:
        values = {name: source[name] for name in source.files}
    caps = np.asarray(RF1_GAIN_CAP_CANDIDATES, dtype=np.float64)
    expected_shape = (
        len(caps),
        len(values["heldout_slide_ids"]),
        len(SCANNERS) - 1,
        100,
    )
    if not (
        int(values["fov"]) == fov
        and int(values["outer_fold"]) == outer_fold
        and int(values["inner_validation_fold"]) == inner_fold
        and np.array_equal(values["caps"], caps)
        and values["material_range_fraction"].shape == expected_shape
        and values["validation_output_power"].shape
        == (len(caps), len(SCANNERS) - 1, len(values["radial_frequency"]))
        and np.isfinite(values["validation_output_power"]).all()
    ):
        raise RuntimeError(f"invalid strict-nested arrays: {path}")
    return values, summary


def main():
    args = parse_args()
    inner_root = Path(args.inner)
    outer_root = Path(args.outer)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    candidate_caps = np.asarray(
        [IDENTITY_CAP, *RF1_GAIN_CAP_CANDIDATES], dtype=np.float64
    )

    inner = {}
    inner_hashes = {}
    outer = {}
    outer_hashes = {}
    for fov in FOVS:
        for outer_fold in range(RF1_FOLDS):
            outer_path = outer_root / f"fov_{fov}_fold_{outer_fold}.npz"
            outer_values, outer_summary = load_fold(outer_path)
            outer[fov, outer_fold] = outer_values
            outer_hashes[f"fov_{fov}_outer_{outer_fold}"] = outer_summary[
                "output_sha256"
            ]
            for inner_fold in range(RF1_FOLDS):
                if inner_fold == outer_fold:
                    continue
                path = inner_root / (
                    f"fov_{fov}_outer_{outer_fold}_inner_{inner_fold}.npz"
                )
                values, summary = load_inner(path, fov, outer_fold, inner_fold)
                inner[fov, outer_fold, inner_fold] = values
                inner_hashes[
                    f"fov_{fov}_outer_{outer_fold}_inner_{inner_fold}"
                ] = summary["output_sha256"]

    selection_rows = []
    evaluation_rows = []
    cell_rows = []
    original_index = list(RF1_GAIN_CAP_CANDIDATES).index(1.25)

    for fov in FOVS:
        radial_frequency = inner[fov, 0, 1]["radial_frequency"]
        for scanner_index, scanner in enumerate(SCANNERS[1:]):
            selected_indices = []
            selected_output_powers = []
            selected_metric_sets = []
            target_powers = []
            base_powers = []
            original_powers = []
            for outer_fold in range(RF1_FOLDS):
                inner_folds = [
                    fold for fold in range(RF1_FOLDS) if fold != outer_fold
                ]
                base_rmse = []
                candidate_rmse = []
                gamut_pass = []
                for inner_fold in inner_folds:
                    values = inner[fov, outer_fold, inner_fold]
                    if not np.array_equal(values["radial_frequency"], radial_frequency):
                        raise RuntimeError("strict nested radial frequency mismatch")
                    target = values["validation_target_power"]
                    base = values["validation_base_power"][scanner_index]
                    base_rmse.append(log_spectrum_rmse(base, target, radial_frequency))
                    local_rmse = []
                    local_gamut = []
                    for candidate_index in range(len(candidate_caps)):
                        power = candidate_power(values, candidate_index, scanner_index)
                        metrics = candidate_metrics(values, candidate_index, scanner_index)
                        local_rmse.append(
                            log_spectrum_rmse(power, target, radial_frequency)
                        )
                        local_gamut.append(
                            gamut_gate(
                                metrics["material_range_fraction"],
                                metrics["projection_rgb_mae"],
                                metrics["final_clamp_mae"],
                            )
                        )
                    candidate_rmse.append(local_rmse)
                    gamut_pass.append(local_gamut)

                pooled_target = sum(
                    inner[fov, outer_fold, fold]["validation_target_power"]
                    for fold in inner_folds
                )
                pooled_base = sum(
                    inner[fov, outer_fold, fold]["validation_base_power"][scanner_index]
                    for fold in inner_folds
                )
                pooled_base_rmse = log_spectrum_rmse(
                    pooled_base, pooled_target, radial_frequency
                )
                pooled_candidate_rmse = np.asarray(
                    [
                        log_spectrum_rmse(
                            sum(
                                candidate_power(
                                    inner[fov, outer_fold, fold],
                                    candidate_index,
                                    scanner_index,
                                )
                                for fold in inner_folds
                            ),
                            pooled_target,
                            radial_frequency,
                        )
                        for candidate_index in range(len(candidate_caps))
                    ]
                )
                base_rmse = np.asarray(base_rmse)
                candidate_rmse = np.asarray(candidate_rmse)
                gamut_pass = np.asarray(gamut_pass)
                (
                    selected,
                    eligible,
                    mean_delta,
                    best_candidate,
                    best_candidate_se,
                    one_se_threshold,
                    within_one_se,
                ) = choose_candidate_one_se(
                    candidate_caps,
                    base_rmse,
                    candidate_rmse,
                    gamut_pass,
                )
                selected_indices.append(selected)
                for candidate_index, cap in enumerate(candidate_caps):
                    selection_rows.append(
                        {
                            "fov": fov,
                            "scanner": scanner,
                            "outer_fold": outer_fold,
                            "candidate": "identity"
                            if candidate_index == 0
                            else f"cap_{cap:g}",
                            "gain_cap": cap,
                            "inner_validation_folds": ",".join(map(str, inner_folds)),
                            "selection_pooled_base_rmse": pooled_base_rmse,
                            "selection_pooled_output_rmse": pooled_candidate_rmse[
                                candidate_index
                            ],
                            "selection_rmse_change": pooled_candidate_rmse[candidate_index]
                            - pooled_base_rmse,
                            "inner_mean_delta_vs_identity": mean_delta[candidate_index],
                            "best_mean_delta_candidate": candidate_index
                            == best_candidate,
                            "best_candidate_delta_se": best_candidate_se,
                            "one_se_threshold": one_se_threshold,
                            "within_one_se": bool(within_one_se[candidate_index]),
                            "all_inner_folds_gamut_pass": bool(
                                gamut_pass[:, candidate_index].all()
                            ),
                            "all_inner_folds_spectrum_nonworse": bool(
                                np.all(
                                    candidate_rmse[:, candidate_index]
                                    <= base_rmse + SPECTRUM_TOLERANCE
                                )
                            ),
                            "eligible": bool(eligible[candidate_index]),
                            "selected": candidate_index == selected,
                            "strict_nested_training_exclusion": True,
                        }
                    )

                values = outer[fov, outer_fold]
                target = values["validation_target_power"]
                base = values["validation_base_power"][scanner_index]
                selected_power = candidate_power(values, selected, scanner_index)
                selected_metrics = candidate_metrics(values, selected, scanner_index)
                before = log_spectrum_rmse(base, target, radial_frequency)
                after = log_spectrum_rmse(selected_power, target, radial_frequency)
                evaluation_rows.append(
                    {
                        "fov": fov,
                        "scanner": scanner,
                        "outer_fold": outer_fold,
                        "heldout_slides": len(values["heldout_slide_ids"]),
                        "selected_candidate": "identity"
                        if selected == 0
                        else f"cap_{candidate_caps[selected]:g}",
                        "selected_gain_cap": candidate_caps[selected],
                        "base_log_spectrum_rmse": before,
                        "output_log_spectrum_rmse": after,
                        "log_spectrum_rmse_change": after - before,
                        "rmse_reduction_percent": 100.0 * (before - after) / before,
                        "outer_spectrum_nonworse": after
                        <= before + SPECTRUM_TOLERANCE,
                        "outer_gamut_pass": gamut_gate(
                            selected_metrics["material_range_fraction"],
                            selected_metrics["projection_rgb_mae"],
                            selected_metrics["final_clamp_mae"],
                        ),
                        "strict_nested_training_exclusion": True,
                    }
                )
                selected_output_powers.append(selected_power)
                selected_metric_sets.append(selected_metrics)
                target_powers.append(target)
                base_powers.append(base)
                original_powers.append(
                    values["validation_output_power"][original_index, scanner_index]
                )

            selected_caps = [float(candidate_caps[index]) for index in selected_indices]
            pooled_target = sum(target_powers)
            pooled_base = sum(base_powers)
            pooled_selected = sum(selected_output_powers)
            pooled_original = sum(original_powers)
            before = log_spectrum_rmse(pooled_base, pooled_target, radial_frequency)
            after = log_spectrum_rmse(pooled_selected, pooled_target, radial_frequency)
            original_after = log_spectrum_rmse(
                pooled_original, pooled_target, radial_frequency
            )
            material = np.concatenate(
                [value["material_range_fraction"] for value in selected_metric_sets]
            )
            projection = np.concatenate(
                [value["projection_fraction"] for value in selected_metric_sets]
            )
            projection_mae = np.concatenate(
                [value["projection_rgb_mae"] for value in selected_metric_sets]
            )
            final_mae = np.concatenate(
                [value["final_clamp_mae"] for value in selected_metric_sets]
            )
            cell_rows.append(
                {
                    "fov": fov,
                    "scanner": scanner,
                    "patches": len(material),
                    "selection_label": format_selection(selected_caps),
                    "selected_identity_folds": sum(
                        np.isclose(value, IDENTITY_CAP) for value in selected_caps
                    ),
                    "selected_caps": ",".join(f"{value:g}" for value in selected_caps),
                    "base_log_spectrum_rmse": before,
                    "noharm_output_log_spectrum_rmse": after,
                    "noharm_log_spectrum_rmse_change": after - before,
                    "noharm_rmse_reduction_percent": 100.0 * (before - after) / before,
                    "original_cap_1p25_output_log_spectrum_rmse": original_after,
                    "original_cap_1p25_log_spectrum_rmse_change": original_after - before,
                    "original_cap_1p25_rmse_reduction_percent": 100.0
                    * (before - original_after)
                    / before,
                    "noharm_minus_original_output_rmse": after - original_after,
                    "material_range_fraction_mean": float(material.mean()),
                    "material_range_fraction_q99": float(np.quantile(material, 0.99)),
                    "projection_fraction_mean": float(projection.mean()),
                    "projection_rgb_mae_mean": float(projection_mae.mean()),
                    "final_clamp_mae_max": float(final_mae.max()),
                    "aggregate_spectrum_nonworse": after
                    <= before + SPECTRUM_TOLERANCE,
                    "aggregate_gamut_pass": gamut_gate(
                        material, projection_mae, final_mae
                    ),
                    "strict_nested_training_exclusion": True,
                }
            )

    selection = pd.DataFrame(selection_rows)
    evaluation = pd.DataFrame(evaluation_rows)
    cell = pd.DataFrame(cell_rows)
    if not (
        len(selection) == 1050
        and len(evaluation) == 75
        and len(cell) == 15
        and selection["strict_nested_training_exclusion"].all()
        and evaluation["strict_nested_training_exclusion"].all()
        and cell["strict_nested_training_exclusion"].all()
        and np.isfinite(
            cell.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
        ).all()
    ):
        raise RuntimeError("incomplete strict-nested RF1.1 population")

    selection_path = output / "candidate_selection.csv"
    evaluation_path = output / "outer_fold_evaluation.csv"
    cell_path = output / "scanner_fov_summary.csv"
    selection.to_csv(selection_path, index=False)
    evaluation.to_csv(evaluation_path, index=False)
    cell.to_csv(cell_path, index=False)
    figure_paths = render_figure(
        cell,
        output,
        args.dpi,
        stem="rf1_noharm_strict_nested_spectrum",
        method_label="Strict-nested no-harm selection",
    )

    identity_selections = int(
        np.isclose(evaluation["selected_gain_cap"], IDENTITY_CAP).sum()
    )
    summary = {
        "analysis": "rf1_noharm_strict_nested_result",
        "parent_rf1_version": RF1_VERSION,
        "outcome_access": False,
        "pfm_feature_access": False,
        "strict_nested_training_exclusion": True,
        "selection_protocol": (
            "For outer fold h, each of four inner gains is fitted on the three folds "
            "excluding h and inner fold j. Identity/cap selection uses only those four "
            "inner validations. Eligible candidates pass gamut and are non-worse in every "
            "inner fold. For fold-wise delta = candidate RMSE - identity RMSE, find the "
            "eligible candidate with minimum mean delta, compute SE = sample SD of its "
            "four deltas (ddof=1) / sqrt(4), then select the smallest-cap eligible candidate "
            "with mean delta <= best mean delta + SE. Selected cap is evaluated once on "
            "outer h using the locked transform/gain fitted on all folds except h."
        ),
        "selection_rule": "one_standard_error_preference_toward_identity",
        "selection_delta": "candidate_log_spectrum_RMSE - identity_log_spectrum_RMSE",
        "selection_se_formula": "std(best_candidate_fold_deltas, ddof=1) / sqrt(4)",
        "algebraic_reuse": [
            "Lab sufficient statistics",
            "raw AT2 training power",
            "raw AT2 inner-validation power",
            "locked outer-fold candidate render after selection",
        ],
        "nonlinear_gpu_rerender": [
            "post-Reinhard inner-training source spectrum",
            "post-Reinhard inner-validation candidate outputs",
        ],
        "fovs": list(FOVS),
        "source_scanners": list(SCANNERS[1:]),
        "outer_folds": RF1_FOLDS,
        "inner_directional_audits": len(inner_hashes),
        "unique_unordered_inner_fits": len(FOVS) * 10,
        "candidate_caps": candidate_caps.tolist(),
        "identity_selections": identity_selections,
        "nonidentity_selections": int(len(evaluation) - identity_selections),
        "outer_fold_spectrum_nonworse": int(
            evaluation["outer_spectrum_nonworse"].sum()
        ),
        "outer_fold_spectrum_total": len(evaluation),
        "outer_fold_gamut_pass": int(evaluation["outer_gamut_pass"].sum()),
        "aggregate_spectrum_nonworse_cells": int(
            cell["aggregate_spectrum_nonworse"].sum()
        ),
        "aggregate_gamut_pass_cells": int(cell["aggregate_gamut_pass"].sum()),
        "original_cap_1p25_spectrum_nonworse_cells": int(
            (cell["original_cap_1p25_log_spectrum_rmse_change"] <= 0).sum()
        ),
        "mean_noharm_rmse_reduction_percent": float(
            cell["noharm_rmse_reduction_percent"].mean()
        ),
        "mean_original_cap_1p25_rmse_reduction_percent": float(
            cell["original_cap_1p25_rmse_reduction_percent"].mean()
        ),
        "s360_identity_selections": int(
            np.isclose(
                evaluation.loc[evaluation["scanner"] == "s360", "selected_gain_cap"],
                IDENTITY_CAP,
            ).sum()
        ),
        "s360_selections": int((evaluation["scanner"] == "s360").sum()),
        "inner_audit_sha256": inner_hashes,
        "outer_audit_sha256": outer_hashes,
        "artifacts": {
            path.name: sha256(path)
            for path in [selection_path, evaluation_path, cell_path, *figure_paths]
        },
        "strict_nested_gate_pass": bool(
            len(inner_hashes) == 60
            and evaluation["outer_gamut_pass"].all()
            and evaluation["outer_spectrum_nonworse"].all()
            and cell["aggregate_gamut_pass"].all()
            and cell["aggregate_spectrum_nonworse"].all()
        ),
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not summary["strict_nested_gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

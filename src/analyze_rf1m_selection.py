"""Aggregate the RF1M strict-nested image-only selection and apply its gates.

Selection uses only the 60 inner cells; the chosen cap is then read once from the
15 outer cells, which were fitted without the evaluated fold.  The locked radial
RF1 cap-1.25 audit is loaded purely as a reference column.

The script exits non-zero when any frozen image-only gate fails, so a dependent
PFM job can never start from a failed candidate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_rf1_noharm_gain import (
    SPECTRUM_TOLERANCE,
    format_selection,
    gamut_gate,
    load_fold,
    render_figure,
    sha256,
)
from build_rf1m_cell import ANALYSIS as CELL_ANALYSIS
from build_rf1m_cell import cell_name
from e5_comparator_population import FOVS, SCANNERS
from e5_reinhard_residual_frequency import (
    RF1_FOLDS,
    RF1_GAIN_CAP_CANDIDATES,
    log_spectrum_rmse,
)
from rf1m_combined import (
    RF1M_CAP_CANDIDATES,
    RF1M_IDENTITY_CAP,
    RF1M_SHRINKAGE_SE,
    RF1M_SIGMAS,
    RF1M_VERSION,
    choose_candidate_k_se,
)


METRIC_NAMES = (
    "material_range_fraction",
    "projection_fraction",
    "projection_rgb_mae",
    "final_clamp_mae",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", default="outputs/rf1m_candidate/cells")
    parser.add_argument("--rf1-audit", default="outputs/e5_rf1_input_fold_audit")
    parser.add_argument("--output", default="outputs/rf1m_candidate/selection")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def load_cell(root: Path, fov: int, stage: str, outer_fold: int, validation_fold: int):
    path = root / f"{cell_name(fov, stage, outer_fold, validation_fold)}.npz"
    summary_path = path.with_suffix(".summary.json")
    if not path.exists() or not summary_path.exists():
        raise FileNotFoundError(path)
    summary = json.loads(summary_path.read_text())
    training_folds = set(summary.get("training_folds", []))
    expected_training = set(range(RF1_FOLDS)) - {outer_fold, validation_fold}
    if not (
        summary.get("analysis") == CELL_ANALYSIS
        and summary.get("rf1m_version") == RF1M_VERSION
        and summary.get("outcome_access") is False
        and summary.get("pfm_feature_access") is False
        and summary.get("stage") == stage
        and summary.get("fov") == fov
        and summary.get("outer_fold") == outer_fold
        and summary.get("validation_fold") == validation_fold
        and training_folds == expected_training
        and summary.get("output_sha256") == sha256(path)
        and summary.get("cell_gate_pass") is True
    ):
        raise RuntimeError(f"invalid RF1M cell: {path}")
    with np.load(path) as source:
        values = {name: source[name] for name in source.files}
    caps = np.asarray(RF1M_CAP_CANDIDATES, dtype=np.float64)
    expected_shape = (len(caps), len(values["heldout_slide_ids"]), len(SCANNERS) - 1, 100)
    if not (
        np.array_equal(values["caps"], caps)
        and np.array_equal(values["sigmas"], np.asarray(RF1M_SIGMAS, dtype=np.float64))
        and values["material_range_fraction"].shape == expected_shape
        and values["validation_output_power"].shape
        == (len(caps), len(SCANNERS) - 1, len(values["radial_frequency"]))
        and np.isfinite(values["validation_output_power"]).all()
        and np.all(values["validation_output_power"] >= 0)
        and np.all(values["validation_base_power"] >= 0)
        and np.all(values["validation_target_power"] >= 0)
    ):
        raise RuntimeError(f"invalid RF1M cell arrays: {path}")
    return values, summary


def candidate_metrics(cell: dict[str, np.ndarray], candidate_index: int, scanner_index: int):
    return {
        name: cell[name][candidate_index, :, scanner_index].reshape(-1)
        for name in METRIC_NAMES
    }


def candidate_power(cell: dict[str, np.ndarray], candidate_index: int, scanner_index: int):
    """Return the validation radial power of one candidate.

    Identity means no correction at all, so its power is the Reinhard base
    itself.  The cell also stores an explicitly rendered identity candidate; it
    agrees with the base only to float32 rounding, which would otherwise trip the
    1e-12 non-worse tolerance.  That render is kept as the audit quantity
    reported by `identity_render_log_spectrum_deviation_max`.
    """
    if candidate_index == 0:
        return cell["validation_base_power"][scanner_index]
    return cell["validation_output_power"][candidate_index, scanner_index]


def main():
    args = parse_args()
    cells_root = Path(args.cells)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    caps = np.asarray(RF1M_CAP_CANDIDATES, dtype=np.float64)
    rf1_cap_index = list(RF1_GAIN_CAP_CANDIDATES).index(1.25)

    inner: dict[tuple[int, int, int], dict] = {}
    outer: dict[tuple[int, int], dict] = {}
    reference: dict[tuple[int, int], dict] = {}
    cell_hashes: dict[str, str] = {}
    for fov in FOVS:
        for outer_fold in range(RF1_FOLDS):
            values, summary = load_cell(cells_root, fov, "outer", outer_fold, outer_fold)
            outer[fov, outer_fold] = values
            cell_hashes[cell_name(fov, "outer", outer_fold, outer_fold)] = summary[
                "output_sha256"
            ]
            reference[fov, outer_fold], _ = load_fold(
                Path(args.rf1_audit) / f"fov_{fov}_fold_{outer_fold}.npz"
            )
            for validation_fold in range(RF1_FOLDS):
                if validation_fold == outer_fold:
                    continue
                values, summary = load_cell(
                    cells_root, fov, "inner", outer_fold, validation_fold
                )
                inner[fov, outer_fold, validation_fold] = values
                cell_hashes[
                    cell_name(fov, "inner", outer_fold, validation_fold)
                ] = summary["output_sha256"]

    selection_rows = []
    evaluation_rows = []
    cell_rows = []
    identity_deviation = 0.0
    for fov in FOVS:
        radial_frequency = outer[fov, 0]["radial_frequency"]
        for scanner_index, scanner in enumerate(SCANNERS[1:]):
            selected_caps = []
            selected_powers = []
            selected_metric_sets = []
            base_powers = []
            target_powers = []
            reference_powers = []
            for outer_fold in range(RF1_FOLDS):
                inner_folds = [fold for fold in range(RF1_FOLDS) if fold != outer_fold]
                base_rmse = []
                candidate_rmse = []
                gamut_pass = []
                for inner_fold in inner_folds:
                    values = inner[fov, outer_fold, inner_fold]
                    if not np.array_equal(values["radial_frequency"], radial_frequency):
                        raise RuntimeError("RF1M radial frequency mismatch")
                    target = values["validation_target_power"]
                    base_rmse.append(
                        log_spectrum_rmse(
                            values["validation_base_power"][scanner_index],
                            target,
                            radial_frequency,
                        )
                    )
                    candidate_rmse.append(
                        [
                            log_spectrum_rmse(
                                candidate_power(values, candidate_index, scanner_index),
                                target,
                                radial_frequency,
                            )
                            for candidate_index in range(len(caps))
                        ]
                    )
                    identity_deviation = max(
                        identity_deviation,
                        abs(
                            log_spectrum_rmse(
                                values["validation_output_power"][0, scanner_index],
                                target,
                                radial_frequency,
                            )
                            - base_rmse[-1]
                        ),
                    )
                    local_gamut = []
                    for candidate_index in range(len(caps)):
                        metrics = candidate_metrics(values, candidate_index, scanner_index)
                        local_gamut.append(
                            gamut_gate(
                                metrics["material_range_fraction"],
                                metrics["projection_rgb_mae"],
                                metrics["final_clamp_mae"],
                            )
                        )
                    gamut_pass.append(local_gamut)

                base_rmse = np.asarray(base_rmse)
                candidate_rmse = np.asarray(candidate_rmse)
                gamut_pass = np.asarray(gamut_pass)
                (
                    selected,
                    eligible,
                    mean_delta,
                    best_candidate,
                    best_candidate_se,
                    shrinkage_threshold,
                    within_threshold,
                ) = choose_candidate_k_se(
                    caps, base_rmse, candidate_rmse, gamut_pass, RF1M_SHRINKAGE_SE
                )
                for candidate_index, cap in enumerate(caps):
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
                            "inner_mean_delta_vs_identity": mean_delta[candidate_index],
                            "best_mean_delta_candidate": candidate_index == best_candidate,
                            "best_candidate_delta_se": best_candidate_se,
                            "shrinkage_threshold": shrinkage_threshold,
                            "within_shrinkage_threshold": bool(within_threshold[candidate_index]),
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
                        else f"cap_{caps[selected]:g}",
                        "selected_gain_cap": caps[selected],
                        "selected_band_gains": ",".join(
                            f"{value:.6f}"
                            for value in values["gains"][selected, scanner_index]
                        ),
                        "base_log_spectrum_rmse": before,
                        "output_log_spectrum_rmse": after,
                        "log_spectrum_rmse_change": after - before,
                        "rmse_reduction_percent": 100.0 * (before - after) / before,
                        "outer_spectrum_nonworse": after <= before + SPECTRUM_TOLERANCE,
                        "outer_gamut_pass": gamut_gate(
                            selected_metrics["material_range_fraction"],
                            selected_metrics["projection_rgb_mae"],
                            selected_metrics["final_clamp_mae"],
                        ),
                        "strict_nested_training_exclusion": True,
                    }
                )
                selected_caps.append(float(caps[selected]))
                selected_powers.append(selected_power)
                selected_metric_sets.append(selected_metrics)
                base_powers.append(base)
                target_powers.append(target)
                reference_powers.append(
                    reference[fov, outer_fold]["validation_output_power"][
                        rf1_cap_index, scanner_index
                    ]
                )

            pooled_target = sum(target_powers)
            before = log_spectrum_rmse(sum(base_powers), pooled_target, radial_frequency)
            after = log_spectrum_rmse(sum(selected_powers), pooled_target, radial_frequency)
            rf1_after = log_spectrum_rmse(
                sum(reference_powers), pooled_target, radial_frequency
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
                        np.isclose(value, RF1M_IDENTITY_CAP) for value in selected_caps
                    ),
                    "selected_caps": ",".join(f"{value:g}" for value in selected_caps),
                    "base_log_spectrum_rmse": before,
                    "rf1m_output_log_spectrum_rmse": after,
                    "rf1m_log_spectrum_rmse_change": after - before,
                    "noharm_rmse_reduction_percent": 100.0 * (before - after) / before,
                    "original_cap_1p25_output_log_spectrum_rmse": rf1_after,
                    "original_cap_1p25_rmse_reduction_percent": 100.0
                    * (before - rf1_after)
                    / before,
                    "rf1m_minus_rf1_output_rmse": after - rf1_after,
                    "material_range_fraction_mean": float(material.mean()),
                    "material_range_fraction_q99": float(np.quantile(material, 0.99)),
                    "projection_fraction_mean": float(projection.mean()),
                    "projection_rgb_mae_mean": float(projection_mae.mean()),
                    "final_clamp_mae_max": float(final_mae.max()),
                    "aggregate_spectrum_nonworse": after <= before + SPECTRUM_TOLERANCE,
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
        len(selection) == len(FOVS) * (len(SCANNERS) - 1) * RF1_FOLDS * len(caps)
        and len(evaluation) == 75
        and len(cell) == 15
    ):
        raise RuntimeError("incomplete RF1M selection population")

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
        stem="rf1m_strict_nested_spectrum",
        method_label="RF1M multiscale + strict no-harm",
    )

    identity_selections = int(
        np.isclose(evaluation["selected_gain_cap"], RF1M_IDENTITY_CAP).sum()
    )
    gate_pass = bool(
        len(cell_hashes) == 75
        and evaluation["outer_spectrum_nonworse"].all()
        and evaluation["outer_gamut_pass"].all()
        and cell["aggregate_spectrum_nonworse"].all()
        and cell["aggregate_gamut_pass"].all()
        and cell["final_clamp_mae_max"].max() <= 1e-6
    )
    summary = {
        "analysis": "rf1m_strict_nested_selection",
        "rf1m_version": RF1M_VERSION,
        "status": "IMAGE_ONLY_SELECTION_NOT_RESULT_LOCKED",
        "outcome_access": False,
        "pfm_feature_access": False,
        "strict_nested_training_exclusion": True,
        "contract": "docs/e5_rf1m_combined_candidate_contract.md",
        "transform": (
            "Training-fold Reinhard, then three native Laplacian bands at sigma 1/2/4 px whose "
            "gains are sqrt(target/source) band energy accumulated directly on training "
            "patches, capped symmetrically, applied as one shared mean-OD residual projected "
            "into the exact RGB8-supporting OD interval."
        ),
        "selection_protocol": (
            "For outer fold h, each of four inner cells is fitted on the three folds excluding "
            "h and inner fold j. Eligible candidates pass the frozen gamut gate and are "
            "spectrum non-worse than identity in every inner fold. For fold-wise delta = "
            "candidate RMSE - identity RMSE, find the eligible candidate with minimum mean "
            "delta, compute SE = sd(delta, ddof=1)/sqrt(4), then select the smallest cap with "
            "mean delta <= best mean delta + SE. The selection is evaluated once on outer fold "
            "h using the transform fitted on all folds except h."
        ),
        "selection_rule": "k_standard_error_preference_toward_identity",
        "shrinkage_se_multiplier": RF1M_SHRINKAGE_SE,
        "amendment": "Amendment 1 (2026-08-03): shrinkage raised from one to two standard errors",
        "pyramid_sigmas_pixels": list(RF1M_SIGMAS),
        "band_energy_source": "native training-patch Laplacian accumulation",
        "identity_definition": "the uncorrected Reinhard base; the rendered identity candidate is an audit only",
        "identity_render_log_spectrum_deviation_max": identity_deviation,
        "fovs": list(FOVS),
        "source_scanners": list(SCANNERS[1:]),
        "outer_folds": RF1_FOLDS,
        "outer_cells": len(FOVS) * RF1_FOLDS,
        "inner_cells": len(FOVS) * RF1_FOLDS * (RF1_FOLDS - 1),
        "candidate_caps": caps.tolist(),
        "identity_selections": identity_selections,
        "nonidentity_selections": int(len(evaluation) - identity_selections),
        "outer_fold_spectrum_nonworse": int(evaluation["outer_spectrum_nonworse"].sum()),
        "outer_fold_spectrum_total": len(evaluation),
        "outer_fold_gamut_pass": int(evaluation["outer_gamut_pass"].sum()),
        "aggregate_spectrum_nonworse_cells": int(
            cell["aggregate_spectrum_nonworse"].sum()
        ),
        "aggregate_gamut_pass_cells": int(cell["aggregate_gamut_pass"].sum()),
        "mean_rf1m_rmse_reduction_percent": float(
            cell["noharm_rmse_reduction_percent"].mean()
        ),
        "mean_locked_rf1_rmse_reduction_percent": float(
            cell["original_cap_1p25_rmse_reduction_percent"].mean()
        ),
        "rf1m_better_than_locked_rf1_cells": int(
            (cell["rf1m_minus_rf1_output_rmse"] < 0).sum()
        ),
        "final_clamp_mae_max": float(cell["final_clamp_mae_max"].max()),
        "s360_identity_selections": int(
            np.isclose(
                evaluation.loc[evaluation["scanner"] == "s360", "selected_gain_cap"],
                RF1M_IDENTITY_CAP,
            ).sum()
        ),
        "cell_sha256": cell_hashes,
        "artifacts": {
            path.name: sha256(path)
            for path in [selection_path, evaluation_path, cell_path, *figure_paths]
        },
        "pfm_access_gate_pass": gate_pass,
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not gate_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

"""Does the staining condition carry any interpretable information about the scanner effect?

The nested fit already says a scanner's spectral slope varies from section to
section.  That alone does not say the *stain* means anything: PLISM applies its 13
staining conditions to 13 different serial sections, so a section-level term
confounds the chemistry with whatever tissue happens to sit on that section.

This asks the question that can be answered.  If the section-level variance were
staining chemistry, a scanner's slope should track how the stain actually renders
-- how dark it is, how much contrast it carries, how much fine detail survives it.
Those are measurable on the section's own AT2 reference, and so are scanner
independent by construction.  Add one as a fixed covariate and see whether it
explains anything.

A null result is the informative one: it says the section term is unstructured,
and that "staining condition" is a label rather than an explanatory variable for
the scanner signature.

**Leave-one-section-out is part of the test, not an afterthought.**  With thirteen
sections a single one has enormous leverage: the sparse-cohort run returned a
coefficient of -0.132 for S60 that fell to -0.002 when one section was dropped,
and the report drew its conclusion from that collapse.  That check was done by
hand and never entered the code, so nobody could tell which of the other
coefficients would have survived it.  It runs here for every scanner and every
covariate, and a coefficient is called stable only if it keeps its sign and its
significance in every fold.

The profiled REML is generalised here to an arbitrary fixed-effects design rather
than modifying the locked PanNormal estimator, and is checked against it on the
intercept-only model before anything else is read.

Contract: docs/e9_plism_native_ert_contract.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg, optimize, stats

from analyze_exp05_tissue_random_slopes import fit_nested_reml
from analyze_plism_random_slopes import REFERENCE, build_replicates, load_patches
from plism_dataset_corrections import drop_blocks, parse_exclusions

COVARIATES = {
    "stain_od": "reference mean optical density — how dark the staining is",
    "stain_contrast": "reference optical-density spread — how much contrast it carries",
    "stain_detail": "log2 reference high-band power — how much fine detail survives it",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/plism_native_psd")
    parser.add_argument("--output", default="outputs/plism_stain_covariate")
    parser.add_argument("--band", default="high")
    parser.add_argument("--gate-um", type=float, default=1.0)
    parser.add_argument("--exclude", default="",
                        help="stain:scanner blocks to drop, or 'default' for the\n                              known-defective table in plism_dataset_corrections")
    parser.add_argument("--per-core", type=int, default=8)
    parser.add_argument("--leave-one-out", action="store_true", default=True,
                        help="refit dropping each section in turn; the sparse cohort's "
                             "conclusion rested on this check and it was never in the code")
    parser.add_argument("--no-leave-one-out", dest="leave_one_out", action="store_false")
    parser.add_argument("--fold-section", default="",
                        help="compute only this held-out section's refit and exit; the folds "
                             "are independent, so an array over sections does in an hour what "
                             "the serial loop takes half a day over")
    return parser.parse_args()


def make_blocks(frame: pd.DataFrame, design_columns: list[str]):
    blocks = []
    counts = frame.groupby("slide_id")["replicate"].nunique()
    if counts.nunique() != 1:
        raise ValueError("replicate counts are not balanced across cores")
    for section, section_frame in frame.groupby("tissue_type", sort=True):
        section_frame = section_frame.sort_values(["slide_id", "replicate"])
        cores = section_frame["slide_id"].drop_duplicates().tolist()
        labels = section_frame["slide_id"].astype(str).to_numpy()
        core_design = np.column_stack([(labels == core).astype(float) for core in cores])
        blocks.append({
            "section": str(section),
            "y": section_frame["log2_relative_transfer"].to_numpy(dtype=float),
            "X": section_frame[design_columns].to_numpy(dtype=float),
            "core_kernel": core_design @ core_design.T,
            "section_kernel": np.ones((len(section_frame),) * 2, dtype=float),
            "identity": np.eye(len(section_frame), dtype=float),
        })
    return blocks


def evaluate(blocks, variances, include_section=True):
    """Profiled REML for a general fixed-effects design."""
    section_var, core_var, residual_var = variances
    width = blocks[0]["X"].shape[1]
    xvx = np.zeros((width, width))
    xvy = np.zeros(width)
    yvy = 0.0
    log_determinant = 0.0
    total = 0
    for block in blocks:
        covariance = core_var * block["core_kernel"] + residual_var * block["identity"]
        if include_section:
            covariance = covariance + section_var * block["section_kernel"]
        factor = linalg.cho_factor(covariance, lower=True, check_finite=False)
        inverse_x = linalg.cho_solve(factor, block["X"], check_finite=False)
        inverse_y = linalg.cho_solve(factor, block["y"], check_finite=False)
        xvx += block["X"].T @ inverse_x
        xvy += block["X"].T @ inverse_y
        yvy += float(block["y"] @ inverse_y)
        log_determinant += float(2.0 * np.log(np.diag(factor[0])).sum())
        total += len(block["y"])
    beta = np.linalg.solve(xvx, xvy)
    quadratic = yvy - float(xvy @ beta)
    sign, log_det_xvx = np.linalg.slogdet(xvx)
    negative_log_likelihood = 0.5 * (
        log_determinant + log_det_xvx + quadratic + (total - width) * np.log(2.0 * np.pi)
    )
    return {
        "negative_log_likelihood": float(negative_log_likelihood),
        "beta": beta,
        "beta_cov": np.linalg.inv(xvx),
        "total_n": total,
    }


def fit(frame: pd.DataFrame, design_columns: list[str]):
    blocks = make_blocks(frame, design_columns)

    def objective(log_variances):
        return evaluate(blocks, np.exp(log_variances))["negative_log_likelihood"]

    best = None
    for scale in ([1.0, 1.0, 1.0], [0.25, 2.0, 1.0], [2.0, 0.25, 1.0]):
        start = np.log(np.array(scale) * np.array([0.01, 0.01, 0.05]))
        result = optimize.minimize(objective, start, method="L-BFGS-B",
                                   bounds=[(-18.0, 5.0)] * 3,
                                   options={"maxiter": 500, "ftol": 1e-11})
        if best is None or result.fun < best.fun:
            best = result
    variances = np.exp(best.x)
    full = evaluate(blocks, variances)
    return {"variances": variances, **full, "nll": float(best.fun)}


def main() -> None:
    args = parse_args()
    patches = load_patches(Path(args.input))
    patches, removed = drop_blocks(patches, parse_exclusions(args.exclude))
    for entry in removed:
        print(f"excluded {entry['stain']}/{entry['scanner']}: "
              f"{entry['rows']:,} patches — {entry['reason']}")
    frame = build_replicates(patches, args.band, args.gate_um, args.per_core)

    # Section-level covariates, all measured on that section's own AT2 reference,
    # so none of them can carry information about the scanner being tested.
    reference = patches.loc[patches["scanner"] == REFERENCE]
    section_stats = reference.groupby("stain").agg(
        stain_od=("od_mean", "mean"),
        stain_contrast=("od_std", "mean"),
        stain_detail=("high", lambda s: np.log2(s.mean())),
    )
    section_stats = (section_stats - section_stats.mean()) / section_stats.std()
    frame = frame.join(section_stats, on="tissue_type")
    frame["intercept"] = 1.0

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"band {args.band}, gate {args.gate_um} um, "
          f"{frame['tissue_type'].nunique()} sections, {frame['slide_id'].nunique()} cores\n")
    print("section covariates, standardised (measured on AT2 only):")
    print(section_stats.round(3).to_string(), "\n")

    if args.fold_section:
        held_out = args.fold_section
        if held_out not in set(frame["tissue_type"]):
            raise KeyError(f"{held_out} is not one of {sorted(set(frame['tissue_type']))}")
        parts_dir = output_dir / f"loo_{args.band}"
        parts_dir.mkdir(parents=True, exist_ok=True)
        kept = frame.loc[frame["tissue_type"] != held_out]
        part = []
        for scanner, block in kept.groupby("scanner"):
            for name in COVARIATES:
                alternative = fit(block, ["intercept", name])
                beta = float(alternative["beta"][1])
                se = float(np.sqrt(alternative["beta_cov"][1, 1]))
                part.append({"held_out": held_out, "scanner": scanner, "covariate": name,
                             "beta": beta, "se": se,
                             "z": beta / se if se > 0 else np.nan,
                             "p": float(2 * stats.norm.sf(abs(beta / se))) if se > 0 else np.nan})
        pd.DataFrame(part).to_csv(parts_dir / f"{held_out}.csv", index=False)
        print(f"fold without {held_out}: {len(part)} coefficients -> "
              f"{parts_dir / f'{held_out}.csv'}")
        return

    rows = []
    for scanner, block in frame.groupby("scanner"):
        null = fit(block, ["intercept"])
        # self-check against the locked estimator on the intercept-only model
        locked = fit_nested_reml(block[["tissue_type", "slide_id", "replicate",
                                        "log2_relative_transfer"]])
        agreement = abs(float(null["beta"][0]) - locked["full"]["fixed_mean"])

        for name in COVARIATES:
            alternative = fit(block, ["intercept", name])
            beta = float(alternative["beta"][1])
            se = float(np.sqrt(alternative["beta_cov"][1, 1]))
            likelihood_ratio = max(0.0, 2.0 * (null["nll"] - alternative["nll"]))
            rows.append({
                "scanner": scanner,
                "covariate": name,
                "beta": beta,
                "se": se,
                "z": beta / se if se > 0 else np.nan,
                "p": float(2 * stats.norm.sf(abs(beta / se))) if se > 0 else np.nan,
                "var_section_null": null["variances"][0],
                "var_section_with": alternative["variances"][0],
                "section_var_explained": 1 - alternative["variances"][0] / null["variances"][0]
                if null["variances"][0] > 0 else np.nan,
                "lrt": likelihood_ratio,
                "intercept_check": agreement,
            })

    table = pd.DataFrame(rows)

    parts_dir = output_dir / f"loo_{args.band}"
    folds = []
    if args.leave_one_out:
        sections = sorted(frame["tissue_type"].unique())
        parts_dir.mkdir(parents=True, exist_ok=True)
        missing = [s for s in sections if not (parts_dir / f"{s}.csv").exists()]
        if missing:
            print(f"computing leave-one-section-out for {len(missing)} of "
                  f"{len(sections)} sections\n")
        for held_out in missing:
            kept = frame.loc[frame["tissue_type"] != held_out]
            part = []
            for scanner, block in kept.groupby("scanner"):
                for name in COVARIATES:
                    alternative = fit(block, ["intercept", name])
                    beta = float(alternative["beta"][1])
                    se = float(np.sqrt(alternative["beta_cov"][1, 1]))
                    part.append({
                        "held_out": held_out, "scanner": scanner, "covariate": name,
                        "beta": beta, "se": se,
                        "z": beta / se if se > 0 else np.nan,
                        "p": float(2 * stats.norm.sf(abs(beta / se))) if se > 0 else np.nan})
            pd.DataFrame(part).to_csv(parts_dir / f"{held_out}.csv", index=False)
            print(f"  fold without {held_out}: {len(part)} coefficients")
        fold_table = pd.concat(
            [pd.read_csv(parts_dir / f"{s}.csv") for s in sections], ignore_index=True)
        fold_table.to_csv(output_dir / f"stain_covariate_{args.band}_loo.csv", index=False)

        # A coefficient counts as stable only if every fold keeps its sign and
        # its significance.  This is the check the sparse conclusion rested on.
        stability = []
        for (scanner, name), block in fold_table.groupby(["scanner", "covariate"]):
            full = table.loc[(table["scanner"] == scanner)
                             & (table["covariate"] == name)].iloc[0]
            same_sign = bool((np.sign(block["beta"]) == np.sign(full["beta"])).all())
            significant = int((block["p"] < 0.05).sum())
            stability.append({
                "scanner": scanner, "covariate": name,
                "beta_full": full["beta"], "p_full": full["p"],
                "beta_min": float(block["beta"].min()),
                "beta_max": float(block["beta"].max()),
                "folds": len(block), "folds_significant": significant,
                "sign_stable": same_sign,
                "stable": bool(same_sign and significant == len(block)
                               and full["p"] < 0.05)})
        stability = pd.DataFrame(stability)
        stability.to_csv(output_dir / f"stain_covariate_{args.band}_stability.csv", index=False)
        table = table.merge(
            stability[["scanner", "covariate", "folds", "folds_significant",
                       "sign_stable", "stable"]],
            on=["scanner", "covariate"], how="left")

    table.to_csv(output_dir / f"stain_covariate_{args.band}.csv", index=False)

    pd.set_option("display.width", 220)
    print(f"intercept-only agreement with the locked estimator: "
          f"max |difference| = {table['intercept_check'].max():.2e}\n")
    for name, description in COVARIATES.items():
        block = table[table["covariate"] == name]
        print(f"=== {name} — {description} ===")
        print(block[["scanner", "beta", "se", "z", "p", "section_var_explained"]]
              .round(4).to_string(index=False))
        print(f"  |beta| max {block['beta'].abs().max():.4f}, "
              f"smallest p {block['p'].min():.3f}, "
              f"section variance explained: median {block['section_var_explained'].median():.3f}")
        if "stable" in block:
            survivors = block.loc[block["stable"].astype(bool), "scanner"].tolist()
            print(f"  stable under leave-one-section-out: "
                  f"{len(survivors)}/{len(block)} — {', '.join(survivors) or 'none'}")
        print()

    (output_dir / "summary.json").write_text(json.dumps({
        "band": args.band,
        "gate_um": args.gate_um,
        "covariates": COVARIATES,
        "sections": int(frame["tissue_type"].nunique()),
        "cores": int(frame["slide_id"].nunique()),
        "intercept_check_max": float(table["intercept_check"].max()),
        "leave_one_out": bool(args.leave_one_out),
        "stable_coefficients": (int(table["stable"].sum()) if "stable" in table else None),
        "coefficients": int(len(table)),
    }, indent=2))
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()

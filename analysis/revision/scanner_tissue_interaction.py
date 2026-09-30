#!/usr/bin/env python3
"""RV11: scanner x tissue interaction as a direct fixed effect.

For every slide x target-scanner contrast d(s, r) (balanced: each slide has all five
target scanners) the model is

    d(s, r) = alpha_r + beta_s + gamma_(s, t(r)) + eps

alpha_r absorbs the tissue main effect and slide differences, beta_s is the mean scanner
effect, gamma is the scanner x tissue term and eps the scanner x slide variation within a
tissue. With a balanced design the additive residual E = d - row mean - column mean + grand
mean splits exactly into SS_gamma (tissue cell means of E) and SS_eps. The interaction is
tested with F = [SS_gamma / (S-1)(T-1)] / [SS_eps / (S-1)(R-T)]; the primary p value comes
from permuting tissue labels across slides.

Inputs: the RQ1 contrasts (13 image measures) and, as a secondary analysis, the RV09 slide x
scanner correction gains. Outputs: tests, effect sizes, tissue-specific scanner effects
m(t, s) and a comparison with the mixed model, under ``results/scanner_tissue_interaction/``.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT = Path(__file__).resolve().parents[2]
CONTRASTS = PROJECT / "analysis/paper/results/direct_slide_lmm/direct_slide_contrasts.csv"
LMM = PROJECT / "analysis/paper/results/direct_slide_lmm/endpoint_summary.csv"
GAINS = Path(__file__).resolve().parent / "results/gain_tissue_dependence/inputs"
OUTPUT = Path(__file__).resolve().parent / "results/scanner_tissue_interaction"
SCANNERS = ("versa", "akoya", "gt450", "s360", "s60")
PERMUTATIONS = 9999
SEED = 20260929


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out


def interaction_ss(residual: np.ndarray, codes: np.ndarray, n_tissues: int) -> float:
    """SS of tissue cell means of the additive residual (rows: slides, columns: scanners)."""
    counts = np.bincount(codes, minlength=n_tissues).astype(float)
    sums = np.zeros((n_tissues, residual.shape[1]))
    np.add.at(sums, codes, residual)
    present = counts > 0
    return float(((sums[present] ** 2).sum(axis=1) / counts[present]).sum())


def analyse(frame: pd.DataFrame, rng: np.random.Generator) -> tuple[dict, pd.DataFrame]:
    table = frame.pivot(index="slide_id", columns="scanner", values="value")[list(SCANNERS)]
    if table.isna().any().any():
        raise ValueError("unbalanced contrasts")
    tissue = frame.drop_duplicates("slide_id").set_index("slide_id")["tissue_type"].reindex(table.index)
    values = table.to_numpy(dtype=float)
    residual = values - values.mean(axis=1, keepdims=True) - values.mean(axis=0, keepdims=True) + values.mean()
    tissues = sorted(tissue.unique())
    codes = tissue.map({name: index for index, name in enumerate(tissues)}).to_numpy()
    n_slides, n_scanners, n_tissues = values.shape[0], values.shape[1], len(tissues)
    ss_total = float((residual ** 2).sum())
    ss_gamma = interaction_ss(residual, codes, n_tissues)
    ss_eps = ss_total - ss_gamma
    df_gamma = (n_scanners - 1) * (n_tissues - 1)
    df_eps = (n_scanners - 1) * (n_slides - n_tissues)
    f_value = (ss_gamma / df_gamma) / (ss_eps / df_eps)
    permuted = np.empty(PERMUTATIONS)
    for index in range(PERMUTATIONS):
        shuffled = rng.permutation(codes)
        g = interaction_ss(residual, shuffled, n_tissues)
        permuted[index] = (g / df_gamma) / ((ss_total - g) / df_eps)
    effects = table.groupby(tissue.to_numpy()).mean()
    effects.index.name = "tissue_type"
    result = {
        "n_slides": n_slides, "n_tissues": n_tissues, "df_interaction": df_gamma, "df_residual": df_eps,
        "ss_interaction": ss_gamma, "ss_residual": ss_eps, "F": f_value,
        "p_F": float(stats.f.sf(f_value, df_gamma, df_eps)),
        "p_permutation": float((1 + (permuted >= f_value).sum()) / (PERMUTATIONS + 1)),
        "tissue_share": ss_gamma / (ss_gamma + ss_eps),
        # Expected tissue share under no interaction, and the chance-corrected partial omega^2.
        "tissue_share_chance": df_gamma / (df_gamma + df_eps),
        "omega_sq": max(0.0, df_gamma * (f_value - 1) / (df_gamma * (f_value - 1) + values.size)),
    }
    for scanner in SCANNERS:
        column = effects[scanner]
        result[f"mean_{scanner}"] = float(values[:, SCANNERS.index(scanner)].mean())
        result[f"between_tissue_sd_{scanner}"] = float(column.std(ddof=1))
        result[f"min_{scanner}"] = float(column.min())
        result[f"min_tissue_{scanner}"] = str(column.idxmin())
        result[f"max_{scanner}"] = float(column.max())
        result[f"max_tissue_{scanner}"] = str(column.idxmax())
    return result, effects


def run(frames: dict[str, pd.DataFrame], label: str, rng: np.random.Generator,
        drop_singleton: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, effect_rows = [], []
    for endpoint, frame in frames.items():
        if drop_singleton:
            counts = frame.drop_duplicates("slide_id").tissue_type.value_counts()
            frame = frame[frame.tissue_type.isin(counts[counts >= 2].index)]
        result, effects = analyse(frame, rng)
        rows.append({"analysis": label, "endpoint": endpoint,
                     "endpoint_label": frame.endpoint_label.iloc[0], "family": frame.family.iloc[0], **result})
        long = effects.reset_index().melt(id_vars="tissue_type", var_name="scanner", value_name="effect")
        long.insert(0, "endpoint", endpoint)
        long.insert(0, "analysis", label)
        effect_rows.append(long)
    table = pd.DataFrame(rows)
    table["q_permutation"] = bh(table.p_permutation.to_numpy())
    table["q_F"] = bh(table.p_F.to_numpy())
    return table, pd.concat(effect_rows, ignore_index=True)


def ordering(effects: pd.DataFrame, endpoint: str) -> dict:
    wide = effects[effects.endpoint == endpoint].pivot(index="tissue_type", columns="scanner", values="effect")
    return {"tissues": int(len(wide)),
            "akoya_lowest": int((wide.idxmin(axis=1) == "akoya").sum()),
            "gt450_highest": int((wide.idxmax(axis=1) == "gt450").sum())}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    contrasts = pd.read_csv(CONTRASTS, dtype={"slide_id": str})
    measures = {endpoint: frame for endpoint, frame in contrasts.groupby("endpoint", sort=False)}
    primary, effects = run(measures, "image_measures", rng, drop_singleton=False)
    sensitivity, _ = run(measures, "image_measures_without_singleton_tissue", rng, drop_singleton=True)
    gain_frames = {}
    for path in sorted(GAINS.glob("*.csv")):
        frame = pd.read_csv(path, dtype={"slide_id": str})
        gain_frames[path.stem] = frame
    gain_tables = []
    for pfm in sorted({stem.split("__")[0] for stem in gain_frames}):
        subset = {stem: frame for stem, frame in gain_frames.items() if stem.startswith(pfm + "__")}
        table, _ = run(subset, f"correction_gains_{pfm}", rng, drop_singleton=False)
        gain_tables.append(table)
    gains = pd.concat(gain_tables, ignore_index=True)

    lmm = pd.read_csv(LMM)[["endpoint", "fraction_scanner_by_tissue", "scanner_specific_tissue_dependence_bh_q",
                            "any_tissue_dependence_bh_q"]]
    comparison = primary[["endpoint", "endpoint_label", "F", "p_permutation", "q_permutation", "tissue_share",
                          "omega_sq"]].merge(
        lmm, on="endpoint", how="left")

    primary.to_csv(OUTPUT / "interaction_tests.csv", index=False)
    sensitivity.to_csv(OUTPUT / "interaction_tests_without_singleton.csv", index=False)
    gains.to_csv(OUTPUT / "gain_interaction_tests.csv", index=False)
    effects.to_csv(OUTPUT / "tissue_scanner_effects.csv", index=False)
    comparison.to_csv(OUTPUT / "comparison_with_mixed_model.csv", index=False)
    order = ordering(effects, "frequency_high")
    (OUTPUT / "summary.json").write_text(json.dumps({
        "permutations": PERMUTATIONS, "seed": SEED, "frequency_high_ordering": order,
        "significant_measures_q05": int((primary.q_permutation < 0.05).sum()),
        "significant_measures_q05_without_singleton": int((sensitivity.q_permutation < 0.05).sum()),
    }, indent=2) + "\n")
    (OUTPUT / "summary.md").write_text(render(primary, sensitivity, gains, comparison, order))
    print((OUTPUT / "summary.md").read_text())


def render(primary, sensitivity, gains, comparison, order) -> str:
    lines = ["# RV11 scanner × tissue interaction as a direct fixed effect", "",
             "Model per measure: d(s, r) = α_r + β_s + γ_(s, t(r)) + ε (balanced; 103 slides × 5 scanners). "
             f"Primary p: {PERMUTATIONS} permutations of tissue labels across slides; BH across measures.", "",
             "## Image measures", "",
             "| Measure | F (144, 264) | p perm | q perm | p F | Tissue share (chance 0.353) | Partial ω² | Mixed-model scanner×tissue fraction | Mixed-model q |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in comparison.merge(primary[["endpoint", "p_F"]], on="endpoint").itertuples():
        lines.append(f"| {row.endpoint_label} | {row.F:.2f} | {row.p_permutation:.4f} | {row.q_permutation:.4f} | "
                     f"{row.p_F:.2e} | {row.tissue_share:.3f} | {row.omega_sq:.3f} | {row.fraction_scanner_by_tissue:.3f} | "
                     f"{row.scanner_specific_tissue_dependence_bh_q:.4f} |")
    lines += ["", f"- q < 0.05: {(primary.q_permutation < 0.05).sum()}/13; without the single-slide tissue: "
              f"{(sensitivity.q_permutation < 0.05).sum()}/13.",
              f"- High-frequency transfer: AKOYA lowest in {order['akoya_lowest']}/{order['tissues']} tissues, "
              f"GT450 highest in {order['gt450_highest']}/{order['tissues']}.", "",
              "## Tissue-specific scanner effects (measure units)", "",
              "| Measure | Scanner | Mean | Between-tissue SD | Min (tissue) | Max (tissue) |", "| --- | --- | --- | --- | --- | --- |"]
    for row in primary.itertuples():
        for scanner in SCANNERS:
            lines.append(f"| {row.endpoint_label} | {scanner} | {getattr(row, 'mean_' + scanner):+.3f} | "
                         f"{getattr(row, 'between_tissue_sd_' + scanner):.3f} | "
                         f"{getattr(row, 'min_' + scanner):+.3f} ({getattr(row, 'min_tissue_' + scanner)}) | "
                         f"{getattr(row, 'max_' + scanner):+.3f} ({getattr(row, 'max_tissue_' + scanner)}) |")
    lines += ["", "## Correction gains (secondary)", "",
              "| PFM | Gain | F | p perm | q perm | Partial ω² |", "| --- | --- | --- | --- | --- | --- |"]
    for row in gains.itertuples():
        lines.append(f"| {row.analysis.replace('correction_gains_', '')} | {row.endpoint_label} | {row.F:.2f} | "
                     f"{row.p_permutation:.4f} | {row.q_permutation:.4f} | {row.omega_sq:.3f} |")
    lines += ["", "## Files", "",
              "- `interaction_tests.csv`, `interaction_tests_without_singleton.csv`, `gain_interaction_tests.csv`",
              "- `tissue_scanner_effects.csv` (m(t, s), input to RV12), `comparison_with_mixed_model.csv`, `summary.json`", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Write the revised manuscript tables from the revision results.

Main tables
* Table 1 ``table_scanner_tissue_interaction.tex``: scanner-specific mean differences for five
  representative image measures and the scanner x tissue variance fraction, both from the
  linear mixed-effects model (``analysis/paper/results/direct_slide_lmm``).
* Table 2 ``table_image_correction_three_axis.tex``: image fidelity, UNI alignment, target
  detectability (best of three probes, within-fold; RV04b) and tissue retrieval (RV04) for
  every image-level correction in PanNormal and PLISM.
* Table 3 ``table_cross_pfm_detectability.tex``: target-distance change and detectability for
  image- and feature-level corrections in four PFMs (RV04, RV04b).

Supplementary tables
* ``table_scanner_tissue_variance_full.tex`` and ``table_scanner_tissue_variance_components.tex``:
  mixed-model scanner means and variance fractions for all 13 measures.
* ``table_pfm_robustness_supp.tex``: scanner sensitivity, robustness and tissue information for
  all model variants (RV13, RV14).
* ``table_detectability_probes_supp.tex``: PanNormal detectability under each of the three
  within-fold probes (RV04b).
* ``table_cross_pfm_distance_supp.tex``: target distance and its change from raw for every method
  in four PFMs and both datasets (RV04).

The fixed-effects permutation test of RV11 is no longer reported in the manuscript.

All values are read from the result files; nothing is typed by hand.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"
TABLES = PROJECT / "00_manuscript/tables"
SCANNERS = ("versa", "akoya", "gt450", "s360", "s60")
SCANNER_LABELS = {"versa": "VERSA", "akoya": "AKOYA", "gt450": "GT450", "s360": "S360", "s60": "S60"}
PFMS = ("uni_v1", "uni2", "virchow2", "hoptimus1")
PFM_LABELS = {"uni_v1": "UNI", "uni2": "UNI2-h", "virchow2": "Virchow2", "hoptimus1": "H-optimus-1"}
LMM = PROJECT / "analysis/paper/results/direct_slide_lmm/endpoint_summary.csv"
# Measure groups of the mixed-model supplementary tables (label, endpoints).
LMM_GROUPS = (("Color and optical density", ("delta_lab_l", "delta_lab_a", "delta_lab_b", "log2_mean_od_ratio")),
              ("Contrast and edges", ("log2_lab_l_sd_ratio", "log2_od_sd_ratio", "log2_gradient_rms_ratio")),
              ("Spatial frequency", ("frequency_low_mid", "frequency_mid", "frequency_high")),
              ("Supplementary measures", ("gradient_dissimilarity", "delta_tissue_fraction", "log2_laplacian_ratio")))
IMAGE_METHODS = ("reinhard", "macenko", "vahadane", "frequency", "combined", "pix2pix", "cyclegan")
FAMILY_SHADE = r"\rowcolor{black!7} "
METHOD_LABELS = {"raw": "Raw", "reinhard": "Reinhard", "macenko": "Macenko", "vahadane": "Vahadane",
                 "pix2pix": "Pix2Pix", "cyclegan": "CycleGAN", "combined": "Color + frequency",
                 "ridge": "Ridge affine", "combat": "ComBat", "ols": "Affine OLS", "frequency": "Frequency"}
ENDPOINT_LABELS = {
    "delta_lab_l": r"$\Delta L^*$", "delta_lab_a": r"$\Delta a^*$", "delta_lab_b": r"$\Delta b^*$",
    "log2_mean_od_ratio": r"Mean OD ratio ($\log_2$)", "log2_lab_l_sd_ratio": r"$L^*$ contrast ratio ($\log_2$)",
    "log2_od_sd_ratio": r"OD contrast ratio ($\log_2$)", "log2_gradient_rms_ratio": r"Gradient energy ratio ($\log_2$)",
    "gradient_dissimilarity": r"$1-$gradient NCC to AT2", "delta_tissue_fraction": r"$\Delta$ tissue fraction",
    "log2_laplacian_ratio": r"Laplacian variance ratio ($\log_2$)",
    "frequency_low_mid": r"Low--mid frequency transfer ($\log_2$)", "frequency_mid": r"Mid-frequency transfer ($\log_2$)",
    "frequency_high": r"High-frequency transfer ($\log_2$)",
}

CLASS_LABELS = {"raw": "Baseline", "reinhard": "Stain normalization", "pix2pix": "Learned translation",
                "combined": "Phenotype-targeted"}


def signed(value: float, digits: int) -> str:
    if round(value, digits) == 0:
        return f"{0:.{digits}f}"
    return f"${value:+.{digits}f}$"


def cells() -> pd.DataFrame:
    table = pd.read_csv(RESULTS / "three_axis_evaluation/cell_statistics.csv")
    return table[table.scanner == "pooled"]


def value(frame: pd.DataFrame, dataset: str, pfm: str, method: str, statistic: str) -> float:
    match = frame[(frame.dataset == dataset) & (frame.pfm == pfm) & (frame.method == method)
                  & (frame.statistic == statistic)]
    return float(match.estimate.iloc[0]) if len(match) else np.nan


def detectability() -> pd.DataFrame:
    """Best of the three within-fold probes (point estimates), RV04b."""
    table = pd.read_csv(RESULTS / "detectability_within_fold/detectability.csv")
    table = table[table.statistic.isin(["linear", "mlp", "knn"])]
    return table.groupby(["dataset", "pfm", "method"]).estimate.max().rename("detect").reset_index()


def detect_value(frame: pd.DataFrame, dataset: str, pfm: str, method: str) -> float:
    match = frame[(frame.dataset == dataset) & (frame.pfm == pfm) & (frame.method == method)]
    return float(match.detect.iloc[0]) if len(match) else np.nan


def math_minus(value: float, digits: int) -> str:
    """Unsigned positives, math-mode minus for negatives."""
    return f"$-{abs(value):.{digits}f}$" if round(value, digits) < 0 else f"{value:.{digits}f}"


def fmt(value: float, digits: int) -> str:
    return "--" if not np.isfinite(value) else f"{value:.{digits}f}"


def table_interaction() -> None:
    """Table 1 and the two mixed-model supplementary tables, all from the linear mixed model."""
    lmm = pd.read_csv(LMM).set_index("endpoint")
    if not ((lmm.any_tissue_dependence_bh_q <= 0.0021) & (lmm.scanner_specific_tissue_dependence_bh_q <= 0.0021)).all():
        raise ValueError("caption states BH q <= 0.002 for every measure; check the LMM results")
    rows = (("delta_lab_a", "Color", "$\\Delta a^*$"), ("log2_lab_l_sd_ratio", "Contrast", "$L^*$ contrast ratio ($\\log_2$)"),
            ("log2_od_sd_ratio", "Contrast", "OD contrast ratio ($\\log_2$)"),
            ("frequency_low_mid", "Frequency", "Low--mid frequency transfer ($\\log_2$)"),
            ("frequency_high", "Frequency", "High-frequency transfer ($\\log_2$)"))
    lines = [r"\begin{table}[pos=H]",
             r"    \caption{Scanner effects on PanNormal image measures and their dependence on tissue type, estimated with a linear mixed-effects model. Mean differences are the scanner-specific means relative to the reference scanner (AT2). Scanner $\times$ tissue is the share of the remaining variance explained by tissue-specific scanner effects. Tissue dependence was significant for every measure (parametric-bootstrap likelihood-ratio test, Benjamini--Hochberg $q\leq0.002$). All 13 measures are given in Supplementary Tables~\ref{table_scanner_tissue_variance_full} and \ref{table_scanner_tissue_variance_components}.}",
             r"    \label{table_scanner_tissue_interaction}", r"    \centering", r"    \small",
             r"    \begin{tabularx}{\textwidth}{@{}lYrrrrrr@{}}", r"        \toprule",
             r"        & & \multicolumn{5}{c}{Mean difference from AT2} & Scanner $\times$ \\",
             r"        \cmidrule(lr){3-7}",
             "        Category & Image measure & " + " & ".join(SCANNER_LABELS[s] for s in SCANNERS) + r" & tissue (\%) \\",
             r"        \midrule"]
    previous = None
    for endpoint, category, label in rows:
        row = lmm.loc[endpoint]
        means = " & ".join(signed(row[f"fixed_{s}"], 2) for s in SCANNERS)
        shown, previous = ("" if category == previous else category), category
        lines.append(f"        {shown} & {label} & {means} & {100 * row.fraction_scanner_by_tissue:.1f} \\\\")
    lines += [r"        \bottomrule", r"    \end{tabularx}", r"\end{table}", ""]
    (TABLES / "table_scanner_tissue_interaction.tex").write_text("\n".join(lines))

    header = " & ".join(SCANNER_LABELS[s] for s in SCANNERS)
    lines = [r"\begin{table}[pos=H]",
             r"    \caption{Scanner-specific mean differences from the reference scanner (AT2) for all 13 PanNormal image measures, estimated with the linear mixed-effects model (Supplementary Methods). Tissue dependence and scanner $\times$ tissue interactions were significant for every measure (Benjamini--Hochberg $q\leq0.002$).}",
             r"    \label{table_scanner_tissue_variance_full}", r"    \centering", r"    \small",
             r"    \begin{tabularx}{\textwidth}{@{}Yrrrrr@{}}", r"        \toprule",
             f"        Image measure & {header} \\\\", r"        \midrule"]
    for index, (group, endpoints) in enumerate(LMM_GROUPS):
        if index:
            lines.append(r"        \addlinespace")
        lines.append(rf"        \rowcolor{{black!5}}\multicolumn{{6}}{{@{{}}l}}{{\textit{{{group}}}}} \\")
        for endpoint in endpoints:
            row = lmm.loc[endpoint]
            means = " & ".join(signed(row[f"fixed_{s}"], 3) for s in SCANNERS)
            lines.append(f"        {ENDPOINT_LABELS[endpoint]} & {means} \\\\")
    lines += [r"        \bottomrule", r"    \end{tabularx}", r"\end{table}", ""]
    (TABLES / "table_scanner_tissue_variance_full.tex").write_text("\n".join(lines))

    lines = [r"\begin{table}[pos=H]",
             r"    \caption{Variance components (\%) of the linear mixed-effects model after removing scanner-specific mean differences, for all 13 PanNormal image measures. Shared tissue and shared slide are effects common to the five target scanners; scanner $\times$ tissue is the tissue-specific scanner effect.}",
             r"    \label{table_scanner_tissue_variance_components}", r"    \centering", r"    \small",
             r"    \begin{tabularx}{\textwidth}{@{}Yrrrr@{}}", r"        \toprule",
             r"        Image measure & Shared tissue & Scanner $\times$ tissue & Shared slide & Residual \\", r"        \midrule"]
    for index, (group, endpoints) in enumerate(LMM_GROUPS):
        if index:
            lines.append(r"        \addlinespace")
        lines.append(rf"        \rowcolor{{black!5}}\multicolumn{{5}}{{@{{}}l}}{{\textit{{{group}}}}} \\")
        for endpoint in endpoints:
            row = lmm.loc[endpoint]
            values = (row.fraction_tissue_shared, row.fraction_scanner_by_tissue, row.fraction_slide_shared,
                      row.fraction_residual)
            lines.append(f"        {ENDPOINT_LABELS[endpoint]} & " + " & ".join(f"{100 * v:.1f}" for v in values) + r" \\")
    lines += [r"        \bottomrule", r"    \end{tabularx}", r"\end{table}", ""]
    (TABLES / "table_scanner_tissue_variance_components.tex").write_text("\n".join(lines))


def table_image_correction(frame: pd.DataFrame, detect: pd.DataFrame) -> None:
    methods = ("raw", "reinhard", "macenko", "vahadane", "pix2pix", "cyclegan", "combined")
    lines = [r"\begin{table}[pos=H]",
             r"    \caption{Image-level correction judged on three axes against real paired target images in PanNormal and PLISM (UNI). Image fidelity: SSIM, LPIPS--VGG16, and image residual over 10 image properties (lower is closer to the target; PanNormal only). Representation: UNI cosine distance to the target and its change from raw (positive = closer). Retained information: target-scanner detectability, the accuracy of the best of three probes (linear, MLP, $k$-NN) separating corrected images from real targets (0.5 = not detectable), and same-tissue retrieval (macro recall). In PLISM, the same cores recur across sections, so retrieval is near its ceiling. PLISM applies the corrections fitted in PanNormal without refitting.}",
             r"    \label{table_image_correction_three_axis}", r"    \centering", r"    \scriptsize",
             r"    \setlength{\tabcolsep}{3pt}", r"    \begin{tabularx}{\textwidth}{@{}lYrrrrrrr@{}}", r"        \toprule",
             r"        & & \multicolumn{3}{c}{Image fidelity} & \multicolumn{2}{c}{Representation} & \multicolumn{2}{c}{Retained information} \\",
             r"        \cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
             r"        Model class & Method & SSIM $\uparrow$ & LPIPS $\downarrow$ & Residual $\downarrow$ & Distance $\downarrow$ & $\Delta$ Raw (\%) $\uparrow$ & Scanner det. $\downarrow$ & Tissue (\%) $\uparrow$ \\",
             r"        \midrule"]
    for dataset, title in (("pannormal", "PanNormal"), ("plism", "PLISM")):
        lines.append(rf"        \rowcolor{{black!5}}\multicolumn{{9}}{{@{{}}l}}{{\textbf{{{title}}}}} \\")
        raw = value(frame, dataset, "uni_v1", "raw", "target_distance")
        for method in methods:
            distance = value(frame, dataset, "uni_v1", method, "target_distance")
            change = "Ref." if method == "raw" else signed(100 * (raw - distance) / raw, 1)
            shade = r"\rowcolor{black!7} " if method in ("reinhard", "macenko", "vahadane", "combined") else ""
            label = CLASS_LABELS.get(method, "")
            lines.append(f"        {shade}{label} & {METHOD_LABELS[method]} & "
                         f"{fmt(value(frame, dataset, 'uni_v1', method, 'image_ssim'), 3)} & "
                         f"{fmt(value(frame, dataset, 'uni_v1', method, 'image_lpips'), 3)} & "
                         f"{fmt(value(frame, dataset, 'uni_v1', method, 'image_image_residual'), 2)} & "
                         f"{distance:.4f} & {change} & "
                         f"{fmt(detect_value(detect, dataset, 'uni_v1', method), 2)} & "
                         f"{fmt(100 * value(frame, dataset, 'uni_v1', method, 'retrieval_macro_recall'), 1)} \\\\")
        if dataset == "pannormal":
            lines.append(r"        \midrule")
    lines += [r"        \bottomrule", r"    \end{tabularx}", r"\end{table}", ""]
    (TABLES / "table_image_correction_three_axis.tex").write_text("\n".join(lines))


def table_cross_pfm(frame: pd.DataFrame, detect: pd.DataFrame) -> None:
    methods = (("raw", "Baseline"), ("reinhard", "Image"), ("combined", ""), ("pix2pix", ""), ("cyclegan", ""),
               ("ridge", "Feature"), ("combat", ""))
    header = " & ".join(rf"\multicolumn{{2}}{{c}}{{{PFM_LABELS[p]}}}" for p in PFMS)
    rules = "".join(rf"\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(PFMS)))
    lines = [r"\begin{table}[pos=H]",
             r"    \caption{Image- and feature-level correction in four PFMs, in PanNormal and PLISM. $\Delta$: change in cosine distance to the real paired target from raw (\%; positive = closer). Det.: accuracy of the best of three probes (linear, MLP, $k$-NN) separating corrected embeddings from real target embeddings, trained and tested within the slides held out from each correction fit (0.5 = not detectable). PLISM applies the corrections fitted in PanNormal without refitting.}",
             r"    \label{table_cross_pfm_detectability}", r"    \centering", r"    \scriptsize",
             r"    \setlength{\tabcolsep}{3pt}", r"    \begin{tabularx}{\textwidth}{@{}lYrrrrrrrr@{}}", r"        \toprule",
             f"        & & {header} \\\\", f"        {rules}",
             "        Correction & Method & " + " & ".join([r"$\Delta$ (\%) $\uparrow$ & Det. $\downarrow$"] * len(PFMS)) + r" \\",
             r"        \midrule"]
    for dataset, title in (("pannormal", "PanNormal"), ("plism", "PLISM")):
        lines.append(rf"        \rowcolor{{black!5}}\multicolumn{{10}}{{@{{}}l}}{{\textbf{{{title}}}}} \\")
        for method, level in methods:
            cells_ = []
            for pfm in PFMS:
                raw = value(frame, dataset, pfm, "raw", "target_distance")
                distance = value(frame, dataset, pfm, method, "target_distance")
                change = "Ref." if method == "raw" else (signed(100 * (raw - distance) / raw, 1) if np.isfinite(distance) else "--")
                cells_ += [change, fmt(detect_value(detect, dataset, pfm, method), 2)]
            shade = FAMILY_SHADE if method in IMAGE_METHODS else ""
            lines.append(f"        {shade}{level} & {METHOD_LABELS[method]} & " + " & ".join(cells_) + r" \\")
        if dataset == "pannormal":
            lines.append(r"        \midrule")
    lines += [r"        \bottomrule", r"    \end{tabularx}", r"\end{table}", ""]
    (TABLES / "table_cross_pfm_detectability.tex").write_text("\n".join(lines))


def table_pfm_robustness() -> None:
    table = pd.read_csv(RESULTS / "sensitivity_robustness/model_table.csv").set_index("model")
    stats = pd.read_csv(RESULTS / "anchor_frequency_diversity/model_statistics.csv")
    wide = stats.pivot_table(index="model", columns="statistic", values="estimate")
    order = ["virchow2", "hoptimus1", "conch_pre", "seal_conch_pre", "seal_uni2", "uni_v1", "uni2", "exaonepath",
             "dinov2", "exaonepath_raw", "plip"]
    groups = {"uni_v1": "Image-only, WSI", "uni2": "Image-only, WSI", "virchow2": "Image-only, WSI",
              "hoptimus1": "Image-only, WSI", "exaonepath": "Image-only, WSI + stain norm.",
              "exaonepath_raw": "Same, without Macenko", "seal_uni2": "ST anchor", "seal_conch_pre": "ST anchor",
              "conch_pre": "Text anchor (figures)", "plip": "Text anchor (web)", "dinov2": "Image-only, natural images"}
    lines = [r"\begin{table}[pos=H]",
             r"    \caption{Frequency sensitivity, scanner robustness, and tissue information for all model variants (PanNormal, raw images). Sensitivity: representation shift under an equal high-band change, divided by the between-tissue distance. Scanner robustness: normalized scanner distance, best-probe detectability of the target scanner (Det.) and PathoROB robustness index (RI; chance 0.089 in this dataset). Tissue: same-tissue retrieval (macro recall). Correction share: fraction of the raw target distance removed by Reinhard (color) and, additionally, by frequency matching. Models are ordered by RI.}",
             r"    \label{table_pfm_robustness}", r"    \centering", r"    \scriptsize",
             r"    \setlength{\tabcolsep}{3pt}", r"    \begin{tabularx}{\textwidth}{@{}lYrrrrrrr@{}}", r"        \toprule",
             r"        & & Sensitivity & \multicolumn{3}{c}{Scanner robustness} & Tissue & \multicolumn{2}{c}{Correction share} \\",
             r"        \cmidrule(lr){3-3}\cmidrule(lr){4-6}\cmidrule(lr){7-7}\cmidrule(lr){8-9}",
             r"        Model & Training signal and data & High band $\downarrow$ & Distance $\downarrow$ & Det. $\downarrow$ & RI $\uparrow$ & Retrieval (\%) $\uparrow$ & Color & Frequency \\",
             r"        \midrule"]
    for name in order:
        row, extra = table.loc[name], wide.loc[name]
        label = row.label.replace("\\n", " ").replace("\n", " ")
        lines.append(f"        {label} & {groups[name]} & {row['normalized_shift_high_d0.25']:.4f} & "
                     f"{row.normalized_distance:.3f} & {extra.detectability_best:.3f} & {row.robustness_index:.2f} & "
                     f"{100 * row.tissue_retrieval:.1f} & {extra.colour_share:.2f} & {math_minus(extra.frequency_share, 3)} \\\\")
    lines += [r"        \bottomrule", r"    \end{tabularx}", r"\end{table}", ""]
    (TABLES / "table_pfm_robustness_supp.tex").write_text("\n".join(lines))


def table_detectability_probes() -> None:
    table = pd.read_csv(RESULTS / "detectability_within_fold/detectability.csv")
    table = table[(table.dataset == "pannormal") & table.statistic.isin(["linear", "mlp", "knn"])]
    methods = (("raw", "Baseline"), ("reinhard", "Image"), ("macenko", ""), ("vahadane", ""), ("combined", ""),
               ("pix2pix", ""), ("cyclegan", ""), ("ridge", "Feature"), ("combat", ""), ("ols", ""))
    header = " & ".join(rf"\multicolumn{{3}}{{c}}{{{PFM_LABELS[p]}}}" for p in PFMS)
    rules = "".join(rf"\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(len(PFMS)))
    lines = [r"\begin{table}[pos=H]",
             r"    \caption{Target-scanner detectability by probe in PanNormal. Balanced accuracy of linear, MLP, and $k$-NN probes separating corrected embeddings from real target embeddings, trained and tested within the slides held out from each correction fit (0.5 = not detectable). Shaded rows are image-level corrections. In PLISM, every method stayed at 0.99 or higher under the linear probe.}",
             r"    \label{table_detectability_probes}", r"    \centering", r"    \scriptsize",
             r"    \setlength{\tabcolsep}{3pt}", r"    \begin{tabularx}{\textwidth}{@{}lYrrrrrrrrrrrr@{}}", r"        \toprule",
             f"        & & {header} \\\\", f"        {rules}",
             "        Correction & Method & " + " & ".join(["Linear & MLP & $k$-NN"] * len(PFMS)) + r" \\",
             r"        \midrule"]
    for method, level in methods:
        cells_ = []
        for pfm in PFMS:
            for probe in ("linear", "mlp", "knn"):
                match = table[(table.pfm == pfm) & (table.method == method) & (table.statistic == probe)]
                cells_.append(fmt(float(match.estimate.iloc[0]), 2))
        shade = FAMILY_SHADE if method in IMAGE_METHODS else ""
        lines.append(f"        {shade}{level} & {METHOD_LABELS[method]} & " + " & ".join(cells_) + r" \\")
    lines += [r"        \bottomrule", r"    \end{tabularx}", r"\end{table}", ""]
    (TABLES / "table_detectability_probes_supp.tex").write_text("\n".join(lines))


def table_cross_pfm_distance(frame: pd.DataFrame) -> None:
    methods = (("raw", "Baseline"), ("reinhard", "Image"), ("macenko", ""), ("vahadane", ""), ("frequency", ""),
               ("combined", ""), ("pix2pix", ""), ("cyclegan", ""), ("ridge", "Feature"), ("combat", ""), ("ols", ""))
    header = " & ".join(rf"\multicolumn{{2}}{{c}}{{{PFM_LABELS[p]}}}" for p in PFMS)
    rules = "".join(rf"\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(PFMS)))
    lines = [r"\begin{table}[pos=H]",
             r"    \caption{PFM cosine distance to the real paired target after each correction, and its change from raw ($\Delta$, \%; positive = closer). PanNormal: 103 slides, five directions; PLISM: 13 sections, three directions, with corrections fitted in PanNormal and applied without refitting. Shaded rows are image-level corrections. Frequency: radial frequency matching alone.}",
             r"    \label{table_cross_pfm_distance}", r"    \centering", r"    \scriptsize",
             r"    \setlength{\tabcolsep}{3pt}", r"    \begin{tabularx}{\textwidth}{@{}lYrrrrrrrr@{}}", r"        \toprule",
             f"        & & {header} \\\\", f"        {rules}",
             "        Correction & Method & " + " & ".join([r"Distance $\downarrow$ & $\Delta$ (\%) $\uparrow$"] * len(PFMS)) + r" \\",
             r"        \midrule"]
    for dataset, title in (("pannormal", "PanNormal"), ("plism", "PLISM")):
        lines.append(rf"        \rowcolor{{black!5}}\multicolumn{{10}}{{@{{}}l}}{{\textbf{{{title}}}}} \\")
        for method, level in methods:
            cells_ = []
            for pfm in PFMS:
                raw = value(frame, dataset, pfm, "raw", "target_distance")
                distance = value(frame, dataset, pfm, method, "target_distance")
                change = "Ref." if method == "raw" else (signed(100 * (raw - distance) / raw, 1) if np.isfinite(distance) else "--")
                cells_ += [fmt(distance, 4), change]
            shade = FAMILY_SHADE if method in IMAGE_METHODS else ""
            lines.append(f"        {shade}{level} & {METHOD_LABELS[method]} & " + " & ".join(cells_) + r" \\")
        if dataset == "pannormal":
            lines.append(r"        \midrule")
    lines += [r"        \bottomrule", r"    \end{tabularx}", r"\end{table}", ""]
    (TABLES / "table_cross_pfm_distance_supp.tex").write_text("\n".join(lines))


def main() -> None:
    frame, detect = cells(), detectability()
    table_interaction()
    table_image_correction(frame, detect)
    table_cross_pfm(frame, detect)
    table_pfm_robustness()
    table_detectability_probes()
    table_cross_pfm_distance(frame)
    for name in ("table_scanner_tissue_interaction", "table_scanner_tissue_variance_full",
                 "table_scanner_tissue_variance_components",
                 "table_image_correction_three_axis", "table_cross_pfm_detectability", "table_pfm_robustness_supp",
                 "table_detectability_probes_supp", "table_cross_pfm_distance_supp"):
        print(f"== {name}")
        print((TABLES / f"{name}.tex").read_text())


if __name__ == "__main__":
    main()

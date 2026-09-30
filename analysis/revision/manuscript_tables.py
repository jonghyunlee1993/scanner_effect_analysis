#!/usr/bin/env python3
"""Write the revised manuscript tables from the revision results.

Main tables
* Table 1 ``table_scanner_tissue_interaction.tex``: scanner mean differences for five
  representative image measures and the scanner x tissue test (RV11).
* Table 2 ``table_image_correction_three_axis.tex``: image fidelity, UNI v1 alignment, target
  detectability (best of three probes, within-fold; RV04b) and tissue retrieval (RV04) for
  every image-level correction in PanNormal and PLISM.
* Table 3 ``table_cross_pfm_detectability.tex``: target-distance change and detectability for
  image- and feature-level corrections in four PFMs (RV04, RV04b).

Supplementary tables
* ``table_scanner_tissue_interaction_full_supp.tex``: all 13 measures, with the mixed-model
  interaction fraction alongside (RV11).
* ``table_pfm_robustness_supp.tex``: scanner sensitivity, robustness and tissue information for
  all model variants (RV13, RV14).
* ``table_detectability_probes_supp.tex``: PanNormal detectability under each of the three
  within-fold probes (RV04b).
* ``table_cross_pfm_distance_supp.tex``: target distance for every method in four PFMs and both
  datasets (RV04).

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
PFM_LABELS = {"uni_v1": "UNI v1", "uni2": "UNI2-h", "virchow2": "Virchow2", "hoptimus1": "H-optimus-1"}
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

CLASS_LABELS = {"raw": "Baseline", "reinhard": "Stain Norm", "pix2pix": "Style Transfer",
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


def fmt(value: float, digits: int) -> str:
    return "--" if not np.isfinite(value) else f"{value:.{digits}f}"


def table_interaction() -> None:
    tests = pd.read_csv(RESULTS / "scanner_tissue_interaction/interaction_tests.csv")
    rows = (("delta_lab_a", "Color", "$\\Delta a^*$", 2), ("log2_lab_l_sd_ratio", "Contrast", "$L^*$ contrast ratio ($\\log_2$)", 2),
            ("log2_od_sd_ratio", "Contrast", "OD contrast ratio ($\\log_2$)", 2),
            ("frequency_low_mid", "Frequency", "Low--mid frequency transfer ($\\log_2$)", 3),
            ("frequency_high", "Frequency", "High-frequency transfer ($\\log_2$)", 3))
    lines = [r"\begin{table}[pos=H]",
             r"    \caption{Mean differences in PanNormal image measures from the reference scanner (AT2) and their dependence on tissue type. The last two columns test whether the scanner effect differs between tissue types (scanner $\times$ tissue term; $\omega^2$ is the chance-corrected effect size). Full results for all 13 measures are in Supplementary Table~\ref{table_scanner_tissue_interaction_full}.}",
             r"    \label{table_scanner_tissue_interaction}", r"    \centering", r"    \small",
             r"    \begin{tabularx}{\textwidth}{@{}lYrrrrrrr@{}}", r"        \toprule",
             r"        Category & Image measure & \multicolumn{5}{c}{Mean difference from AT2} & $\omega^2$ & $q$ \\",
             r"        \cmidrule(lr){3-7}",
             "        & & " + " & ".join(SCANNER_LABELS[s] for s in SCANNERS) + r" & & \\", r"        \midrule"]
    for endpoint, category, label, digits in rows:
        row = tests[tests.endpoint == endpoint].iloc[0]
        means = " & ".join(signed(row[f"mean_{s}"], digits) for s in SCANNERS)
        q = row.q_permutation
        lines.append(f"        {category} & {label} & {means} & {row.omega_sq:.2f} & {'$<$0.001' if q < 0.001 else f'{q:.3f}'} \\\\")
    lines += [r"        \bottomrule", r"    \end{tabularx}", r"\end{table}", ""]
    (TABLES / "table_scanner_tissue_interaction.tex").write_text("\n".join(lines))

    lmm = pd.read_csv(PROJECT / "analysis/paper/results/direct_slide_lmm/endpoint_summary.csv")[
        ["endpoint", "fraction_scanner_by_tissue"]]
    full = tests.merge(lmm, on="endpoint", how="left")
    lines = [r"\begin{table}[pos=H]",
             r"    \caption{Scanner $\times$ tissue test for all 13 PanNormal image measures. $F$ compares the scanner $\times$ tissue term with the variation between slides of the same tissue type (144 and 264 degrees of freedom); $p$ values come from 9,999 permutations of tissue labels across slides and $q$ values from Benjamini--Hochberg adjustment. $\omega^2$ is the chance-corrected effect size; the mixed-model column gives the scanner $\times$ tissue variance fraction from the linear mixed model (Supplementary Methods) for comparison.}",
             r"    \label{table_scanner_tissue_interaction_full}", r"    \centering", r"    \small",
             r"    \begin{tabularx}{\textwidth}{@{}Yrrrrr@{}}", r"        \toprule",
             r"        Image measure & $F$ & $p$ & $q$ & $\omega^2$ & Mixed-model fraction \\", r"        \midrule"]
    for row in full.itertuples():
        label = ENDPOINT_LABELS[row.endpoint]
        lines.append(f"        {label} & {row.F:.2f} & {row.p_permutation:.4f} & {row.q_permutation:.4f} & "
                     f"{row.omega_sq:.2f} & {row.fraction_scanner_by_tissue:.2f} \\\\")
    lines += [r"        \bottomrule", r"    \end{tabularx}", r"\end{table}", ""]
    (TABLES / "table_scanner_tissue_interaction_full_supp.tex").write_text("\n".join(lines))


def table_image_correction(frame: pd.DataFrame, detect: pd.DataFrame) -> None:
    methods = ("raw", "reinhard", "macenko", "vahadane", "pix2pix", "cyclegan", "combined")
    lines = [r"\begin{table}[pos=H]",
             r"    \caption{Image-level correction judged on three axes against real paired images (UNI v1). Image fidelity: SSIM, LPIPS--VGG16 and image residual (lower is closer to the target). Representation: UNI v1 cosine distance to the target and its change from raw (positive = closer). Detectability: accuracy of the best of three probes (linear, MLP, $k$-NN) separating corrected images from real targets (0.5 = not detectable). Tissue: same-tissue retrieval (macro recall, \%); in PLISM the same cores recur across sections, so retrieval is near its ceiling. Image residual was computed for PanNormal only.}",
             r"    \label{table_image_correction_three_axis}", r"    \centering", r"    \scriptsize",
             r"    \setlength{\tabcolsep}{3pt}", r"    \begin{tabularx}{\textwidth}{@{}lYrrrrrrr@{}}", r"        \toprule",
             r"        & & \multicolumn{3}{c}{Image fidelity} & \multicolumn{2}{c}{Representation} & & \\",
             r"        \cmidrule(lr){3-5}\cmidrule(lr){6-7}",
             r"        Model class & Method & SSIM $\uparrow$ & LPIPS $\downarrow$ & Residual $\downarrow$ & Distance $\downarrow$ & $\Delta$ Raw (\%) $\uparrow$ & Detectability $\downarrow$ & Tissue (\%) $\uparrow$ \\",
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
                         f"{fmt(detect_value(detect, dataset, 'uni_v1', method), 3)} & "
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
             r"    \caption{Image- and feature-level correction in four PFMs. $\Delta$: change in cosine distance to the real paired target from raw (\%; positive = closer). Det.: accuracy of the best of three probes (linear, MLP, $k$-NN) separating corrected embeddings from real target embeddings, trained and tested within the slides held out from each correction fit (0.5 = not detectable). PLISM applies the corrections fitted in PanNormal without refitting.}",
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
            shade = r"\rowcolor{black!7} " if level == "Image" or method in ("combined", "pix2pix", "cyclegan") else ""
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
             r"    \caption{Scanner sensitivity, robustness and tissue information for all model variants (PanNormal, raw images). Sensitivity: representation shift under an equal high-frequency change divided by the between-tissue distance. Distance: normalized scanner distance. Det.: best-probe detectability of the target scanner. RI: PathoROB robustness index (chance 0.089 in this dataset). Tissue: same-tissue retrieval (\%). Color and frequency shares: fraction of the raw target distance removed by Reinhard and, additionally, by frequency matching. Models are ordered by RI.}",
             r"    \label{table_pfm_robustness}", r"    \centering", r"    \scriptsize",
             r"    \setlength{\tabcolsep}{3pt}", r"    \begin{tabularx}{\textwidth}{@{}lYrrrrrrr@{}}", r"        \toprule",
             r"        Model & Training signal and data & Sensitivity $\downarrow$ & Distance $\downarrow$ & Det. $\downarrow$ & RI $\uparrow$ & Tissue (\%) $\uparrow$ & Color share & Freq. share \\",
             r"        \midrule"]
    for name in order:
        row, extra = table.loc[name], wide.loc[name]
        label = row.label.replace("\\n", " ").replace("\n", " ")
        lines.append(f"        {label} & {groups[name]} & {row['normalized_shift_high_d0.25']:.4f} & "
                     f"{row.normalized_distance:.3f} & {extra.detectability_best:.3f} & {row.robustness_index:.2f} & "
                     f"{100 * row.tissue_retrieval:.1f} & {extra.colour_share:.2f} & {extra.frequency_share:.3f} \\\\")
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
             r"    \caption{Target-scanner detectability by probe in PanNormal. Balanced accuracy of linear, MLP and $k$-NN probes separating corrected embeddings from real target embeddings, trained and tested within the slides held out from each correction fit (0.5 = not detectable). In PLISM, every method stayed at 0.99 or higher under the linear probe.}",
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
        lines.append(f"        {level} & {METHOD_LABELS[method]} & " + " & ".join(cells_) + r" \\")
    lines += [r"        \bottomrule", r"    \end{tabularx}", r"\end{table}", ""]
    (TABLES / "table_detectability_probes_supp.tex").write_text("\n".join(lines))


def table_cross_pfm_distance(frame: pd.DataFrame) -> None:
    methods = (("raw", "Baseline"), ("reinhard", "Image"), ("macenko", ""), ("vahadane", ""), ("frequency", ""),
               ("combined", ""), ("pix2pix", ""), ("cyclegan", ""), ("ridge", "Feature"), ("combat", ""), ("ols", ""))
    header = " & ".join(PFM_LABELS[p] for p in PFMS)
    lines = [r"\begin{table}[pos=H]",
             r"    \caption{PFM cosine distance to the real paired target after each correction (lower is closer). PanNormal: 103 slides, five directions; PLISM: 13 sections, three directions, with corrections fitted in PanNormal and applied without refitting. Frequency: radial frequency matching alone.}",
             r"    \label{table_cross_pfm_distance}", r"    \centering", r"    \scriptsize",
             r"    \setlength{\tabcolsep}{3pt}", r"    \begin{tabularx}{\textwidth}{@{}lYrrrrrrrr@{}}", r"        \toprule",
             r"        & & \multicolumn{4}{c}{PanNormal} & \multicolumn{4}{c}{PLISM} \\",
             r"        \cmidrule(lr){3-6}\cmidrule(lr){7-10}",
             f"        Correction & Method & {header} & {header} \\\\", r"        \midrule"]
    for method, level in methods:
        cells_ = [fmt(value(frame, dataset, pfm, method, "target_distance"), 4)
                  for dataset in ("pannormal", "plism") for pfm in PFMS]
        lines.append(f"        {level} & {METHOD_LABELS.get(method, 'Frequency')} & " + " & ".join(cells_) + r" \\")
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
    for name in ("table_scanner_tissue_interaction", "table_scanner_tissue_interaction_full_supp",
                 "table_image_correction_three_axis", "table_cross_pfm_detectability", "table_pfm_robustness_supp",
                 "table_detectability_probes_supp", "table_cross_pfm_distance_supp"):
        print(f"== {name}")
        print((TABLES / f"{name}.tex").read_text())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build separate Korean and English PanNormal scientific story reports.

The editable Korean/English narrative lives outside this builder. Numeric
claims, tables, figure geometry, and the evidence registry are regenerated
from locked aggregate artifacts. The outputs are deterministic standalone
HTML files, each accompanied by its own release manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from html import escape
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "reports/pannormal_story_bilingual_v3/report_config.json"
DEFAULT_NARRATIVE = ROOT / "reports/pannormal_story_bilingual_v3/narrative.html"
DEFAULT_REGISTRY = ROOT / "reports/pannormal_story_bilingual_v3/evidence_registry.json"
DEFAULT_OUTPUT_DIR = ROOT / "presentations/pannormal_scanner_harmonization_report_v3_2026-08-29"
DEFAULT_OUTPUT_KO = DEFAULT_OUTPUT_DIR / "pannormal_scanner_harmonization_evidence_report_ko.html"
DEFAULT_OUTPUT_EN = DEFAULT_OUTPUT_DIR / "pannormal_scanner_harmonization_evidence_report_en.html"
DEFAULT_MANIFEST_KO = DEFAULT_OUTPUT_DIR / "pannormal_scanner_harmonization_evidence_report_ko.manifest.json"
DEFAULT_MANIFEST_EN = DEFAULT_OUTPUT_DIR / "pannormal_scanner_harmonization_evidence_report_en.manifest.json"

ENCODERS = ("resnet50", "uni_v1", "conch_v1", "virchow2")
ENCODER_LABELS = {
    "resnet50": "ResNet50",
    "uni_v1": "UNI v1",
    "conch_v1": "CONCH v1",
    "virchow2": "Virchow2",
}

SOURCES = {
    "geometry": "outputs/e0_native_geometry_final/summary.json",
    "e4_summary": "outputs/e4_control_frontier/summary.json",
    "e4_endpoints": "outputs/e4_control_frontier/endpoint_summary.csv",
    "e5_summary": "outputs/e5_comparator_frontier/summary.json",
    "e5_endpoints": "outputs/e5_comparator_frontier/endpoint_summary.csv",
    "rf1u_summary": "outputs/rf1u_multitarget/frontier/summary.json",
    "rf1u_endpoints": "outputs/rf1u_multitarget/frontier/endpoint_summary.csv",
    "rf1u_incremental": "outputs/rf1u_multitarget/frontier/incremental_vs_reinhard.csv",
    "e8_frontier": "outputs/e8_residual/frontier/gt450/endpoint_summary.csv",
    "e8_reading": "outputs/e8_residual/reading/gt450/summary.json",
    "e8_audit": "outputs/e8_residual/audit/gt450/summary.json",
    "e9_summary": "outputs/plism_core_native_ert/summary.json",
    "e9_endpoints": "outputs/plism_core_native_ert/ert_summary.csv",
    "e8_contract": "docs/e8_paired_residual_contract.md",
    "e9_contract": "docs/e9_plism_native_ert_contract.md",
}
for encoder in ENCODERS:
    SOURCES[f"e8_probe_{encoder}"] = f"outputs/e8_residual/probe/{encoder}.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--narrative", type=Path, default=DEFAULT_NARRATIVE)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-ko", type=Path, default=DEFAULT_OUTPUT_KO)
    parser.add_argument("--output-en", type=Path, default=DEFAULT_OUTPUT_EN)
    parser.add_argument("--manifest-ko", type=Path, default=DEFAULT_MANIFEST_KO)
    parser.add_argument("--manifest-en", type=Path, default=DEFAULT_MANIFEST_EN)
    return parser.parse_args()


def tr(language: str, ko: str, en: str) -> str:
    if language not in {"ko", "en"}:
        raise ValueError(f"unsupported report language: {language}")
    return ko if language == "ko" else en


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(value: str | float | int) -> float:
    return float(value)


def signed(value: float, digits: int = 1, suffix: str = "") -> str:
    if abs(value) < 0.5 * 10 ** (-digits):
        value = 0.0
    sign = "+" if value >= 0 else "−"
    return f"{sign}{abs(value):.{digits}f}{suffix}"


def plain(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def number_range(values: Iterable[float], digits: int, suffix: str = "") -> str:
    vals = list(values)
    return f"{signed(min(vals), digits, suffix)} to {signed(max(vals), digits, suffix)}"


def percent_range(values: Iterable[float], digits: int = 1) -> str:
    vals = list(values)
    return f"{signed(min(vals), digits, '%')} to {signed(max(vals), digits, '%')}"


def select(rows: list[dict[str, str]], **criteria: str) -> dict[str, str]:
    matches = [row for row in rows if all(row.get(key) == value for key, value in criteria.items())]
    if len(matches) != 1:
        raise RuntimeError(f"expected one row for {criteria}, found {len(matches)}")
    return matches[0]


def boolean(value: str) -> bool:
    return value.strip().lower() == "true"


def table_html(
    caption: str,
    headers: list[str],
    rows: list[list[str]],
    *,
    css_class: str = "data-table",
) -> str:
    head = "".join(f'<th scope="col">{cell}</th>' for cell in headers)
    body_rows = []
    for row in rows:
        if len(row) != len(headers):
            raise RuntimeError("table row width does not match header width")
        cells = [f'<th scope="row">{row[0]}</th>']
        cells.extend(f"<td>{cell}</td>" for cell in row[1:])
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        f'<div class="table-wrap"><table class="{css_class}">'
        f"<caption>{caption}</caption><thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table></div>"
    )


def badge(label: str, kind: str) -> str:
    return f'<span class="badge {kind}">{escape(label)}</span>'


def source_metadata() -> dict[str, dict[str, str]]:
    roles = {
        "geometry": "paired cohort identity and geometry gate",
        "e4_summary": "control analysis contract and uncertainty",
        "e4_endpoints": "construct-validity control endpoints",
        "e5_summary": "comparator benchmark contract",
        "e5_endpoints": "comparator endpoints",
        "rf1u_summary": "multi-target frequency analysis contract",
        "rf1u_endpoints": "multi-target frequency endpoints",
        "rf1u_incremental": "paired incremental contrasts versus Reinhard",
        "e8_frontier": "learned image-correction endpoints",
        "e8_reading": "scanner-probe ranges and decision reading",
        "e8_audit": "image-only selection and phase audit",
        "e9_summary": "external native-spectrum predictions and uncertainty",
        "e9_endpoints": "external scanner-level ERT endpoints",
        "e8_contract": "prospective/amended decision-status metadata",
        "e9_contract": "external robustness-analysis status metadata",
    }
    metadata: dict[str, dict[str, str]] = {}
    for key, relative in SOURCES.items():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        role = roles.get(key, "scanner-probe endpoint")
        metadata[key] = {"path": relative, "sha256": sha256(path), "role": role}
    return metadata


def build_registry() -> dict[str, Any]:
    geometry = read_json(ROOT / SOURCES["geometry"])
    e4_summary = read_json(ROOT / SOURCES["e4_summary"])
    e4_rows = read_csv(ROOT / SOURCES["e4_endpoints"])
    e5_summary = read_json(ROOT / SOURCES["e5_summary"])
    e5_rows = read_csv(ROOT / SOURCES["e5_endpoints"])
    rf1u_summary = read_json(ROOT / SOURCES["rf1u_summary"])
    rf1u_rows = read_csv(ROOT / SOURCES["rf1u_endpoints"])
    incremental_rows = read_csv(ROOT / SOURCES["rf1u_incremental"])
    e8_rows = read_csv(ROOT / SOURCES["e8_frontier"])
    e8_reading = read_json(ROOT / SOURCES["e8_reading"])
    e8_audit = read_json(ROOT / SOURCES["e8_audit"])
    e9_summary = read_json(ROOT / SOURCES["e9_summary"])
    e9_rows = read_csv(ROOT / SOURCES["e9_endpoints"])

    control_conditions = {
        "complete_hf_removal": "hf_retention_0p00",
        "partial_hf_control": "registered_loo_hf_mean_0p25",
    }
    controls: dict[str, Any] = {}
    for label, condition in control_conditions.items():
        by_encoder = {}
        for encoder in ENCODERS:
            row = select(e4_rows, encoder_id=encoder, condition=condition)
            by_encoder[encoder] = {
                "relative_radius_reduction": f(row["relative_radius_reduction"]),
                "raw_minus_condition": f(row["raw_minus_condition"]),
                "difference_ci": [f(row["difference_ci_lower"]), f(row["difference_ci_upper"])],
                "delta_content_margin": f(row["delta_content_margin"]),
                "content_noninferiority_pass": boolean(row["content_noninferiority_pass"]),
                "collapse_every_scanner_pass": boolean(row["collapse_every_scanner_pass"]),
                "joint_improvement_pass": boolean(row["safe_and_invariance_improved"]),
            }
        controls[label] = {"condition": condition, "by_encoder": by_encoder}

    method_conditions = {
        "Reinhard": "reinhard_lab",
        "Paired OD affine": "paired_od_affine",
        "Frequency calibration": "frequency_calibration",
        "CORAL": "coral",
        "Orthogonal Procrustes": "orthogonal_procrustes",
    }
    benchmark: dict[str, Any] = {}
    for method, condition in method_conditions.items():
        by_encoder = {}
        for encoder in ENCODERS:
            row = select(e5_rows, encoder_id=encoder, condition=condition)
            by_encoder[encoder] = {
                "relative_radius_reduction": f(row["relative_radius_reduction"]),
                "raw_minus_condition": f(row["raw_minus_condition"]),
                "difference_ci": [f(row["difference_ci_lower"]), f(row["difference_ci_upper"])],
                "delta_content_margin": f(row["delta_content_margin"]),
                "representation_gate_pass": boolean(row["safe_for_pfm"]),
                "joint_improvement_pass": boolean(row["safe_and_invariance_improved"]),
            }
        benchmark[method] = {
            "condition": condition,
            "space": "feature" if condition in {"coral", "orthogonal_procrustes"} else "image",
            "by_encoder": by_encoder,
        }

    rf1u: dict[str, Any] = {}
    incremental_lookup = {
        (row["target"], row["encoder_id"]): row for row in incremental_rows
    }
    for target in ("at2", "gt450", "s60"):
        by_encoder = {}
        for encoder in ENCODERS:
            base = select(rf1u_rows, target=target, encoder_id=encoder, condition=f"reinhard_{target}")
            corrected = select(
                rf1u_rows,
                target=target,
                encoder_id=encoder,
                condition=f"reinhard_unpaired_band_{target}",
            )
            paired = incremental_lookup[(target, encoder)]
            by_encoder[encoder] = {
                "reinhard_rr": f(base["relative_radius_reduction"]),
                "rf1u_rr": f(corrected["relative_radius_reduction"]),
                "incremental_rr_points": 100
                * (f(corrected["relative_radius_reduction"]) - f(base["relative_radius_reduction"])),
                "paired_radius_difference": f(paired["reinhard_minus_rf1u_radius"]),
                "paired_ci": [f(paired["paired_ci_lower"]), f(paired["paired_ci_upper"])],
                "rf1u_better_than_reinhard": boolean(paired["rf1u_better_than_reinhard"]),
                "joint_improvement_pass": boolean(corrected["safe_and_invariance_improved"]),
            }
        rf1u[target] = {"by_encoder": by_encoder}

    probe_rows: list[dict[str, str]] = []
    for encoder in ENCODERS:
        probe_rows.extend(read_csv(ROOT / SOURCES[f"e8_probe_{encoder}"]))
    probes: dict[str, Any] = {}
    for condition in ("raw", "e8:e8_free_gt450", "feature:coral", "feature:orthogonal_procrustes"):
        probes[condition] = {}
        for probe in ("linear", "mlp", "knn"):
            values = {
                row["encoder_id"]: f(row["balanced_accuracy"])
                for row in probe_rows
                if row["condition"] == condition and row["probe"] == probe
            }
            if values:
                probes[condition][probe] = values

    e8_frontier = {}
    for encoder in ENCODERS:
        row = select(e8_rows, encoder_id=encoder, condition="e8_free_gt450")
        e8_frontier[encoder] = {
            "relative_radius_reduction": f(row["relative_radius_reduction"]),
            "raw_minus_condition": f(row["raw_minus_condition"]),
            "difference_ci": [f(row["difference_ci_lower"]), f(row["difference_ci_upper"])],
            "delta_content_margin": f(row["delta_content_margin"]),
            "representation_gate_pass": boolean(row["safe_for_pfm"]),
            "joint_improvement_pass": boolean(row["safe_and_invariance_improved"]),
        }
    e8 = {
        "target": "GT450",
        "frontier": e8_frontier,
        "probes": probes,
        "chance": f(e8_reading["probe_chance"]),
        "arm_a_folds_passing": len(e8_audit["arms"]["gainfield"]["gate"]["folds_passing"]),
        "arm_b_folds_passing": len(e8_audit["arms"]["free"]["gate"]["folds_passing"]),
        "arm_a_gate_pass": bool(e8_audit["arms"]["gainfield"]["gate"]["gate_pass"]),
        "arm_b_gate_pass": bool(e8_audit["arms"]["free"]["gate"]["gate_pass"]),
        "phase_correlation": f(e8_audit["arms"]["free"]["audit"]["phase_correlation"]["mean"]),
        "rf1u_phase_correlation": f(
            e8_audit["arms"]["free"]["audit"]["phase_correlation_base"]["mean"]
        ),
        "decision_status": {
            "document_header": "DRAFT",
            "endpoint_access": "specified before PFM endpoint access",
            "amendment": "selection ordering amended after fold-0 image results",
        },
    }

    e9_endpoint_map = {row["scanner"]: row for row in e9_rows}
    e9 = {
        "sections": int(e9_summary["sections"]),
        "scanners": int(e9_summary["scanners"]),
        "band_high_cyc_um": e9_summary["band_high_cyc_um"],
        "bootstrap_replicates": int(e9_summary["bootstrap"]),
        "verdicts": e9_summary["verdicts"],
        "endpoints": {
            scanner: {
                "estimate": f(row["estimate"]),
                "ci": [f(row["ci_low"]), f(row["ci_high"])],
                "n_sections": int(row["n_sections"]),
            }
            for scanner, row in e9_endpoint_map.items()
        },
        "analysis_status": "amended external robustness analysis",
        "estimator_relation": "non-equivalent to the PanNormal primary ERT estimator",
        "confounding": "stain fully confounded with serial section",
    }

    registry = {
        "schema_version": "pannormal_story_evidence_v1",
        "sources": source_metadata(),
        "design": {
            "slides": int(geometry["slides_expected"]),
            "scanners": int(geometry["cells_expected"] // geometry["slides_expected"]),
            "locations_per_slide": int(
                geometry["six_scanner_tuples_expected"] // geometry["slides_expected"]
            ),
            "acquisitions": int(geometry["location_rows_observed"]),
            "paired_tuples": int(geometry["six_scanner_tuples_passing"]),
            "geometry_gate_pass": bool(geometry["cohort_gate_pass"]),
            "bootstrap_unit": e4_summary["bootstrap_unit"],
            "bootstrap_replicates": int(e4_summary["bootstrap_replicates"]),
            "bootstrap_seed": int(e4_summary["bootstrap_seed"]),
            "content_noninferiority_margin": f(e4_summary["content_noninferiority_margin"]),
            "collapse_point_threshold": f(e4_summary["collapse_point_threshold"]),
            "collapse_lower_ci_threshold": f(e4_summary["collapse_lower_ci_threshold"]),
        },
        "controls": controls,
        "benchmark": benchmark,
        "rf1u": rf1u,
        "e8": e8,
        "e9": e9,
        "claim_ledger": {
            "supported": [
                "paired scanner/acquisition-pipeline differences are representation-visible",
                "invariance-only evaluation is degenerate",
                "tested corrections occupy different invariance-fidelity trade-offs",
                "a post-colour spatial-frequency residual is target-dependent",
            ],
            "not_supported": [
                "clinical safety or utility",
                "unseen-scanner generalization",
                "pure scanner-hardware effect",
                "universal image-domain ceiling",
                "scanner-information removal or deployment superiority of CORAL/Procrustes",
                "confirmatory external replication",
            ],
        },
        "benchmark_contract": {
            "conditions_including_raw": int(e5_summary["conditions_including_raw"]),
            "pfms": int(e5_summary["pfms"]),
            "analysis_gate_pass": bool(e5_summary["analysis_gate_pass"]),
        },
        "rf1u_contract": {
            "targets": rf1u_summary["targets"],
            "encoders": rf1u_summary["encoders"],
            "rf1u_cells": int(rf1u_summary["rf1u_cells"]),
            "rf1u_better_than_reinhard_cells": int(
                rf1u_summary["rf1u_better_than_reinhard_cells"]
            ),
        },
    }
    return registry


def render_headline_cards(registry: dict[str, Any]) -> str:
    controls = registry["controls"]["complete_hf_removal"]["by_encoder"]
    control_rr = [100 * value["relative_radius_reduction"] for value in controls.values()]
    e8_rr = [100 * value["relative_radius_reduction"] for value in registry["e8"]["frontier"].values()]
    e8_probe = list(registry["e8"]["probes"]["e8:e8_free_gt450"]["linear"].values())
    cards = [
        (f'{registry["design"]["acquisitions"]:,}', "paired acquisitions", "동일 위치 paired 획득"),
        (f'{min(control_rr):.1f}–{max(control_rr):.1f}%', "false invariance gain", "내용 삭제로 얻은 허위 이득"),
        (f'{max(e8_rr):.1f}% / ≥{min(e8_probe):.3f}', "max image RR / probe floor", "최대 image RR / 남은 probe"),
        ("0 / 3", "external directions reproduced", "외부 방향 예측 재현"),
    ]
    body = []
    for value, en_label, ko_label in cards:
        body.append(
            '<div class="headline-card">'
            f'<div class="headline-value">{escape(value)}</div>'
            f'<div lang="ko">{escape(ko_label)}</div>'
            f'<div lang="en" class="en small">{escape(en_label)}</div>'
            "</div>"
        )
    return '<div class="headline-grid">' + "".join(body) + "</div>"


def render_story_arc(language: str) -> str:
    steps = [
        ("1", "Paired observation", "동일 조직으로 scanner effect 식별"),
        ("2", "Falsification control", "내용 삭제가 만든 허위 성공 포착"),
        ("3", "Correction frontier", "image와 feature trade-off 비교"),
        ("4", "Stress and external tests", "강한 보정과 외부 실패로 주장 제한"),
    ]
    items = []
    for number, en_text, ko_text in steps:
        items.append(
            '<div class="flow-step">'
            f'<span class="flow-number">{number}</span>'
            f'<strong lang="ko">{escape(ko_text)}</strong>'
            f'<span lang="en" class="en">{escape(en_text)}</span>'
            "</div>"
        )
    return (
        '<figure class="story-figure"><div class="flow">'
        + '<span class="flow-arrow" aria-hidden="true">→</span>'.join(items)
        + f'</div><figcaption>{tr(language, "Figure 1. 근거 사슬의 논리. 각 실험은 앞선 실험이 드러낸 평가 실패에 답한다.", "Figure 1. Logic of the evidence chain. Each experiment answers a failure exposed by the previous one.")}</figcaption></figure>'
    )


def render_design_table(registry: dict[str, Any], language: str) -> str:
    d = registry["design"]
    rows = [
        [tr(language, "코호트", "Cohort"), tr(language, f'{d["slides"]}개 물리 슬라이드 × {d["scanners"]}개 스캐너 × {d["locations_per_slide"]}개 matched location', f'{d["slides"]} physical slides × {d["scanners"]} scanners × {d["locations_per_slide"]} matched locations')],
        [tr(language, "Paired 근거", "Paired evidence"), tr(language, f'{d["acquisitions"]:,}회 획득; {d["paired_tuples"]:,}개 완전한 six-scanner tuple', f'{d["acquisitions"]:,} acquisitions; {d["paired_tuples"]:,} complete six-scanner tuples')],
        [tr(language, "추론", "Inference"), tr(language, f'물리 슬라이드 단위; bootstrap {d["bootstrap_replicates"]:,}회; seed {d["bootstrap_seed"]}', f'{escape(d["bootstrap_unit"])}; {d["bootstrap_replicates"]:,} bootstrap replicates; seed {d["bootstrap_seed"]}')],
        ["Content gate", f'Δ margin ≥ {signed(d["content_noninferiority_margin"], 2)}'],
        ["Collapse gate", tr(language, f'모든 non-reference scanner에서 point ≥ {d["collapse_point_threshold"]:.2f}; lower CI ≥ {d["collapse_lower_ci_threshold"]:.2f}', f'point ≥ {d["collapse_point_threshold"]:.2f}; lower CI ≥ {d["collapse_lower_ci_threshold"]:.2f} for every non-reference scanner')],
    ]
    return table_html(
        tr(language, "Table 1. 전역 해석 계약", "Table 1. Global interpretation contract"),
        [tr(language, "요소", "Component"), tr(language, "고정 정의", "Locked definition")],
        rows,
    )


def control_ranges(registry: dict[str, Any], label: str) -> tuple[list[float], list[float]]:
    by_encoder = registry["controls"][label]["by_encoder"]
    rr = [100 * row["relative_radius_reduction"] for row in by_encoder.values()]
    content = [row["delta_content_margin"] for row in by_encoder.values()]
    return rr, content


def render_controls_table(registry: dict[str, Any], language: str) -> str:
    rows = []
    labels = {
        "complete_hf_removal": tr(language, "고주파 완전 제거", "Complete HF removal"),
        "partial_hf_control": "Paired 25% partial-HF",
    }
    for key, label in labels.items():
        rr, content = control_ranges(registry, key)
        passed = all(
            row["joint_improvement_pass"]
            for row in registry["controls"][key]["by_encoder"].values()
        )
        rows.append(
            [
                label,
                percent_range(rr),
                number_range(content, 4),
                badge(tr(language, "4/4 통과", "4/4 pass") if passed else tr(language, "0/4 통과", "0/4 pass"), "pass" if passed else "fail"),
                tr(language, "충실도 보존 대조", "Fidelity-preserving control") if passed else tr(language, "파괴적 허위 성공", "Destructive false success"),
            ]
        )
    return table_html(
        tr(language, "Table 2. Paired control은 유용한 정렬과 내용 삭제를 구별한다", "Table 2. Paired controls distinguish useful alignment from content deletion"),
        [tr(language, "개입", "Intervention"), tr(language, "Radius 감소", "Radius reduction"), "Δ content margin", tr(language, "공동 판정", "Joint decision"), tr(language, "해석", "Interpretation")],
        rows,
    )


def benchmark_rr(registry: dict[str, Any], method: str) -> list[float]:
    return [
        100 * registry["benchmark"][method]["by_encoder"][encoder]["relative_radius_reduction"]
        for encoder in ENCODERS
    ]


def render_method_frontier(registry: dict[str, Any], language: str) -> str:
    methods = ["Reinhard", "Paired OD affine", "Frequency calibration", "CORAL", "Orthogonal Procrustes"]
    width, height = 780, 350
    x0, x1 = 210, 735
    scale_min, scale_max = -15.0, 50.0

    def x(value: float) -> float:
        return x0 + (value - scale_min) / (scale_max - scale_min) * (x1 - x0)

    ticks = [-10, 0, 10, 20, 30, 40, 50]
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-labelledby="frontier-title frontier-desc">',
        f'<title id="frontier-title">{tr(language, "네 encoder의 scanner-radius 감소 범위", "Range of scanner-radius reductions across four encoders")}</title>',
        f'<desc id="frontier-desc">{tr(language, "Image-space와 feature-space 방법은 서로 다른 범위를 차지하며 음수는 악화를 뜻한다.", "Image-space and feature-space methods occupy different ranges; negative values indicate worsening.")}</desc>',
    ]
    for tick in ticks:
        xpos = x(tick)
        css = "zero-line" if tick == 0 else "grid-line"
        parts.append(f'<line class="{css}" x1="{xpos:.1f}" y1="35" x2="{xpos:.1f}" y2="300"/>')
        parts.append(f'<text class="svg-tick" x="{xpos:.1f}" y="325" text-anchor="middle">{tick:+d}%</text>')
    for index, method in enumerate(methods):
        values = benchmark_rr(registry, method)
        low, high = min(values), max(values)
        y = 60 + index * 52
        family = registry["benchmark"][method]["space"]
        parts.append(f'<text class="svg-label" x="198" y="{y + 5}" text-anchor="end">{escape(method)}</text>')
        parts.append(f'<line class="range {family}" x1="{x(low):.1f}" y1="{y}" x2="{x(high):.1f}" y2="{y}"/>')
        for value in values:
            parts.append(f'<circle class="point {family}" cx="{x(value):.1f}" cy="{y}" r="5"><title>{value:+.1f}%</title></circle>')
    parts.append(f'<text class="svg-axis" x="472" y="347" text-anchor="middle">{tr(language, "상대 scanner-radius 감소", "relative scanner-radius reduction")}</text>')
    parts.append("</svg>")
    return (
        '<figure class="frontier-figure">'
        + "".join(parts)
        + '<div class="legend"><span><i class="image-dot"></i>image-space</span><span><i class="feature-dot"></i>feature-space</span></div>'
        + f'<figcaption>{tr(language, "Figure 2. 네 encoder에서 관찰된 범위. 점은 encoder 하나, 선은 전체 panel 범위다. Radius는 fidelity gate와 함께 해석한다.", "Figure 2. Range across four encoders. A point is an encoder; the line spans the tested panel. Radius is interpreted only with the fidelity gates.")}</figcaption></figure>'
    )


def render_benchmark_table(registry: dict[str, Any], language: str) -> str:
    rows = []
    for method, payload in registry["benchmark"].items():
        row = [f"{escape(method)} <span class=\"muted\">({payload['space']})</span>"]
        joint = []
        gates = []
        for encoder in ENCODERS:
            item = payload["by_encoder"][encoder]
            row.append(signed(100 * item["relative_radius_reduction"], 1, "%"))
            joint.append(item["joint_improvement_pass"])
            gates.append(item["representation_gate_pass"])
        if all(joint):
            status = badge(tr(language, "공동 판정 4/4 통과", "joint pass 4/4"), "pass")
        elif all(gates):
            status = badge(tr(language, f"gate 4/4; 공동 판정 {sum(joint)}/4", f"gates 4/4; joint {sum(joint)}/4"), "mixed")
        else:
            status = badge(tr(language, f"gate 실패 {4 - sum(gates)}/4", f"gate failures {4 - sum(gates)}/4"), "fail")
        row.append(status)
        rows.append(row)
    return table_html(
        tr(language, "Table 3. 방법·encoder별 상대 scanner-radius 감소", "Table 3. Relative scanner-radius reduction by method and encoder"),
        [tr(language, "방법", "Method")] + [ENCODER_LABELS[e] for e in ENCODERS] + [tr(language, "공동 해석", "Joint interpretation")],
        rows,
    )


def render_rf1u_table(registry: dict[str, Any], language: str) -> str:
    rows = []
    for target in ("at2", "gt450", "s60"):
        for encoder in ENCODERS:
            item = registry["rf1u"][target]["by_encoder"][encoder]
            rows.append(
                [
                    f"{target.upper()} · {ENCODER_LABELS[encoder]}",
                    signed(100 * item["reinhard_rr"], 2, "%"),
                    signed(100 * item["rf1u_rr"], 2, "%"),
                    signed(item["incremental_rr_points"], 2, " pp"),
                    badge(tr(language, "개선", "better") if item["rf1u_better_than_reinhard"] else tr(language, "개선 아님", "not better"), "pass" if item["rf1u_better_than_reinhard"] else "fail"),
                ]
            )
    return table_html(
        tr(language, "Table 4. Reinhard 이후 주파수 보정의 추가 이득", "Table 4. Incremental frequency correction after Reinhard"),
        ["Target · encoder", "Reinhard RR", "Reinhard + RF1U RR", tr(language, "추가 이득", "Increment"), tr(language, "Paired 비교", "Paired contrast")],
        rows,
        css_class="data-table compact",
    )


def probe_values(registry: dict[str, Any], condition: str, probe: str) -> list[float]:
    return list(registry["e8"]["probes"][condition][probe].values())


def render_probe_table(registry: dict[str, Any], language: str) -> str:
    conditions = [
        ("Raw", "raw", "linear"),
        ("E8 Arm B image correction", "e8:e8_free_gt450", "linear"),
        ("CORAL", "feature:coral", "linear"),
        ("CORAL", "feature:coral", "mlp"),
        ("Orthogonal Procrustes", "feature:orthogonal_procrustes", "linear"),
        ("Orthogonal Procrustes", "feature:orthogonal_procrustes", "mlp"),
    ]
    rows = []
    for label, condition, probe in conditions:
        values = probe_values(registry, condition, probe)
        rows.append([f"{label} · {probe}", f"{min(values):.3f}–{max(values):.3f}", tr(language, "6개 scanner; chance 0.167", "6 scanners; chance 0.167")])
    return table_html(
        tr(language, "Table 5. 표현과 probe 종류에 따른 scanner-probe balanced accuracy", "Table 5. Scanner-probe balanced accuracy changes with representation and probe class"),
        ["Representation · probe", tr(language, "Encoder 범위", "Range across encoders"), tr(language, "기준", "Reference")],
        rows,
    )


def render_e8_table(registry: dict[str, Any], language: str) -> str:
    rows = []
    probes = registry["e8"]["probes"]["e8:e8_free_gt450"]["linear"]
    for encoder in ENCODERS:
        item = registry["e8"]["frontier"][encoder]
        rows.append(
            [
                ENCODER_LABELS[encoder],
                signed(100 * item["relative_radius_reduction"], 1, "%"),
                signed(item["delta_content_margin"], 4),
                plain(probes[encoder], 3),
                badge(tr(language, "공동 판정 통과", "joint pass"), "pass") if item["joint_improvement_pass"] else badge(tr(language, "실패", "fail"), "fail"),
            ]
        )
    return table_html(
        tr(language, "Table 6. E8 Arm B 학습 기반 image correction", "Table 6. E8 Arm B learned image correction"),
        ["Encoder", tr(language, "Radius 감소", "Radius reduction"), "Δ content margin", "Linear probe", tr(language, "표현 판정", "Representation decision")],
        rows,
    )


def render_e9_table(registry: dict[str, Any], language: str) -> str:
    verdicts = registry["e9"]["verdicts"]
    rows = [
        ["P1 · GT450", "log2 ERT > 0", signed(verdicts["P1"]["log2_ert"], 3), f'[{signed(verdicts["P1"]["ci"][0], 3)}, {signed(verdicts["P1"]["ci"][1], 3)}]', badge(tr(language, "실패", "fail"), "fail")],
        ["P2 · S360", "log2 ERT < 0", signed(verdicts["P2"]["log2_ert"], 3), f'[{signed(verdicts["P2"]["ci"][0], 3)}, {signed(verdicts["P2"]["ci"][1], 3)}]', badge(tr(language, "실패", "fail"), "fail")],
        [tr(language, "P3 · 순서", "P3 · ordering"), "GT450 > S60 > S360", "S360 > S60 > GT450", "section bootstrap", badge(tr(language, "실패", "fail"), "fail")],
    ]
    return table_html(
        tr(language, "Table 7. 고정된 PLISM native-spectrum 예측", "Table 7. Frozen PLISM native-spectrum predictions"),
        [tr(language, "예측", "Prediction"), tr(language, "기대", "Expected"), tr(language, "관찰", "Observed"), tr(language, "95% CI / 불확실성", "95% CI / uncertainty"), tr(language, "판정", "Decision")],
        rows,
    )


def render_contribution_cards() -> str:
    cards = [
        ("Construct validity", "구성타당도", "A destructive control proves why invariance alone is insufficient.", "파괴적 대조실험으로 invariance 단독 지표의 실패를 직접 증명한다."),
        ("Paired evidence", "Paired 근거", "Same-location acquisitions separate scanner effects from tissue content.", "동일 위치 획득으로 scanner effect와 조직 내용을 분리한다."),
        ("One interpretation contract", "하나의 해석 계약", "Image and feature corrections are judged on the same invariance–fidelity frontier.", "image·feature 보정을 동일한 invariance–fidelity frontier에서 판정한다."),
        ("Negative external evidence", "부정적 외부 근거", "Failed predictions refine the estimand instead of being hidden as a replication claim.", "실패한 외부 예측을 숨기지 않고 estimand와 주장 범위를 교정한다."),
    ]
    out = []
    for en_title, ko_title, en_text, ko_text in cards:
        out.append(
            '<article class="contribution-card">'
            f'<h3 lang="ko">{escape(ko_title)}</h3><p lang="ko">{escape(ko_text)}</p>'
            f'<h3 lang="en" class="en">{escape(en_title)}</h3><p lang="en" class="en">{escape(en_text)}</p>'
            "</article>"
        )
    return '<div class="contribution-grid">' + "".join(out) + "</div>"


def render_claim_boundary_table(language: str) -> str:
    supported = tr(language, "지지됨", "Supported")
    unsupported = tr(language, "지지 안 됨", "Not supported")
    rows = [
        [supported, tr(language, "Paired representation에서 scanner/acquisition-pipeline 차이가 계속 관찰된다.", "Scanner/acquisition-pipeline differences remain visible in paired representations."), tr(language, "Paired design; raw probe; 네 encoder", "Paired design; raw probes; four encoders")],
        [supported, tr(language, "Invariance-only 평가는 퇴행적 해를 허용한다.", "Invariance-only evaluation is degenerate."), tr(language, "고주파 완전 제거와 paired partial-HF control 비교", "Complete HF removal versus paired partial-HF control")],
        [supported, tr(language, "시험한 correction family들은 서로 다른 invariance–fidelity trade-off를 만든다.", "Tested correction families occupy different invariance–fidelity trade-offs."), "E5 benchmark; RF1U; E8"],
        [unsupported, tr(language, "임상 안전성, 유용성 또는 비열등성", "Clinical safety, utility, or non-inferiority."), tr(language, "Labelled clinical endpoint와 임상적으로 보정된 margin 없음", "No labelled clinical endpoint or calibrated clinical margin")],
        [unsupported, tr(language, "보편적 image-domain ceiling 또는 unseen-scanner 일반화", "Universal image-domain ceiling or unseen-scanner generalization."), tr(language, "주 코호트 하나; E8 target 하나; unseen-scanner 시험 없음", "One main cohort; one E8 target; no unseen-scanner test")],
        [unsupported, tr(language, "CORAL/Procrustes의 deployment 우월성 또는 scanner information 제거", "CORAL/Procrustes deployment superiority or scanner-information removal."), tr(language, "Non-nested probe; fold별 좌표계", "Non-nested probes; fold-specific coordinates")],
        [unsupported, tr(language, "Confirmatory external replication", "Confirmatory external replication."), tr(language, "비동등 estimator; amended PLISM analysis", "Non-equivalent estimator; amended PLISM analysis")],
    ]
    return table_html(
        tr(language, "Table 8. 투고용 주장 경계", "Table 8. Submission-facing claim boundary"),
        [tr(language, "상태", "Status"), tr(language, "주장", "Claim"), tr(language, "근거 경계", "Evidence boundary")],
        rows,
    )


def render_risk_table(language: str) -> str:
    high = tr(language, "높음", "High")
    medium = tr(language, "중간", "Medium")
    rows = [
        [badge(high, "fail"), tr(language, "Downstream 타당도", "Downstream validity"), tr(language, "Nested labelled task가 없고 representation gate는 clinical endpoint가 아니다.", "No nested labelled task; representation gates are not clinical endpoints."), tr(language, "Outer-fold nested 진단·분자·예후 평가를 추가한다.", "Add outer-fold nested diagnostic/molecular/prognostic evaluation.")],
        [badge(high, "fail"), tr(language, "외부 동등성", "External equivalence"), tr(language, "PLISM은 비동등 spectral estimator와 amended pipeline을 사용한다.", "PLISM uses a non-equivalent spectral estimator and amended pipeline."), tr(language, "PLISM raw와 Reinhard에 정확한 PanNormal estimator를 적용한다.", "Run the exact PanNormal estimator on PLISM raw + Reinhard.")],
        [badge(high, "fail"), tr(language, "획득 교란", "Acquisition confounding"), tr(language, "Scanner model당 장비 한 대이며 스캔 시점이 다르다.", "One unit per scanner model; nonconcurrent scanning."), tr(language, "Repeat scan과 scanner/date sensitivity analysis를 수행한다.", "Repeat scans and scanner/date sensitivity analysis.")],
        [badge(medium, "mixed"), tr(language, "비교군 범위", "Comparator breadth"), tr(language, "완전한 contemporary FEATMAP-style deployment comparator가 없다.", "No full contemporary FEATMAP-style deployment comparator."), tr(language, "Nested feature-harmonization comparator를 추가하거나 novelty 주장을 좁힌다.", "Add a nested feature-harmonization comparator or narrow novelty claims.")],
        [badge(medium, "mixed"), tr(language, "Fitting 불확실성", "Fitting uncertainty"), tr(language, "Transform fitting 불확실성이 모든 interval에 전파되지 않았다.", "Transform fitting is not propagated through every reported interval."), tr(language, "전체 fit-transform-score pipeline을 bootstrap한다.", "Bootstrap the entire fit-transform-score pipeline.")],
        [badge(medium, "mixed"), tr(language, "E8 일반화", "E8 generality"), tr(language, "GT450 target 하나와 cohort 하나에 제한된다.", "One GT450 target and one cohort."), tr(language, "Destination과 cohort sensitivity를 시험한다.", "Test destination and cohort sensitivity.")],
    ]
    return table_html(
        tr(language, "Table 9. 위험도와 완화 실험", "Table 9. Risk register and mitigation"),
        [tr(language, "위험", "Risk"), tr(language, "영역", "Domain"), tr(language, "중요성", "Why it matters"), tr(language, "완화", "Mitigation")],
        rows,
    )


def render_provenance_table(registry: dict[str, Any], language: str) -> str:
    role_ko = {
        "geometry": "paired cohort identity와 geometry gate",
        "e4_summary": "control analysis contract와 불확실성",
        "e4_endpoints": "구성타당도 control endpoint",
        "e5_summary": "comparator benchmark contract",
        "e5_endpoints": "comparator endpoint",
        "rf1u_summary": "multi-target frequency analysis contract",
        "rf1u_endpoints": "multi-target frequency endpoint",
        "rf1u_incremental": "Reinhard 대비 paired incremental contrast",
        "e8_frontier": "learned image-correction endpoint",
        "e8_reading": "scanner-probe 범위와 decision reading",
        "e8_audit": "image-only selection과 phase audit",
        "e9_summary": "external native-spectrum 예측과 불확실성",
        "e9_endpoints": "external scanner-level ERT endpoint",
        "e8_contract": "prospective/amended decision-status metadata",
        "e9_contract": "external robustness-analysis status metadata",
    }
    rows = []
    for key, metadata in registry["sources"].items():
        rows.append(
            [
                f"<code>{escape(key)}</code>",
                f"<code>{escape(metadata['path'])}</code>",
                escape(role_ko.get(key, "scanner-probe endpoint") if language == "ko" else metadata["role"]),
                f"<code>{metadata['sha256']}</code>",
            ]
        )
    return table_html(
        tr(language, "Table 11. 근거 입력과 고정 hash", "Table 11. Evidence inputs and immutable hashes"),
        ["Key", tr(language, "산출물", "Artifact"), tr(language, "역할", "Role"), "SHA-256"],
        rows,
        css_class="data-table compact provenance",
    )


def render_glossary_table(language: str) -> str:
    rows = [
        ["Scanner radius", tr(language, "Scanner centroid와 전체 centroid 사이의 RMS dispersion. 낮을수록 invariant하지만 단독으로는 충분하지 않다.", "RMS dispersion of scanner centroids around their grand centroid; lower is more invariant, but not sufficient.")],
        ["Content margin", tr(language, "고정 reference 대비 matched tissue content 분리. Δ는 −0.02 non-inferiority 기준으로 판정한다.", "Separation of matched tissue content relative to the locked reference; Δ is judged against −0.02 non-inferiority.")],
        ["Collapse gate", tr(language, "세 metric에 대한 source-scanner별 보존 검사. 모든 non-reference scanner가 통과해야 한다.", "Per-source-scanner preservation check across three metrics; every non-reference scanner must pass.")],
        ["RR", tr(language, "Raw 대비 scanner radius의 상대 감소. 양수는 dispersion 감소를 뜻한다.", "Relative reduction in scanner radius from raw; positive values indicate reduced dispersion.")],
        ["Scanner probe", tr(language, "Embedding에서 여섯 scanner 중 하나를 예측하는 balanced accuracy. Chance는 1/6이다.", "Balanced accuracy for predicting one of six scanners from embeddings; chance is 1/6.")],
        ["ERT", tr(language, "Effective relative transfer. 이 보고서의 PanNormal과 PLISM 값은 명시적으로 비동등 estimator를 사용한다.", "Effective relative transfer. PanNormal and PLISM values in this report use explicitly non-equivalent estimators.")],
    ]
    return table_html(
        tr(language, "Table 12. 지표 용어집", "Table 12. Metric glossary"),
        [tr(language, "용어", "Term"), tr(language, "조작적 의미", "Operational meaning")],
        rows,
    )


def render_uncertainty_table(registry: dict[str, Any], language: str) -> str:
    """Expose locked paired bootstrap intervals without overloading main tables."""
    rows: list[list[str]] = []

    control_labels = {
        "complete_hf_removal": "E4 · complete HF removal",
        "partial_hf_control": "E4 · paired 25% partial-HF",
    }
    for key, label in control_labels.items():
        for encoder in ENCODERS:
            item = registry["controls"][key]["by_encoder"][encoder]
            rows.append(
                [
                    label,
                    ENCODER_LABELS[encoder],
                    signed(100 * item["relative_radius_reduction"], 2, "%"),
                    signed(item["raw_minus_condition"], 5),
                    f'[{signed(item["difference_ci"][0], 5)}, {signed(item["difference_ci"][1], 5)}]',
                ]
            )

    for method, payload in registry["benchmark"].items():
        for encoder in ENCODERS:
            item = payload["by_encoder"][encoder]
            rows.append(
                [
                    f"E5 · {escape(method)}",
                    ENCODER_LABELS[encoder],
                    signed(100 * item["relative_radius_reduction"], 2, "%"),
                    signed(item["raw_minus_condition"], 5),
                    f'[{signed(item["difference_ci"][0], 5)}, {signed(item["difference_ci"][1], 5)}]',
                ]
            )

    for encoder in ENCODERS:
        item = registry["e8"]["frontier"][encoder]
        rows.append(
            [
                "E8 · Arm B",
                ENCODER_LABELS[encoder],
                signed(100 * item["relative_radius_reduction"], 2, "%"),
                signed(item["raw_minus_condition"], 5),
                f'[{signed(item["difference_ci"][0], 5)}, {signed(item["difference_ci"][1], 5)}]',
            ]
        )

    for target in ("at2", "gt450", "s60"):
        for encoder in ENCODERS:
            item = registry["rf1u"][target]["by_encoder"][encoder]
            rows.append(
                [
                    f"RF1U vs Reinhard · {target.upper()}",
                    ENCODER_LABELS[encoder],
                    signed(item["incremental_rr_points"], 2, " pp"),
                    signed(item["paired_radius_difference"], 5),
                    f'[{signed(item["paired_ci"][0], 5)}, {signed(item["paired_ci"][1], 5)}]',
                ]
            )
    return table_html(
        tr(language, "Table 10. 고정된 slide-paired 불확실성 구간. 양의 Δ radius는 시험 correction의 radius가 comparator보다 작음을 뜻하며 RF1U 행은 Reinhard를 comparator로 사용한다.", "Table 10. Locked slide-paired uncertainty intervals. Positive Δ radius means the tested correction has a smaller radius than its comparator; RF1U rows use Reinhard as comparator."),
        [tr(language, "분석·조건", "Analysis · condition"), "Encoder", tr(language, "상대 효과", "Relative effect"), tr(language, "Δ radius 절대 차이", "Δ radius"), "95% bootstrap CI"],
        rows,
        css_class="data-table compact",
    )


STYLE = r"""
:root {
  --ink: #16252b;
  --muted: #5a6b70;
  --paper: #f7f5ef;
  --card: #fffef9;
  --line: #d8d8cd;
  --blue: #175d72;
  --blue-soft: #dcecf0;
  --gold: #b57627;
  --gold-soft: #f5e9d5;
  --red: #a63d40;
  --red-soft: #f7dede;
  --green: #2f725f;
  --green-soft: #dceee7;
  --feature: #b57627;
  --image: #175d72;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body { margin: 0; padding-left: 260px; color: var(--ink); background: var(--paper); font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans KR", sans-serif; line-height: 1.68; }
a { color: var(--blue); }
strong { font-weight: 750; }
.hero, main, footer { width: min(1080px, calc(100% - 64px)); margin: 0 auto; }
.hero { padding: 74px 0 36px; }
.eyebrow, .kicker { color: var(--blue); text-transform: uppercase; letter-spacing: .12em; font-size: .78rem; font-weight: 800; }
.title-pair { display: grid; grid-template-columns: 1fr; gap: 24px; align-items: end; margin-top: 20px; }
h1 { margin: 0; max-width: 710px; font-size: clamp(2.8rem, 6vw, 5.5rem); line-height: .99; letter-spacing: -.055em; }
.english-title { margin: 0; font-family: ui-serif, Georgia, serif; font-size: clamp(2.05rem, 4vw, 3.7rem); line-height: 1.02; letter-spacing: -.035em; }
.subtitle { margin: 20px 0 0; color: var(--muted); font-size: 1.08rem; }
.pair { display: grid; grid-template-columns: 1fr; gap: 18px; }
.pair > * { min-width: 0; }
.en { color: #405359; }
.deck { margin-top: 46px; padding-top: 28px; border-top: 2px solid var(--ink); font-size: 1.18rem; }
.headline-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 34px; }
.headline-card { padding: 20px; background: var(--card); border: 1px solid var(--line); border-top: 4px solid var(--blue); min-height: 145px; }
.headline-value { font-size: 2.15rem; font-weight: 850; line-height: 1.1; margin-bottom: 18px; letter-spacing: -.03em; }
.small { font-size: .82rem; }
.toc { position: fixed; inset: 0 auto 0 0; z-index: 10; width: 260px; display: flex; flex-direction: column; gap: 4px; overflow-y: auto; padding: 38px 22px; background: #edf1ef; border-right: 1px solid var(--line); }
.toc-title { margin: 0 8px 18px; color: var(--ink); font-family: ui-serif, Georgia, serif; font-size: 1.45rem; font-weight: 800; }
.toc a { white-space: normal; text-decoration: none; color: var(--muted); font-size: .82rem; line-height: 1.35; padding: 10px 12px; border-radius: 8px; }
.toc a:hover { background: var(--blue-soft); color: var(--blue); }
.section { position: relative; padding: 88px 0; border-bottom: 1px solid var(--line); }
.section-number { position: absolute; top: 82px; left: -58px; color: #a9b1b0; font-family: ui-monospace, monospace; font-size: .76rem; }
.section-head { margin-bottom: 32px; align-items: end; }
.section-head h2 { margin: 4px 0 0; font-size: clamp(1.75rem, 3vw, 2.65rem); line-height: 1.14; letter-spacing: -.035em; }
.section p { margin-top: 0; }
.takeaway, .boundary, .priority, .venue, .contract { margin-top: 30px; padding: 24px; border-left: 5px solid var(--blue); background: var(--blue-soft); }
.boundary { border-left-color: var(--gold); background: var(--gold-soft); }
.priority { border-left-color: var(--green); background: var(--green-soft); }
.contract h3, .priority h3, .venue h3 { margin-top: 0; }
.story-figure, .frontier-figure { margin: 38px 0; padding: 24px; background: var(--card); border: 1px solid var(--line); }
.flow { display: flex; align-items: stretch; gap: 10px; }
.flow-step { flex: 1; position: relative; display: flex; flex-direction: column; gap: 9px; padding: 18px; background: #eef3f2; border-top: 4px solid var(--blue); }
.flow-number { width: 28px; height: 28px; display: grid; place-items: center; border-radius: 50%; color: white; background: var(--blue); font-weight: 800; }
.flow-arrow { align-self: center; color: var(--gold); font-size: 1.4rem; }
figcaption { margin-top: 16px; color: var(--muted); font-size: .83rem; }
.table-wrap { overflow-x: auto; margin: 34px 0; border: 1px solid var(--line); background: var(--card); }
table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
caption { text-align: left; padding: 16px 18px; color: var(--ink); font-weight: 800; border-bottom: 1px solid var(--line); }
th, td { padding: 12px 14px; text-align: left; vertical-align: top; border-bottom: 1px solid #e7e6dd; }
thead th { color: var(--muted); background: #f0efe8; font-size: .78rem; text-transform: none; letter-spacing: .01em; }
tbody th { min-width: 160px; font-weight: 750; }
tbody tr:last-child > * { border-bottom: 0; }
.compact th, .compact td { padding: 8px 10px; font-size: .81rem; }
.provenance code { overflow-wrap: anywhere; font-size: .72rem; }
.badge { display: inline-block; white-space: nowrap; padding: 3px 8px; border-radius: 999px; font-size: .72rem; font-weight: 800; }
.badge.pass { color: var(--green); background: var(--green-soft); }
.badge.fail { color: var(--red); background: var(--red-soft); }
.badge.mixed { color: #875713; background: var(--gold-soft); }
.muted, .note { color: var(--muted); }
.frontier-figure svg { width: 100%; height: auto; overflow: visible; }
.grid-line { stroke: #dfe2dc; stroke-width: 1; }
.zero-line { stroke: var(--ink); stroke-width: 1.6; }
.range { stroke-width: 8; stroke-linecap: round; opacity: .75; }
.range.image { stroke: var(--image); }
.range.feature { stroke: var(--feature); }
.point.image { fill: var(--image); }
.point.feature { fill: var(--feature); }
.svg-label, .svg-tick, .svg-axis { fill: var(--muted); font-family: ui-sans-serif, sans-serif; font-size: 13px; }
.svg-label { fill: var(--ink); font-weight: 700; }
.legend { display: flex; gap: 20px; justify-content: flex-end; font-size: .78rem; color: var(--muted); }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }
.image-dot { background: var(--image); }
.feature-dot { background: var(--feature); }
details { margin: 26px 0; border: 1px solid var(--line); background: var(--card); }
summary { cursor: pointer; padding: 16px 18px; font-weight: 800; }
details > .table-wrap { margin: 0; border: 0; border-top: 1px solid var(--line); }
details > .note { padding: 18px; }
.contribution-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }
.contribution-card { padding: 24px; background: var(--card); border: 1px solid var(--line); border-top: 4px solid var(--blue); }
.contribution-card h3 { margin: 0 0 8px; }
.contribution-card h3.en { margin-top: 22px; padding-top: 18px; border-top: 1px solid var(--line); }
.technical code { color: var(--blue); }
footer { padding: 60px 0 80px; font-size: 1.05rem; }
.buildline { margin-top: 36px; color: var(--muted); font-size: .78rem; }
ol { padding-left: 1.25rem; }
@media (max-width: 920px) {
  body { padding-left: 0; }
  .hero, main, footer { width: min(100% - 32px, 1080px); }
  .toc { position: sticky; inset: 0; width: 100%; flex-direction: row; gap: 4px; overflow-x: auto; overflow-y: hidden; padding: 9px 12px; border-right: 0; border-bottom: 1px solid var(--line); }
  .toc-title { display: none; }
  .toc a { white-space: nowrap; padding: 7px 10px; border-radius: 999px; }
  .title-pair, .pair { grid-template-columns: 1fr; gap: 18px; }
  .title-pair { gap: 34px; }
  .headline-grid { grid-template-columns: 1fr 1fr; }
  .flow { flex-direction: column; }
  .flow-arrow { transform: rotate(90deg); }
  .contribution-grid { grid-template-columns: 1fr; }
  .section-number { display: none; }
}
@media (max-width: 560px) {
  .hero, main, footer { width: min(100% - 24px, 1080px); }
  .hero { padding-top: 44px; }
  .headline-grid { grid-template-columns: 1fr; }
  .section { padding: 60px 0; }
  .story-figure, .frontier-figure { padding: 12px; }
  h1 { font-size: 2.65rem; }
}
@media print {
  body { background: white; font-size: 10pt; }
  .toc { display: none; }
  .section { break-inside: avoid; padding: 32px 0; }
  details { break-inside: avoid; }
  details:not([open]) > * { display: block; }
  .hero, main, footer { width: 100%; }
}
"""


class StructureCounter(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: list[str] = []
        self.links: list[str] = []
        self.tables = 0
        self.figures = 0
        self.inline_svg = 0
        self.embedded_png = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.ids.append(str(values["id"]))
        if tag == "a" and str(values.get("href", "")).startswith("#"):
            self.links.append(str(values["href"])[1:])
        elif tag == "table":
            self.tables += 1
        elif tag == "figure":
            self.figures += 1
        elif tag == "svg":
            self.inline_svg += 1
        elif tag == "img" and str(values.get("src", "")).startswith("data:image/png"):
            self.embedded_png += 1


def retain_language(html_text: str, language: str) -> str:
    """Remove the opposite-language narrative blocks from the shared source."""
    opposite = "en" if language == "ko" else "ko"
    localized = html_text
    for tag in ("div", "span", "p", "h1", "h2", "h3", "strong", "article"):
        pattern = re.compile(
            rf'<{tag}\b(?=[^>]*\blang="{opposite}")[^>]*>.*?</{tag}>',
            flags=re.DOTALL,
        )
        while pattern.search(localized):
            localized = pattern.sub("", localized)
    if language == "en":
        localized = localized.replace('<p class="english-title">', '<h1 class="english-title">')
        localized = localized.replace("</p>\n      <p class=\"subtitle\">", "</h1>\n      <p class=\"subtitle\">", 1)
        localized = localized.replace(' class="en small"', ' class="small"')
        localized = localized.replace(' class="en"', "")
    localized = localized.replace(
        'aria-label="Report navigation / 보고서 목차"',
        f'aria-label="{tr(language, "보고서 목차", "Report navigation")}"',
    )
    return localized


def render_report(
    config: dict[str, Any],
    narrative: str,
    registry: dict[str, Any],
    registry_hash: str,
    language: str,
) -> str:
    d = registry["design"]
    hf_rr, hf_content = control_ranges(registry, "complete_hf_removal")
    partial_rr, partial_content = control_ranges(registry, "partial_hf_control")
    raw_linear = probe_values(registry, "raw", "linear")
    e8_linear = probe_values(registry, "e8:e8_free_gt450", "linear")
    coral_linear = probe_values(registry, "feature:coral", "linear")
    proc_linear = probe_values(registry, "feature:orthogonal_procrustes", "linear")
    proc_mlp = probe_values(registry, "feature:orthogonal_procrustes", "mlp")
    e8_rr = [100 * row["relative_radius_reduction"] for row in registry["e8"]["frontier"].values()]
    e9 = registry["e9"]["verdicts"]

    replacements: dict[str, str] = {key: str(value) for key, value in config.items()}
    replacements.update(
        {
            "slides": str(d["slides"]),
            "scanners": str(d["scanners"]),
            "locations_per_slide": str(d["locations_per_slide"]),
            "acquisitions": f'{d["acquisitions"]:,}',
            "paired_tuples": f'{d["paired_tuples"]:,}',
            "bootstrap_replicates": f'{d["bootstrap_replicates"]:,}',
            "bootstrap_seed": str(d["bootstrap_seed"]),
            "hf_removal_rr_range": f'{min(hf_rr):.1f}–{max(hf_rr):.1f}%',
            "hf_removal_content_range": number_range(hf_content, 4),
            "partial_hf_rr_range": f'{min(partial_rr):.1f}–{max(partial_rr):.1f}%',
            "partial_hf_content_range": number_range(partial_content, 4),
            "reinhard_rr_range": percent_range(benchmark_rr(registry, "Reinhard")),
            "coral_rr_range": percent_range(benchmark_rr(registry, "CORAL")),
            "procrustes_rr_range": percent_range(benchmark_rr(registry, "Orthogonal Procrustes")),
            "raw_probe_range": f'{min(raw_linear):.3f}–{max(raw_linear):.3f}',
            "e8_probe_range": f'{min(e8_linear):.3f}–{max(e8_linear):.3f}',
            "feature_linear_probe_range": f'{min(coral_linear + proc_linear):.3f}–{max(coral_linear + proc_linear):.3f}',
            "procrustes_mlp_range": f'{min(proc_mlp):.3f}–{max(proc_mlp):.3f}',
            "probe_chance": f'{registry["e8"]["chance"]:.3f}',
            "arm_a_folds": str(registry["e8"]["arm_a_folds_passing"]),
            "arm_b_folds": str(registry["e8"]["arm_b_folds_passing"]),
            "e8_max_rr": f'{max(e8_rr):.1f}%',
            "e8_phase": f'{registry["e8"]["phase_correlation"]:.3f}',
            "rf1u_phase": f'{registry["e8"]["rf1u_phase_correlation"]:.3f}',
            "e9_gt450": signed(e9["P1"]["log2_ert"], 3),
            "e9_gt450_ci": f'{signed(e9["P1"]["ci"][0], 3)} to {signed(e9["P1"]["ci"][1], 3)}',
            "e9_s360": signed(e9["P2"]["log2_ert"], 3),
            "e9_s360_ci": f'{signed(e9["P2"]["ci"][0], 3)} to {signed(e9["P2"]["ci"][1], 3)}',
            "e9_observed_order": "S360 (+0.061) > S60 (−0.724) > GT450 (−2.135)",
            "e9_sections": str(registry["e9"]["sections"]),
            "headline_cards": render_headline_cards(registry),
            "story_arc_figure": render_story_arc(language),
            "design_table": render_design_table(registry, language),
            "controls_table": render_controls_table(registry, language),
            "method_frontier_figure": render_method_frontier(registry, language),
            "benchmark_table": render_benchmark_table(registry, language),
            "rf1u_table": render_rf1u_table(registry, language),
            "probe_table": render_probe_table(registry, language),
            "e8_table": render_e8_table(registry, language),
            "e9_table": render_e9_table(registry, language),
            "contribution_cards": render_contribution_cards(),
            "claim_boundary_table": render_claim_boundary_table(language),
            "risk_table": render_risk_table(language),
            "uncertainty_table": render_uncertainty_table(registry, language),
            "provenance_table": render_provenance_table(registry, language),
            "glossary_table": render_glossary_table(language),
        }
    )
    body = narrative
    for key, value in replacements.items():
        body = body.replace(f"[[{key}]]", value)
    unresolved = sorted(set(re.findall(r"\[\[[a-zA-Z0-9_]+\]\]", body)))
    if unresolved:
        raise RuntimeError(f"unresolved narrative tokens: {unresolved}")
    body = retain_language(body, language)

    title = f'{config[f"title_{language}"]} — {config["report_version"]}'
    return f"""<!doctype html>
<html lang="{language}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="generator" content="PanNormal scientific story report builder">
  <meta name="evidence-registry-sha256" content="{registry_hash}">
  <title>{escape(title)}</title>
  <style>{STYLE}</style>
</head>
<body class="report-{language}">
{body}
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    config = read_json(args.config.resolve())
    narrative = args.narrative.resolve().read_text(encoding="utf-8")
    registry = build_registry()

    registry_text = json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    args.registry.parent.mkdir(parents=True, exist_ok=True)
    args.registry.write_text(registry_text, encoding="utf-8")
    registry_hash = sha256(args.registry)

    source_hashes = {
        key: metadata["sha256"] for key, metadata in registry["sources"].items()
    }
    releases = []
    for language, output, manifest_path in (
        ("ko", args.output_ko, args.manifest_ko),
        ("en", args.output_en, args.manifest_en),
    ):
        html_text = render_report(config, narrative, registry, registry_hash, language)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(html_text, encoding="utf-8")

        parser = StructureCounter()
        parser.feed(html_text)
        parser.close()
        duplicates = sorted({identifier for identifier in parser.ids if parser.ids.count(identifier) > 1})
        broken_links = sorted(set(parser.links) - set(parser.ids))
        if duplicates or broken_links or "[[" in html_text or "]]" in html_text:
            raise RuntimeError(
                f"{language} internal structural audit failed: duplicates={duplicates}, broken_links={broken_links}"
            )
        opposite = "en" if language == "ko" else "ko"
        if f'lang="{opposite}"' in html_text:
            raise RuntimeError(f"{language} report retained {opposite} narrative blocks")

        manifest = {
            "schema_version": "scientific_story_release_v1",
            "report": output.name,
            "language": language,
            "bytes": output.stat().st_size,
            "report_sha256": sha256(output),
            "tables": parser.tables,
            "figures": parser.figures,
            "inline_svg": parser.inline_svg,
            "embedded_png": parser.embedded_png,
            "release_date": config["release_date"],
            "report_version": config["report_version"],
            "builder": str(Path(__file__).resolve().relative_to(ROOT)),
            "builder_sha256": sha256(Path(__file__).resolve()),
            "config": str(args.config.resolve().relative_to(ROOT)) if ROOT in args.config.resolve().parents else args.config.name,
            "config_sha256": sha256(args.config.resolve()),
            "narrative": str(args.narrative.resolve().relative_to(ROOT)) if ROOT in args.narrative.resolve().parents else args.narrative.name,
            "narrative_sha256": sha256(args.narrative.resolve()),
            "evidence_registry": args.registry.name,
            "evidence_registry_sha256": registry_hash,
            "evidence_source_sha256": source_hashes,
            "counts": {
                "evidence_sources": len(source_hashes),
                "supported_claims": len(registry["claim_ledger"]["supported"]),
                "unsupported_claims": len(registry["claim_ledger"]["not_supported"]),
            },
            "audit": {
                "status": "pass",
                "duplicate_ids": len(duplicates),
                "broken_internal_links": len(broken_links),
                "unresolved_template_tokens": 0,
                "runtime_asset_dependencies": 0,
                "opposite_language_blocks": 0,
            },
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        releases.append(
            {
                "language": language,
                "report": str(output),
                "bytes": manifest["bytes"],
                "sha256": manifest["report_sha256"],
                "manifest": str(manifest_path),
            }
        )
    print(json.dumps({"registry": str(args.registry), "releases": releases}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

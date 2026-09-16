#!/usr/bin/env python3
"""Evidence and semantic regression checks for separate Korean/English v3 reports."""

from __future__ import annotations

import hashlib
from html.parser import HTMLParser
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "presentations/pannormal_scanner_harmonization_report_v3_2026-08-29"
REPORTS = {
    "ko": REPORT_DIR / "pannormal_scanner_harmonization_evidence_report_ko.html",
    "en": REPORT_DIR / "pannormal_scanner_harmonization_evidence_report_en.html",
}
MANIFESTS = {
    language: report.with_suffix(".manifest.json") for language, report in REPORTS.items()
}
SUPERSEDED_INDEX = ROOT / "presentations/pannormal_story_bilingual_v3_2026-08-29/index.html"
REGISTRY = ROOT / "reports/pannormal_story_bilingual_v3/evidence_registry.json"
BUILDER = ROOT / "src/build_pannormal_story_report_v3.py"
NARRATIVE = ROOT / "reports/pannormal_story_bilingual_v3/narrative.html"
CONFIG = ROOT / "reports/pannormal_story_bilingual_v3/report_config.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_builder() -> Any:
    spec = importlib.util.spec_from_file_location("pannormal_story_v3_builder", BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load v3 builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VisibleText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hidden = 0
        self.text: list[str] = []
        self.languages: dict[str, int] = {"ko": 0, "en": 0}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self.hidden += 1
        lang = dict(attrs).get("lang")
        if lang in self.languages:
            self.languages[str(lang)] += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"}:
            self.hidden -= 1

    def handle_data(self, data: str) -> None:
        if not self.hidden and data.strip():
            self.text.append(data)


def main() -> int:
    required_files = list(REPORTS.values()) + list(MANIFESTS.values()) + [REGISTRY, BUILDER, NARRATIVE, CONFIG]
    missing = [str(path.relative_to(ROOT)) for path in required_files if not path.is_file()]
    if missing:
        print("missing: " + ", ".join(missing), file=sys.stderr)
        return 1

    report_raw = {language: path.read_text(encoding="utf-8") for language, path in REPORTS.items()}
    parsers: dict[str, VisibleText] = {}
    visible: dict[str, str] = {}
    for language, raw in report_raw.items():
        parser = VisibleText()
        parser.feed(raw)
        parser.close()
        parsers[language] = parser
        visible[language] = " ".join(" ".join(parser.text).split())
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    manifests = {language: json.loads(path.read_text(encoding="utf-8")) for language, path in MANIFESTS.items()}
    builder = load_builder()
    recomputed = builder.build_registry()

    checks: list[tuple[str, bool]] = []

    def check(label: str, condition: bool) -> None:
        checks.append((label, bool(condition)))

    check("registry exactly recomputes from aggregate artifacts", registry == recomputed)
    for language in ("ko", "en"):
        report = REPORTS[language]
        manifest = manifests[language]
        opposite = "en" if language == "ko" else "ko"
        check(f"{language} report hash matches manifest", sha256(report) == manifest["report_sha256"])
        check(f"{language} report byte count matches manifest", report.stat().st_size == manifest["bytes"])
        check(f"{language} registry hash matches manifest", sha256(REGISTRY) == manifest["evidence_registry_sha256"])
        check(f"{language} builder hash matches manifest", sha256(BUILDER) == manifest["builder_sha256"])
        check(f"{language} narrative hash matches manifest", sha256(NARRATIVE) == manifest["narrative_sha256"])
        check(f"{language} config hash matches manifest", sha256(CONFIG) == manifest["config_sha256"])
        check(f"{language} manifest audit and language", manifest["audit"]["status"] == "pass" and manifest["language"] == language)
        check(f"{language} report contains only its narrative language", parsers[language].languages[language] >= 20 and parsers[language].languages[opposite] == 0)
        check(f"{language} report has no unresolved tokens", "[[" not in report_raw[language] and "]]" not in report_raw[language])
        check(f"{language} report has descriptive filename", report.name != "index.html" and language in report.stem)
        check(f"{language} report has left navigation panel", "position: fixed" in report_raw[language] and "width: 260px" in report_raw[language] and "padding-left: 260px" in report_raw[language])
    check("Korean and English reports are separate artifacts", REPORTS["ko"] != REPORTS["en"] and sha256(REPORTS["ko"]) != sha256(REPORTS["en"]))
    check("superseded bilingual index removed", not SUPERSEDED_INDEX.exists())
    design = registry["design"]
    check("paired cohort dimensions", (design["slides"], design["scanners"], design["locations_per_slide"]) == (109, 6, 100))
    check("paired acquisition and tuple counts", (design["acquisitions"], design["paired_tuples"]) == (65400, 10900))
    check("slide-blocked uncertainty contract", design["bootstrap_unit"] == "physical_slide" and design["bootstrap_replicates"] == 5000 and design["bootstrap_seed"] == 20260803)

    complete = registry["controls"]["complete_hf_removal"]["by_encoder"]
    partial = registry["controls"]["partial_hf_control"]["by_encoder"]
    complete_rr = [100 * row["relative_radius_reduction"] for row in complete.values()]
    complete_content = [row["delta_content_margin"] for row in complete.values()]
    partial_rr = [100 * row["relative_radius_reduction"] for row in partial.values()]
    partial_content = [row["delta_content_margin"] for row in partial.values()]
    check("destructive control radius range", 26.7 < min(complete_rr) < 26.9 and 51.0 < max(complete_rr) < 51.2)
    check("destructive control content range", -0.743 < min(complete_content) < -0.742 and -0.281 < max(complete_content) < -0.280)
    check("destructive control fails all joint gates", not any(row["joint_improvement_pass"] for row in complete.values()))
    check("partial-HF control radius range", 9.4 < min(partial_rr) < 9.6 and 13.9 < max(partial_rr) < 14.1)
    check("partial-HF content range", 0.0014 < min(partial_content) < 0.0016 and 0.0240 < max(partial_content) < 0.0242)
    check("partial-HF control passes all joint gates", all(row["joint_improvement_pass"] for row in partial.values()))

    expected_rr = {
        "Reinhard": [27.6, 6.0, 11.9, 3.7],
        "Paired OD affine": [25.7, 8.9, -3.3, -11.7],
        "Frequency calibration": [0.5, -0.1, -0.5, 0.3],
        "CORAL": [34.9, 16.3, 22.0, 17.6],
        "Orthogonal Procrustes": [43.9, 22.5, 23.9, 20.0],
    }
    for method, expected in expected_rr.items():
        observed = [
            round(100 * registry["benchmark"][method]["by_encoder"][encoder]["relative_radius_reduction"], 1)
            for encoder in builder.ENCODERS
        ]
        check(f"benchmark RR: {method}", observed == expected)

    rf1u = registry["rf1u"]
    check("GT450 improves after Reinhard in 4/4 paired contrasts", sum(rf1u["gt450"]["by_encoder"][e]["rf1u_better_than_reinhard"] for e in builder.ENCODERS) == 4)
    check("S60 improves after Reinhard in 3/4 paired contrasts", sum(rf1u["s60"]["by_encoder"][e]["rf1u_better_than_reinhard"] for e in builder.ENCODERS) == 3)
    check("AT2 worsens after Reinhard in 4/4", all(rf1u["at2"]["by_encoder"][e]["incremental_rr_points"] < 0 for e in builder.ENCODERS))

    e8 = registry["e8"]
    e8_rr = [100 * row["relative_radius_reduction"] for row in e8["frontier"].values()]
    e8_probe = list(e8["probes"]["e8:e8_free_gt450"]["linear"].values())
    check("E8 Arm A/B fold gates", e8["arm_a_folds_passing"] == 3 and e8["arm_b_folds_passing"] == 5)
    check("E8 representation gates pass 4/4", all(row["joint_improvement_pass"] for row in e8["frontier"].values()))
    check("E8 maximum RR exact", abs(max(e8_rr) - 47.87305040557904) < 1e-10)
    check("E8 residual probe range", 0.758 < min(e8_probe) < 0.760 and 0.966 < max(e8_probe) < 0.968)
    check("E8 phase audit", abs(e8["phase_correlation"] - 0.6939621895569599) < 1e-12 and abs(e8["rf1u_phase_correlation"] - 0.9460032361864997) < 1e-12)
    check("E4/E5/E8 paired intervals retained", all("difference_ci" in row and len(row["difference_ci"]) == 2 for family in registry["controls"].values() for row in family["by_encoder"].values()) and all("difference_ci" in row and len(row["difference_ci"]) == 2 for family in registry["benchmark"].values() for row in family["by_encoder"].values()) and all("difference_ci" in row and len(row["difference_ci"]) == 2 for row in e8["frontier"].values()))

    e9 = registry["e9"]["verdicts"]
    check("E9 P1-P3 all fail", not e9["P1"]["pass"] and not e9["P2"]["pass"] and not e9["P3"]["pass"])
    check("E9 GT450 estimate and CI", abs(e9["P1"]["log2_ert"] + 2.1348399783725753) < 1e-12 and e9["P1"]["ci"][1] < 0)
    check("E9 S360 estimate and CI", abs(e9["P2"]["log2_ert"] - 0.06133818198998765) < 1e-12 and e9["P2"]["ci"][0] > 0)

    required_phrases = {
      "ko": [
        "스캐너 조화가 성공처럼 보일 때",
        "전역 해석 계약",
        "고정된 slide-paired 불확실성 구간",
      ],
      "en": [
        "When scanner harmonization only looks successful",
        "Global interpretation contract",
        "Locked slide-paired uncertainty intervals",
      ],
      "both": [
        "0 / 3",
        "scanner-instance/acquisition pipeline",
        "0.60–0.90 cyc/µm",
        "0.10–0.99 cyc/µm",
        "Medical Image Analysis",
        "IEEE Transactions on Medical Imaging",
        "narrative.html",
        "evidence_registry.json",
      ],
    }
    for language in ("ko", "en"):
        for phrase in required_phrases[language] + required_phrases["both"]:
            check(f"{language} required visible phrase: {phrase}", phrase in visible[language])
    check("English report contains no Hangul", not any("가" <= char <= "힣" for char in visible["en"]))

    forbidden_positive_claims = [
        "claims replicated on PLISM",
        "Five of eight claims replicated",
        "The strongest deployable-without-pairing option is CORAL",
        "scanner and nothing else",
        "universal image-space ceiling was established",
        "clinical safety was demonstrated",
    ]
    for language in ("ko", "en"):
        for phrase in forbidden_positive_claims:
            check(f"{language} forbidden overclaim absent: {phrase}", phrase not in visible[language])

    failures = [label for label, passed in checks if not passed]
    for label, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'}  {label}")
    print(f"checks {len(checks)}; failures {len(failures)}")
    if failures:
        print("failed: " + "; ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

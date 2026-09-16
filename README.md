# Scanner Spectrum

동일 조직의 multi-scanner H&E 영상에서 scanner-associated variation과 병리
foundation-model 표현 변화를 분석한 연구 저장소다. 핵심 결론은 scanner separability
감소만으로는 조화를 입증할 수 없고, paired evaluation과 content/representation fidelity
gate가 함께 필요하다는 것이다.

## Final release

- E0--E7: result-locked core
- E8: learned paired residual stress test
- E9: PLISM external robustness analysis; 고정 예측 P1--P3 모두 실패
- [국문 보고서](presentations/pannormal_scanner_harmonization_report_v3_2026-08-29/pannormal_scanner_harmonization_evidence_report_ko.html)
- [English report](presentations/pannormal_scanner_harmonization_report_v3_2026-08-29/pannormal_scanner_harmonization_evidence_report_en.html)
- 편집 source: `reports/pannormal_story_bilingual_v3/`

## Reproduction

```bash
sbatch --export=ALL,JOB_CONDA_PREFIX="$CONDA_PREFIX",JOB_WORKDIR="$PWD" scripts/report_v3_validate.sbatch
sbatch --export=ALL,JOB_CONDA_PREFIX="$CONDA_PREFIX",JOB_WORKDIR="$PWD" scripts/pannormal_core_results_audit.sbatch
```

보고서 검증은 임시 디렉터리에서 두 번 재생성한 뒤 국문·영문 HTML, release manifest와
evidence registry를 보존된 release와 byte 단위로 비교한다. Core audit은 8개 result lock,
Main Figure 1--6과 130개 artifact의 hash를 확인한다.

## Active code

```text
src/build_pannormal_story_report_v3.py   deterministic final-report builder
src/audit_pannormal_core_results.py      locked-result integrity gate
src/prenorm/                             reusable frequency, pairing and embedding library
scripts/_verify_report_v3.py             release semantic/hash checks
scripts/report_v3_validate.sbatch        report reproduction entrypoint
scripts/pannormal_core_results_audit.sbatch core-lock entrypoint
```

`docs/`에는 일반 설명 문서를 두지 않는다. 그 안의 8개 Markdown 파일은 최종 report와
core-lock hash가 직접 참조하는 immutable execution input이므로 이름과 내용을 유지한다.

## Artifact retention

`outputs/`에는 core-lock artifact와 최종 보고서의 직접 입력만 남긴다. 유지 기준과 정리
규모는 `reports/pannormal_story_bilingual_v3/output_retention_manifest.json`에 기록한다.
대용량 원본과 feature store는 `data/`, 최종 standalone 문서는 `presentations/`에 둔다.
로그, 임시 파일, 과거 실험 script/config/output은 보존하지 않으며 Git history의 tracked
파일만 복구 경로로 사용한다.

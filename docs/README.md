# Documentation

## Active

- [Manuscript storyline](storyline.md): 최종 핵심 주장, 논문 절별 내용,
  주장–실험–결과 매핑, validity gate와 실행 우선순위
- [Final study protocol](final_study_protocol.md): 최종 논문 스토리라인, 내부 근거,
  multi-PFM 실험, PLISM 외부 검증, 통계 및 실행 순서
- [E0 registration sentinel audit](e0_registration_sentinel_audit.md): current registered
  WSI와 existing VALIS rigid의 geometry/ERT 비교, 실패 원인과 population 후속 gate
- [E0 registration cohort audit](e0_registration_cohort_audit.md): current route의
  109-slide geometry, six-scanner tuple retention과 ERT sensitivity
- [E0 six-scanner feature manifest](e0_feature_manifest.md): 109 × 100 corrected centers,
  deterministic replacement, per-scanner integer offsets와 512 px FOV gate
- [E0 resampling provenance and alias audit](e0_resampling_alias_audit.md): historical
  interpolation/MPP provenance, frozen 2D alias gate와 native explicit-AA 결정
- [E0 primary native-AA pilot](e0_primary_native_aa_pilot.md): same-scanner native geometry
  recovery, per-location residual QC와 cohort 확장 gate
- [E0 native geometry cohort contract](e0_native_geometry_cohort.md): 65,400-row
  native-WSI manifest schema, frozen recovery gate와 targeted VALIS fallback 기준

`storyline.md`는 현재 논문의 과학적 방향과 claim architecture의 기준이고,
`final_study_protocol.md`는 세부 데이터·모델·구현 계약을 제공한다. 두 문서가
충돌하면 storyline의 최신 방향을 우선하고 protocol을 후속 갱신한다. 결과별 세부
manifest와 실행 로그는 별도 파일로 분리한다.

## Archived

- [2026-07-31 pre-final archive](archive/2026-07-31_pre_final/README.md): 초기
  canonicalization 방향, Exp-01/02 실행 로그, common-space feasibility 계획,
  이전 synthesis와 중단된 시도

아카이브 문서는 당시 판단의 provenance를 보존하지만 현재의 주장이나 실행 계약으로
사용하지 않는다.

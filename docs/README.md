# Canonical documentation

이 디렉터리는 현재 연구 주장, 재현 계약, locked result와 진행 중인 RF1 improvement에
필요한 문서만 유지한다. 초기 계획, superseded diagnostic/pilot, 과거 archive와 manuscript
working draft는 2026-08-03에 제거했으며 삭제 전 상태는
`checkpoints/prenorm_postcleanup_20260803.bundle`에서 복구할 수 있다.

## Coordination and manuscript basis

- [RF1 improvement handoff](2026-08-03_rf1_improvement_handoff.md): 현재 결과, 해석,
  live jobs, 다음 실행 순서와 cleanup history
- [Manuscript storyline](storyline.md): 최종 핵심 주장과 claim--experiment--result 구조
- [Final study protocol](final_study_protocol.md): 데이터, 분석, 통계와 실행 계약
- [Storyline completion audit](storyline_completion_audit.md): E0--E7 lock 및 artifact 완료 근거

`storyline.md`가 과학적 방향의 기준이고 `final_study_protocol.md`가 세부 구현 계약이다.
충돌 시 storyline을 우선하고 protocol을 갱신한다.

## E0--E3: population and feature basis

- [Six-scanner feature manifest](e0_feature_manifest.md): 109 × 100 corrected centers와
  scanner별 geometry gate
- [Native geometry cohort contract](e0_native_geometry_cohort.md): 65,400-row native-AA
  population과 targeted VALIS fallback
- [PFM execution contract](e0_pfm_contract.md): ResNet50, UNI v1, CONCH v1, Virchow2
  extraction contract
- [Scanner acquisition metadata](e0_scanner_acquisition_metadata.md): 654 native WSI header
  audit와 외부 확인이 필요한 metadata
- [Locked-results audit code](../src/audit_e0_e3_locked_results.py) 및
  [Figure builder](../src/build_e0_e3_main_figures.py): E0--E3 artifact 검증과 Figure 1--3 재현

## E4: physical controls

- [Pre-outcome decision record](e4_e7_decision_record.md): E4--E7 endpoint와 claim rule
- [Control-population contract](e4_control_population_contract.md)
- [Control-population results](e4_control_population_results.md)

## E5: comparators and RF1

- [Comparator execution contract](e5_comparator_execution_contract.md)
- [Comparator population results](e5_comparator_population_results.md)
- [Macenko supplement contract](e5_macenko_supplement_contract.md)
- [Macenko supplement results](e5_macenko_supplement_results.md)
- [Reinhard residual-frequency rationale](reinhard_frequency_extension.md)
- [RF1 execution contract](e5_reinhard_residual_frequency_contract.md)
- [RF1 locked results](e5_reinhard_residual_frequency_results.md)
- [RF1 visual comparison](e5_rf1_visual_comparison.md)
- [RF1 improvement pilot contract](e5_rf1_improvement_pilot_contract.md)
- [RF1M combined-candidate contract](e5_rf1m_combined_candidate_contract.md): multiscale +
  strict no-harm 후보와 Amendment 1
- [RF1M image-only results](e5_rf1m_combined_candidate_results.md): per-fold no-harm gate 실패로
  PFM 접근 차단

## E6--E7: heterogeneity and tissue evidence

- [Heterogeneity execution contract](e6_heterogeneity_execution_contract.md)
- [Baseline-spectrum predictor contract](e6_baseline_spectrum_predictor_contract.md)
- [Heterogeneity results](e6_heterogeneity_results.md)
- [Grouped tissue-probe contract](e7_tissue_probe_execution_contract.md)
- [Grouped tissue-probe results](e7_tissue_probe_results.md)
- [PanNormal core audit code](../src/audit_pannormal_core_results.py): E0--E7 locks와
  Main Figure 1--6 통합 drift gate

## Document boundary

- Generated figures, embeddings, result locks와 logs는 `outputs/` 및 `logs/`에 있고 Git
  문서가 아니다.
- 별도 manuscript draft나 DOCX를 이 repository에서 유지하지 않는다. 논문 작성 시
  canonical storyline, protocol, locked-results 문서에서 새로 생성한다.
- superseded 문서를 다시 active tree로 복원하지 않는다. 과거 판단이 필요할 때만 Git
  bundle을 별도 경로에 clone해 조회한다.

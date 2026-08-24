# Canonical documentation

이 디렉터리는 현재 연구 주장, 재현 계약, locked result와 진행 중인 RF1 improvement에
필요한 문서만 유지한다. 초기 계획, superseded diagnostic/pilot, 과거 archive와 manuscript
working draft는 2026-08-03에 제거했으며 삭제 전 상태는
`checkpoints/prenorm_postcleanup_20260803.bundle`에서 복구할 수 있다.

## Coordination and manuscript basis

- [Manuscript storyline](storyline.md): 최종 핵심 주장과 claim--experiment--result 구조.
  2026-08-23에 E8(learned paired baseline)과 E9(PLISM external validation)를 반영했고,
  과거 판본의 E8/E9 ID 혼선(구: E8=PLISM, E9=background)을 실행 계약 기준으로 정정했다
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

- [Pre-outcome decision record](e4_e7_decision_record.md): E4--E7 endpoint와 claim rule.
  FROZEN 문서이므로 본문을 수정하지 않고 **Amendment A (2026-08-23)**로 post-core 상태만
  덧붙였다 — 원문의 "PLISM 제외·후속 protocol로 연기"는 실제로 지켜진 결정 기록이다
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
- [RF1M slide-adaptive feasibility](e5_rf1m_slide_adaptive_feasibility.md): slide별 gain 추정은
  AKOYA/GT450/S60에서만 성립
- [RF1U multi-target contract](e5_rf1u_multitarget_contract.md): unpaired band-wise 조건과
  runtime equivalence amendment
- [RF1U multi-target results](e5_rf1u_multitarget_results.md): 타겟이 방법보다 결정적 —
  AT2 0/4, GT450·S60 7/8 safe+improved

## E6--E7: heterogeneity and tissue evidence

- [Heterogeneity execution contract](e6_heterogeneity_execution_contract.md)
- [Baseline-spectrum predictor contract](e6_baseline_spectrum_predictor_contract.md)
- [Heterogeneity results](e6_heterogeneity_results.md)
- [Grouped tissue-probe contract](e7_tissue_probe_execution_contract.md)
- [Grouped tissue-probe results](e7_tissue_probe_results.md)
- [PanNormal core audit code](../src/audit_pannormal_core_results.py): E0--E7 locks와
  Main Figure 1--6 통합 drift gate

## E8: learned paired baseline

- [Paired residual execution contract](e8_paired_residual_contract.md): pix2pix 계열
  paired residual regression baseline. Arm A `gainfield`는 RF1U 위의 공간가변 band gain
  (보장 유지), arm B `free`는 무제약 OD residual = **ceiling probe**. §10에 §12 ceiling
  주장의 지지/반증 조건을 사전 등록했고, CycleGAN 제외 근거를 명시했다. **Amendment 1
  (2026-08-23)**: 미실행으로 남은 PLISM external validation의 범위를 사전 고정했다 — scanner
  교집합은 4개(`AT2, GT450, S360, S60`)뿐이라 zero-shot 경로만 external validation이고
  chance는 0.25, arm B는 ceiling probe이므로 "성공"은 probe가 **높게 유지**되는 것이며,
  encoder panel 불일치 때문에 두 panel을 모두 추출해야 한다.
- [Paired residual results](e8_paired_residual_results.md): §10 사전등록 판정 **supported** —
  arm B(무제약 학습 residual)는 4/4 PFM에서 content/collapse gate를 통과하고 ResNet50에서
  RR +47.87%로 패널 최고인데도 linear scanner probe가 0.759 아래로 내려가지 않았다. arm A
  `gainfield`는 per-source leg에서 3/5 fold만 통과해 encoding이 차단됐다. §5는 MLP probe에서
  Procrustes 0.614–0.778 vs CORAL 0.074–0.269로 두 feature 방법이 갈라짐을 기록한다.
- [E8 results digest](e8_results_digest.md): condition 계열(ceiling control, moment ladder,
  destination sweep, variance components, acquisition provenance)의 원자료
- 실행: [`build_e8_cache.py`](../src/build_e8_cache.py) →
  [`train_e8_residual.py`](../src/train_e8_residual.py) →
  [`audit_e8_residual.py`](../src/audit_e8_residual.py) (게이트+hallucination, encoding 차단) →
  [`extract_e8_residual_features.py`](../src/extract_e8_residual_features.py) →
  [`analyze_e8_frontier.py`](../src/analyze_e8_frontier.py)
- 검증: [`_verify_e8_results_doc.py`](../scripts/_verify_e8_results_doc.py)가 results 문서에
  인용된 82개 수치를 artifact와 대조하고,
  [`_verify_cross_doc.py`](../scripts/_verify_cross_doc.py)가 storyline/protocol/audit/README/
  report가 같은 E8·E9 수치에 동의하는지 35개 assertion으로 대조한다

## E9: PLISM external validation

- [Native ERT contract](e9_plism_native_ert_contract.md): interpolation/alignment audit,
  effective-transfer 명명 규칙, Amendment 1--3. Amendment 3이 core-grid 재구축과 함께
  Amendment 2의 per-patch `response >= 0.3` 권고를 철회하고 block-level exclusion으로 대체
- [Core-grid results](e9_plism_core_results.md): 116,831 location, 817,817 measurement,
  tissue coverage 88.2%, 46개 명명 tissue, 13 stain × 7 scanner, encoder panel UNI2-h /
  CONCHv1.5 / H-optimus-1. 8개 주장 중 5개 재현, 2건 정정(3항 분해 철회, CORAL 실패는
  추정 아티팩트). `src/build_e9_core_results.py`가 분석 출력에서 직접 생성한다
- 실행: `scripts/e9_core_results_doc.sbatch` (문서 생성은 login node OOM으로 SLURM 필수)
- 검증: [`_verify_conditions.py`](../scripts/_verify_conditions.py)가 2,177,672개 condition
  vector를 checkpoint contract/finiteness/frozen eval grid에 대조

## Document boundary

- Generated figures, embeddings, result locks와 logs는 `outputs/` 및 `logs/`에 있고 Git
  문서가 아니다.
- 별도 manuscript draft나 DOCX를 이 repository에서 유지하지 않는다. 논문 작성 시
  canonical storyline, protocol, locked-results 문서에서 새로 생성한다.
- superseded 문서를 다시 active tree로 복원하지 않는다. 과거 판단이 필요할 때만 Git
  bundle을 별도 경로에 clone해 조회한다.
- 2026-08-24 정리에서 제거한 기록과 그 이유, 대체 위치는
  [superseded records](superseded_records_20260824.md)에 있고 복구는
  `checkpoints/prenorm_postcleanup_20260824.bundle`에서 한다. 그 문서는 삭제된 발표본이
  담고 있던, 이후 반박된 두 주장(“scanner style은 분리 가능하다”, “feature-space 보정은
  확장성이 없다”)도 기록한다.

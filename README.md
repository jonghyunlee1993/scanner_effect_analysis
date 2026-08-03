# Scanner Spectrum

`scanner-spectrum`은 동일 조직을 여러 스캐너로 촬영한 H&E 영상에서 scanner
variation을 물리적 주파수 전달과 pathology foundation model(PFM) 표현 변화로
연결하는 연구 프로젝트다.

## 현재 논문 가설

스캐너 효과는 존재하지만, 모든 조직에 동일하게 적용되는 하나의 전역적인
`scanner style`은 아니다. 관측된 차이는 다음 세 요소의 합으로 보는 것이 더
정확하다.

1. 전역 affine으로 상당 부분 교정 가능한 저주파 색·조명 차이
2. 스캐너가 전달하거나 잃는 고주파 정보
3. scanner × tissue 및 scanner × slide 상호작용

따라서 저주파 외관을 맞추는 것은 가능하지만, 그것만으로 PFM 표현이 정렬되지는
않는다. 반대로 고주파를 제거해 scanner 분류가 어려워지는 것은 진정한 invariance가
아니라 생물학적 정보까지 사라진 representation collapse일 수 있다.

109개 내부 슬라이드의 PanNormal E0--E7 분석은 scanner main effect와 tissue/slide별
상호작용, four-PFM invariance--fidelity frontier, exact LOTO와 grouped tissue probe를
완료했다. White-space background는 tissue-domain transfer의 추가 설명력을 거의 제공하지
않았다. 여덟 result lock과 Main Figure 1--6은 통합 artifact audit을 통과했다. PLISM은
후속 extension으로 보류한다.

## 문서

- [최종 논문 스토리라인과 실행 청사진](docs/storyline.md)
- [최종 연구 프로토콜과 논문 스토리라인](docs/final_study_protocol.md)
- [RF1 improvement handoff](docs/2026-08-03_rf1_improvement_handoff.md)
- [문서 인덱스](docs/README.md)

현재 논문 주장과 서사는 `storyline.md`, 세부 실험 계약은
`final_study_protocol.md`를 기준으로 한다. 충돌하는 과학적 방향은 최신
`storyline.md`를 우선한다. superseded 문서와 manuscript working draft는 active tree에서
제거했으며 삭제 전 Git bundle에서만 조회한다.

## 활성 분석 코드

```text
src/prenorm/exp01/                       registered split와 주파수 분해
src/prenorm/exp02/                       LF affine, pairwise, HF trajectory
src/run_exp05_spectral_pilot.py          patch-level spectrum 추출
src/analyze_exp05_spectral_cohort.py     109-slide scanner spectrum 분석
src/analyze_exp05_tissue_random_slopes.py tissue/slide interaction 분석
src/run_exp06_raw_nps_pilot.py           raw background NPS 추출
src/finalize_exp06_raw_nps_pilot.py      NPS 산출물 정리
src/build_exp07_background_manifest.py   109 × 6 background job manifest
src/run_exp07_background_cell.py         raw white-space patch 추출
src/aggregate_exp07_background.py        background cohort 집계
src/analyze_exp08_background_mixed_effects.py background 증분 mixed model
scripts/exp05_*.sbatch                    tissue spectrum cohort jobs
scripts/exp06_*.sbatch                    raw NPS pilot jobs
scripts/exp07_*.sbatch                    background extraction jobs
scripts/exp08_*.sbatch                    background mixed-model jobs
presentations/scanner_drift_2026-07-28/   LF/HF 기전 발표 자료
```

## 데이터 계약

대용량 HDF5, parquet index, embedding, checkpoint와 결과물은 Git으로 추적하지
않고 로컬 `data/`, `outputs/`, `logs/`에서 관리한다.

Internal AT2/GT450/VERSA/Akoya lattice와 AT2/S60 lattice는 서로 다른 spatial
grid다. evaluation group에는 반드시 `lattice_id`를 포함하며, 서로 다른 lattice의
`(slide_id, tuple_id)`를 같은 registered location으로 취급하지 않는다. 모든 train,
validation, bootstrap split은 tile이 아니라 physical slide 단위로 수행한다.

## 현재 단계

PanNormal core E0--E7, Main Figure 1--6 및 결과 artifact 잠금은 완료됐다. 현재 작업은
저자·윤리·acquisition-record 정보와 공개 repository/release 정보를 채우는 원고 마감
단계다. 추가 PFM과 PLISM은 core 결론과 분리된 post-core extension이다.

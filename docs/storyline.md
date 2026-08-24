# Fidelity-constrained scanner harmonization: manuscript storyline

**Status:** PanNormal core E0--E7 completed and result-locked. Two post-core extensions have
since been executed and are no longer deferred: **E8**, the learned paired residual baseline,
whose §10 pre-registered ceiling reading returned `supported`; and **E9**, PLISM external
validation, rebuilt on a 116,831-location core grid with a 2026 encoder panel.

**Updated:** 2026-08-23

> **ID note.** Earlier revisions of this document used `E8` for the PLISM extension and `E9` for
> background characterization. The executed contracts renumbered them: **E8 = learned paired
> residual baseline** ([`e8_paired_residual_contract.md`](e8_paired_residual_contract.md)),
> **E9 = PLISM external validation**
> ([`e9_plism_native_ert_contract.md`](e9_plism_native_ert_contract.md)). Background
> characterization keeps its supplemental role but no longer carries an experiment ID. §6 and
> §5.3 below follow the executed numbering.

**Target journal:** *Medical Image Analysis*
**Working title:** *Fidelity-constrained evaluation of scanner harmonization in computational pathology: a paired frequency-resolved framework*

> 이 문서는 논문의 최종 서사, 주장–실험–결과의 연결, 본문 배치와 실행 우선순위를
> 정의한다. 세부 구현은 [`final_study_protocol.md`](final_study_protocol.md)를
> 참고하되, 두 문서가 충돌하면 현재의 과학적 방향은 이 문서를 우선한다.

**빠른 탐색:** 핵심 논리는 §1–4, 원고 절별 배치는 §5, 실험 매핑은 §6, 그림·표는
§7, 주장 사용 규칙은 §8–9, 실제 실행 순서와 stop/go 기준은 §10에 있다.

## 1. 한 문장 핵심 주장

**Scanner separability만으로는 representation alignment와 content collapse를 구별할 수
없으며, paired acquisition과 scanner-relevant frequency intervention을 이용해 scanner
invariance를 content-fidelity 제약 아래 평가해야 한다.**

주파수는 보존된 생물학적 정보량을 알려주는 유일한 축이 아니다. 본 연구에서 주파수를
사용하는 이유는 optical transfer, sampling/resampling, sharpening 및 block compression과
같은 주요 scanner-associated operator가 frequency-selective signature를 가지며, 개입
연산을 정확히 지정하고 출력에서 독립적으로 검증할 수 있기 때문이다.

## 2. 논문의 논리 구조

논문은 다음 순서로 진행한다.

```text
Scanner shift가 존재한다
        ↓
기존의 invariance-only 평가는 퇴화해를 허용한다
        ↓
paired data와 물리적으로 대응되는 control이 필요하다
        ↓
frequency-resolved intervention으로 collapse·alignment·divergence를 만든다
        ↓
scanner effect가 frequency와 content에 의존하는지 먼저 검증한다
        ↓
invariance–fidelity frontier에서 실제 correction을 평가한다
        ↓
fidelity를 만족한 조건에서만 invariance gain을 순위화한다
        ↓
      tissue heterogeneity와 representation/structural fidelity에서 결론을 검증한다
```

Part I의 scanner spectrum 분석은 독립적인 characterization story가 아니다. Part II의
평가 control과 correction benchmark가 물리적으로 타당한지를 뒷받침하는 전제다.

## 3. 동결된 원칙과 아직 동결하지 않은 선택

### 3.1 동결된 원칙

1. 핵심 명제는 “harmonization 자체가 식별 불가능하다”가 아니라
   **“scanner invariance만으로는 content fidelity에 대한 결론을 식별할 수 없다”**이다.
2. 2차원 invariance–fidelity frontier를 primary presentation으로 사용한다.
3. 단일 요약 지표는 fidelity gate를 통과한 방법에만 secondary ranking으로 보고한다.
4. 추론, bootstrap, cross-validation과 split의 기본 단위는 physical slide다.
5. Registration, resampling, aliasing과 noise floor는 nuisance appendix가 아니라 ERT
   해석의 validity gate다.
6. Native-AA ERT와 nested tissue/slide variance 결과는 E0 validity gate 이후의 audited
   primary 값으로 사용한다. Historical ERT와 85.5% descriptive fraction은 legacy
   provenance로만 보존한다.
7. 실제 correction benchmark에는 image-space와 feature-space 방법을 모두 포함한다.
8. Protocol의 유용성은 separability-only 결론과 fidelity-constrained decision의
   불일치로 평가한다. 특정 rank reversal을 사후에 찾지 않는다.
9. Core PFM panel은 **ResNet50, UNI v1, CONCH v1, Virchow2**로 동결한다. 추가 PFM은
   PanNormal core 분석 완료 후 확장한다.
10. 분석 위치는 현재 선정된 **109 slides × 100 locations**를 유지하고, 기존의 5개
    20-patch replicate를 sampling uncertainty 단위로 유지한다.
11. 모델마다 같은 biological center를 사용하되 training contract에 맞는 native physical
    FOV를 사용한다. 모든 모델에 동일한 256 px crop을 강제하지 않는다.
12. TRIDENT는 encoder, 공식 preprocessing과 checkpoint loading에 사용하고, paired crop
    정체성과 residual alignment는 이 연구의 manifest/DataLoader에서 통제한다.
13. 현재 zero-centered ±16 px local search는 paired feature extraction의 최종 정합으로
    사용하지 않는다. 다만 이 진단만으로 upstream WSI registration 전체를 폐기하지 않는다.
14. 현재 사용할 수 있는 biological label은 normal tissue type뿐이다. 따라서 PanNormal의
    hard gate를 `biological fidelity`라고 부르지 않고 **content/representation fidelity**로
    제한하며, tissue type은 secondary coarse-content endpoint로 사용한다.
15. ~~PLISM은 PanNormal core 연구의 완료 조건에서 제외하고 후속 frozen extension으로
    보류한다.~~ — 유지되었고, 그 extension이 2026-08-22/23에 **E9**로 실행됐다. Core의
    endpoint, margin, alignment contract와 correction benchmark를 먼저 동결한 뒤 시작한다는
    조건은 지켜졌다. 결과는 [`e9_plism_core_results.md`](e9_plism_core_results.md).

### 3.2 결과를 보기 전에 추가로 동결할 선택

아래 항목의 수식, 집계 단위, threshold와 leakage 방지 규칙은
[`e4_e7_decision_record.md`](e4_e7_decision_record.md)에 모았으며, 2026-08-03 사용자의
`전부 승인`으로 outcome 확인 전에 동결했다.

- ~~TRIDENT commit과 모델별 checkpoint revision/feature layer 동결~~ — commit
  `a6305acf`, ResNet50 `78f3ecfd`, UNI v1 `b55a5ec6`, CONCH v1 `f9ca9f87`,
  Virchow2 `31586458`; checkpoint SHA-256은 각각 `065b941a`, `56ef09b4`,
  `40a9644b`, `14244fba`. Pinned runtime의 A100 smoke test에서 공식 transform
  출력 224/224/448/224 px, feature 1,024/1,024/512/2,560-D와 bit-identical
  repeated eval을 모두 통과
- ~~Primary content-fidelity endpoint 하나와 collapse guardrail의 계층~~
- ~~Endpoint별 허용 margin과 slide-clustered uncertainty 기준~~
- ~~Scanner dispersion (R)의 정확한 정의와 raw-radius 최소값 \(\epsilon\)~~
- ~~Registration, model-FOV bounds와 aliasing audit의 수치 통과 기준~~ — SIFT inlier,
  reprojection, anisotropy, scale, NCC, ±120 boundary, strict 512 px native bounds와
  q05/q50/q95 alias ≤5% gate를 동결한 채 전수 통과
- ~~최소 image-space/feature-space comparator 목록~~
- ~~109-slide candidate replacement와 targeted native-rigid rerun의 수치 승격 기준~~ —
  geometry-only route-min, deterministic reserve와 unchanged all-100 gate로 완료

Self-cosine은 retrieval보다 비순환적인 representation-preservation diagnostic이지만,
그 자체를 biological fidelity라고 부르지 않는다. 아무 변화도 주지 않는 방법을 선호하고
유용한 nuisance removal도 벌점 줄 수 있으므로, primary content-fidelity endpoint와
collapse guardrail을 함께 사용하고 tissue-type 결과는 secondary evidence로 해석한다.
독립적인 biological/downstream label이 추가되기 전에는 biological non-inferiority를
주장하지 않는다.

## 4. 형식적 출발점: invariance-only 평가의 퇴화

### 4.1 본문에 넣을 명제

Embedding \(Z=g(X)\)와 scanner label \(S\)의 결합분포에만 의존하는 invariance functional
\(I(Z,S)\)를 생각한다. Scanner 정보를 완전히 제거하는 상수 사상 \(g(X)=c\)는 모든
scanner에서 동일한 embedding을 생성하므로 많은 invariance functional을 최대로 만든다.
그러나 이 사상은 content \(C\)에 대한 정보를 보존하지 않는다.

따라서 다음이 성립한다.

> Scanner invariance functional alone induces no ordering with respect to content
> fidelity; a degenerate constant representation can be optimal.

이 명제는 주파수를 필요로 하지 않는다. 주파수 intervention은 이 퇴화 가능성을 실제
영상과 PFM에서 경험적으로 드러내고, scanner acquisition mechanism에 대응되는 control
family를 제공한다.

### 4.2 피해야 할 표현

- “모든 harmonization은 비식별이다.”
- “주파수만이 정보량이 알려진 유일한 축이다.”
- “HF retention 0.5는 생물학적 정보를 절반 보존한다.”
- “Scanner separability 감소는 곧 harmonization 성공이다.”

## 5. 논문 목차와 절별 배치

### 5.1 Introduction

#### Paragraph 1 — 문제와 임상적 맥락

- WSI scanner는 광학, sampling, focus, sharpening, color pipeline과 compression을 통해
  동일 슬라이드에 장비별 acquisition signature를 만든다.
- PFM도 이 signature를 인코딩하며 cross-scanner prediction agreement와 calibration이
  변할 수 있다.
- 목적은 scanner sensitivity의 존재를 다시 증명하는 것이 아니라 correction을 어떻게
  평가해야 하는지 정하는 것이다.

#### Paragraph 2 — 기존 평가의 핵심 결손

- Scanner classifier accuracy, centroid radius와 domain separability는 invariance만 본다.
- Complete blur나 constant representation도 이러한 지표를 개선한다.
- 따라서 alignment와 collapse를 구별하려면 fidelity가 독립 축으로 필요하다.
- 위의 degeneracy proposition을 짧게 제시한다.

#### Paragraph 3 — 왜 paired frequency-resolved control인가

- Paired acquisition은 같은 biological content를 scanner 사이에서 고정한다.
- Optical transfer, sampling/resampling, sharpening과 block compression은 주파수 선택적
  signature를 가진다.
- Frequency intervention은 coefficient operation을 정확히 지정하고, attenuation,
  amplification, phase cancellation과 paired consensus라는 서로 다른 failure/control
  상태를 만들 수 있다.
- Frequency가 모든 scanner nuisance를 설명하거나 biological information을 직접
  정량화한다는 주장은 하지 않는다.

#### Paragraph 4 — 연구 설계와 기여

기여는 네 가지로 제한한다.

1. Invariance-only evaluation이 degenerate optimum을 허용함을 형식화한다.
2. Paired spectrum과 registration/resampling audit를 포함하는 frequency-resolved
   evaluation protocol을 제안한다.
3. Collapse, partial oracle와 divergence control이 정의하는 invariance–fidelity
   frontier에서 실제 correction을 평가한다.
4. 네 PFM과 tissue/slide strata에서 fidelity-constrained decision의 재현성과 content
   dependence를 검증한다.

### 5.2 Methods

#### 2.1 Paired cohort와 분석 단위

배치할 내용:

- 109 physical slides, 37 tissue types, 6 scanners
- AT2 reference와 scanner별 acquisition metadata
- slide–scanner당 100 paired locations, 5개의 20-patch replicate
- slide-blocked inference와 tissue hierarchy
- IRB, data governance, exclusion 및 missingness

#### 2.2 Common grid, registration과 resampling audit

이 절은 ERT보다 먼저 둔다.

- Output pixel이 AT2 reference와 동일한 물리 좌표를 나타내는지 검증
- Native MPP, target MPP, scale factor, transform family와 interpolation kernel
- TIFF/OME/OpenSlide metadata 불일치와 최종 frequency-axis source
- Slide-level rigid 또는 affine offset prior → 좁은 local integer refinement
- Scanner·slide별 shift distribution, boundary saturation, failure와 padding
- Registration audit가 Part I과 Part II에 미치는 영향 구분
- Old vs corrected registration의 paired sensitivity

정합 의사결정은 다음 순서로 동결한다.

1. 기존 registered WSI와 offset은 canonical center 후보, native→target geometry 복구와 QC에
   사용하되 primary RGB pixel source로 사용하지 않는다.
2. Original 100-location route gate가 실패하면 기존 common coordinate pool에서 512 px-aware
   deterministic replacement로 slide당 100개를 복구한다.
3. 각 scanner의 native WSI와 same-scanner historical output 사이의 global similarity/affine
   transform을 복구하고, frozen center마다 low-pass local residual을 추정한다.
4. 최종 crop은 native WSI에서만 읽어 `Lanczos3 reduction → residual bicubic affine`으로
   AT2 0.5052 µm/px grid에 직접 투영한다. Historical registered pixel은 reconstruction QC에만
   사용한다.
5. Global inlier, local NCC, search-boundary, 512 px bounds 또는 six-scanner completeness gate가
   실패한 scanner–slide cell만 fallback 대상으로 삼는다. 먼저 보존된 VALIS rigid geometry를
   같은 100개 center와 동일 gate로 감사하고, 전부 통과한 cell은 native→rigid transform을
   복구한 뒤 reconstructed-native↔rigid same-scanner local residual(NCC ≥0.75, ±120
   non-boundary)을 합성하되 RGB는 native WSI에서 직접 읽는다. 이 감사가 실패하거나 output이 없을 때만 해당
   cell의 VALIS rigid/affine를 native WSI부터 다시 수행한다. Strict native-FOV bound만 실패한
   location은 outcome-blind common-pool 순위의 다음 후보로 교체한다. Non-rigid 결과는
   interpolation 자체가 spectrum과 PFM feature에 미치는 영향을 별도로 통과하기 전에는
   primary로 쓰지 않는다.

현재 확인된 진단:

- AKOYA boundary saturation: 54.46%
- AKOYA 109 slides 중 61개에서 patch 절반 이상이 ±16 px 경계에 도달
- 슬라이드별 dominant edge fraction 중앙값: 79.2%
- 현재 Exp05의 zero-centered ±16 px search는 slide-coherent residual을 포착하지 못함
- 15-slide/15-tissue sentinel에서 current WSI + integer refinement는 slide–scanner cell
  64/75 (85.3%), 기존 VALIS rigid는 57/75 (76.0%)로 frozen route gate를 모두 통과하지 못함
- Current old→corrected high-band ERT의 median absolute delta는 0.00165 log2, q95는
  0.01667로 작았으나 current↔VALIS route delta는 median 0.26210 log2로 큼
- 따라서 existing VALIS를 cohort-wide로 승격하지 않으며, 109-slide candidate-pool
  replacement 후에도 실패하는 scanner–slide cell만 native rigid rerun 대상으로 삼음
- 109-slide current-route audit에서 slide–scanner cell 458/545 (84.0%)가 90/100 gate를
  통과함: GT450 93.6%, VERSA 77.1%, AKOYA 58.7%, S60 92.7%, S360 98.2%
- 원래 100개 중 exact six-scanner location은 slide median 84개(최소 13개)였으나 common
  coordinate pool은 slide당 1,040–31,892개이므로 outcome-blind replacement를 먼저 감사함
- 109-slide old→corrected high-band ERT median absolute delta는 scanner별
  0.00002–0.00384 log2였고 q95는 최대 0.08075(AKOYA)였음
- Common-pool replacement 후 109 slides × 100 = 10,900 locations 모두 six-scanner
  geometry와 shifted 512 px FOV를 통과함; original 8,203개 유지, 2,697개 교체
- Slide당 replacement 중앙값은 20개, 최대 89개였고 candidate 평가 최대값은 855/3,000으로
  native rigid rerun trigger에 도달한 slide는 없었음
- 보존된 96-slide VALIS rigid provenance에서 native→AT2 export는 VALIS 1.2.0/
  libvips 8.15.3 bicubic 단일 affine warp였고 explicit anti-alias prefilter는 없었음
- GT450/VERSA의 유효 native px/output px 중앙값은 1.922/1.842였으며, moving
  scanner 출력 TIFF에 native MPP를 다시 써서 AT2 grid metadata가 잘못된 export bug를
  cohort 수준에서 확인함
- Same-scanner native geometry recovery pilot에서 global SIFT similarity의 thumbnail reprojection
  median은 GT450 0.94 px, VERSA 0.19 px였고, frozen center별 local residual 적용 후 40개
  patch의 NCC median은 각각 0.9960/0.9962였음
- 단 GT450의 target-grid local residual magnitude q95가 51.1 px여서 global similarity 하나만
  primary transform으로 사용하는 것은 기각함. Per-location residual을 manifest에 저장하고
  boundary 실패 cell만 from-scratch VALIS로 승격함
- Native explicit-AA real-tissue pilot에서 historical bicubic 대비 high-band amplitude retention은
  GT450 0.773, VERSA 0.776이었고 low–mid는 0.998로 거의 유지됨. 이는 synthetic audit의
  alias 억제 비용과 방향이 일치하지만, 실제 cohort alias gate를 대신하지는 않음

주의:

- NCC threshold로 patch를 단순 제거하지 않는다. Scanner blur가 NCC를 낮추므로 selection
  bias가 생길 수 있다.
- Direction consistency는 slide-coherent displacement를 지지하지만 pure rigid translation을
  단독으로 증명하지 않는다. Constant와 affine prior를 비교한다.

#### 2.3 Resampling aliasing audit

Target grid가 0.5052 µm/px이면 sampling frequency는 1.9794 cycles/µm, Nyquist는
0.9897 cycles/µm다. 1차원에서 output 0.60–0.90 cycles/µm로 접히는 첫 mirror band는
약 1.079–1.379 cycles/µm다. GT450과 VERSA의 downsampling 경로를 우선 감사하되,
2D aliasing은 방향에 따라 더 넓은 source 영역을 혼합함을 명시한다.

합성 audit:

- 여러 방향과 위상의 sinusoid sweep
- 2D white/broadband noise 또는 zone plate
- 실제 source→reference transform과 동일한 interpolation chain
- Output frequency × input frequency mixing matrix
- 0.60–0.90 band의 alias-to-in-band ratio
- Passband attenuation, ringing 및 angular anisotropy
- 원 파이프라인과 explicit anti-aliased pipeline 비교

사전 동결 gate는 0.60–0.90 cycles/µm에서 sinusoid mixing과 broadband noise의
alias/true-in-band power가 모두 5% 이하이며 q05/q50/q95 실제 transform profile 전체가
통과해야 한다.

현재 audit 결과:

- 최종 native geometry 218개 GT450/VERSA transform의 q05/q50/q95 profile을 다시
  감사했다. 원 bicubic의 q50 alias/true-in-band power는 GT450에서 sinusoid
  1.545/white noise 1.115, VERSA에서 1.319/0.960으로 실패함
- `Lanczos3 reduction → residual bicubic affine`의 q50은 GT450 0.0148/0.0173,
  VERSA 0.0154/0.0169였고, 전체 q05/q50/q95의 최악값도 sinusoid 0.0154,
  white noise 0.0190으로 5% gate를 통과함
- Explicit-AA의 q50 high-band amplitude retention 중앙값은 GT450 0.775, VERSA
  0.766이고 전체 profile 최대 angular anisotropy는 3.639 dB이므로 이 필터링 비용을
  함께 보고함
- 따라서 native-AA common grid와 0.60–0.90 cycles/µm primary band는 E0c gate를
  통과했다. E1–E3는 이 grid에서만 재산출함

Alias term이 무시할 수 없으면 단일 multiplicative transfer 해석이 깨진다.

\[
P_Y(\mathbf f) \approx
\sum_{\mathbf k\in\mathbb Z^2}
|G(\mathbf f+\mathbf kF_s)|^2P_X(\mathbf f+\mathbf kF_s)+P_N(\mathbf f).
\]

이 경우 선택지는 다음 순서다.

1. Proper anti-aliasing을 포함해 common grid를 다시 생성한다.
2. Primary band를 aliasing이 허용 기준 이하인 구간으로 제한한다.
3. 재처리가 불가능하면 `scanner ERT` 대신
   `effective spectrum after registration/resampling`으로 명명한다.

#### 2.4 Effective relative transfer와 estimator sensitivity

- RGB→mean optical density, patch mean removal, 2D Hann, FFT
- 2D power average 후 radial summary
- \(\sqrt{P_s/P_{AT2}}\)와 0.03–0.10 cycles/µm anchor normalization
- Low–mid 0.10–0.30, mid 0.30–0.60, high 0.60–0.90의 현재 정의
- Unnormalized power ratio와 anchor-normalized ERT를 함께 보고
- 2D/angular spectrum을 resampling/compression audit에 보고
- Anchor-band sensitivity와 background/noise-floor sensitivity

Audited native-AA high-band fold 0.335와 1.478을 포함한 ERT는 anchor-normalized amplitude
transfer이며 절대 HF power ratio가 아니다. `amplification`, `attenuation`은 anchor 대비
spectral shape를 뜻한다고 명시한다. Historical 0.297과 1.595는 primary 결과로 사용하지 않는다.

기존 raw background NPS는 tissue registration과 다른 interpolation 경로를 사용했으므로
registered tissue power에서 직접 빼지 않는다. Noise correction은 동일한 processing chain의
background를 사용하고, subtraction 후 음수값과 SNR threshold sensitivity를 보고한다.

현재 E0d에서는 Exp07의 outcome-blind accepted native glass 좌표 65,400개를 사후 재선별 없이
최종 `Lanczos3 reduction → residual bicubic affine` 체인으로 다시 렌더링하고 E1과 동일한
natural-log mean OD/Hann/radial periodogram을 적용했다. 전체 47,088 radial bin에서 subtraction
후 음수는 0개였다. SNR threshold 1/2/5/10을 모두 보고하며, 이 결과는 detector NPS가 아니라
동일 전처리 이후의 operational background floor로만 해석한다.

#### 2.5 Scanner, tissue, slide와 sampling 분해

- Scanner fixed mean
- Tissue에 대한 scanner contrast의 random slope
- Tissue 안 slide에 대한 scanner contrast의 random slope
- 20-patch replicate sampling error
- Scanner subset, 특히 AKOYA 포함/제외 sensitivity
- Scanner main percentage는 고정된 장비 집합의 descriptive fraction임을 명시

#### 2.6 단일 blur–sharpen 축으로의 환원 가능성

- Scanner spectrum을 사전 정의한 AT2 blur/sharpen intervention manifold에 투영
- Equivalent gain과 off-axis residual을 분리
- 20-patch replicate covariance를 이용한 Mahalanobis lack-of-fit
- Held-out patch/slide cross-validation과 bootstrap CI

허용되는 결론:

> The predefined scalar blur–sharpen family does not explain the scanner spectrum
> family within sampling uncertainty.

모든 가능한 1차원 표현이 불가능하다고 주장하지 않는다.

#### 2.7 Invariance–fidelity evaluation protocol

Control family:

| Control | 역할 | 기대 위치 |
|---|---|---|
| Raw | 기준 | 원점 |
| Complete HF attenuation | collapse control | invariance↑, fidelity↓↓ |
| Global coefficient mean | phase-cancellation collapse control | invariance↑, fidelity↓↓ |
| Registered same-location HF mean | partial image-space oracle | invariance↑, fidelity 유지 가능 |
| Full paired target replacement | grouping/assay oracle | radius≈0, deployable하지 않음 |
| HF boost/sharpening | nuisance-amplification control | invariance↓, fidelity diagnostic 유지 가능 |

Sharpening은 content-fidelity endpoint를 통과하기 전까지 `content-preserving anti-control`이라고
부르지 않고 `nuisance-amplification` 또는 `divergence control`로 부른다.

#### 2.8 평가 대상 correction

최소 benchmark:

**Image-space primary:** Reinhard, paired OD affine, frozen frequency calibration

**Image-space Supplement:** Macenko

**Feature-space primary:** CORAL, orthogonal Procrustes

이 범위는 outcome 확인 전 decision record에서 동결했다. Learned normalization과 FEATMAP은
동일 paired-data/FOV/CV 조건을 공정하게 재현하는 별도 extension으로 남기고 core 결과에
사후 추가하지 않았다. 모든 core 방법은 train slide에서만 적합하고 held-out slide에서
평가했으며, exact LOTO sensitivity에서는 held-out tissue 전체를 fit에서 제외했다.

#### 2.9 PFM panel과 endpoint

Core panel은 다음 네 모델로 동결한다.

| PFM | TRIDENT encoder ID | Native input contract | Feature dim | 역할 |
|---|---|---:|---:|---|
| ResNet50 | `resnet50` | 256 px @ 20× (~129.3 µm) | 1,024 | Natural-image CNN baseline |
| UNI v1 | `uni_v1` | 256 px @ 20× (~129.3 µm) | 1,024 | Pathology SSL ViT |
| CONCH v1 | `conch_v1` | 512 px @ 20× (~258.7 µm) | 512 | Vision-language pathology model |
| Virchow2 | `virchow2` | 224 px @ 20× (~113.2 µm) | 2,560 | Large pathology SSL model |

ResNet50은 TRIDENT가 노출하는 1,024-dimensional feature contract를 사용하며 일반적인
최종 2,048-dimensional ResNet50 feature와 혼용하지 않는다. 실제 실행 전 TRIDENT commit,
checkpoint revision/hash, precision, normalization과 output layer를 frozen manifest에 기록한다.

Patch/feature extraction contract:

- 현재 `selected_patches.csv`의 100개 좌표를 256 px AT2 top-left가 아니라 canonical center
  \((x+128, y+128)\)로 해석한다.
- 모든 scanner와 PFM은 같은 canonical biological center를 공유한다.
- 각 PFM은 위 native input 크기를 사용한다. 따라서 CONCH v1의 FOV는 다른 세 모델보다
  넓으며, 결과는 모델 안의 raw-relative effect를 먼저 계산한 뒤 계층적으로 종합한다.
- 512 px FOV의 bounds, tissue coverage와 padding을 feature를 보기 전에 감사하고, 실패 위치는
  기존 common coordinate pool에서 결정론적으로 대체한다.
- 이 감사로 생성된 `e0_integer_512_v1` manifest를 모든 PFM/condition의 유일한 location
  source로 사용하고 legacy `selected_patches.csv`를 직접 읽지 않는다.
- TRIDENT의 slide segmentation/patch sampling을 다시 돌리지 않는다. 연구 DataLoader가
  corrected RGB crop을 만들고 TRIDENT encoder factory와 공식 eval transform을 호출한다.
- Embedding row는 `slide_id, tissue_type, scanner, location_id, replicate_id, center, FOV,
  alignment_version, image_condition, encoder/checkpoint`를 보존한다.

이 설계는 10,900 paired locations와 raw 기준 65,400 scanner crops를 만든다. 100 locations/
slide는 109개 physical slide에 걸친 population scanner effect와 5 × 20 replicate sampling
uncertainty에는 충분한 core design으로 간주한다. 반면 37개 tissue type의 class-balanced
supervised biology task를 보장하지는 않으므로 tissue-type endpoint는 secondary로 유지한다.

Invariance endpoint:

- Paired location-centered scanner dispersion \(R\)
- Scanner balanced accuracy
- Scanner centroid/radius
- Paired consensus gain
- Cross-scanner same-location retrieval은 paired alignment endpoint로 사용

Fidelity/collapse endpoint:

- Within-scanner raw→corrected self-cosine
- Neighborhood preservation
- Embedding variance, effective rank, pairwise distance
- Same-location content margin/retrieval
- Tissue-type retrieval 또는 grouped linear probe는 secondary coarse-content endpoint

#### 2.10 Frontier와 요약 통계

Primary result는 invariance와 fidelity의 2차원 Pareto/frontier다. 단일 지표는
secondary decision aid로만 사용한다.

Fidelity-constrained relative radius reduction은 다음과 같이 정의한다.

\[
\mathrm{FC\text{-}RR}_m = 1-\frac{R_m}{R_{raw}}.
\]

보고 조건:

- Primary content-fidelity endpoint의 사전 동결 허용 기준을 통과
- Collapse guardrail 통과
- \(R_{raw}\ge\epsilon\); 그렇지 않으면 N/A
- 음수값을 자르지 않음
- Full paired oracle은 분모가 아니라 assay positive control
- Slide-blocked bootstrap CI 보고
- PFM별 within-model ratio를 계산한 뒤 계층적으로 종합

여러 fidelity endpoint를 모두 hard equivalence gate로 두지 않는다. Primary 하나를
선택하고 slide-clustered uncertainty와 margin을 사전 지정한다. Secondary fidelity,
tissue-type 결과와 collapse endpoint는 계층적으로 해석한다. 이 판정은 biological
non-inferiority가 아니라 representation/content preservation 판정이다.

#### 2.11 Correction 효과의 content dependence

E2의 scanner × content와 E5의 benchmark를 연결하는 핵심 절이다.

- `scanner × correction × tissue` interaction
- Tissue에 대한 correction effect의 random slope
- Slide 안 paired contrast
- 독립 morphology annotation은 없으므로 사전 고정한 baseline spectrum 변수만 correction
  benefit의 exact tissue-held-out predictor로 분석
- Mean effect가 유사해도 tissue별 부호가 바뀌는지 보고

#### 2.12 Content fidelity와 제한된 tissue-type validation

PanNormal에는 normal tissue type 외의 독립 biological/downstream label이 없다. 따라서
현재 연구에서 직접 판정할 범위를 다음과 같이 제한한다.

1. **Primary representation/content endpoint:** same-location content margin을 outcome 확인
   전에 동결해 완료했다.
2. **Collapse guardrail:** embedding variance, effective rank와 pairwise-distance를 모든
   source scanner에서 평가했다.
3. **Secondary coarse content:** held-out-slide tissue-centroid probe와 scanner별
   tissue-neighborhood agreement를 완료했다.
4. **Structural sensitivity:** 독립 annotation/QC 계약이 없어 nucleus endpoint를 core에서
   생략했다.

37개 tissue class 중 상당수의 slide 수가 적으므로 patch를 독립 표본으로 세지 않는다.
Split과 bootstrap의 단위는 physical slide이며, tissue endpoint는 class imbalance와
minimum-slide sensitivity를 함께 보고한다. 별도의 labeled cohort가 추가되지 않으면
`biological fidelity`, `clinical validation` 또는 powered downstream non-inferiority를
주장하지 않는다.

#### 2.13 Background와 noise characterization

- 기존 65,400 white-space patch 결과는 supplemental characterization으로 이동
- Background covariate가 tissue transfer의 held-out prediction을 개선하지 않았다는
  결과를 간단히 보고
- Main role은 acquisition QC와 registered-domain noise-floor sensitivity
- Detector NPS, DQE 또는 absolute MTF로 부르지 않음

#### 2.14 PLISM external extension — 실행 완료 (E9)

PLISM은 PanNormal core 분석과 원고의 완료 조건이 아니었고, 그 순서는 지켜졌다. PanNormal의
endpoint, margin, alignment contract와 correction benchmark를 먼저 동결·완료한 뒤 별도
frozen extension으로 2026-08-22/23에 실행했다. 아래는 실행 전에 정한 원칙과 그 결과다.

| 사전 원칙 | 결과 |
|---|---|
| PanNormal outcome과 threshold를 PLISM 결과 확인 전에 동결 | 지켜짐. Amendment 3에서도 estimator/band/threshold/bootstrap 불변, cohort만 교체 |
| 한 stain condition의 scanner-only contrast를 primary로 지정 | 지켜짐 |
| Scanner × stain은 secondary | 지켜짐. Nesting을 두 방향(staining condition 상위 / tissue type 상위)으로 모두 보고 |
| Aligned tile group/core를 blocking unit으로 사용 | 지켜짐. Blocking 단위는 section과 core-grid location |
| HF 결과보다 interpolation audit를 먼저 제시 | 지켜짐. 비참조 measurement 700,986건, median residual 0.091–0.113 µm, 1 µm gate 99.57% |
| Native scan이 없으면 `effective transfer after PLISM preprocessing`으로 명명 | **발동하지 않음.** 91개 native WSI를 resampling 없이 native resolution으로 읽었다. 다만 tissue 위 effective transfer이지 MTF가 아니라는 제한은 유지 |

결과는 §5.3의 3.10과 [`e9_plism_core_results.md`](e9_plism_core_results.md)에 있다.

#### 2.15 통계와 재현성

- Slide-blocked bootstrap와 cross-validation
- Effect size와 CI 우선
- Primary family만 confirmatory testing
- Secondary family에 BH correction
- Correction ranking의 불확실성과 tissue/PFM interaction 보고
- Code commit, frozen config, seeds, scanner metadata 및 failed-run manifest 공개

### 5.3 Results

결과는 아래 순서대로 쓴다. E0 validity gate가 끝나기 전에는 E1–E3 숫자를 초록이나
제목을 지지하는 확정 결과로 사용하지 않는다.

#### 3.1 Common-grid, registration과 resampling validity — E0

먼저 보여줄 결과:

- Registered output이 AT2 physical grid를 따르는 근거
- Scanner별 native→target scale factor와 실제 interpolation kernel
- Old registration의 shift와 boundary saturation
- Coarse-to-fine 후 residual, failure와 padding 변화
- GT450/VERSA의 alias mixing과 anti-aliased pipeline 비교
- Anchor와 noise-floor sensitivity
- Same-chain high-band background/tissue power 중앙값은 AT2 0.00048, GT450 0.00534,
  VERSA 0.00250, AKOYA 0.00569, S60 0.00378, S360 0.00238이었다. Noise subtraction의
  high-band ERT 변화 중앙값은 비참조 scanner에서 -0.0014~-0.0039 log2였고 5,000회
  slide bootstrap CI도 모두 0.006 log2 이내였다
- SNR≥10에서 high-band eligible slide는 GT450/S60/AT2 109/109, VERSA/S360 108/109,
  AKOYA 106/109였으며, 47,088개 전체 bin에서 음수 subtraction은 없었다

결론 형식:

> The primary spectral range and estimator were retained/restricted after prespecified
> registration and resampling audits.

이 절이 통과하지 않으면 이후 결과는 scanner optics가 아니라 preprocessing 이후의
effective signature로 제한한다.

#### 3.2 Audited paired scanner spectrum — E1

- Unnormalized power와 anchor-normalized ERT를 나란히 제시
- Scanner별 curve, CI와 slide spread
- Native-AA high-band fixed fold ERT는 GT450 1.478 (log2 0.563), VERSA 0.938
  (-0.092), AKOYA 0.335 (-1.578), S60 1.044 (0.062), S360 0.842 (-0.248)
- Historical→native-AA high-band median absolute delta는 scanner별 0.073–0.134 log2였고
  q95는 AKOYA에서 0.430으로 가장 컸다. Low–mid median absolute delta는 모두
  0.016 log2 이하였으므로 resampling revision은 주로 고주파 해석을 갱신함
- Anchor 변경의 high-band median absolute delta는 GT450 0.008, VERSA 0.018,
  S60 0.037, S360 0.012 log2 이하였으나 AKOYA는 upper anchor에서 0.117
  (q95 0.290)이므로 scanner별 민감도를 함께 보고함
- Optical MTF가 아니라 effective relative spectrum이라는 제한을 반복

핵심 질문:

> Scanner acquisition/preprocessing은 동일 content에서 재현 가능한 frequency-selective
> signature를 만드는가?

#### 3.3 Scanner spectrum은 content에 의존한다 — E2

- Scanner fixed mean
- Scanner × tissue, scanner × slide와 sampling variation
- AKOYA 포함/제외 sensitivity
- High band의 tissue-between-slide variance fraction은 GT450 20.5%, AKOYA 31.3%,
  S60 27.6%, VERSA 6.5%, S360 3.3%였다. Tissue variance의 BH q-value는 각각
  0.0499, 0.0059, 0.0145, 0.3510, 0.4607이므로 모든 scanner에 동일한 tissue effect를
  주장하지 않음
- 기존 85.5% scanner fraction은 폐기하고 현재 nested model의 scanner별 variance
  component를 사용함

핵심 결론:

> Scanner signature exists as a population tendency, but its observed magnitude is a
> scanner × content response.

#### 3.4 단일 blur–sharpen scalar로 충분하지 않다 — E3

- Nearest equivalent HF gain
- Cross-validated off-axis lack-of-fit와 CI
- 어떤 frequency region과 scanner에서 residual이 남는지
- Leave-one-slide-out manifold와 matched-gain held-out scalar null을 사용했을 때 모든
  scanner의 median excess off-axis RMS 95% bootstrap CI가 0보다 컸다: GT450
  0.081–0.116, VERSA 0.095–0.107, AKOYA 0.256–0.316, S60 0.093–0.111,
  S360 0.093–0.107 log2
- GT450은 equivalent gain이 상한 2.0에 도달한 slide가 90.8%이므로 off-axis residual과
  함께 사전 정의한 gain 범위 부족도 별도로 보고함

이 결과는 frequency control family가 단일 retention scalar가 아니라 attenuation,
boost, phase-cancellation과 paired consensus를 포함해야 하는 이유를 제공한다.

#### 3.5 Controls가 attainable invariance–fidelity region을 정의한다 — E4

- 436/436 model--slide shard와 2,354,400/2,354,400 control embedding이 population audit을
  통과했다. 추론은 109 physical slide equal weight와 동결된 5,000 bootstrap을 사용했다.
- Complete HF attenuation은 PFM별 scanner radius를 26.8--51.1% 줄였지만 Δ content margin이
  −0.2805--−0.7425였고 네 PFM 모두 fidelity decision에 실패했다.
- LOSO global training HF mean 완전 치환도 radius를 26.2--46.7% 줄였지만 Δ content margin
  −0.2721--−0.7331과 collapse-ratio lower CI 최저 0.166--0.377로 네 PFM 모두 실패했다.
- Registered same-location leave-one-scanner-out HF mean 25% oracle은 ResNet50, UNI v1,
  CONCH v1, Virchow2에서 각각 RR 12.5%, 9.5%, 14.0%, 10.7%였고, content와 모든 source-scanner
  collapse gate를 통과하면서 네 모델 모두 invariance CI 하한이 0보다 컸다.
- HF retention 0.75는 네 모델 모두 invariance를 개선했지만 UNI v1의 content lower CI
  −0.0292와 CONCH v1의 −0.0205가 frozen −0.0200 margin을 넘지 못해 common-safe가 아니었다.
- Paired oracle 외 simple attenuation/boost의 safe-and-improved 조건은 model-specific이었다:
  ResNet50은 retention 0.75와 boost 1.25/1.50, Virchow2는 retention 0.75와 boost 1.25를
  통과했으나 UNI v1과 CONCH v1에는 해당 조건이 없었다.

따라서 낮은 scanner radius는 content loss/collapse와 진짜 paired alignment 양쪽에서 모두
나올 수 있다. E4는 evaluation principle과 attainable safe region을 population 수준에서
입증하지만 paired oracle은 deployable correction이 아니며 실제 방법 선택은 E5에 남는다.
전체 결과와 provenance는 [`e4_control_population_results.md`](e4_control_population_results.md)에
잠갔다.

#### 3.6 실제 correction은 common-safe region과 method-specific rejection을 함께 보인다 — E5

- Input-only audit에서 CORAL shrinkage `0.05`와 frequency gain cap `1.03`을 고정한 뒤,
  Reinhard, paired OD affine, frequency calibration, CORAL와 orthogonal Procrustes를
  109-fold LOSO로 적합했다.
- 436/436 image shard의 784,800 embedding과 436/436 feature shard의 523,200 embedding,
  총 1,308,000개가 identity, raw-AT2 equality, finite norm와 source/statistic/hash audit을
  통과했다.
- Reinhard, CORAL와 orthogonal Procrustes는 네 PFM 모두 content/collapse gate를 통과하면서
  invariance CI 하한이 0보다 큰 four-of-four common-safe + improved 방법이었다.
- Orthogonal Procrustes는 네 PFM 모두에서 가장 큰 safe RR을 보였다: ResNet50 43.9%,
  UNI v1 22.5%, CONCH v1 23.9%, Virchow2 20.0%. CORAL은 34.9%, 16.3%, 22.0%, 17.6%,
  Reinhard는 27.6%, 6.0%, 11.9%, 3.7%였다.
- Frequency calibration은 four-of-four fidelity-safe였지만 보수적 cap 아래 UNI v1과
  CONCH v1의 invariance를 개선하지 못했다.
- UNI v1 paired OD affine은 RR +8.9%와 positive invariance CI를 보였지만 VERSA
  variance-trace point ratio 0.8865가 collapse point gate 0.90에 미달했다. 따라서
  `invariance-positive but fidelity-unsafe`인 실제 method/PFM cell이 하나 존재했다.
- 반면 최고 방법 자체는 네 PFM 모두 separability-only와 fidelity-constrained ranking에서
  orthogonal Procrustes로 일치했다. 특정 rank reversal을 사후 선택하지 않는 원칙에 따라
  이 null top-rank disagreement도 그대로 보고한다.

전체 20개 actual method × PFM cell, scanner-specific content reversal과 provenance는
[`e5_comparator_population_results.md`](e5_comparator_population_results.md)에 잠갔다.
E5는 실제 방법에서도 fidelity gate가 개별 positive invariance 판정을 바꿀 수 있음을
보였지만, feature-space 정렬이 independent biology 또는 clinical utility를 보존했다는
주장은 하지 않는다.

#### 3.7 Correction 효과는 tissue와 slide에 따라 달라진다 — E6

- E5의 exact LOSO prediction에서 슬라이드당 5개 disjoint 20-location replicate를 만들고,
  20개 radius-benefit cell과 100개 source-scanner content-change cell 각각에 tissue와
  slide-within-tissue random slope를 적합했다. 전체 및 minimum-three sensitivity를 합친
  240개 full/reduced REML fit은 모두 수렴하고 독립 result lock을 통과했다.
- Full 37-tissue 분석에서 radius benefit은 18/20 cell, content change는 71/100 cell에서
  prespecified family별 BH q<0.05였다. 31 tissue/98 slide minimum-three sensitivity에서는
  각각 17/20과 70/100이었다.
- Frozen global winner인 orthogonal Procrustes는 ResNet50과 UNI v1에서 37/37 tissue,
  CONCH v1에서 36/37, Virchow2에서 34/37 tissue의 predicted radius benefit 1위를
  유지했다. 나머지 네 tissue에서는 CORAL이 1위였고 모두 3-slide class였다.
- 독립 morphology annotation은 없다. 따라서 post-E6 secondary 계약에서 AT2 band power
  3개와 across-scanner transfer SD 3개만 사용했다. Exact tissue-held-out ridge의 predictive
  R² CI 하한이 0보다 큰 것은 20개 중 ResNet50 frequency, UNI Reinhard, UNI paired OD의
  세 cell뿐이었다. CORAL/Procrustes의 큰 평균 gain은 이 단순 spectrum 변수로 예측되지
  않아 universal response predictor 주장은 지지되지 않는다.
- Exact 37-fold correction LOTO population도 872/872 shard와 1,308,000/1,308,000 embedding
  audit을 통과했다. LOSO 대비 fidelity-safe 및 safe-and-improved 판정 변화는 20개 cell
  모두 0이었고, common-safe + improved인 Reinhard/CORAL/Procrustes 결론도 유지됐다.
  최대 RR 변화는 Virchow2 CORAL의 −3.21 percentage point였다. 이는 primary LOSO를
  대체하지 않는 unseen-tissue transfer sensitivity다.

전체 결과, BLUP, class-size sensitivity와 provenance는
[`e6_heterogeneity_results.md`](e6_heterogeneity_results.md)에 기록했다.

#### 3.8 Content fidelity와 tissue-type evidence — E7

- E5의 prespecified content margin과 variance/effective-rank/pairwise-distance collapse
  guardrail이 primary fidelity decision을 제공한다. E7은 이를 대체하지 않는 coarse
  tissue-type secondary다.
- Raw AT2만으로 held-out-slide centroid를 적합하고 singleton tissue를 제외해 36 tissue,
  108 slide를 평가했다. 25,920 source row와 432 AT2 row가 identity/finiteness/hash gate를
  통과했고, tissue→slide 5,000회 hierarchical bootstrap을 사용했다.
- Primary LOSO에서 Procrustes top-1 변화는 네 PFM 모두 CI가 0을 포함했지만 matched-AT2
  centroid-profile agreement는 네 PFM 모두 증가했다(+0.00347--+0.0402, 모든 CI 하한 >0).
  Exact LOTO에서도 profile agreement 증가는 유지됐으나 UNI v1과 Virchow2의
  correct-tissue margin은 감소했다.
- CORAL은 primary E5 gate를 통과했지만 centroid-profile agreement가 LOSO와 LOTO 모두 네
  PFM에서 감소했다. UNI v1 top-1은 LOSO −2.92 pp, LOTO −3.72 pp였고, minimum-three
  sensitivity에서는 top-1 CI가 0을 포함했으나 top-5/margin/profile 감소는 남았다.
- 따라서 어떤 correction도 모든 tissue metric과 PFM을 일률적으로 개선하지 않았다.
  E7은 global collapse가 없다는 것과 모든 class-relative geometry가 보존된다는 것이
  동치가 아님을 보여주지만, E5의 frozen safe/unsafe 판정을 사후 변경하지 않는다.

Figure 6과 전체 결과/provenance는
[`e7_tissue_probe_results.md`](e7_tissue_probe_results.md)에 잠갔다. Self-cosine 또는
tissue type만으로 독립 biological fidelity를 확정하지 않으며 nucleus/spatial endpoint는
annotation/QC 계약이 없어 PanNormal core에서 생략한다.

#### 3.9 Image-space의 천장은 방법이 아니라 도메인의 성질이다 — E8

이 절은 core 원고의 필수 결과가 아니지만, §4의 degeneracy 명제와 §3.6의 image/feature 대비를
**사전등록된 학습 baseline**으로 마감한다. 계약 §10은 embedding이 존재하기 전에 지지/반증
조건을 고정했고, 어떤 threshold도 endpoint를 본 뒤 수정하지 않았다.

- **Arm A `gainfield`** (RF1U 위의 공간가변 band gain, 보장 유지)는 image-only gate에서
  탈락했다. Pooled paired MAE는 다섯 fold 모두 개선했으나 두 fold에서 5개 source 중 3개만
  개선해 4개 요건(`E8_GATE_MIN_SOURCES`)을 못 채웠고, arm gate의 4/5 fold 요건에 3/5로
  미달했다. **평균적으로 나빠서가 아니라 scanner 사이에서 고르지 못해 기각됐다.** 따라서
  §2.11의 scanner × tissue interaction은 이 parameterization 안에서는 모델링 가치가
  확립되지 않았고, 세 숫자짜리 analytic gain이 자기 family 안에서 충분하다.
- **Arm B `free`** (무제약 OD residual, ceiling probe)는 네 PFM 모두 content
  non-inferiority와 collapse gate를 통과하면서 ResNet50 RR **+47.87%** — feature space를
  포함해 이 연구 전체 최대치 — 를 냈고, linear scanner probe는 **0.759--0.967**에 머물렀다.
  같은 조건의 CORAL/Procrustes는 0.082--0.162다. 사전등록 판정은 **supported**이며 반증
  구간(2개 이상 모델에서 0.30 미만)에는 어떤 모델도 접근하지 않았다.
- **천장의 상한.** E4의 destructive control(0.808--0.950), paired oracle(0.928--0.992),
  그리고 arm B(0.759--0.967)가 세 방향에서 같은 벽을 가리킨다. 한계는 correction family의
  성질도, 최적화 강도의 성질도 아니다.
- **비선형 probe가 두 feature 방법을 가른다.** MLP probe에서 Procrustes는 0.614--0.778,
  CORAL은 0.074--0.269다. Procrustes는 scanner별 강체 회전이라 within-scanner 거리를 정확히
  보존하므로 국소 기하가 그대로 남고, CORAL은 2차 모멘트를 whitening·recolouring해 그 모양을
  바꾼다. **radius를 가장 많이 줄인 방법이 scanner 정보를 가장 많이 남긴다** — invariance
  지표와 실제 scanner 제거의 괴리가 feature space 안에서, content 손실 없이 재현된 것이다.
  따라서 배포 권고는 "feature space"가 아니라 **CORAL**로 좁혀야 한다.
- **정직한 단서.** Arm B의 phase correlation 평균은 0.694로 RF1U의 0.946보다 낮다. 천장은
  구조적으로 공격적인 correction까지 견딘다는 뜻이지, 보수적인 학습 모델을 시험해 실패했다는
  뜻이 아니다.

전체 수치와 provenance는 [`e8_paired_residual_results.md`](e8_paired_residual_results.md)에
있고, condition 계열(moment ladder, destination sweep, variance components, acquisition
provenance)은 [`e8_results_digest.md`](e8_results_digest.md)에 있다.

#### 3.10 외부 코호트에서의 재현과 두 건의 정정 — E9

PLISM을 116,831 location, 817,817 scanner-to-reference measurement, 46개 명명 tissue,
13 staining condition × 7 scanner, tissue coverage 88.2%로 재구축하고 2026년 encoder
panel(UNI2-h, CONCHv1.5, H-optimus-1)로 다시 측정했다. 여덟 개 주장 중 다섯이 재현됐다.

- **목적지 규칙은 강하게 재현된다.** Destination detail power 대 fitted mean log gain의
  Spearman은 7개 목적지에서 **+0.964**이고, AT2는 패널에서 detail power가 가장 낮으며
  18개 cell 중 15개가 AT2 쪽으로 감쇠한다.
- **그러나 encoder 이득으로는 넘어가지 않는다.** "detail이 풍부한 목적지를 겨냥하면 encoder가
  더 이득"은 3개 모델 중 1개에서만 성립해 **unconfirmed**다. 축소된 형태가 §3.9와 정확히
  맞물린다: 목적지 규칙은 이미지에 관한 법칙이지 표현에 관한 법칙이 아니다.
- **정정 1 — 3항 분해는 살아남지 못한다.** Raw fine-band power를 optical-density **표준편차**의
  제곱으로 나눈 contrast audit에서 초과분은 PLISM 1.40×, PanNormal 1.36×로 사실상 같다.
  sampling을 맞추면 사라진다는 8월판의 결론은 철회한다. 살아남는 것은 raw band power가
  contrast에 지배된다는 사실이다.
- **정정 2 — CORAL 실패는 추정 아티팩트였다.** "고차원 encoder에서 CORAL이 실패한다"는 단서는
  재현되지 않는다. 표본이 충분하면 3/3에서 safe + improved이고, break-even은 768차원에서
  6.5 samples/dim, 1536차원 두 encoder에서 32.5로 **encoder마다 측정해야 한다**.
- **색을 맞추기 전에는 주파수가 식별되지 않는다.** Raw band power의 scanner 순위는 잠긴 transfer
  순위와 **ρ = −1.000**이고, Reinhard 이후 **+0.800**이 된다. 이는 §3.6의 RF1 ordering 결과
  (주파수 보정은 단독으로는 0에 가깝고 Reinhard 뒤에서만 유용하다)와 **같은 사실**이며, 두
  코호트에서 반대 방향으로 측정된 것이다. Methods의 규칙으로 승격한다.
- **단일 지렛대점.** `HR` 섹션(다른 어느 섹션보다 3 표준편차 이상 짙은 염색)이 세 분석을
  떠받친다: stain covariate가 6개 scanner 중 1개만 전 fold 생존하고 나머지 둘은 `HR` 없이
  무너지며, feature 보정이 아무 효과가 없는 유일한 섹션이다. 각주가 아니라 본문에 둔다.

결과는 [`e9_plism_core_results.md`](e9_plism_core_results.md), 계약과 Amendment 3은
[`e9_plism_native_ert_contract.md`](e9_plism_native_ert_contract.md)에 있다.

#### 3.11 Background 결과 — Supplement 중심

- Background feature가 tissue ERT의 held-out prediction을 일관되게 개선하지 않음
- Noise/QC characterization으로서의 제한된 역할
- 동일 processing chain의 noise-floor sensitivity
- 65,400개 glass patch의 post-render QC 유지율은 98.34%였고, 사후 QC 실패 patch도
  결과에서 제외하지 않았다. Maximum black-pixel fraction은 0.00140이었다

### 5.4 Discussion

#### 4.1 Invariance-only evaluation의 한계

- Formal degeneracy와 empirical collapse control을 연결
- Scanner separability를 폐기하자는 것이 아니라 fidelity와 함께 조건화하자는 주장

#### 4.2 Scanner spectrum의 역할과 제한

- Frequency는 scanner-relevant control axis이며 biological information meter가 아님
- Effective relative spectrum과 absolute optical MTF를 구분
- Aliasing, noise와 content dependence를 해석에 포함

#### 4.3 Universal scanner style의 한계

- Population tendency와 scanner × content response를 함께 논의
- Tissue-adaptive 또는 uncertainty-aware correction의 필요성
- Universal impossibility 같은 과도한 명제는 피함

#### 4.4 Image-space와 feature-space harmonization

- Image-space의 공유 가능성과 hallucination/irreversibility
- Feature-space의 직접성과 representation/content geometry distortion 위험
- 방법 family가 아니라 fidelity-constrained generalization으로 판단

#### 4.5 Clinical and evaluation implications

- 실제 배포에서는 scanner robustness뿐 아니라 독립 labeled cohort의 calibration과
  biological non-inferiority를 추가로 평가해야 함
- PanNormal의 content-fidelity 결과를 clinical validation으로 해석하지 않음

#### 4.6 Limitations

- Absolute MTF가 아님
- Registration/resampling preprocessing의 영향
- 제한된 scanner 집합
- 일부 tissue의 낮은 slide 수
- PFM 및 downstream label의 범위
- Tissue type만 사용할 수 있어 독립 biological fidelity를 직접 검증하지 못함

## 6. 실험–주장–배치 매핑

| ID | 실험 | 핵심 질문 | 상태 | 본문 배치 | 완료/통과 조건 |
|---|---|---|---|---|---|
| E0a | Common physical grid audit | 모든 scanner pixel이 AT2 물리 좌표를 따르는가? | **완료** | Methods 2.2, Results 3.1 | 65,400 native-AA patch identity/hash gate 통과 |
| E0b | Coarse-to-fine registration | AKOYA 경계 포화가 ERT를 교란하는가? | **완료** | Methods 2.2, Results 3.1 | 65,400-row/10,900-tuple six-scanner/512 px gate 통과 |
| E0c | 2D aliasing audit | GT450/VERSA downsampling이 high band를 오염하는가? | **완료** | Methods 2.3, Results 3.1 | 최종 transform q05/q50/q95에서 ≤5% 통과 |
| E0d | Anchor/noise-floor audit | ERT shape가 anchor와 noise에 견고한가? | **완료** | Methods 2.4, Results 3.1/Supplement | 65,400 same-chain glass patch, 음수 subtraction 0/47,088, SNR 1/2/5/10 보고 |
| E1 | Paired ERT | Scanner별 frequency signature가 재현되는가? | **완료** | Results 3.2 | Native-AA 109-slide locked tables |
| E2 | Hierarchical decomposition | Signature가 tissue/slide에 의존하는가? | **완료** | Results 3.3 | 37-tissue/slide nested REML 완료 |
| E3 | Scalar reducibility test | 단일 blur–sharpen 축으로 충분한가? | **완료** | Results 3.4 | LOSO lack-of-fit와 5,000 bootstrap CI |
| E4 | Control population | Collapse, oracle, divergence가 frontier를 정의하는가? | **완료** | Results 3.5 | 436 shards, 2,354,400 embeddings, 4 PFM, slide-bootstrap CI |
| E5 | Correction benchmark | Protocol이 실제 방법 선택을 바꾸는가? | **완료** | Results 3.6 | 872 shards, 1,308,000 embeddings, five-method LOSO frontier와 Figure 5 |
| E6 | Content-dependent correction | Correction benefit이 tissue/slide에 따라 달라지는가? | **완료** | Results 3.7 | 240 REML fit + 37-fold exact LOTO transfer lock |
| E7 | Content/tissue fidelity | Invariance gain이 representation과 coarse tissue content를 보존하는가? | **완료** | Results 3.8 | 36-class grouped probe + 5,000 hierarchical bootstrap + Figure 6 |
| E8 | Learned paired residual baseline | 학습된 image-space correction이 천장을 넘는가? | **완료** | Results 3.9 | §10 사전등록 판정 `supported`; arm A gate 탈락, arm B 4/4 safe |
| E9 | PLISM external validation | 결론이 외부 scanner/stain에서 재현되는가? | **완료** | Results 3.10 | 116,831-location core grid, 8개 주장 중 5개 재현, 2건 정정 |
| — | Background characterization | Background가 tissue ERT를 설명하는가? | 완료 | Supplement/Results 3.11 | 기존 null + registered noise sensitivity |

## 7. Figure와 Table 설계

### Main figures

**Figure 1 — Study design and estimator validity**

- Paired acquisition과 분석 단위
- Invariance-only degeneracy 개념도
- Outcome-blind registration recovery와 최종 geometry/render retention
- GT450/VERSA resampling mixing audit와 same-chain background floor

**Figure 2 — Audited frequency-resolved scanner signatures**

- Scanner별 normalized ERT cohort curve와 slide spread
- High-band fixed effect와 95% CI
- Scalar blur–sharpen LOSO projection residual과 slide-bootstrap CI
- Unnormalized radial power와 전체 2D mixing은 Supplement에 배치

**Figure 3 — Scanner × content structure**

- Tissue/slide/sampling decomposition
- Band별 tissue share와 BH q-value
- Tissue-specific slopes. Scanner fixed mean은 Figure 2B, AKOYA/route sensitivity는
  Supplement에 배치

**Figure 4 — Control-bounded invariance–fidelity frontier**

- ~~Raw, collapse, global mean, paired oracle, sharpening~~
- ~~PFM별 frontier와 bootstrap CI~~
- ~~Content non-inferiority와 every-scanner collapse guardrail~~
- PNG/PDF 및 source/output hash 잠금 완료

**Figure 5 — Image- and feature-space correction benchmark**

- ~~Actual methods의 full method × PFM frontier~~
- ~~Separability-only ranking 대 fidelity-constrained decision~~
- ~~Content와 every-scanner collapse gate~~
- PNG/PDF 및 source/output hash 잠금 완료

**Figure 6 — Content preservation and tissue-level evidence**

- ~~Primary content-fidelity endpoint와 collapse guardrail~~
- ~~Tissue-type secondary endpoint와 class-size sensitivity~~
- ~~Tissue class-size와 minimum-slide sensitivity~~
- PNG/PDF 및 source/output hash 잠금 완료

### Main tables

| Table | 내용 |
|---|---|
| Table 1 | Cohort, scanner, native/target MPP, scale, interpolation, compression |
| Table 2 | Audited scanner spectrum과 content variance, 95% CI |
| Table 3 | PFM × method FC-RR, fidelity gate, collapse status와 decision |
| Table 4 | Content-fidelity, collapse guardrail과 tissue-type secondary endpoint |

Table 1의 native-header recoverable field는 654/654 scanner--slide file에서 감사했다.
Objective NA와 firmware, VERSA/AKOYA exact manufacturer/model, S60/S360 JPEG quality만
operator/acquisition-record 확인 항목으로 남는다. Table 2는 E1--E3 locked result를 한 행의
scanner summary로 결합한다.
Table 3의 full PFM × method grid는 E5 result lock, Table 4의 content/collapse/tissue-type
endpoint는 E5와 E7 result lock에서 재현되며 Figure 6과 함께 잠겼다.

### Supplement

- Registration shift/failure 전체 분포
- 2D alias mixing과 angular spectrum 전체 결과
- Anchor band 선택 sensitivity
- Registered-domain background/noise sensitivity
- Scanner subset과 tissue minimum-slide filters
- Full method × PFM endpoint grid
- Clipping/out-of-gamut, runtime와 scalability
- Mixed-model diagnostics와 singular-fit 처리

## 8. 현재 수치의 사용 규칙

### Legacy provenance로만 보존할 수치

- GT450 high-band ERT 1.595
- AKOYA high-band ERT 0.297
- High-band scanner main descriptive fraction 85.5%

이 수치는 historical registered-image 분석의 provenance로만 남긴다. E0 audit 이후 primary는
native-AA GT450 1.478, AKOYA 0.335와 scanner별 nested tissue/slide variance component다.
초록, 제목, Highlights와 최종 결론에는 audited table만 사용한다.

### 현재도 사용할 수 있는 진단적 결과

- Invariance-only control이 collapse를 허용한다는 mechanistic pilot
- LF pixel alignment가 UNI alignment를 보장하지 않았다는 pilot
- Background covariate의 held-out tissue-transfer 설명력 null result
- AKOYA의 slide-coherent registration boundary saturation 진단

이들 역시 pilot 또는 diagnostic이라는 범위를 명시한다.

## 9. 허용되는 주장과 금지되는 주장

### Validity gate 이후 허용 가능한 주장

- Scanner separability alone is non-identifying with respect to content fidelity.
- Frequency-resolved controls are physically matched to major scanner-associated operators.
- Scanner signatures are frequency- and content-dependent.
- A predefined scalar blur–sharpen family is insufficient within sampling uncertainty.
- Fidelity-constrained evaluation can change the decision produced by separability-only ranking.
- Correction effects vary across tissue, slide and PFM.

### 금지하거나 제한할 주장

- 주파수는 보존 정보량이 알려진 유일한 축이다.
- 현재 ERT는 scanner의 absolute MTF다.
- GT450 high-band signature는 aliasing 때문이라고 확정한다.
- AKOYA attenuation은 registration failure 때문이라고 확정한다.
- Self-cosine이 높으면 biological content가 보존됐다.
- Full paired target은 deployable oracle harmonization이다.
- Scanner가 보편적으로 variation의 85.5%를 설명한다.
- PLISM은 독립 환자 clinical validation이다.
- 하나의 universal canonical image는 원리적으로 불가능하다.

## 10. 실행 순서와 stop/go 기준

### Phase 0 — Data and estimator validity

1. ~~Common grid와 transform/interpolation provenance 회수~~ — 96-slide exact rigid provenance,
   VALIS/libvips bicubic·MPP overwrite 확인
2. ~~Existing offsets에서 constant 대 affine slide prior 진단~~ — 109-slide five-fold
   `prior_fits.csv`의 2,725 scanner-fold fit에서 affine 1,544, constant 1,181을 선택하고
   downstream geometry audit에 전달
3. ~~Sentinel 10–20 slides 선정~~
   - AKOYA registration worst/median/best
   - GT450/VERSA aliasing risk
   - Tissue와 slide strata 균형
4. ~~현재 registered WSI의 corrected integer crop과 기존 VALIS rigid output 대조~~
5. ~~109-slide common-pool에서 100-location/512 px manifest 복구~~
6. ~~Native geometry gate 실패 cell에 한해서만 preserved rigid를 감사하고 필요한 cell은
   native WSI에서 VALIS registration 재실행~~ — primary 654 cell 중 61개가 실패했으며,
   preserved audit와 targeted from-scratch similarity를 거쳐 20개 cell을 진단했다. Similarity가
   primary보다 불량한 cell은 unchanged gate 아래 primary를 유지했고, 별도 affine diagnostic도
   이를 개선하지 못했다. 이 선택에는 PFM, tissue 또는 downstream outcome을 사용하지 않았다.
7. ~~GT450/VERSA 한-slide same-scanner geometry recovery + native explicit-AA patch pilot~~
   — local residual 후 NCC median 0.996, boundary failure 0/80; global transform 단독은 GT450에서
   불충분하므로 per-location residual 유지
8. ~~109 slides × 100 locations에서 scanner별 native transform과 local residual manifest를
   구축하고 512 px/six-scanner QC를 통과~~ — global inlier/scale/reprojection gate와 ±120 px
   local NCC/boundary/native-bounds gate를 유지했다. Route-min candidate plan은 22개 slide의
   44개 location만 outcome-blind reserve로 교체했으며, 최종 65,400/65,400 scanner-location,
   654/654 scanner–slide cell과 10,900/10,900 six-scanner tuple이 통과했다. 최종 RGB source는
   모두 native WSI다.
9. ~~Frozen native geometry로 anti-aliased 512 px patch/grid 생성과 입력 무결성·historical
   sensitivity 감사~~
   — 109/109 shard와 65,400/65,400 patch의 SHA-256, geometry identity 및 black/white render
   gate 통과. Historical 대비 `geometry × resampling` band delta 산출 완료
10. ~~최종 native transform 2D alias mixing과 E0d sensitivity~~ — original bicubic 실패,
    explicit-AA는 전체 q05/q50/q95에서 통과. Anchor sensitivity와 65,400-patch same-chain
    background/noise-floor audit도 완료했으며, 전체 47,088 bin에서 음수 subtraction은 없고
    high-band median ERT 변화는 비참조 scanner에서 0.0039 log2 이하였다

**Stop:** Common physical coordinates가 보장되지 않음.

**Revise:** Alias leakage가 사전 기준을 넘거나 registration correction이 ERT를 실질적으로
변경함. Common grid/band/estimator를 고친 뒤 재시작.

**Go:** Primary band와 corrected registration contract를 동결할 수 있음.

### Phase 1 — E1–E3 rebuild

1. ~~109-slide ERT 재산출~~
2. ~~Scanner × tissue/slide model 재적합~~
3. ~~Scalar reducibility LOSO test와 5,000 slide bootstrap~~
4. ~~E0--E3 locked table과 artifact provenance 생성~~ — 22개 source artifact SHA-256,
   26개 locked result row와 6개 figure component를 전수 감사해 통과
5. ~~Main Figure 1--3 journal-layout composite 조립~~ — E0--E3 locked source만 사용한
   PNG/PDF 3쌍을 생성하고 source/output SHA-256과 시각 검수를 완료

**Go:** 결과가 E0 contract를 통과하고 slide-bootstrap uncertainty가 완전함.

### Phase 2 — Evaluation protocol population test

1. ~~4개 core PFM의 TRIDENT commit/checkpoint/preprocessing contract 동결~~
2. ~~Frozen 109 × 100 canonical-center manifest에서 모델별 native FOV crop을 직접 생성~~
3. ~~TRIDENT sampling을 재실행하지 않고 동일 RGB crop에 네 encoder feature를 추출~~
   — 436/436 model-slide shard, 261,600/261,600 embedding 전수 audit 통과
4. ~~**E4--E7 outcome을 열기 전에** primary scanner-dispersion/content endpoint, margin,
   collapse guardrail, comparator, cross-validation과 multi-PFM claim rule을 사용자 승인으로 동결~~
   — 2026-08-03 전부 승인; every-source-scanner collapse gate와 Supplement-only Macenko 포함
5. ~~E4 control population 실행~~ — 436/436 shard, 2,354,400/2,354,400 embedding과
   91,560 slide-result row audit 통과; paired HF 25% oracle만 four-of-four common-safe + improved
6. ~~Minimal E5 image+feature benchmark~~ — 872/872 shard, 1,308,000/1,308,000 embedding
7. ~~E5 method FC-RR 및 fidelity-constrained frontier 산출~~ — Reinhard/CORAL/orthogonal
   Procrustes four-of-four common-safe + improved; UNI paired OD invariance-positive but unsafe

**Go:** 실제 correction에서 invariance와 fidelity를 함께 판정할 수 있음. E4만으로 논문을
완결하지 않는다.

### Phase 3 — Heterogeneity and content fidelity

1. ~~E6 tissue/slide correction effect~~ — full/minimum-three 240 REML fit과 exact LOTO 완료
2. ~~E7 primary representation/content endpoint와 collapse guardrail~~ — E5 authoritative
   gate와 E7 secondary의 역할을 분리해 잠금
3. ~~Tissue-type secondary, tissue class-size 보고와 minimum-slide sensitivity~~ — 36/108
   full-evaluable과 31/98 sensitivity 완료

**Go:** 사전 동결한 content-fidelity 판정과 slide-level inference가 완전하고, tissue-type
결과의 표본수 제한이 명시됨.

### Phase 4 — PanNormal manuscript lock

- ~~초록 수치를 locked tables와 대조~~
- ~~결과 확인 후 method, PFM 또는 threshold를 선택하지 않음~~
- ~~Null과 failed-fidelity method를 frontier에 그대로 포함~~
- ~~Main/Supplement 배치와 terminology audit~~
- ~~`biological fidelity`와 `clinical validation` 과장 여부 최종 점검~~
- Elsevier highlights/AI disclosure 정책은 확인했고, MedIA submission portal의
  journal-specific abstract/figure 조건은 실제 투고 직전에 재확인

### Post-core extensions — E5-RF1, E8과 E9 실행 완료; 추가 PFM은 보류

1. **E5-RF1 result-locked:** five-fold cross-fitted Reinhard 이후 72-bin mean-OD residual
   spectrum을 shared-OD gamut projection으로 교정했다. Input-only gate가 선택한 gain cap은
   1.25였고, final hard-clamp MAE는 0이었다. Spectrum RMSE는 15 FOV×scanner cell 중
   12개에서 감소했지만 S360 세 cell은 악화됐다.
2. RF1은 네 PFM 모두 raw 대비 common-safe + improved였고 RR은 ResNet50 29.7%, UNI v1
   6.7%, CONCH v1 14.7%, Virchow2 4.7%였다. Paired Reinhard 대비 scanner radius도 네 PFM
   모두 추가 감소했지만 크기는 +0.61--+2.75 RR percentage point의 점진적 개선이었다.
3. Tissue-type secondary top-1은 UNI에서만 full/min-3 모두 Reinhard 대비 개선됐고,
   나머지는 혼합 또는 null이었다. 따라서 biological claim은 확대하지 않는다.
4. 현재 원고에서는 CycleGAN을 추가하지 않는다. 이 조건은 유지되며 그 근거는
   [`e8_paired_residual_contract.md`](e8_paired_residual_contract.md) §11에 명시했다:
   unpaired adversarial objective는 이 코호트의 pixel registration — 가장 큰 자산 — 을
   버리고, ceiling question에서는 paired arm에 지배당한다.
5. **완료 — E8.** 요구했던 새 계약(paired scanner-conditioned residual, image-only nested
   selection, hallucination audit, 사전등록 reading)을 먼저 동결한 뒤 실행했다. Arm A는
   image-only gate에서 탈락했고 arm B는 §10 판정 `supported`를 냈다. PLISM external
   validation은 arm B에 대해 여전히 미실행이며, 이것이 E8의 남은 공백이다.
6. **완료 — E9.** PLISM interpolation/alignment audit(비참조 measurement 700,986건에서
   median residual 0.091--0.113 µm, 1 µm gate 통과율 99.57%), 한 stain의 scanner-only
   primary, scanner × stain secondary, internal--external effect comparison을 모두 수행했다.
   SQ file-swap 정정이 core grid 전반에 적용됐고, Amendment 2의 per-patch `response >= 0.3`
   권고는 Amendment 3에서 block-level exclusion + 두 variant 병기로 대체돼 철회됐다.
7. Core 결과와 무관하게 추가 PFM panel 및 checkpoint contract를 별도 동결 — E9가 UNI2-h,
   CONCHv1.5, H-optimus-1로 사실상 착수했으나 PanNormal core에서의 재현은 미실행
8. 추가 PFM에서 core endpoint와 claim rule을 그대로 재현 — 보류
9. **남은 최대 공백:** labeled downstream endpoint. 이 코호트의 유일한 biological label이
   normal tissue type이므로 모든 fidelity 측정이 representation 수준에 머문다. 목적지 효과가
   downstream task로 이어지는지 — detail-poor target을 겨냥하면 subtype/biomarker 정확도가
   실제로 떨어지는지 — 가 "careful measurement"를 "decisions change"로 바꿀 단일 변경이다.

## 11. 예상 초록의 구조

### Background

PFM의 scanner separability가 보고되어 왔지만, separability 감소는 true alignment와
content collapse를 구별하지 못한다.

### Methods

109-slide paired cohort에서 registration/resampling audit를 거친 effective spectrum을
추정하고, control-bounded invariance–fidelity frontier에서 image/feature correction을 여러
PFM에 평가한다. 추론은 slide-blocked이며 prespecified content-fidelity endpoint, collapse
guardrail과 tissue-type secondary endpoint를 사용한다.

### Results

Native-AA scanner spectrum은 scanner별 population signature와 유의한 tissue/slide
heterogeneity를 보였고 사전 정의한 scalar blur--sharpen family로 환원되지 않았다.
Destructive controls와 paired oracle은 낮은 scanner radius가 collapse와 genuine alignment
양쪽에서 생길 수 있음을 보였다. Actual LOSO에서 Reinhard, CORAL와 Procrustes는 네 PFM
모두 safe + improved였고 exact LOTO에서도 판정 변화가 없었다. Tissue secondary에서는
Procrustes가 네 PFM의 paired centroid-profile agreement를 높였지만 어떤 correction도 모든
tissue metric을 보존하거나 개선하지는 않았다. 사전등록된 learned paired residual baseline은
네 PFM 모두 fidelity gate를 통과하고 최대 radius 감소를 내면서도 linear scanner probe를
0.76 아래로 내리지 못했고, ground-truth paired oracle 역시 넘지 못했다 — image-space의
한계는 correction family의 성질이 아니라 도메인의 성질이다. 독립 공개 코호트(13 staining
condition × 7 scanner, 2026 encoder panel)에서 여덟 주장 중 다섯이 재현됐다.

### Conclusion

Scanner harmonization은 scanner signal을 최소화하는 문제가 아니라 content fidelity를
보존하면서 scanner-associated variation을 줄이는 constrained evaluation problem이다.
Image space에는 oracle과 학습 baseline이 함께 가리키는 한계가 있고, 남은 여유는 feature
space — 구체적으로 CORAL — 에 있으나 그 공간은 invariance-only 평가가 가장 신뢰할 수 없는
공간이므로, 기여는 둘을 가려내는 fidelity-constrained protocol과 벽의 위치를 그린 지도다.

## 12. Highlights 초안

- Scanner separability cannot distinguish alignment from representation collapse.
- Paired spectra reveal frequency- and content-dependent scanner signatures.
- Physical controls bound the attainable invariance–fidelity region.
- Corrections are ranked only after passing a prespecified fidelity constraint.
- A pre-registered learned baseline and a paired oracle meet the same image-space ceiling.
- The framework is tested across seven PFMs, two cohorts and tissue/slide strata.

Highlights는 제출 시 Elsevier의 최신 글자 수 제한을 다시 확인한다.

## 13. 문서 관리 규칙

- 이 문서는 storyline과 claim architecture의 기준이다.
- [`final_study_protocol.md`](final_study_protocol.md)는 세부 데이터·모델·구현
  계약을 담으며, 다음 갱신에서 이 storyline에 맞춰 E0와 fidelity contract를 반영한다.
- 별도 manuscript working draft와 archive는 active tree에서 유지하지 않는다. 제출용
  원고는 이 storyline, protocol과 locked-results 문서에서 새로 생성한다.
- 삭제된 과거 판단이 필요하면 `checkpoints/prenorm_postcleanup_20260803.bundle`을 별도
  경로에 clone해 조회하고 active documentation으로 복원하지 않는다.
- 결과 문서에는 `확정`, `잠정`, `pilot`, `미완` 상태를 명시한다.

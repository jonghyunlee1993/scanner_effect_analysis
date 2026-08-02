# Fidelity-constrained scanner harmonization: manuscript storyline

**Status:** active storyline and execution blueprint

**Updated:** 2026-08-02

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
6. 현재 ERT 수치와 85.5% variance fraction은 validity gate 재검증 전까지 잠정값이다.
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
15. PLISM은 PanNormal core 연구의 완료 조건에서 제외하고 후속 frozen extension으로
    보류한다.

### 3.2 결과를 보기 전에 추가로 동결할 선택

- TRIDENT commit, 모델별 정확한 checkpoint revision/hash, preprocessing 및 feature layer
- Primary content-fidelity endpoint 하나와 collapse guardrail의 계층
- Endpoint별 허용 margin과 slide-clustered uncertainty 기준
- Scanner dispersion (R)의 정확한 정의와 raw-radius 최소값 \(\epsilon\)
- Registration, model-FOV bounds와 aliasing audit의 수치 통과 기준
- 최소 image-space/feature-space comparator 목록
- Sentinel slide 구성과 `existing corrected crop` 대 `VALIS rigid` 승격 기준

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

정합 의사결정은 다음 세 단계로 동결한다.

1. 현재 registered WSI에서 slide-level constant/affine prior와 local integer refinement를
   사용해 paired crop을 다시 만든다.
2. Sentinel slide에서 이 결과를 이미 생성된 VALIS rigid output과 동일 좌표·동일 FOV로
   대조한다.
3. 두 경로가 사전 동결한 residual/padding/paired-similarity 기준을 만족하지 못할 때만
   native WSI에서 VALIS rigid registration을 다시 수행한다. Non-rigid 결과는 interpolation
   자체가 spectrum과 PFM feature에 미치는 영향을 별도로 통과하기 전에는 primary로 쓰지
   않는다.

현재 확인된 진단:

- AKOYA boundary saturation: 54.46%
- AKOYA 109 slides 중 61개에서 patch 절반 이상이 ±16 px 경계에 도달
- 슬라이드별 dominant edge fraction 중앙값: 79.2%
- 현재 Exp05의 zero-centered ±16 px search는 slide-coherent residual을 포착하지 못함

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

0.297과 1.595는 anchor-normalized amplitude transfer이며 절대 HF power ratio가 아니다.
`amplification`, `attenuation`은 anchor 대비 spectral shape를 뜻한다고 명시한다.

기존 raw background NPS는 tissue registration과 다른 interpolation 경로를 사용했으므로
registered tissue power에서 직접 빼지 않는다. Noise correction은 동일한 processing chain의
background를 사용하고, subtraction 후 음수값과 SNR threshold sensitivity를 보고한다.

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

**Image-space**

- Reinhard/Macenko
- RGB 또는 OD affine
- Paired LF affine
- 주파수 기반 attenuation/calibration 방법 1개
- 가능하면 학습 기반 normalization 1개

**Feature-space**

- CORAL
- Orthogonal Procrustes 또는 ComBat
- 접근 가능하고 공정한 조건이면 FEATMAP

전체 방법을 무리하게 늘리기보다 동결된 4개 PFM에서 image 1–2개, feature 1–2개의
최소 완결 benchmark를 먼저 수행한다. 모든 방법은 train slide에서만 적합하고 held-out
slide와 tissue에서 평가한다.

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
- Tissue morphology 또는 baseline spectrum이 correction benefit을 예측하는지 분석
- Mean effect가 유사해도 tissue별 부호가 바뀌는지 보고

#### 2.12 Content fidelity와 제한된 tissue-type validation

PanNormal에는 normal tissue type 외의 독립 biological/downstream label이 없다. 따라서
현재 연구에서 직접 판정할 범위를 다음과 같이 제한한다.

1. **Primary representation/content endpoint:** same-location retrieval/content margin,
   raw→corrected geometry preservation 후보 중 하나를 결과 확인 전에 동결한다.
2. **Collapse guardrail:** embedding variance, effective rank와 pairwise-distance 유지.
3. **Secondary coarse biology:** tissue-type retrieval 또는 grouped linear probe와
   scanner별 tissue-neighborhood agreement.
4. **Optional structural sensitivity:** 사전 QC를 통과한 nucleus segmentation/count/boundary
   agreement. 이는 독립 biological ground truth로 승격하지 않는다.

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

#### 2.14 PLISM deferred extension

PLISM은 PanNormal core 분석과 원고의 완료 조건이 아니다. PanNormal의 endpoint, margin,
alignment contract와 correction benchmark를 먼저 동결하고 완료한 뒤 별도 frozen extension으로
시작한다. 그때 적용할 원칙은 다음과 같다.

- PanNormal outcome과 threshold를 PLISM 결과 확인 전에 동결
- 한 stain condition의 scanner-only contrast를 primary로 지정
- Scanner × stain은 secondary
- Aligned tile group/core를 blocking unit으로 사용
- HF 결과보다 interpolation audit를 먼저 제시
- Native scan이 없으면 `effective transfer after PLISM preprocessing`으로 명명

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

결론 형식:

> The primary spectral range and estimator were retained/restricted after prespecified
> registration and resampling audits.

이 절이 통과하지 않으면 이후 결과는 scanner optics가 아니라 preprocessing 이후의
effective signature로 제한한다.

#### 3.2 Audited paired scanner spectrum — E1

- Unnormalized power와 anchor-normalized ERT를 나란히 제시
- Scanner별 curve, CI와 slide spread
- 현재 GT450 1.595, AKOYA 0.297 등의 수치는 재등록·alias audit 후 갱신
- Optical MTF가 아니라 effective relative spectrum이라는 제한을 반복

핵심 질문:

> Scanner acquisition/preprocessing은 동일 content에서 재현 가능한 frequency-selective
> signature를 만드는가?

#### 3.3 Scanner spectrum은 content에 의존한다 — E2

- Scanner fixed mean
- Scanner × tissue, scanner × slide와 sampling variation
- AKOYA 포함/제외 sensitivity
- 현재 85.5%는 잠정 descriptive value로만 유지

핵심 결론:

> Scanner signature exists as a population tendency, but its observed magnitude is a
> scanner × content response.

#### 3.4 단일 blur–sharpen scalar로 충분하지 않다 — E3

- Nearest equivalent HF gain
- Cross-validated off-axis lack-of-fit와 CI
- 어떤 frequency region과 scanner에서 residual이 남는지

이 결과는 frequency control family가 단일 retention scalar가 아니라 attenuation,
boost, phase-cancellation과 paired consensus를 포함해야 하는 이유를 제공한다.

#### 3.5 Controls가 attainable invariance–fidelity region을 정의한다 — E4

- Complete blur와 global mean이 낮은 radius와 collapse를 함께 만드는지
- Partial paired oracle이 fidelity를 유지하며 radius를 줄이는지
- Sharpening이 nuisance divergence quadrant를 채우는지
- 동결된 4개 PFM에서 slide-bootstrap frontier와 CI

이 절이 논문의 evaluation principle을 경험적으로 입증한다.

#### 3.6 실제 correction의 평가는 separability-only 결론을 바꾼다 — E5

- Image-space 및 feature-space correction의 frontier
- Raw 대비 FC-RR와 fidelity pass/fail
- Separability ranking과 fidelity-constrained decision의 일치/불일치
- 특정 rank reversal을 선택적으로 강조하지 않고 전체 method grid를 보고

최소 성공 조건:

- 적어도 하나의 실제 correction이 population 수준으로 평가됨
- Separability improvement와 fidelity 결과가 함께 보고됨
- Protocol을 적용했을 때 단일 separability 지표가 숨긴 의사결정 차이가 드러남

#### 3.7 Correction 효과는 tissue와 slide에 따라 달라진다 — E6

- Tissue-specific correction slope
- PFM × tissue × correction interaction
- Overall winner가 모든 tissue에서 winner인지 평가
- 어떤 baseline spectrum/morphology가 correction response를 예측하는지

#### 3.8 Content fidelity와 tissue-type evidence — E7

- Prespecified primary representation/content endpoint와 margin
- Variance, effective rank와 pairwise-distance collapse guardrail
- Tissue-type retrieval/grouped probe와 cross-scanner tissue-neighborhood agreement
- Tissue class imbalance와 minimum-slide sensitivity
- Optional nucleus/spatial structural sensitivity

Self-cosine이 높거나 tissue type이 유지된다는 이유만으로 독립 biological fidelity를
확정하지 않는다. 이 절의 결론은 representation/content preservation으로 제한한다.

#### 3.9 PLISM post-core extension — E8, 현재 보류

이 절은 PanNormal core 원고의 필수 결과가 아니다. 후속 extension을 시작할 경우에만
interpolation audit, 한 stain의 scanner-only primary, scanner × stain secondary와
internal–external effect comparison을 사전 동결 계획에 따라 추가한다.

#### 3.10 Background 결과 — E9, Supplement 중심

- Background feature가 tissue ERT의 held-out prediction을 일관되게 개선하지 않음
- Noise/QC characterization으로서의 제한된 역할
- 동일 processing chain의 noise-floor sensitivity

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
| E0a | Common physical grid audit | 모든 scanner pixel이 AT2 물리 좌표를 따르는가? | 부분 확인 | Methods 2.2, Results 3.1 | Transform, scale, MPP source 명시 |
| E0b | Coarse-to-fine registration | AKOYA 경계 포화가 ERT를 교란하는가? | 진단 완료, 재등록 미완 | Methods 2.2, Results 3.1 | Old/new paired sensitivity와 residual QC |
| E0c | 2D aliasing audit | GT450/VERSA downsampling이 high band를 오염하는가? | 미완 | Methods 2.3, Results 3.1 | Mixing matrix와 허용 band 동결 |
| E0d | Anchor/noise-floor audit | ERT shape가 anchor와 noise에 견고한가? | 부분 완료 | Methods 2.4, Results 3.1/Supplement | Registered-chain sensitivity와 SNR 보고 |
| E1 | Paired ERT | Scanner별 frequency signature가 재현되는가? | **잠정** | Results 3.2 | E0 통과 후 109-slide 재산출 |
| E2 | Hierarchical decomposition | Signature가 tissue/slide에 의존하는가? | **잠정** | Results 3.3 | Corrected ERT로 재적합 |
| E3 | Scalar reducibility test | 단일 blur–sharpen 축으로 충분한가? | 부분 완료 | Results 3.4 | CV lack-of-fit와 bootstrap CI |
| E4 | Control population | Collapse, oracle, divergence가 frontier를 정의하는가? | Pilot | Results 3.5 | 109 slides, 4 PFM, CI |
| E5 | Correction benchmark | Protocol이 실제 방법 선택을 바꾸는가? | 미완 | Results 3.6 | Image+feature comparator와 held-out slide |
| E6 | Content-dependent correction | Correction benefit이 tissue/slide에 따라 달라지는가? | 미완 | Results 3.7 | Primary interaction/random slope |
| E7 | Content/tissue fidelity | Invariance gain이 representation과 coarse tissue content를 보존하는가? | 미완 | Results 3.8 | Primary content endpoint + collapse guardrail + tissue secondary |
| E8 | PLISM post-core extension | 결론이 외부 scanner/stain에서 재현되는가? | **보류** | Future/Results 3.9 | PanNormal 완료 후 별도 frozen plan |
| E9 | Background characterization | Background가 tissue ERT를 설명하는가? | 완료 | Supplement/Results 3.10 | 기존 null + registered noise sensitivity |

## 7. Figure와 Table 설계

### Main figures

**Figure 1 — Study design and estimator validity**

- Paired acquisition과 분석 단위
- Invariance-only degeneracy 개념도
- AKOYA registration saturation 전후
- GT450/VERSA resampling mixing audit

**Figure 2 — Audited frequency-resolved scanner signatures**

- 2D/radial spectrum
- Unnormalized power와 normalized ERT
- Scanner별 cohort curve와 slide spread
- Scalar blur–sharpen projection residual

**Figure 3 — Scanner × content structure**

- Scanner fixed mean
- Tissue/slide/sampling decomposition
- Tissue-specific slopes와 AKOYA subset sensitivity

**Figure 4 — Control-bounded invariance–fidelity frontier**

- Raw, collapse, global mean, paired oracle, sharpening
- PFM별 frontier와 bootstrap CI
- Collapse guardrail

**Figure 5 — Image- and feature-space correction benchmark**

- Actual methods의 frontier
- Separability-only ranking 대 fidelity-constrained decision
- Tissue/PFM별 correction heterogeneity

**Figure 6 — Content preservation and tissue-level evidence**

- Primary content-fidelity endpoint와 collapse guardrail
- Tissue-type secondary endpoint와 class-size sensitivity
- Optional structural sensitivity

### Main tables

| Table | 내용 |
|---|---|
| Table 1 | Cohort, scanner, native/target MPP, scale, interpolation, compression |
| Table 2 | Audited scanner spectrum과 content variance, 95% CI |
| Table 3 | PFM × method FC-RR, fidelity gate, collapse status와 decision |
| Table 4 | Content-fidelity, collapse guardrail과 tissue-type secondary endpoint |

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

### 잠정적으로만 보존할 수치

- GT450 high-band ERT 1.595
- AKOYA high-band ERT 0.297
- High-band scanner main descriptive fraction 85.5%
- Scanner × tissue/slide fractions

이 수치는 기존 분석의 provenance로 남기되 E0 audit와 재등록 후 교체한다. 초록, 제목,
Highlights와 최종 결론에는 audited table만 사용한다.

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

1. Common grid와 transform/interpolation provenance 회수
2. Existing offsets에서 constant 대 affine slide prior 진단
3. Sentinel 10–20 slides 선정
   - AKOYA registration worst/median/best
   - GT450/VERSA aliasing risk
   - Tissue와 slide strata 균형
4. 현재 registered WSI의 corrected integer crop과 기존 VALIS rigid output 대조
5. 두 경로가 모두 기준 미달일 때만 native WSI에서 VALIS rigid 재실행
6. `old/new registration × original/anti-aliased resampling` factorial audit
7. 2D alias mixing, anchor와 noise-floor sensitivity

**Stop:** Common physical coordinates가 보장되지 않음.

**Revise:** Alias leakage가 사전 기준을 넘거나 registration correction이 ERT를 실질적으로
변경함. Common grid/band/estimator를 고친 뒤 재시작.

**Go:** Primary band와 corrected registration contract를 동결할 수 있음.

### Phase 1 — E1–E3 rebuild

1. 109-slide ERT 재산출
2. Scanner × tissue/slide model 재적합
3. Scalar reducibility test 강화
4. Main Figure 1–3의 locked tables 생성

**Go:** 결과가 E0 contract를 통과하고 slide-bootstrap uncertainty가 완전함.

### Phase 2 — Evaluation protocol population test

1. 4개 core PFM의 TRIDENT commit/checkpoint/preprocessing contract 동결
2. 109 × 100 canonical-center manifest와 모델별 native-FOV bounds audit
3. E4 control population 실행
4. Primary content endpoint, margin과 collapse guardrail 확정
5. Minimal E5 image+feature benchmark
6. FC-RR 및 frontier 산출

**Go:** 실제 correction에서 invariance와 fidelity를 함께 판정할 수 있음. E4만으로 논문을
완결하지 않는다.

### Phase 3 — Heterogeneity and content fidelity

1. E6 tissue/slide correction effect
2. E7 primary representation/content endpoint와 collapse guardrail
3. Tissue-type secondary와 optional structural sensitivity
4. Remaining PFM panel 확장 여부 결정

**Go:** 사전 동결한 content-fidelity 판정과 slide-level inference가 완전하고, tissue-type
결과의 표본수 제한이 명시됨.

### Phase 4 — PanNormal manuscript lock

- 초록 수치를 locked tables와 대조
- 결과 확인 후 method, PFM 또는 threshold를 선택하지 않음
- Null과 failed-fidelity method를 frontier에 그대로 포함
- Main/Supplement 배치와 terminology audit
- `biological fidelity`와 `clinical validation` 과장 여부 최종 점검
- MedIA Guide for Authors, highlights와 disclosure 최종 확인

### Post-core extension — PLISM, 현재 보류

1. PLISM interpolation audit
2. 한 stain scanner-only primary
3. Scanner × stain secondary
4. Internal–external effect comparison

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

다음 네 결과만 최종 locked 수치로 채운다.

1. Audited scanner spectrum과 content dependence
2. Collapse/oracle control이 만드는 attainable region
3. Actual correction에서 separability-only 대 fidelity-constrained decision
4. Representation/content preservation과 tissue-type secondary result

### Conclusion

Scanner harmonization은 scanner signal을 최소화하는 문제가 아니라 content fidelity를
보존하면서 scanner-associated variation을 줄이는 constrained evaluation problem이다.

## 12. Highlights 초안

- Scanner separability cannot distinguish alignment from representation collapse.
- Paired spectra reveal frequency- and content-dependent scanner signatures.
- Physical controls bound the attainable invariance–fidelity region.
- Corrections are ranked only after passing a prespecified fidelity constraint.
- The framework is tested across four PFMs and tissue/slide strata.

Highlights는 제출 시 Elsevier의 최신 글자 수 제한을 다시 확인한다.

## 13. 문서 관리 규칙

- 이 문서는 storyline과 claim architecture의 기준이다.
- [`final_study_protocol.md`](final_study_protocol.md)는 세부 데이터·모델·구현
  계약을 담으며, 다음 갱신에서 이 storyline에 맞춰 E0와 fidelity contract를 반영한다.
- [`manuscript/scanner_spectrum_media_draft_ko.md`](manuscript/scanner_spectrum_media_draft_ko.md)는
  원고 working draft이며 E0 통과 전의 숫자는 잠정 표시를 유지한다.
- 과거 판단과 실행 로그는 [`archive/`](archive/)에서 provenance로만 보존한다.
- 결과 문서에는 `확정`, `잠정`, `pilot`, `미완` 상태를 명시한다.

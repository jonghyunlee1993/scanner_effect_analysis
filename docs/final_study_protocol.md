# Scanner Spectrum: final study protocol

**Status:** final-experiment design, 2026-07-31

**Primary target:** *Medical Image Analysis*
**Role of this document:** 논문 주장, 분석 단위, 비교군과 external validation을
동결하는 기준 문서

## 1. 한 문장 결론

병리 스캐너 효과는 하나의 전역적인 style layer가 아니라, **교정 가능한 저주파
외관 차이와 scanner가 보존·손실하는 고주파 정보가 조직 및 슬라이드 내용과
상호작용한 결과**이며, 외관 정규화 또는 scanner 분류 정확도의 감소만으로 PFM의
생물학적 invariance를 보장할 수 없다.

이 연구의 차별점은 “PFM이 scanner-sensitive하다”를 다시 증명하는 데 있지 않다.
동일 조직의 paired acquisition을 사용하여 그 sensitivity를 물리적 전달 특성,
조직 의존성, representation fidelity의 세 층으로 분해하고, 어떤 정규화가 실제
정보 보존인지 혹은 collapse인지 구별하는 데 있다.

## 2. 논문의 중심 서사

### 2.1 Scanner effect는 존재하지만 절대적인 scanner style은 아니다

스캐너마다 평균적으로 반복되는 차이는 분명하다. 특히 고주파 전달량에서는 scanner
main effect가 크다. 그러나 같은 scanner contrast도 tissue type과 physical slide에
따라 크기와 때로는 방향이 달라진다. 따라서 가장 정확한 표현은 다음과 같다.

> Scanner style exists as a population-level tendency, but the observed image and
> representation shift is a scanner × content response rather than a detachable,
> globally constant style.

`tissue type`은 단순 공변량이 아니다. 유방과 신장처럼 조직군마다 핵 밀도, gland
boundary, stroma texture와 염색 분포가 다르면 동일한 optical transfer function이
서로 다른 관측 차이를 만든다. 같은 tissue type 안에서도 section quality, 초점,
염색량과 morphology가 달라 slide-specific response가 남는다.

### 2.2 저주파와 고주파는 서로 다른 문제다

- **Low frequency (LF):** 색, 조명, stain-like appearance의 상당 부분은
  scanner-specific affine transform으로 맞출 수 있다.
- **High frequency (HF):** blur, sampling, sharpening, compression과 phase가
  얽혀 있다. 이미 잃은 구조 정보는 전역 변환으로 복구할 수 없다.
- **Representation:** LF pixel error가 줄어도 PFM scanner separability가 그대로일
  수 있다. HF를 없애 scanner separability만 줄이면 content도 함께 사라질 수 있다.

따라서 평가는 항상 `scanner invariance`와 `biological fidelity`를 함께 측정한다.
scanner centroid가 가까워진 결과만으로 성공이라 부르지 않는다.

### 2.3 왜 global image transformation이 충분하지 않은가

전역 변환이 모든 조직에서 잘 작동하려면 scanner 차이가 내용과 독립적이어야 한다.
하지만 내부 데이터는 scanner main effect 위에 scanner × tissue와 scanner × slide
variation이 남는다는 것을 보여준다. LF correction은 유용한 calibration 단계이지만,
HF reconstruction은 입력에 없는 정보를 요구하고 조직에 따른 error mode를 만든다.

따라서 최종 연구는 하나의 universal canonical image를 만드는 경쟁이 아니다.
대신 다음 질문에 답한다.

1. 어느 주파수 대역까지 전역 교정이 안전한가?
2. 어느 지점부터 scanner와 content의 상호작용이 지배적인가?
3. PFM별로 어떤 주파수 손실에 민감하며, invariance–fidelity frontier는 어디인가?
4. 이미지 수준 보정과 feature 수준 보정 중 어떤 접근이 어떤 조건에서 유효한가?

## 3. 현재 내부 근거

### 3.1 Cohort와 분석 단위

- 109 physical slides
- 37 tissue types
- 6 scanners
- slide–scanner당 tissue patch 100개
- 100개를 20-patch disjoint replicate 5개로 나누어 sampling uncertainty 추정
- AT2를 상대 transfer reference로 사용
- low–mid, mid, high의 세 주파수 band를 사전 정의

패치는 관측치지만 독립적인 biological replicate가 아니다. 추론과 split의 기본 단위는
physical slide이며, tissue type은 slide를 묶는 상위 biological grouping으로 취급한다.

### 3.2 Scanner spectrum

AT2 대비 high-band fold transfer의 cohort estimate는 다음과 같다.

| Scanner | High-band transfer vs AT2 |
|---|---:|
| GT450 | 1.595 |
| VERSA | 1.011 |
| AKOYA | 0.297 |
| S60 | 0.994 |
| S360 | 0.784 |

이는 scanner를 다섯 개 범주 비교에만 두지 않고, task-relevant spectral transfer를
가진 연속 좌표로 다룰 수 있음을 보여준다. 다만 tissue PSD에서 얻은 값은 엄밀한
absolute MTF가 아니라 **effective relative transfer**다. noise correction, sampling,
sharpening과 compression의 영향을 분리하기 전에는 MTF라는 용어를 과도하게 쓰지 않는다.

### 3.3 Scanner, tissue와 slide의 분해

다섯 non-reference scanner를 균형 있게 놓은 100-patch mean의 descriptive variance
decomposition은 다음과 같다.

| Band | Scanner main | Scanner × tissue | Scanner × slide | Sampling |
|---|---:|---:|---:|---:|
| Low–mid | 71.1% | 6.6% | 21.2% | 1.0% |
| Mid | 85.5% | 3.2% | 10.7% | 0.5% |
| High | 85.5% | 2.5% | 11.2% | 0.8% |

여기서 scanner는 연구 대상인 고정된 장비 집합이므로 `scanner main %`는 formal
random-effect variance가 아니라 balanced marginal descriptive variance다. 장비 구성에
민감하며, 특히 AKOYA를 제외하면 scanner main fraction은 low–mid/mid/high에서 각각
약 5%/28%/65%로 감소한다. 따라서 “scanner가 전체 variation의 85.5%를 보편적으로
설명한다”와 같은 문장은 쓰지 않는다.

각 scanner에서 between-slide variance 중 tissue type이 설명하는 비율도 균일하지 않다.

| Band | GT450 | VERSA | AKOYA | S60 | S360 |
|---|---:|---:|---:|---:|---:|
| Low–mid | 29% | 39% | 60% | 58% | 0% |
| Mid | 15% | 6% | 52% | 40% | 3% |
| High | 17% | 11% | 28% | 27% | 3% |

scanner × tissue variance는 15 scanner-band 조합 중 8개에서 BH 보정 후 유의했다.
S60은 AT2와 population mean이 가깝지만 tissue interaction이 상당했고, S360은 tissue
share가 낮은 대신 slide-specific variation이 컸다. 이것이 “scanner × slide style”을
고려해야 하는 직접적인 이유다.

관련 내부 그림:

- [`figure1_cohort_spectral_overview.png`](../outputs/exp05_spectral_cohort_109/figure1_cohort_spectral_overview.png)
- [`figure2_mixed_effects.png`](../outputs/exp05_spectral_cohort_109/figure2_mixed_effects.png)
- [`figure3_blur_sharpen_bridge.png`](../outputs/exp05_spectral_cohort_109/figure3_blur_sharpen_bridge.png)
- [`figure4_tissue_random_slopes.png`](../outputs/exp05_spectral_cohort_109/figure4_tissue_random_slopes.png)

`outputs/`는 Git으로 추적하지 않으므로 이 링크는 분석 환경에서만 유효하다.

### 3.4 LF correction의 범위

기존 paired pilot에서 LF MAE는 raw 0.2531에서 requested affine 0.0297로 감소했고,
clipping을 포함한 operational transform에서는 0.0366이었다. requested transform 기준
약 88%의 LF error reduction이며 SSIM은 0.281에서 0.433으로 증가했다.

그러나 LF scanner balanced accuracy가 internal lattice에서 0.813→0.556,
S60 lattice에서 0.928→0.780으로 감소한 반면, UNI scanner balanced accuracy는 각각
0.998→0.998과 0.998→1.000으로 사실상 변하지 않았다. 즉 픽셀 수준 LF alignment와
PFM representation alignment는 같은 문제가 아니다.

sharp paired target cosine gain도 평균 -0.0202였고, 14 scanner-location 조합 중
7개는 개선, 7개는 악화했다. 14개 중 6개는 두 test slide 사이에서 효과 방향이
바뀌었다. 표본이 작은 mechanistic pilot이므로 population estimate로 인용하지 않고,
최종 실험의 가설과 intervention 범위를 정하는 근거로만 쓴다.

### 3.5 HF intervention이 보여준 실패 모드

- 1,024 train images의 global coefficient HF mean은 individual HF RMS의 4.27%만
  보존했다. phase cancellation 때문에 실질적으로 blur negative control이 됐다.
- registered same-location HF mean 25%는 scanner radius 0.095→0.069,
  consensus gain +0.028, self cosine 0.957, retrieval 1.0으로 유망한 oracle positive
  control이었다.
- HF를 완전히 제거하면 self cosine 0.065, consensus gain -0.781,
  content margin 약 0, retrieval 0.67이었다. scanner radius는 0.040으로 작아졌지만
  이는 invariance가 아니라 collapse다.
- sharpening 1.0은 self cosine 0.828을 유지했지만 radius를 0.095→0.120으로 늘렸다.
- 같은 HF intervention도 LF reference에 따라 반응이 달랐고, LF/HF recombination은
  일부 조건에서 out-of-gamut clipping을 일으켰다. AKOYA LF reference에서는
  non-AKOYA source pixel의 9–19%가 clipped되었다.

이 결과도 3개 location의 mechanistic trajectory이므로 최종 population inference로
사용하지 않는다. 최종 연구에서는 동일한 intervention을 109 slides와 여러 PFM에
확장한다.

### 3.6 Background가 주는 것과 주지 않는 것

Exp-07에서 109 × 6 × 100, 총 65,400개의 raw white-space patch를 추출했다.
Exp-08은 within-slide AT2 contrast에 background NPS와 OD feature를 추가하여 tissue
transfer 설명력이 증가하는지 평가했다.

- 15 scanner-band 모델 중 AIC는 3개, BIC는 0개에서 background model을 선호했다.
- likelihood-ratio result는 15개 모두 BH 보정 후 유의하지 않았다.
- held-out prediction은 2/15에서만 개선되었고 크기는 +0.69%, +0.09%였다.
  나머지 13개에서는 악화했다.
- background coefficient 75개 중 5개만 nominally significant였고 BH 보정 후 0개였다.
- background patch acceptance sensitivity도 15개 모두 nominally significant하지 않았다.

결론은 background가 쓸모없다는 것이 아니다. White space는 scanner dark/white balance,
noise fingerprint, compression artifact와 acquisition QA를 보여준다. 그러나 tissue가 없는
background는 tissue-dependent optical transfer를 식별할 수 없고, 현재 결과에서는
tissue-domain spectrum 차이에 대한 실질적인 증분 설명력이 없었다.

따라서 background는 다음 역할로 제한한다.

- supplemental scanner/noise characterization
- acquisition QC와 failed-scan detection
- tissue spectrum의 noise-floor sensitivity analysis

background patch를 global transform의 핵심 학습 입력으로 사용하거나, 이를 독립적인
full NPS/MTF 측정이라고 주장하지 않는다. 진정한 detector NPS/DQE에는 통제된 균일 노출과
입력 광자 통계가 필요하다.

## 4. 최종 multi-PFM 실험

### 4.1 연구 질문

1. scanner spectral distance가 PFM representation drift를 예측하는가?
2. 그 관계의 slope는 PFM, tissue type과 slide에 따라 달라지는가?
3. LF affine은 외관, representation, biology 중 어디까지 개선하는가?
4. HF retention/sharpening은 scanner invariance와 content fidelity 사이에 어떤
   frontier를 만드는가?
5. image-space transform과 feature-space harmonization 중 어떤 것이 새로운 tissue,
   slide와 scanner에 더 잘 일반화되는가?

### 4.2 PFM panel

단순히 모델 수를 늘리지 않고 학습 패러다임과 입력 전처리가 다른 6–8개 모델을
사전 선택한다. 후보군은 다음과 같다.

- UNI 또는 UNI2-h
- Virchow2
- Prov-GigaPath
- H-Optimus 계열
- CONCH
- Phikon 또는 Hibou
- stain-normalized pathology model(예: EXAONEPath 계열, 접근 가능할 때)
- DINOv2 또는 ImageNet ResNet natural-image baseline

정확한 모델·checkpoint·입력 크기·색상 전처리·feature layer를 outcome을 보기 전에
lock한다. 같은 모델 계열의 중복보다 서로 다른 inductive bias를 우선한다.

### 4.3 Intervention matrix

모든 PFM에 동일한 physical field of view와 평가 계약을 적용한다.

| Family | Conditions | 역할 |
|---|---|---|
| Original | raw | 기준 |
| LF | scanner-pair affine | calibratable appearance test |
| HF attenuation | retention 0.75, 0.50, 0.25, 0 | invariance–collapse trajectory |
| HF boost | gain 1.25, 1.50, 2.00 | sharpening/noise amplification test |
| HF oracle | registered same-location mean, weight 0.25 | attainable image-space positive control |
| HF global | global coefficient mean | phase-cancellation negative control |
| Feature-space | CORAL, Procrustes/ComBat, 가능하면 FEATMAP | representation-level positive comparator |

LF와 HF intervention은 순서를 명시하고, recombination 후 clipping fraction을 반드시
보고한다. oracle은 실용적 deployable method가 아니라 정보 보존의 상한을 확인하는
positive control이다.

### 4.4 Outcomes

**Image/physics**

- LF MAE와 SSIM
- out-of-gamut/clipping fraction
- radial PSD와 band-wise relative transfer
- background-corrected sensitivity estimate
- 가능하면 slanted-edge 또는 resolution target absolute anchor

**Representation**

- paired cosine similarity와 consensus gain
- scanner balanced accuracy
- cross-scanner same-location retrieval
- scanner centroid distance/radius
- biological content margin와 neighborhood preservation
- collapse diagnostic: embedding variance, effective rank, pairwise distance

**Biology**

- tissue-type classification/retrieval
- nucleus segmentation agreement
- nucleus count, area, eccentricity와 boundary fidelity
- 가능하면 cell graph 또는 spatial neighborhood preservation

scanner invariance endpoint는 최소 하나의 content-fidelity endpoint와 짝지어 해석한다.
예를 들어 scanner BACC가 감소해도 retrieval, effective rank와 nuclear boundary가 함께
무너지면 실패다.

### 4.5 Statistical model

scanner와 intervention은 우리가 비교하려는 고정 효과로 둔다. PFM과 frequency band도
고정 효과이며, tissue type과 slide가 scanner/intervention에 다르게 반응하도록 random
slope 또는 계층적 partial pooling을 사용한다.

개념적 모델은 다음과 같다.

```text
outcome ~ scanner * intervention * PFM * band
        + (1 + scanner + intervention | tissue_type)
        + (1 + scanner + intervention | slide_id)
```

실제 모델은 관측 수와 singular fit을 고려하여 단계적으로 적합한다.

1. scanner × PFM × intervention의 primary fixed effects
2. tissue random intercept + scanner random slope
3. slide random intercept + scanner/intervention random slope
4. 필요할 때 band별 모델로 분리하거나 Bayesian partial pooling 적용

scanner 자체를 random slope라고 부르지 않는다. **tissue와 slide에 대한 scanner
contrast의 random slope**가 정확한 표현이다. 모든 bootstrap과 CV는 physical slide를
block으로 사용하며 patch-level pseudo-replication을 피한다. confidence interval과
effect size를 우선 보고하고 family별 multiple testing에 BH correction을 적용한다.

Primary endpoint는 outcome 확인 전에 제한한다.

1. paired embedding consensus gain
2. cross-scanner same-location retrieval
3. tissue content margin 또는 tissue retrieval
4. nuclear boundary/count fidelity

scanner BACC와 spectral association은 중요한 secondary/mechanistic endpoints다.

## 5. PLISM external validation

### 5.1 왜 PLISM인가

[PLISM](https://p024eb.github.io/)은 동일한 biological TMA를 여러 scanner와 stain
condition으로 구성한 controlled factorial resource다. 내부 데이터와 scanner가 일부
겹치면서도 조직 구성, acquisition site와 registration pipeline이 독립적이어서 최종
주장의 재현성을 시험하기에 적합하다.

[Owkin PLISM tiles](https://huggingface.co/datasets/owkin/plism-dataset-tiles)는
약 1,481,298개의 224 × 224 tile, 0.5 µm/px, 약 146 GB를 제공하며 CC BY 4.0이다.
7 scanner × 13 H&E stain conditions의 91개 조건이 있고, 조건당 16,278개 tile과
46 tissue types가 포함된다. 원 PLISM 연구는 3,417개의 aligned image group을 91개
조건에 걸쳐 제공한다. 원 자료와 라이선스는 [Ochi et al.](https://doi.org/10.1038/s41597-024-03122-5)을
따른다.

중요하게도 91개 WSI/condition은 91명의 독립 환자 cohort가 아니다. 같은 TMA를 반복
촬영·염색한 기술적 factorial dataset이므로 PLISM의 역할은 **external mechanistic
validation**이지 독립적인 clinical validation이 아니다.

### 5.2 내부 데이터와의 연결

metadata audit 전의 예상 overlap은 AT2, GT450, S60, S360이며 PLISM은 P, S210, SQ 등
내부에 없는 장비를 추가한다. vendor/model naming과 reference acquisition은 첫 pilot에서
반드시 원 metadata로 확정한다.

PLISM에서는 다음 두 질문을 분리한다.

1. 동일 stain condition에서 scanner contrast가 내부 spectrum–representation 관계를
   재현하는가?
2. scanner effect가 stain condition에 따라 달라지는가(scanner × stain)?

첫 질문이 primary다. 13 stain 전체의 scanner × stain 분석은 tissue/content interaction을
한 단계 확장하는 secondary analysis다.

### 5.3 Frozen external protocol

- 내부 데이터에서 정한 band boundary, LF transform family, intervention grid, endpoint와
  success criterion을 PLISM 결과를 보기 전에 동결한다.
- PLISM으로 내부 LF affine이나 threshold를 재학습하지 않는다.
- 내부 영상은 PLISM에 맞춰 0.5 µm/px의 동일 224-pixel physical field of view로 crop한다.
  FFT 분석을 위해 PLISM tile을 임의 resize하지 않는다.
- PLISM tile row를 독립 표본으로 세지 않는다. tissue/core/aligned tile group을 block 또는
  random effect로 사용한다.
- primary scanner-only 분석은 stain condition을 고정한다.
- external effect의 방향, 크기와 CI를 내부 결과와 나란히 보고한다.

PLISM tile은 Elastix registration과 common reference로의 resampling을 거쳤기 때문에
interpolation이 HF spectrum을 바꿀 수 있다. 따라서:

- PFM 및 LF 외부 검증에는 registered tile을 그대로 사용할 수 있다.
- HF/relative-transfer 결과는 먼저 identity/resampling interpolation audit을 수행한다.
- native/non-warped scan을 사용할 수 없다면 `scanner MTF`가 아니라 **effective transfer
  after PLISM preprocessing**이라고 명시한다.

또한 [PLISM benchmark](https://github.com/owkin/plism-benchmark)는 이미 다수의
extractor를 순위화한다. 본 연구는 그 leaderboard를 반복하지 않고, paired physics와
invariance–fidelity mechanism을 외부에서 검증하는 데 PLISM을 사용한다.

## 6. 선행 연구와 위치

최근 연구는 PFM의 scanner sensitivity를 대규모로 보여주기 시작했다.

- [Thiringer et al. (2026)](https://arxiv.org/abs/2601.04163): 384 breast slides,
  5 scanners, 14 PFMs에서 scanner embedding과 calibration shift 분석
- [ScanGen (2025)](https://arxiv.org/abs/2507.22092): 323 lung specimens,
  6 scanners, 5 PFMs와 contrastive mitigation
- [Henriksen et al. (2026)](https://arxiv.org/abs/2602.22347): 27,042 WSIs,
  8 PFMs에서 scanner robustness와 downstream loss
- [FEATMAP (2026 preprint)](https://www.biorxiv.org/content/10.64898/2026.07.02.736184v1.full):
  paired multi-scanner data와 feature-level affine harmonization, PLISM external analysis

우리의 기여는 규모 경쟁이나 또 하나의 robustness leaderboard가 아니다.

1. paired tissue spectrum으로 scanner difference를 연속적인 물리 축으로 표현한다.
2. scanner main effect와 scanner × tissue/slide response를 분리한다.
3. LF calibration과 HF information loss를 분리한다.
4. scanner invariance가 biological fidelity를 보존했는지 명시적으로 검증한다.
5. 여러 PFM과 PLISM에서 같은 기전이 재현되는지 시험한다.

이 framing은 *Medical Image Analysis*에 적합할 수 있다. 다만 multi-PFM population
experiment, biological endpoint, external validation이 모두 완성되어야 하며, 현재
내부 mechanistic pilot만으로는 충분하지 않다.

## 7. 논문 그림 계획

1. **Scanner spectrum:** scanner별 radial spectrum, relative transfer curve와 연속 축
2. **Content interaction:** scanner main, scanner × tissue, scanner × slide decomposition
3. **LF calibration across PFMs:** image alignment와 representation alignment의 분리
4. **HF invariance–fidelity frontier:** attenuation, oracle, global mean, sharpening trajectory
5. **PFM × scanner × tissue:** 모델별 susceptibility와 task-band pre-flight relation
6. **PLISM validation:** frozen internal prediction의 외부 재현과 scanner × stain 확장

Supplement에는 background/NPS sensitivity, interpolation audit, clipping, mixed-model
diagnostics, scanner subset sensitivity와 legacy pilot를 둔다.

## 8. 허용되는 주장과 금지할 주장

### 지지 가능한 주장

- Scanner identity has a strong population-level spectral signature.
- The magnitude of that signature is modified by tissue and slide content.
- LF appearance correction alone does not guarantee PFM alignment.
- Reduced scanner separability can reflect either genuine alignment or information collapse.
- Background noise characterization is useful for QC but adds little explanatory power for
  tissue-domain transfer in this cohort.

### 현재 근거로 지지하지 않는 주장

- 하나의 common canonical image가 수학적으로 불가능하다.
- tissue가 scanner보다 항상 더 중요하다.
- LF는 중요하지 않고 HF만 중요하다.
- 모든 PFM이 같은 방식으로 반응한다.
- white-space background만으로 detector MTF/NPS/DQE를 완전히 측정했다.
- PLISM이 독립 환자 clinical validation이다.

## 9. 최종 실행 순서와 stop/go 기준

### Phase A — contract freeze

- scanner/tissue/slide metadata audit
- PFM panel, checkpoints, FOV와 preprocessing lock
- band boundary와 intervention implementation test
- primary outcomes와 mixed-model formula lock

**Go:** 동일 입력에서 모든 PFM의 deterministic extraction과 sample identity 검사가 통과.

### Phase B — PLISM feasibility pilot

- dataset metadata와 scanner naming 확인
- 한 stain, 일부 scanner/tissue로 download and decode test
- aligned group ID와 blocking unit 검증
- resampling/interpolation spectrum audit

**Go:** same-content grouping이 재현되고 HF preprocessing bias를 정량화할 수 있음.

### Phase C — internal population experiment

- 109 slides × 100 patches × 6 scanners
- raw/LF/HF/feature-space conditions
- shared PFM embeddings와 image/biology metrics
- slide-blocked bootstrap와 hierarchical model

**Go:** 최소 3개 PFM paradigm에서 primary endpoints가 완전하고 QC failure가 사전 기준 이내.

### Phase D — frozen PLISM validation

- 한 stain의 scanner-only primary replication
- scanner × stain secondary analysis
- internal–external effect comparison

### Phase E — locked figures and manuscript

- outcome을 본 뒤 조건·모델을 선택하지 않는다.
- 모든 scanner subset, background와 interpolation sensitivity를 supplement에 공개한다.
- null result도 invariance–fidelity map에 그대로 포함한다.

## 10. 구현 및 provenance

핵심 코드는 다음에 있다.

- [`../src/analyze_exp05_spectral_cohort.py`](../src/analyze_exp05_spectral_cohort.py)
- [`../src/analyze_exp05_tissue_random_slopes.py`](../src/analyze_exp05_tissue_random_slopes.py)
- [`../src/analyze_exp08_background_mixed_effects.py`](../src/analyze_exp08_background_mixed_effects.py)
- [`../src/prenorm/exp02/pairwise.py`](../src/prenorm/exp02/pairwise.py)
- [`../src/prenorm/exp02/trajectory.py`](../src/prenorm/exp02/trajectory.py)

최종 단계 이전의 계획과 실행 로그는
[`archive/2026-07-31_pre_final/`](archive/2026-07-31_pre_final/README.md)에 보존한다.

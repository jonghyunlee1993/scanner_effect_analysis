# Scanner Spectrum: final study protocol

**Status:** PanNormal core E0--E7 completed and result-locked, 2026-08-03;
PLISM remains a deferred post-core extension

**Primary target:** *Medical Image Analysis*
**Role of this document:** 논문 주장, 분석 단위, 내부 비교군과 PanNormal 완료 조건을
동결하는 기준 문서. PLISM은 core 완료 후 별도 protocol로 시작한다.

## 1. 한 문장 결론

병리 스캐너 효과는 하나의 전역적인 style layer가 아니라, **교정 가능한 저주파
외관 차이와 scanner가 보존·손실하는 고주파 정보가 조직 및 슬라이드 내용과
상호작용한 결과**이며, 외관 정규화 또는 scanner 분류 정확도의 감소만으로 PFM의
representation/content preservation을 보장할 수 없다.

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

따라서 평가는 항상 `scanner invariance`와 `representation/content fidelity`를 함께
측정한다. PanNormal에는 tissue type 외의 독립 biology label이 없으므로 이를 biological
non-inferiority 또는 clinical validation으로 부르지 않는다.
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

654개 native scanner--slide WSI header를 전수 감사했으며 654/654가 열리고 scanner별
109/109 file의 MPP가 일관됐다. Manuscript-facing metadata는 다음과 같다.

| Scanner | Header model evidence | Native MPP | Nominal magnification/objective | Full-resolution encoding | Product/software |
|---|---|---:|---|---|---|
| AT2 | Aperio AT2 | 0.505200 | 20× | SVS JPEG/RGB Q70 | Aperio Image Library v12.0.16 |
| GT450 | Aperio Leica Biosystems GT450 | 0.262407 | 40× | SVS JPEG/YCC Q91 | GT450 v1.5.1 |
| VERSA | Header label `Versa`; exact model pending | 0.274200 | AppMag 20× | SVS JPEG/YCC Q75 | version absent |
| AKOYA | PerkinElmer-QPI/Akoya; exact model pending | 0.499899 | 20× scan profile; 10× objective string | QPTIFF JPEG/RGB Q70 | Fusion 2.2.0 |
| S60 | Hamamatsu S60; C13210 | 0.442595 | 20× | NDPI JPEG/YCC; quality absent | NZAcquire 3.1.70 |
| S360 | Hamamatsu S360; C13220 | 0.460320 | 20× | NDPI JPEG/YCC; quality absent | NZAcquire 3.2.20 |

Objective NA와 firmware는 모든 장비에서, VERSA/AKOYA exact manufacturer/model과
S60/S360 JPEG quality는 해당 header에서 회수할 수 없다. 이 필드는 operator 또는 acquisition
record 확인 전까지 미상으로 유지하며 software string을 firmware로 대체하지 않는다.

### 3.2 E0 common-grid와 estimator validity

모든 primary RGB는 historical registered TIFF가 아니라 native WSI에서 생성한다. Frozen
geometry는 65,400/65,400 scanner-location, 654/654 scanner-slide cell과 10,900/10,900
six-scanner tuple에서 512-pixel bounds/identity gate를 통과했다. Final renderer는 native
image를 affine 최소 singular value로 Lanczos3 reduction한 뒤 residual bicubic affine을
target grid에서 평가한다.

최종 GT450/VERSA transform q05/q50/q95 profile의 합성 audit에서 original single-pass
bicubic은 alias gate를 실패했지만 explicit-AA chain의 최악 alias/true-in-band ratio는
sinusoid 0.0154, broadband noise 0.0190으로 사전 기준 0.05 이하였다. Anchor sensitivity는
primary 0.03--0.10 cycles/micrometre와 세 대안을 비교했다.

Exp07의 outcome-blind native glass 좌표 65,400개를 같은 affine/AA chain으로 다시 렌더링한
E0d에서는 47,088 tissue/background radial bin 모두 noise subtraction 후 양수였다. High-band
background/tissue power 중앙값은 0.00048--0.00569였고, corrected-minus-raw ERT 변화
중앙값은 비참조 scanner에서 -0.0014~-0.0039 log2였다. 이는 operational background
floor이며 detector NPS, DQE 또는 absolute MTF가 아니다.

### 3.3 Audited scanner spectrum

AT2 대비 high-band fold transfer의 cohort estimate는 다음과 같다.

| Scanner | High-band fold | log2 estimate (95% CI) |
|---|---:|---:|
| GT450 | 1.478 | 0.563 (0.532, 0.594) |
| VERSA | 0.938 | -0.092 (-0.151, -0.033) |
| AKOYA | 0.335 | -1.578 (-1.674, -1.483) |
| S60 | 1.044 | 0.062 (0.022, 0.102) |
| S360 | 0.842 | -0.248 (-0.318, -0.177) |

이는 scanner를 다섯 개 범주 비교에만 두지 않고, task-relevant spectral transfer를
가진 연속 좌표로 다룰 수 있음을 보여준다. 다만 tissue PSD에서 얻은 값은 엄밀한
absolute MTF가 아니라 **effective relative transfer**다. Explicit anti-aliasing과
same-chain noise-floor sensitivity를 통과했어도 optics, sharpening, compression과 tissue
content가 결합된 관측량이므로 MTF라는 용어를 쓰지 않는다.

### 3.4 Scanner, tissue와 slide의 분해

109-slide native-AA replicate table에 diagonal tissue scanner-slope와 slide-within-tissue
scanner-slope REML을 적합했다. High band에서 between-slide scanner-slope variance 중 tissue
type의 비율과 boundary-mixture LRT의 BH q-value는 다음과 같다.

| Scanner | Tissue fraction of between-slide variance | BH q |
|---|---:|---:|
| GT450 | 0.205 | 0.0499 |
| VERSA | 0.065 | 0.3510 |
| AKOYA | 0.313 | 0.00594 |
| S60 | 0.276 | 0.0145 |
| S360 | 0.033 | 0.4607 |

따라서 content dependence는 scanner에 따라 다르며, AKOYA와 S60에서는 tissue component가
뚜렷하지만 VERSA와 S360은 slide-within-tissue variation이 지배적이었다. Tissue type별
slide 수는 1--6개로 대부분 3개이므로 tissue result는 coarse content evidence이고 일부
tissue의 slope는 강하게 shrinkage된다.

관련 내부 그림:

- [`figure1_cohort_spectral_overview.png`](../outputs/e1_native_aa_spectral_109/figure1_cohort_spectral_overview.png)
- [`figure2_mixed_effects.png`](../outputs/e1_native_aa_spectral_109/figure2_mixed_effects.png)
- [`figure3_blur_sharpen_bridge.png`](../outputs/e1_native_aa_spectral_109/figure3_blur_sharpen_bridge.png)
- [`figure4_tissue_random_slopes.png`](../outputs/e1_native_aa_spectral_109/figure4_tissue_random_slopes.png)

`outputs/`는 Git으로 추적하지 않으므로 이 링크는 분석 환경에서만 유효하다.

### 3.5 Scalar blur--sharpen reducibility

AT2 FixedLaplacianPyramid detail-gain 0--2 family를 사전 정의하고 leave-one-physical-slide-out
projection을 수행했다. Scanner curve와 held-out matched scalar null의 off-axis RMS 차이
중앙값에 5,000회 slide bootstrap을 적용했을 때 95% CI는 GT450 0.081--0.116, VERSA
0.095--0.107, AKOYA 0.256--0.316, S60 0.093--0.111, S360 0.093--0.107이었다. 다섯 scanner
모두 CI 하한이 0보다 커서 **사전 정의한** scalar family를 기각했다. 이는 모든 가능한
1차원 representation이 불가능하다는 주장이 아니다. GT450은 gain=2 boundary에 90.8%가
도달했으므로 family 범위 부족도 함께 보고한다.

### 3.6 LF correction의 pilot 범위

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

### 3.7 HF intervention pilot이 보여준 실패 모드

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

### 3.8 Background가 주는 것과 주지 않는 것

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

Same-chain E0d의 post-render glass QC 통과율은 98.34%였고, QC 실패 patch도 결과에서
제외하지 않았다. SNR>=10 high-band eligibility는 AT2/GT450/S60 109/109, VERSA/S360
108/109, AKOYA 106/109였다.

background patch를 global transform의 핵심 학습 입력으로 사용하거나, 이를 독립적인
full NPS/MTF 측정이라고 주장하지 않는다. 진정한 detector NPS/DQE에는 통제된 균일 노출과
입력 광자 통계가 필요하다.

## 4. 최종 multi-PFM 실험

### 4.1 연구 질문

1. scanner spectral distance가 PFM representation drift를 예측하는가?
2. 그 관계의 slope는 PFM, tissue type과 slide에 따라 달라지는가?
3. LF affine은 외관과 representation/content geometry 중 어디까지 개선하는가?
4. HF retention/sharpening은 scanner invariance와 content fidelity 사이에 어떤
   frontier를 만드는가?
5. image-space transform과 feature-space harmonization 중 어떤 것이 새로운 tissue,
   slide와 scanner에 더 잘 일반화되는가?

### 4.2 PFM panel

Core panel은 다음 네 모델로 동결한다. 추가 모델은 core 결과 이후의 확장이다.

| PFM | TRIDENT encoder ID | Native input | Feature dim | 역할 |
|---|---|---:|---:|---|
| ResNet50 | `resnet50` | 256 px | 1,024 | natural-image CNN baseline |
| UNI v1 | `uni_v1` | 256 px | 1,024 | pathology SSL ViT |
| CONCH v1 | `conch_v1` | 512 px | 512 | vision-language pathology model |
| Virchow2 | `virchow2` | 224 px | 2,560 | large pathology SSL model |

모든 crop은 0.5052 micrometres/pixel의 같은 canonical biological center를 사용한다.
TRIDENT sampling을 재실행하지 않고 audited 512-pixel native-AA grid에서 model FOV를
center crop한다. 436/436 model-slide shard와 261,600/261,600 embedding의 identity, shape,
dtype와 checkpoint contract가 outcome 분석 전에 통과했다.

### 4.3 Intervention matrix

모든 PFM에 동일한 physical field of view와 평가 계약을 적용한다.

| Family | Conditions | 역할 |
|---|---|---|
| Original | raw | 기준 |
| LF | paired OD affine | calibratable appearance test |
| HF attenuation | retention 0.75, 0.50, 0.25, 0 | invariance–collapse trajectory |
| HF boost | gain 1.25, 1.50, 2.00 | sharpening/noise amplification test |
| HF oracle | registered same-location mean, weight 0.25 | attainable image-space positive control |
| HF global | global coefficient mean | phase-cancellation negative control |
| Image comparator | Reinhard, paired OD affine, frozen frequency calibration | deployable baseline family |
| Feature comparator | CORAL, orthogonal Procrustes | representation-level comparator |

LF와 HF intervention은 순서를 명시하고, recombination 후 clipping fraction을 반드시
보고한다. Oracle은 실용적 deployable method가 아니라 정보 보존의 상한을 확인하는
positive control이다. Comparator 최종 범위와 CV rule은 E4--E7 decision record가 사용자
승인을 받을 때 동결한다. 2026-08-03 outcome 계산 전에 전부 승인되어 현재 frozen 상태다.

### 4.4 Outcomes

**Image/physics**

- LF MAE와 SSIM
- out-of-gamut/clipping fraction
- radial PSD와 band-wise relative transfer
- background-corrected sensitivity estimate

**Representation**

- paired cosine similarity와 consensus gain
- scanner balanced accuracy
- cross-scanner same-location retrieval
- scanner centroid distance/radius
- representation/content margin와 neighborhood preservation
- collapse diagnostic: embedding variance, effective rank, pairwise distance

**Coarse tissue evidence**

- tissue-type retrieval 또는 grouped probe는 secondary endpoint
- tissue class-size와 minimum-slide sensitivity
- nucleus/spatial endpoint는 annotation과 독립 contract를 확보할 때만 optional

scanner invariance endpoint는 하나의 prespecified primary representation/content endpoint와
collapse guardrail에 짝지어 해석한다. Scanner BACC가 감소해도 content margin, effective
rank 또는 pairwise geometry가 무너지면 실패다. Tissue type만 유지된다는 이유로 biological
fidelity를 확정하지 않는다.

### 4.5 Statistical model

scanner와 intervention은 우리가 비교하려는 고정 효과로 둔다. PFM과 frequency band도
고정 효과이며, tissue type과 slide가 scanner/intervention에 다르게 반응하도록 random
slope 또는 계층적 partial pooling을 사용한다.

개념적 모델은 다음과 같다.

```text
outcome ~ scanner * intervention * PFM
        + tissue-specific scanner/intervention slopes
        + slide-within-tissue scanner/intervention slopes
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

Primary invariance radius, content-margin non-inferiority, collapse thresholds, comparator
범위, held-out split과 4-PFM claim rule은 E4--E7 outcome을 열기 전 하나의 decision
record에 동결한다. 해당 계약은
[`e4_e7_decision_record.md`](e4_e7_decision_record.md)에 기록했으며 2026-08-03 사용자의
`전부 승인`으로 outcome 확인 전에 확정했다. Scanner BACC, consensus gain, retrieval,
neighborhood와 tissue type은 secondary/mechanistic endpoints다.

### 4.6 Locked E4 control outcome

E4는 436/436 model--slide shard와 2,354,400/2,354,400 control embedding을 감사한 뒤
109-slide equal-weight, 5,000-bootstrap contract로 분석했다. Complete HF attenuation과
LOSO global HF mean은 scanner radius를 각각 26.8--51.1%, 26.2--46.7% 낮췄지만 네 PFM
모두 content/collapse fidelity decision에 실패했다. 따라서 낮은 radius만으로는 alignment와
collapse를 구별할 수 없다는 사전 가정이 population 수준에서 확인되었다.

Registered same-location leave-one-scanner-out HF mean 25% oracle은 ResNet50, UNI v1,
CONCH v1과 Virchow2에서 RR 12.5%, 9.5%, 14.0%, 10.7%를 보였고 네 모델 모두 content
non-inferiority와 every-source-scanner collapse gate를 통과했다. 이 조건만 four-of-four
common-safe이면서 네 PFM에서 invariance CI 하한이 0보다 컸다. 반면 simple attenuation과
boost의 safe-and-improved 판정은 PFM별로 달랐다. 상세 수치와 provenance는
[`e4_control_population_results.md`](e4_control_population_results.md)에 잠갔다.

Paired oracle은 deployable normalization이 아니다. E4의 역할은 실패 영역과 attainable safe
region을 정의하고 frozen evaluation rule이 실제로 판별력을 갖는지 확인하는 것이다. 실제
image/feature correction 선택과 generalization 주장은 E5 이후에만 가능하다.

### 4.7 Locked E5 actual-comparator outcome

E5는 input-only audit에서 CORAL shrinkage `0.05`와 frequency gain cap `1.03`을 고정한
뒤 Reinhard, paired OD affine, frequency calibration, CORAL와 orthogonal Procrustes를
109-fold LOSO로 평가했다. 436/436 image shard의 784,800 embedding과 436/436 feature
shard의 523,200 embedding, 총 1,308,000개가 population audit을 통과했다.

Reinhard, CORAL와 orthogonal Procrustes는 네 PFM 모두에서 content non-inferiority,
every-source-scanner collapse와 invariance-improvement CI를 통과했다. Orthogonal
Procrustes의 RR은 ResNet50 43.9%, UNI v1 22.5%, CONCH v1 23.9%, Virchow2 20.0%로
네 PFM에서 가장 컸다. CORAL은 34.9%, 16.3%, 22.0%, 17.6%, Reinhard는 27.6%,
6.0%, 11.9%, 3.7%였다. Frequency calibration은 common-safe였지만 UNI v1과 CONCH
v1의 invariance를 개선하지 못했다.

UNI v1 paired OD affine은 RR +8.9%와 positive invariance CI를 보였으나 VERSA
variance-trace point ratio 0.8865가 collapse threshold 0.90보다 낮아 최종 unsafe였다.
따라서 실제 방법에서 하나의 `invariance-positive but fidelity-unsafe` cell이 확인되었다.
다만 최고 방법은 네 PFM 모두 unconstrained와 fidelity-constrained ranking에서 orthogonal
Procrustes로 동일했다. 이 null top-rank disagreement는 선택적으로 재해석하지 않는다.
상세 grid와 provenance는
[`e5_comparator_population_results.md`](e5_comparator_population_results.md)에 잠갔다.
Feature-space 결과는 frozen representation 정렬의 근거이며 independent biological
restoration 또는 clinical utility의 증거가 아니다.

## 5. PLISM post-core extension — 보류

PLISM은 PanNormal core 분석과 원고의 완료 조건이 아니다. E4--E7 및 내부 manuscript
lock을 먼저 끝낸 뒤 별도의 frozen protocol과 decision record로 시작한다. 아래 내용은 그때
재검토할 설계 메모이며 현재 실행을 승인하지 않는다.

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
4. scanner invariance가 representation/content geometry를 보존했는지 명시적으로 검증한다.
5. 네 core PFM에서 model-specific 결과와 공통적으로 안전한 correction을 구분한다.

E6는 240개 full/sensitivity REML fit과 exact 37-fold LOTO transfer를 완료했고, E7은
36 tissue/108 slide grouped probe와 31 tissue/98 slide sensitivity를 완료했다. E0--E7의
8개 result lock과 Main Figure 1--6은 통합 core audit에서 재검증됐다. PLISM external
mechanistic extension과 독립 biological endpoint는 후속 범위다.

## 7. 논문 그림 계획

1. **Study design and estimator validity:** paired acquisition, invariance-only degeneracy,
   outcome-blind registration recovery, final geometry/render 및 alias/noise gates
2. **Audited scanner spectrum:** normalized ERT cohort spread, high-band fixed effect와 scalar
   LOSO projection residual. Unnormalized power와 전체 2D mixing은 Supplement
3. **Scanner × content structure:** tissue/slide/sampling decomposition, band별 tissue share와
   tissue-specific slopes. Scanner fixed mean은 Figure 2B, route/AKOYA sensitivity는 Supplement
4. **Control-bounded frontier:** raw, collapse, oracle, sharpening과 four-PFM CI
5. **Correction benchmark:** image/feature comparator와 fidelity-constrained decision
6. **Content preservation:** primary content endpoint, collapse와 tissue-type secondary

Supplement에는 background/NPS sensitivity, interpolation audit, clipping, mixed-model
diagnostics, scanner subset sensitivity와 legacy pilot를 둔다.

Main Figure 1--3은 `outputs/e0_e3_main_figures/`에 PNG/PDF로 완성했으며, 각 source와
output hash는 같은 디렉터리의 `summary.json` 및 E0--E3 locked-results audit에 기록한다.
Figure 4도 `outputs/e4_control_figure/`에 PNG/PDF로 완성하고 E4 result lock에 기록했다.
Figure 5는 `outputs/e5_comparator_figure/`에 PNG/PDF로 완성하고 E5 result lock에
기록했다. Figure 6은 `outputs/e7_tissue_probe/`에 PNG/PDF로 완성하고 E7 result lock과
PanNormal core 통합 audit에 기록했다.

## 8. 허용되는 주장과 금지할 주장

### 지지 가능한 주장

- Scanner identity has a strong population-level spectral signature.
- The magnitude of that signature is modified by tissue and slide content.
- LF appearance correction alone does not guarantee PFM alignment.
- Reduced scanner separability can reflect either genuine alignment or information collapse.
- A paired same-location HF oracle identifies an attainable fidelity-preserving invariance region
  across the four frozen PFMs, whereas complete blur and global HF averaging do not.
- Reinhard, CORAL and orthogonal Procrustes are common safe-and-improved across the four
  frozen PFMs under exact LOSO; one UNI paired-OD cell improves invariance but fails collapse.
- Exact leave-one-tissue-out fitting changes none of the 20 safe or safe-and-improved decisions.
- Coarse tissue geometry is method and PFM dependent: Procrustes improves paired centroid-profile
  agreement across all four PFMs, whereas no correction improves every tissue metric.
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

- ~~scanner/tissue/slide metadata audit~~
- ~~PFM panel, checkpoints, FOV와 preprocessing lock~~
- ~~band boundary와 E0--E3 estimator implementation test~~
- ~~primary E4--E7 outcomes, margin, comparator와 mixed-model/CV formula lock~~ —
  2026-08-03 전부 승인으로 outcome 확인 전에 동결

**Go:** 동일 입력에서 모든 PFM의 deterministic extraction과 sample identity 검사가 통과.

### Phase B — internal population experiment

- ~~109 slides × 100 patches × 6 scanners의 raw/E4 control population~~
- ~~4-PFM E4 control embeddings와 representation/content/collapse metrics~~
- ~~E4 physical-slide bootstrap와 control-bounded frontier~~
- ~~E5 image/feature-space 실제 correction의 LOSO cross-fit~~ — 872/872 shard,
  1,308,000 embedding, five-method frontier와 Figure 5/result lock 완료
- ~~E6--E7 correction heterogeneity와 tissue-type secondary~~ — 240 REML fit, exact LOTO와
  36-class grouped probe/Figure 6/result lock 완료

**Go:** 네 core PFM의 primary endpoint, collapse guardrail과 slide-blocked uncertainty가
완전하고 실제 image/feature comparator가 포함됨.

### Phase C — content heterogeneity and manuscript lock

- ~~tissue/slide correction heterogeneity~~
- ~~tissue-type secondary와 sample-size limitation~~
- ~~locked figures/tables와 claim terminology audit~~

### Post-core — frozen PLISM extension, 현재 보류

- 한 stain scanner-only primary
- scanner × stain secondary
- interpolation audit 선행
- internal--external effect comparison

## 10. 구현 및 provenance

핵심 코드는 다음에 있다.

- [`../src/analyze_exp05_spectral_cohort.py`](../src/analyze_exp05_spectral_cohort.py)
- [`../src/analyze_exp05_tissue_random_slopes.py`](../src/analyze_exp05_tissue_random_slopes.py)
- [`../src/analyze_exp08_background_mixed_effects.py`](../src/analyze_exp08_background_mixed_effects.py)
- [`../src/render_e0_native_aa_shard.py`](../src/render_e0_native_aa_shard.py)
- [`../src/analyze_e0_native_aa_ert_sensitivity.py`](../src/analyze_e0_native_aa_ert_sensitivity.py)
- [`../src/analyze_e0d_same_chain_noise_floor.py`](../src/analyze_e0d_same_chain_noise_floor.py)
- [`../src/run_e1_native_aa_spectral_slide.py`](../src/run_e1_native_aa_spectral_slide.py)
- [`../src/analyze_e3_scalar_reducibility_cv.py`](../src/analyze_e3_scalar_reducibility_cv.py)
- [`../src/audit_e0_e3_locked_results.py`](../src/audit_e0_e3_locked_results.py)
- [`../src/build_e0_e3_main_figures.py`](../src/build_e0_e3_main_figures.py)
- [`../src/audit_e0_scanner_acquisition_metadata.py`](../src/audit_e0_scanner_acquisition_metadata.py)
- [`../src/extract_e0_pfm_features.py`](../src/extract_e0_pfm_features.py)
- [`../src/build_e4_global_hf_reference.py`](../src/build_e4_global_hf_reference.py)
- [`../src/extract_e4_control_features.py`](../src/extract_e4_control_features.py)
- [`../src/audit_e4_control_features.py`](../src/audit_e4_control_features.py)
- [`../src/analyze_e4_control_frontier.py`](../src/analyze_e4_control_frontier.py)
- [`../src/audit_e4_control_results.py`](../src/audit_e4_control_results.py)
- [`../src/build_e4_control_figure.py`](../src/build_e4_control_figure.py)
- [`../src/prenorm/exp02/pairwise.py`](../src/prenorm/exp02/pairwise.py)
- [`../src/prenorm/exp02/trajectory.py`](../src/prenorm/exp02/trajectory.py)

최종 단계 이전의 계획, 실행 로그와 superseded diagnostic 문서는 active tree에서
제거했다. 필요 시 `checkpoints/prenorm_postcleanup_20260803.bundle`을 별도 경로에
clone해 조회한다.
E0--E3 잠금 상태는 `outputs/e0_e3_locked_results/`의 source artifact manifest,
locked result table, figure-component/main-figure manifest와 summary로 기계 검증한다. 이
audit은 E4--E7 PFM outcome을 읽지 않으며 Main Figure 1--3은 locked E0--E3 source만으로
완성한다. E4는 별도로 `outputs/e4_control_results_lock/`의 13-artifact manifest와
result-lock summary를 통과했으며 Main Figure 4의 source/output hash를 포함한다.

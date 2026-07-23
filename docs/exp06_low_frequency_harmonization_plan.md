# Exp-06: AT2-anchored low-frequency scanner harmonization

작성일: 2026-07-15  
상태: **LF16 pilot 완료 — hard safety 및 baseline gate 실패로 no-go**  
프로젝트 범위: PFM feature alignment와 분리된 RGB image-level scanner harmonization

실행 결과와 계획 수정 근거는
[`exp06_execution_log.md`](exp06_execution_log.md)에 기록한다. Stage-0에서 image-only idempotence가
비자명한 paired color correction과 구조적으로 충돌함을 확인해 diagnostic으로 재분류했다. 수정된 hard gate로
LF16을 평가했으나 S60 clipping과 constant-residual collapse가 확인되어 LF8/LF4로 진행하지 않았다.

## 0. 결정 요약

Exp-02–05는 image fidelity와 scanner invariance를 동시에 개선하려 했지만, source high-frequency를
보존하면 scanner signature도 전달되고 이를 adversarially 제거하면 morphology 및 reference fidelity가
악화되는 trade-off를 반복해서 확인했다. 또한 canonical AT2 자체가 raw AT2에서 이동하면서
canonical-to-canonical 유사도 개선이 실제 AT2 normalization인지 공통 canonical drift인지 식별하기
어려웠다.

Exp-06은 목표를 다음과 같이 축소하고 명확히 한다.

> 입력에서 관측된 high-frequency evidence는 생성·삭제하지 않고 그대로 통과시키며, 상대적으로
> 교정 가능하고 안전한 low-frequency color/tone/illumination 차이만 AT2 rendering convention으로
> harmonize한다.

핵심 결정은 다음과 같다.

1. Akoya는 학습, hyperparameter 선택, checkpoint 선택에서 제외한다. 기존 분석에 사용된 Akoya는
   untouched external validation이 아니라 locked **unseen-scanner challenge set**으로 정의한다.
2. 기존 external S60를 학습 scanner로 편입한다. 학습 pair는 `AT2–GT450`, `AT2–VERSA`,
   `AT2–S60`이며 scanner pair를 균등 sampling한다.
3. 모델은 perfect-reconstruction Laplacian pyramid의 허용된 low band만 수정한다. 허용되지 않은
   high band는 architecture 수준에서 그대로 복사한다.
4. 첫 실험에는 GAN, GRL, reconstruction branch, high-frequency AT2 regression, PFM loss를 넣지 않는다.
5. scanner classification과 정보 보존은 low/high/full image에서 분리해 평가한다. Full-image UNI는
   학습과 model selection에 사용하지 않는 sealed secondary audit으로만 유지한다.
6. immediate pilot은 기존 6/2/2 slide split과 epoch 10/30 checkpoint를 사용한다. Pilot을 통과한
   단일 bandwidth만 grouped slide CV로 재검증한다.

이 연구의 주장은 더 이상 “모든 scanner를 raw AT2와 구분할 수 없게 만든다”가 아니다. Exp-06의
주장은 **high-frequency evidence를 보존한 conservative low-frequency style harmonization**이다.

## 1. 기존 실험과 방향 전환의 맥락

### 1.1 M1과 high-frequency skip

초기 v4 M1은 image distance를 줄였지만 internal SSIM과 S60 focus가 크게 하락했다. High-band loss를
강화하면 fidelity가 개선됐으나 scanner invariance가 악화됐다. Phase-preserving high-resolution skip은
처음으로 image-space safety gate를 통과하고 AT2 identity와 S60 focus를 크게 회복했지만, source의
scanner-specific high band도 함께 전달했다.

이 단계에서 얻은 핵심 결과는 다음과 같다.

- source-supported detail을 보존하려면 high-resolution path가 필요하다.
- source high-pass를 그대로 canonical path에 전달하면 morphology와 scanner phase를 분리할 수 없다.
- image fidelity와 scanner invariance는 같은 방향으로 자동 개선되지 않는다.

### 1.2 Exp-02: dual residual

Exp-02는 canonical morphology residual과 reconstruction-only scanner residual을 분리했다. 단일 HR skip보다
internal distance `0.104298→0.103302`, SSIM `0.589048→0.597028`, UNI same-location R@5
`0.767460→0.799664`로 개선됐다. 그러나 scanner probe는 `0.997671`로 계속 거의 완벽했고,
Akoya canonical query의 nearest neighbors도 자기 scanner에 머물렀다.

Residual audit에서 VERSA morphology residual의 AT2 coherence는 `-0.0174`였고, source luminance
high-pass 자체가 scanner-specific phase를 포함한다는 한계가 드러났다. Branch 이름을 morphology와
scanner로 분리하는 것만으로 실제 정보 분리가 보장되지 않았다.

### 1.3 Exp-03a/03c: nuisance consistency와 regional distillation

Exp-03a는 same-scanner acquisition nuisance consistency를, Exp-03c는 regional relation distillation을
추가했다. Exp-03a는 internal distance `0.102827`, SSIM `0.599945`, S60 focus retention `0.5590`으로
fidelity와 OOD detail 보존을 소폭 개선했다. 그러나 scanner probe는 `0.997783`으로 개선되지 않았다.
Exp-03c는 Exp-03a와 aggregate 차이가 사실상 없었다.

따라서 augmentation과 regional supervision은 fidelity regularizer로는 작동했지만 source high-frequency
scanner signature를 제거하지 못했다.

### 1.4 Exp-04a/04b: high-frequency GRL

Exp-04a는 Exp-03a continuation control이고 Exp-04b는 canonical high band에 multi-class GRL을 추가했다.
Epoch 30 비교는 다음과 같다.

| 지표 | Exp-04a | Exp-04b | 해석 |
|---|---:|---:|---|
| location-excluded scanner purity@5 ↓ | 0.9069 | **0.8840** | 일반화 scanner clustering 감소 |
| same-location R@5 ↑ | 0.7968 | **0.8162** | paired retrieval 개선 |
| same-location cosine ↑ | 0.7787 | **0.7984** | paired embedding 개선 |
| scanner probe BACC ↓ | 0.9966 | 0.9963 | 독립 분류 가능성은 그대로 |
| standard distance ↓ | **0.1014** | 0.1016 | image distance 동등 |
| standard SSIM ↑ | 0.6041 | **0.6071** | 소폭 개선 |

GRL은 neighborhood 구조에는 영향을 주었지만 scanner information을 제거하지 못했다. 이 결과로
scanner 평균 하나가 아니라 `AT2 대 각 scanner` 및 frequency band별 진단이 필요해졌다.

### 1.5 Exp-05a/05b: AT2-pairwise adversary

Exp-05a는 RMS-normalized high texture를, Exp-05b는 low color, band energy, edge, Laplacian,
spectral slope 등 acquisition statistics를 사용해 AT2 대 각 scanner를 binary adversarially 정렬했다.

Exp-05a online validation adversary BACC는 epoch 10→30에서 GT450 `0.625→0.574`, VERSA
`0.660→0.582`로 낮아졌지만 Akoya는 `0.977→0.991`로 계속 완벽히 구분됐다. Exp-05b는 전체
`0.675→0.723`, Akoya `0.790→0.926`으로 오히려 시간이 갈수록 분류가 쉬워졌다.

Independent slide-LOSO high-band probe는 다음을 보였다. Chance BACC는 0.5이며 낮을수록 좋다.

| Scanner | Raw high | Exp-04b@30 | Exp-05a@10 | Exp-05a@30 | Exp-05b@30 | Exp-05a@30 RMS-normalized high |
|---|---:|---:|---:|---:|---:|---:|
| GT450 | 0.989 | 0.923 | 0.939 | **0.899** | 0.933 | 0.899 |
| VERSA | 0.773 | 0.855 | **0.811** | 0.816 | 0.873 | **0.703** |
| Akoya | 0.997 | 0.987 | 0.992 | 0.991 | 0.992 | 0.985 |

Exp-05a는 GT450/VERSA의 일부 일반화 high-frequency signature를 줄였지만 Akoya high-frequency는
전혀 해결하지 못했다. 매우 낮은 band `low_sigma16`에서도 Exp-05a@30 BACC가 GT450 `0.701`,
VERSA `0.558`, Akoya `0.725`로 나타나 low-frequency color/statistics에도 scanner 정보가 남았다.

### 1.6 Canonical drift와 validity 문제

Exp-05a@30은 S60 canonical→canonical AT2 UNI cosine을 `0.8332`까지 올렸지만, canonical S60→raw
AT2 cosine은 `0.7695`로 Exp-04b의 `0.7874`보다 낮았다. Internal AT2 standard SSIM도 Exp-04b
`0.9219`에서 Exp-05a `0.9060`으로 하락했다.

즉 두 scanner output이 서로 가까워지는 동안 AT2 output도 raw AT2에서 이동했다. 기존 objective는

```text
N(x_scanner) ≈ N(x_AT2)
```

를 장려하지만 `N(x_AT2)=x_AT2`를 구조적으로 고정하지 않는다. 모든 output에 공통 transform을 적용해도
loss를 만족할 수 있으므로, canonical-to-canonical 개선만으로 실제 AT2 normalization을 주장하기 어렵다.

이 validity 문제와 high-frequency 비가역성이 Exp-06 방향 전환의 직접적인 근거다.

## 2. Exp-06 연구 질문과 가설

### 2.1 Primary question

> AT2를 명시적 rendering anchor로 고정하고 input high-frequency band를 정확히 보존할 때, GT450,
> VERSA, S60의 low-frequency scanner appearance를 held-out slide에서 일관되게 줄일 수 있는가?

### 2.2 Secondary questions

1. 교정 가능한 bandwidth를 `low_sigma16 → low_sigma8 → low_sigma4`로 넓힐 때 scanner leakage 감소와
   morphology 보존 사이의 trade-off는 어디에서 시작되는가?
2. Learned correction이 Reinhard/Macenko, paired affine color transform, fixed common smoothing보다
   실제로 우수한가?
3. 학습에서 완전히 제외한 Akoya에서 low-frequency correction은 일반화하고 high-frequency evidence는
   훼손하지 않는가?
4. Full image에서 independent ResNet18 및 sealed UNI same-location consistency가 유지되는가?

### 2.3 가설

- **H1:** scanner classification의 low-frequency 성분은 paired AT2 supervision으로 줄일 수 있다.
- **H2:** high band를 architecture에서 복사하면 focus, edge, nuclei geometry와 scanner-specific detail을
  입력 그대로 보존할 수 있다.
- **H3:** correction bandwidth를 넓힐수록 low-band alignment는 좋아지지만 어느 level 이후 morphology와
  unseen-scanner 안전성이 악화된다.
- **H4:** Akoya의 full/high scanner signature는 남더라도 low-band appearance는 개선할 수 있다.
- **H5:** full-image scanner BACC는 high-band pass-through 때문에 chance까지 내려가지 않는다. 이는 본
  설계의 실패 조건이 아니다.

## 3. 주장 범위

### 3.1 주장하는 것

- paired, registered data에서 학습한 AT2-anchored low-frequency scanner harmonization
- input-supported high-frequency evidence의 보존
- 학습 scanner의 held-out slide와 unseen Akoya에 대한 band-specific 분석
- correction bandwidth에 따른 fidelity–harmonization Pareto curve
- 특정 PFM loss에 의존하지 않는 RGB preconditioner

### 3.2 주장하지 않는 것

- Akoya에서 소실된 detail의 faithful 복원
- output이 raw AT2와 pixel-level로 indistinguishable하다는 주장
- full-image scanner classification을 chance까지 낮춘다는 주장
- low-frequency가 모든 scanner에서 완전히 가역적이라는 주장
- Akoya가 이미 연구 설계에 사용된 사실을 무시한 pristine external validation 주장
- UNI/PFM embedding을 직접 최적화했다는 주장

## 4. 데이터 계약

### 4.1 학습 및 평가 scanner

| 역할 | Scanner | 사용 방식 |
|---|---|---|
| reference | AT2 | paired low-band target, identity anchor |
| training target | GT450 | internal store의 paired AT2–GT450 |
| training target | VERSA | internal store의 paired AT2–VERSA |
| training target | S60 | external store를 재분할한 paired AT2–S60 |
| unseen challenge | Akoya | 학습/loss/model selection에서 제외 |

Akoya tensor를 학습 batch에 싣지 않으며 normalization statistic, scanner adversary, threshold calibration에도
사용하지 않는다. Akoya 결과는 model과 bandwidth level을 internal validation에서 고정한 뒤 한 번 산출한다.

### 4.2 현재 store와 통합 방식

- Internal index: `data/index/index_10slide_v3.parquet`, 13,051 locations, 현재 scanner
  `AT2/GT450/VERSA/Akoya`, 고정 6/2/2 slide split
- S60 index: `data/index/external_s60_10slide_index_v1.parquet`, 7,164 locations,
  `AT2/S60`, 현재 모두 `test`로 표시

두 store는 같은 10개 slide를 사용하지만 별도 AT2 lattice와 sampling budget으로 생성됐다. 따라서 네 scanner가
동일 coordinate에 모두 존재하는 tuple을 강제하지 않는다. Training loader는 다음 두 paired dataset을 교대로
sampling한다.

```text
Dataset A: AT2–GT450–VERSA tuples from prenorm_store_10slide_v3
Dataset B: AT2–S60 pairs from external_s60_10slide_store_v1
```

Dataset B에는 기존 main split의 slide assignment를 새로 부여한다.

| Split | Slides |
|---|---|
| train | `12.5_4`, `12.5_17`, `2-8_1`, `2-8_14`, `8-12_5`, `8-12_9` |
| val | `12.5_27`, `2-8_3` |
| test | `8-12_14`, `2-8_2` |

S60 train/val/test의 paired coverage, `q_reg`, residual shift, padding, focus distribution을 먼저 audit한다.
S60가 기존 scanner보다 낮은 registration quality를 보이면 low-band supervision weight만 `q_reg`로 낮추고
scanner별 loss weight를 임의 조정하지 않는다.

### 4.3 Sampling

한 optimizer epoch에서 target scanner pair를 균등 sampling한다.

```text
P(GT450 pair) = P(VERSA pair) = P(S60 pair) = 1/3
```

Pair 내부 geometry augmentation은 공유한다. Exp-06a에는 blur, resampling, unsharp, noise처럼 bandwidth를
바꾸는 nuisance augmentation을 사용하지 않는다. Color/brightness augmentation도 첫 pilot에서는 끄고 objective
효과를 분리한다.

### 4.4 Akoya challenge protocol

Primary Akoya challenge는 locked test slide `8-12_14`, `2-8_2`만 사용한다. 같은 train slide의 다른
scanner tissue를 모델이 본 상태에서 Akoya를 평가하면 unseen scanner와 unseen tissue가 혼동되므로 train-slide
Akoya 결과는 별도 transductive diagnostic으로만 기록한다.

## 5. Frequency contract

### 5.1 Perfect-reconstruction pyramid

고정 Laplacian pyramid를 사용한다.

\[
P_K(x)=\left(L_K(x), B_{K-1}(x),\ldots,B_0(x)\right)
\]

\[
x=P_K^{-1}\left(L_K(x),B_{K-1}(x),\ldots,B_0(x)\right)
\]

모델이 허용된 band 집합 `A`만 수정하고 나머지 band는 input tensor를 그대로 재사용한다.

\[
N_A(x)=P_K^{-1}\left(\widehat L_K,\{\widehat B_k:k\in A\},\{B_k(x):k\notin A\}\right)
\]

Gaussian blur의 단순 `x-low(x)`도 diagnostic에는 유지할 수 있지만, training graph에는 reconstruction 오차가
명시된 perfect-reconstruction pyramid를 사용한다.

### 5.2 Correction level

| Level | 학습 가능한 성분 | 항상 통과하는 성분 | 목적 |
|---|---|---|---|
| LF16 | coarsest low, 약 `sigma16` 대응 | `8–16`보다 높은 모든 band | 가장 보수적인 primary model |
| LF8 | LF16 + `8–16` band | `4–8`보다 높은 band | 넓은 shading/tone 외 mid-low correction |
| LF4 | LF8 + `4–8` band | `2–4` 및 `high_sigma2` | morphology trade-off 탐색 |

`2–4`와 `high_sigma2`에 learned AT2-like signal을 추가하는 실험은 현재 범위에 포함하지 않는다.

### 5.3 Clipping과 anti-cheating

RGB range clipping은 copied high band를 간접 변경할 수 있다. 따라서 다음을 모두 기록한다.

- pre-clipping output의 copied-band max/mean absolute error
- clipping된 pixel fraction과 scanner별 분포
- post-clipping high-band energy ratio 및 cross-coherence
- correction residual의 power spectrum

Clipping이 safety threshold를 넘으면 bounded low residual의 amplitude를 낮춘다. High-band identity loss로
사후 보정하지 않고 architecture/range constraint를 수정한다.

## 6. 모델 구조

### 6.1 Exp-06 core

```text
query RGB ── fixed Laplacian pyramid ──┬─ low/coarse band ─────────────┐
                                      │                               │
slide low-band context ─ set encoder ─┴─ bounded low residual model   │
                                                                      ├─ replace allowed low bands
input high/mid bands ──────────────────────────────────────────────────┘
                                                                      │
                                                     exact pyramid reconstruction
                                                                      │
                                                           harmonized RGB
```

모델은 기존 dual-residual canonicalizer를 이어 학습하지 않고 새 minimal low-frequency normalizer로 시작한다.
기존 checkpoint에는 high-band 전달과 canonical drift가 이미 결합되어 있으므로 weight reuse가 실험 해석을
흐릴 수 있다.

### 6.2 Context

기존 query-disjoint slide context contract를 재사용하되 context encoder 입력도 허용된 low band로 제한한다.
Context는 slide acquisition appearance를 추정하고 query content를 직접 전달하지 못하도록 low resolution,
spatial dispersion, permutation invariance를 유지한다.

Correction network는 scanner ID를 입력으로 받지 않는다. Pilot에서 `is_reference` metadata는 AT2 hard bypass에만
사용한다. 이 선택은 reference-aware existence test이며, scanner metadata가 전혀 없는 deployment는 후속 ablation으로
분리한다.

### 6.3 AT2 anchor

Reference-aware pilot에서는 wrapper 수준에서 다음을 보장한다.

```text
if scanner == AT2:
    output = input
else:
    output = low_frequency_normalizer(input, slide_context)
```

따라서 AT2 identity는 loss weight에 의존하지 않는다. 모든 scanner를 같은 wrapper로 처리하는 metadata-free variant는
LF16 feasibility가 확인된 뒤 별도 실험으로 수행한다.

## 7. Loss

Exp-06a LF16의 초기 objective는 다음과 같다.

\[
L = \lambda_{pair}L_{pair}^{low}
  + \lambda_{ssim}L_{ssim}^{low}
  + \lambda_{res}L_{residual}
  + \lambda_{tv}L_{smooth}
  + \lambda_{idem}L_{idempotence}^{low}
\]

항별 역할은 다음과 같다.

- `L_pair_low`: corrected target low band와 same-location raw AT2 low band의 quality-weighted robust L1
- `L_ssim_low`: low band의 local structure/contrast 보조
- `L_residual`: correction magnitude와 color matrix/gamma deviation 제한
- `L_smooth`: correction field의 spatial total variation
- `L_idempotence_low`: 두 번째 적용 시 low band가 더 이동하지 않도록 제한

초기 권장 상대 weight는 `pair_low=5`, `ssim_low=1`, `residual=0.05`, `smooth=0.05`,
`idempotence=0.5`이며 one-slide overfit과 gradient-mass audit 후 고정한다. Scanner별 loss weight는 동일하게
유지하고 registration quality만 per-sample weight로 사용한다.

다음 항은 사용하지 않는다.

- high-frequency paired AT2 loss
- source reconstruction/rerender loss
- scanner GRL/adversarial loss
- GAN discriminator
- UNI/PFM/ResNet feature loss
- Akoya supervision

## 8. 실험군

### 8.1 Baseline

| ID | 방법 | 질문 |
|---|---|---|
| B0 | identity | 아무 교정도 하지 않은 기준 |
| B1 | Reinhard 또는 Macenko | 고전적 stain normalization으로 충분한가? |
| B2 | paired affine RGB/OD + gamma | scanner difference가 저차원 color transform으로 설명되는가? |
| B3-LF16/8/4 | fixed common smoothing | scanner BACC 감소가 단순 bandwidth 손실로 설명되는가? |
| B4 | Exp-04b@30 frozen | 이전 best balanced model과 비교 |

### 8.2 Learned experiment

| ID | 허용 correction | 초기화 | Epoch | 목적 |
|---|---|---|---:|---|
| Exp-06a | LF16 only | scratch | 30 | primary conservative harmonizer |
| Exp-06b | LF16 + 8–16 | same seed/scratch | 30 | 첫 mid-band 확장 |
| Exp-06c | LF16 + 8–16 + 4–8 | same seed/scratch | 30 | morphology trade-off 경계 |

각 run은 epoch 10과 30 checkpoint를 저장한다. Exp-06b/c는 06a 결과를 보고 순차적으로 설계하지 않고
band contract를 미리 고정한다. 다만 06a가 hard safety gate를 실패하면 더 넓은 band 실험은 실행하지 않는다.

### 8.3 후속 ablation

Primary pilot 후 필요한 경우에만 수행한다.

- no-context: slide context의 실제 기여
- soft-reference: AT2 hard bypass 없이 identity loss만 사용
- metadata-free: 모든 scanner에 동일 correction path 적용
- common-band output: learned high addition 없이 모든 scanner, AT2에 동일 fixed smoothing 적용

Full-image GAN과 learned high-frequency synthesis는 후속 목록에도 포함하지 않는다.

## 9. 학습 및 checkpoint 운용

### 9.1 Pilot

- split: 기존 6 train / 2 val / 2 test slide
- epochs: 30
- exact checkpoints: epoch 10, epoch 30
- checkpoint selection: low-band validation composite와 hard safety gate를 분리
- optimizer state: run 간 공유하지 않음
- data worker RNG와 pair sampler seed 고정

Validation composite는 `paired low distance + low SSIM + residual regularization`만 사용한다. Scanner probe,
ResNet18, UNI는 training checkpoint selection에 넣지 않는다.

### 9.2 Confirmatory grouped CV

Pilot에서 hard gate를 통과하고 best classical baseline을 이긴 하나의 bandwidth만 다음 단계로 보낸다.

- outer: 5-fold grouped by slide, 각 fold 2개 test slide
- inner: remaining slide 중 validation slide 분리
- 동일 slide의 모든 scanner view는 같은 fold
- Akoya는 각 outer test slide에서 challenge evaluation만 수행

10-slide 결과는 proof-of-concept이며 population generalization claim에는 추가 slide/scanner cohort가 필요하다.

## 10. 평가 계획

### 10.1 Low-frequency primary evaluation

Scanner별로 다음을 raw, classical baseline, learned output에서 비교한다.

- paired AT2 low-band robust distance
- low-band SSIM
- RGB/OD color distance와 stain-vector deviation
- low-band contrast 및 histogram distance
- frozen ResNet18 low-band scanner probe, slide-grouped CV
- handcrafted color/statistics scanner probe
- target scanner별, slide별 paired delta와 confidence interval

Low-band classifier 입력은 동일한 fixed filter와 upsampling을 사용한다. Filter boundary나 normalization artifact가
label shortcut이 되지 않도록 raw/output에 같은 preprocessing을 적용한다.

### 10.2 High-frequency safety evaluation

Copied band에서는 개선을 기대하지 않고 identity를 요구한다.

- output-input band MAE와 max error
- band energy ratio
- input-output cross-coherence
- focus와 Laplacian energy
- edge-map agreement
- StarDist mask를 이용한 nuclei count, centroid, area, eccentricity stability
- high-band scanner BACC raw 대 output

High-band scanner BACC가 동일하게 유지되는 것은 예상 결과다. BACC가 낮아졌다면 먼저 clipping, smoothing,
range compression에 의한 정보 손실을 의심한다.

### 10.3 Full-image evaluation

- paired raw AT2 distance/SSIM: secondary, band별 원인을 함께 보고
- frozen ImageNet ResNet18 same-location cosine/R@K와 location-excluded scanner probe
- morphology and focus safety
- full-image scanner BACC: diagnostic only, go/no-go 단독 기준 아님
- sealed UNI same-location cosine/R@5와 raw paired AT2 cosine
- qualitative panels: input, corrected low, copied bands, reconstructed full, paired AT2, residual spectrum

ResNet18은 학습 loss에 사용하지 않는 primary independent feature observer다. UNI는 PFM alignment 프로젝트와
혼동하지 않도록 secondary sealed audit으로만 보고하며, Exp-06의 성공 정의에 단독으로 사용하지 않는다.

### 10.4 Akoya challenge

Akoya에서는 다음 질문만 평가한다.

1. Low-band paired AT2 distance와 color/statistics가 개선되는가?
2. Copied high band가 input과 동일한가?
3. Full-image morphology, nuclei, focus가 악화되지 않는가?
4. Correction magnitude, clipping, context uncertainty가 training scanner 범위 밖으로 벗어나는가?

Akoya full/high scanner BACC가 높거나 output이 raw AT2보다 blurred한 것은 본 contract에서 자동 실패가 아니다.
모델이 새 detail을 생성하거나 input evidence를 삭제하는 것이 실패다.

## 11. Go/no-go 기준

### 11.1 Hard safety gate

다음 중 하나라도 실패하면 해당 bandwidth는 no-go다.

1. AT2 hard-bypass output이 input과 bitwise 동일하지 않음
2. copied pyramid band의 pre-clipping max absolute error가 float32 기준 `1e-6`을 초과함
3. clipping pixel fraction이 `0.1%`를 넘거나 copied high-band energy ratio가 `[0.995, 1.005]`,
   input-output cross-coherence가 `0.999` 범위를 벗어남
4. output-input StarDist nuclei count 상대 변화가 `1%`를 넘거나 centroid-matched F1이 `0.99` 미만임
5. Akoya에 새로운 unsupported high-frequency energy나 edge가 생성됨

위 수치는 conservative pilot margin이다. Stage 0 round-trip/quantization audit에서 더 엄격하게 조정할 수는 있지만,
learned test output이나 Akoya를 본 뒤 완화하지 않는다.

Stage-0 classical audit에서 paired affine correction은 held-out low MAE를 `79.7–93.8%` 줄였지만 image-only
idempotence MAE가 `0.150–0.576`이었다. 비자명한 full-rank affine transform `T(x)=Ax+b`가 idempotent이려면
`A²=A`, `Ab=0`이어야 하며, full-rank인 경우 `A=I`가 되어 correction이 사라진다. 반대로 idempotent
fixed-target Reinhard는 VERSA/S60 efficacy를 악화시켰다. 따라서 두 번째 적용 MAE `1/255`는 hard safety가
아니라 diagnostic으로 보고한다. 반복 적용이 deployment requirement라면 image-only loss로 해결하지 않고
명시적 normalized-state contract 또는 구조적 projection을 별도 설계한다.

### 11.2 Primary efficacy gate

Hard gate를 통과한 모델 중 다음을 모두 만족해야 한다.

1. GT450, VERSA, S60 각각에서 held-out paired low-band distance가 raw보다 개선
2. 세 scanner 모두에서 low-band scanner BACC가 raw보다 감소; 평균만 좋아지고 한 scanner가 악화되면 실패
3. best classical low-frequency baseline보다 paired metric 또는 scanner BACC에서 일관된 추가 이득
4. 개선 방향이 두 test slide에서 모두 같거나 grouped CV에서 fold majority가 같음
5. full-image ResNet18 same-location R@5 하락이 `0.01` absolute 이내이고 hard morphology gate를 통과

Pilot efficacy의 초기 effect-size 기준은 scanner별 paired low-band distance `5%` 이상 감소와 low-band BACC
`0.02` absolute 이상 감소다. 두 test slide뿐인 pilot에서는 이를 확정적 유의성으로 해석하지 않고, 최종 판정은
선택된 한 모델의 grouped CV에서 fold-level paired delta와 confidence interval로 확인한다.

Low-band BACC를 chance로 만드는 것은 필수 조건이 아니다. High-frequency를 보존하는 이상 full-image BACC가
높게 남을 수 있으며 이를 숨기지 않고 band별로 보고한다.

### 11.3 Bandwidth 선택

가장 넓은 correction level이 아니라 다음 lexicographic rule로 선택한다.

1. hard safety gate 통과
2. 모든 training scanner에서 low-band efficacy gate 통과
3. internal validation에서 best classical baseline보다 비열등
4. 위 조건을 만족하는 모델 중 가장 좁은 correction bandwidth 선택

즉 LF16으로 충분하면 LF8/LF4를 선택하지 않는다.

Akoya는 bandwidth 선택에 사용하지 않는다. Internal data로 model과 level을 완전히 고정한 뒤 challenge를 한 번
실행한다. Akoya safety가 실패하면 다른 level을 Akoya에 맞춰 재선택하지 않고 “unseen-scanner deployment no-go”로
기록하며, 이후 새 cohort를 사용한 별도 연구로 넘어간다.

## 12. 구현 및 실행 순서

### Stage 0 — 데이터와 filter audit

1. S60 index에 기존 slide split을 부여하는 non-destructive joined dataset view 작성
2. scanner pair 균형 sampler 작성
3. scanner별 valid pair 수, `q_reg`, focus, residual shift, clipping range 보고서 생성
4. perfect-reconstruction pyramid의 round-trip 및 copied-band identity unit test
5. LF16/LF8/LF4 band energy와 raw scanner probe baseline 산출

### Stage 1 — 고전적 baseline

1. identity
2. Reinhard/Macenko
3. paired affine RGB/OD + gamma
4. fixed common smoothing LF16/LF8/LF4

모든 baseline은 동일 test slide와 band-specific evaluator를 사용한다.

### Stage 2 — Exp-06a LF16

1. one-batch finite forward/backward
2. AT2 bitwise bypass 및 copied-band identity smoke
3. one-slide overfit: paired low loss 감소와 high identity 동시 확인
4. 30 epoch training, epoch 10/30 저장
5. internal GT450/VERSA/S60 test 평가
6. hard/efficacy gate 판정
7. gate 통과 후 locked Akoya challenge 한 번 실행

### Stage 3 — bandwidth 확장

Exp-06a가 hard gate를 통과한 경우 LF8, 필요할 경우 LF4를 같은 protocol로 실행한다. Band를 추가할 때
모델 width, optimizer, context, data, loss weight를 변경하지 않는다.

### Stage 4 — confirmatory CV

선택된 단일 bandwidth와 best classical baseline만 5-fold grouped slide CV로 비교한다. 이 결과 후에만
metadata-free variant나 더 큰 cohort 확장을 논의한다.

## 13. 예상 결과와 해석

### 시나리오 A — LF16이 성공

Low-band paired alignment와 scanner BACC가 세 training scanner에서 개선되고 high/morphology가 유지되면
프로젝트의 최소 주장은 성립한다. LF8/LF4는 개선 폭이 필요한 경우에만 진행한다.

### 시나리오 B — classical transform과 동등

Learned model이 paired affine 또는 Macenko/Reinhard를 이기지 못하면 deep style transfer의 필요성은 지지되지
않는다. 결과 자체는 scanner difference가 저차원 color transform으로 충분히 설명된다는 유의미한 결론이다.

### 시나리오 C — low correction도 morphology를 변경

LF16에서도 nuclei/edge/idempotence가 실패하면 image-level harmonization 범위를 더 줄이거나 fixed color transform만
사용해야 한다. 더 넓은 band나 GAN으로 진행하지 않는다.

### 시나리오 D — training scanner 성공, Akoya 실패

Akoya low-band에 해를 주지 않고 high band를 보존했다면 unsupported OOD로 보고 conservative raw fallback을
정의할 수 있다. Akoya에 큰 low residual, clipping, morphology 변화가 발생하면 context uncertainty 또는 OOD reject
mechanism이 구조적으로 필요하다.

### 시나리오 E — full scanner BACC가 높게 유지

Low-band BACC는 감소했지만 high/full BACC가 높다면 예상된 결과다. 이 프로젝트는 high-frequency scanner
signature 제거를 주장하지 않으며, 결과 보고에서 이를 명시한다.

## 14. 재현 산출물

각 run은 다음을 저장한다.

- config와 resolved config
- epoch 10/30 checkpoints
- train/val history와 scanner-pair sampling count
- internal per-image 및 per-slide low/high/full metrics
- independent ResNet18 probe features와 grouped predictions
- sealed UNI embeddings와 metric summary
- Akoya challenge metrics와 OOD/context diagnostics
- input/low/band/residual/full qualitative panels
- self-contained HTML report
- unit/integration test 결과와 SLURM job metadata

## 15. 최종 판정 문장 템플릿

성공 시:

> Exp-06은 input high-frequency evidence와 AT2 identity를 구조적으로 보존하면서 GT450, VERSA,
> S60의 low-frequency scanner appearance를 held-out slide에서 일관되게 감소시켰다. 이는 full scanner
> invariance나 lost-detail recovery가 아니라 conservative low-frequency RGB harmonization의 근거다.

실패 시:

> Fixed high-frequency preservation과 AT2 anchoring 아래에서도 learned low-frequency correction이 classical
> baseline을 이기지 못했거나 morphology safety를 만족하지 못했다. 따라서 현 데이터에서 deep image-level
> style transfer의 추가 복잡성은 정당화되지 않으며, fixed color normalization 또는 프로젝트 범위 축소가 타당하다.

# Exp-07: Contract-spectrum low-frequency scanner harmonization

작성일: 2026-07-15  
개정: 2026-07-15 (배포 계약 A/B/C/O 분리, few-shot 누수 방지, composite range-safe 경로,
nested slide CV와 scanner-LOO, Layer 2 진입 조건 및 결과 해석 재정의)  
상태: **제안 (Stage 0–1 설계 확정 전, 미구현)**  
프로젝트 범위: PFM feature alignment와 분리된 **image-level** scanner harmonization  
선행 문서: [`exp06_low_frequency_harmonization_plan.md`](exp06_low_frequency_harmonization_plan.md),
[`exp06_execution_log.md`](exp06_execution_log.md)

---

## 0. 한 줄 요약과 예상 결론

Exp-07은 "deep harmonizer가 성공할 것"을 전제로 한 개발 계획이 아니다. Primary contribution은
**배포 시 허용되는 정보량에 따라 저주파 scanner harmonization이 어디까지 가능한지 측정하는 계약별
한계 지도(contract map)**다.

계약은 다음 네 개로 분리한다.

- **A — known-scanner supervised:** scanner identity를 알고, 그 scanner의 train source–AT2 pair로
  미리 fit한 계수를 적용한다.
- **B — unseen-scanner few-shot:** 새 scanner의 소량 paired AT2 calibration으로 계수를 추정한 뒤,
  calibration에 사용하지 않은 slide에 적용한다.
- **C — unseen-scanner zero-shot:** test-time scanner label과 paired reference 없이 source image와
  slide context만 사용한다. 학습 과정에서 얻은 고정 AT2 population prior는 허용한다.
- **O — evaluation oracle:** 평가 tuple의 paired AT2까지 사용해 계수를 fit한다. 배포 불가능하며
  optimistic ceiling으로만 보고한다.

A와 B는 단순히 정보량만 다른 계약이 아니다. A는 **이미 본 scanner**, B는 **새 scanner**라는 domain
shift까지 포함한다. 따라서 A→B→C를 단조로운 성능 순서로 가정하지 않고, 각 계약을 별도의 배포 질문으로
해석한다.

현재 증거가 가장 강하게 지지하는 예상 결론은 다음과 같다.

> A의 scanner-specific affine은 강하지만 range-unsafe하다. B는 소량의 잘 분산된 calibration으로
> 실용적인 성능에 도달할 가능성이 있으나 아직 측정되지 않았다. C의 metadata-free unseen-scanner
> 일반화는 현재 지지되지 않으며 성공 가능성이 낮다. Deep residual은 안전한 analytic base 이후에도
> slide-cross-fitted 예측 가능 잔차가 남을 때만 정당화된다.

---

## 1. Exp-02–06가 보인 것과 보이지 않은 것

### 1.1 보인 것

- 테스트한 모델과 목적함수 전반에서 image fidelity와 scanner invariance 사이의 강한 trade-off가
  반복됐다.
- source high-frequency를 보존하면 scanner signature도 전달됐고, 이를 줄이려 하면 선명도·형태 또는
  AT2 reference fidelity가 악화됐다.
- Exp-06의 fixed pyramid는 LF16 round-trip max error `2.38e-7`, copied coefficient error `0`을
  달성했다. 즉 **허용한 low band만 수정하는 architecture contract 자체는 실현 가능**하다.
- Exp-06 learned residual은 scanner·channel·image·공간 위치와 무관하게 정확히 `-0.12`로 붕괴했다.
- scanner-specific train-paired affine은 held-out slide에서 low MAE를 GT450 `93.8%`, VERSA `79.7%`,
  S60 `92.3%` 감소시켰다. 저주파 차이의 큰 부분이 단순 색 변환으로 설명될 가능성을 지지한다.

### 1.2 보이지 않은 것

- 위 결과는 모든 image-level harmonization이 정보이론적으로 불가능하다는 증명이 아니다.
- "저주파는 well-posed, 고주파는 제거 불가"는 정리가 아니라 작업가설이다.
- 특히 C에서는 tissue/stain 양, section/fixation, background 비율과 scanner response가 source 통계에
  함께 들어가므로 저주파조차 완전히 식별 가능하지 않다.
- A의 강한 affine 결과는 C의 unseen-scanner 일반화를 뒷받침하지 않는다.
- StarDist 안정성이나 copied high band만으로 진단 정보 전체가 보존됐다고 말할 수 없다.

따라서 Exp-07의 주장은 **저주파에 한정된, 계약 의존적이고 경험적인 harmonization 한계 측정**이다.
남는 고주파 batch effect는 숨기지 않고 band별로 정량화하여 downstream/feature-level 프로젝트로 인계한다.

---

## 2. 계약별 질문과 affine 용어

### 2.1 네 계약

| 계약 | Test-time에 허용되는 정보 | 핵심 질문 | 현재 증거 |
|---|---|---|---|
| **A** known-scanner | scanner identity + train-fit coefficient | 이미 본 scanner에서 안전한 고전 보정이 가능한가? | low-MAE 79.7–93.8% 감소, range-unsafe |
| **B** few-shot | 새 scanner의 소량 paired calibration | 몇 slide/patch면 A에 가까워지는가? | 미측정 |
| **C** zero-shot | source query + dispersed slide context + 고정 AT2 prior | scanner metadata 없이 새 scanner를 보정할 수 있는가? | source-only RGB Reinhard가 VERSA/S60 악화 |
| **O** oracle | 평가 tuple의 paired AT2 | 이 transform family가 낼 수 있는 optimistic ceiling은 얼마인가? | 미측정, 배포 불가 |

### 2.2 네 종류의 estimator를 혼용하지 않는다

| # | 이름 | 계수 추정 방식 | 대응 계약 |
|---|---|---|---|
| ① | **supervised scanner-specific affine** | scanner별 train source–AT2 pair로 fit | A |
| ② | **source-only reference-stat normalization** | test source/context 통계와 train에서 고정한 AT2 population 통계 | C |
| ③ | **few-shot paired affine** | 새 scanner의 calibration pair 일부로 fit하고 별도 slide에서 평가 | B |
| ④ | **evaluation-paired oracle affine** | 평가 slide의 source–AT2 pair로 **per-eval-slide pooled** fit (per-image fit 아님); 동일 tuple 성능은 optimistic ceiling | O |

기존 `79.7–93.8%`는 ①의 결과다. 현재 코드도 `--scanner`가 필수이며 scanner별 train pair에서
`fit_statistics`를 실행한다. ②에 가장 가까운 기존 결과는 RGB Reinhard이고, VERSA/S60 low MAE를 각각
`46.3%`, `23.9%` 악화시켰다. 따라서 ①의 숫자로 ②/C의 가능성을 주장하지 않는다.

### 2.3 B few-shot calibration protocol

B는 paired calibration과 평가가 섞이면 쉽게 낙관적으로 보인다. 다음을 고정한다.

1. **Calibration slide와 evaluation slide를 완전히 분리한다.** 같은 slide의 nearby patch를 fit과
   평가에 나누는 것은 허용하지 않는다.
2. Calibration budget은 slide 수와 patch 수를 함께 보고한다. 초기 learning curve는
   `(1 slide, 8 patches)`, `(1, 32)`, `(2, 128)`, `(4, 512)`로 사전 고정한다. 단 scanner-LOO에서
   held-out scanner의 slide 수가 제한적이므로, 최상위 budget(4 slide)에서도 **evaluation slide가 최소
   1개 남는지**를 Stage 0에서 index 수준으로 확인하고, 남지 않으면 그 budget은 해당 scanner에서 보고하지
   않는다.
3. Patch는 WSI 좌표 bin과 tissue/focus bin에 분산하고 `q_reg` 기준을 통과한 pair만 사용한다.
4. 각 budget은 calibration draw를 최소 10회 반복하고 slide 단위 변동과 함께 보고한다.
5. Evaluation tuple, threshold와 hyperparameter는 calibration coefficient fit에 사용하지 않는다.
6. 실제 배포에서 동일 scanner로 AT2 calibration slide를 재스캔할 수 없다면 B는 실용 계약이 아니므로,
   결과 해석에서 그 비용과 전제를 명시한다.

### 2.4 C가 허용하는 prior

C의 "source-only"는 **test-time에 paired target이나 scanner label을 사용하지 않는다**는 뜻이다.
Train scanner의 pair로 학습한 일반 함수와 train AT2에서 고정한 population statistic은 허용한다. 다만
held-out scanner의 통계, threshold, coefficient 또는 paired AT2를 model selection에 쓰면 C가 아니다.
Slide context는 같은 source slide에서 spatially dispersed하고 query와 겹치지 않게 뽑는다. Context에
paired AT2, scanner label 또는 held-out scanner에서 미리 계산한 population statistic을 넣지 않는다.

---

## 3. 실현 가능성 사전 판단

| 목표 | 판단 | 이유 |
|---|---|---|
| 허용 low band만 수정하고 high band 보존 | **높음** | Exp-06 fixed pyramid로 검증 |
| A의 held-out slide 개선 | **높음** | 이미 강한 train-paired affine 결과 존재 |
| Composite range-safe affine | **중간** | 항상 feasible한 보수적 경로는 있으나 correction이 거의 0이 될 수 있음 |
| B의 few-shot calibration | **중간** | global transform은 소량 pair로 fit 가능하지만 tissue/slide 다양성이 필요 |
| C의 unseen-scanner zero-shot | **낮음** | scanner domain 3개, source 통계의 식별성 문제, 기존 source-only baseline 악화 |
| Deep residual의 analytic base 대비 추가 이득 | **낮음** | affine 후 MAE `0.015–0.033`; registration/noise floor 가능성 |
| task-level 정보 보존 주장 | **현재 불가** | task label과 frozen task model 계약이 없음 |

따라서 **Stage 0–1은 go**, Layer 2는 사전적으로 no-go에 가까운 조건부 가설이다.

---

## 4. 전체 설계

```text
input RGB
   │
   ├─ Fixed Laplacian pyramid ─ coarse low ─► L1: contract-specific global transform
   │                                             │
   │                                             ├─► L1.5: spatial-affine (A/B 중심)
   │                                             │
   │                                             └─► L2: bounded residual (조건부)
   │
   └─ copied mid/high coefficients ─────────────────► exact reconstruction
                                                        │
                                             composite range-safe projection
                                                        │
                                                      output
```

공통 원칙은 다음과 같다.

- Low band만 수정한다.
- Range projection도 **low coefficient를 feasible set 안에서 조정**해야 하며 copied coefficient를 직접
  바꾸지 않는다.
- 출력 clamp로 안전을 만들지 않는다. Clamp는 copied detail을 바꾸므로 failure다.
- 각 layer는 이전 layer 대비 추가 이득이 cross-validation에서 입증될 때만 허용한다.

### 4.1 Layer 1 — global transform

- A: ① scanner-specific train-paired affine
- B: ③ few-shot paired affine
- C: ② source/context statistic 기반 transform
- O: ④ evaluation-paired affine

RGB/OD 공간에서 diagonal scale+offset, 3×3+bias, ridge affine, gamma 포함 변환을 비교한다. 표현력이 큰
모델이 자동으로 우월한 것이 아니다. 작은 데이터에서 coefficient variance와 out-of-range correction이
커질 수 있으므로 inner CV에서 regularization을 선택한다.

### 4.2 Layer 1.5 — spatial-affine field

Global affine이 잡지 못하는 대표적 후보는 비네팅·불균일 조명처럼 scanner/slide 좌표에 따라 부드럽게
변하는 low-frequency correction이다. 그러나 nonlinear color response나 tissue-dependent stain interaction도
global affine이 잡지 못하므로, spatial variation이 **유일한** 잔차 원인이라고 가정하지 않는다.

Primary spatial-affine baseline은 다음처럼 제한한다.

- 좌표: patch-local 좌표가 아니라 정규화한 WSI/scan 좌표 `(u,v)∈[-1,1]²`
- field: degree 1–2 polynomial 또는 동등한 매우 저차원 basis
- fit: `q_reg` weighted ridge regression
- regularization/차수: inner slide CV에서 선택
- 평가: fit에 쓰지 않은 slide에서만
- range: spatial-affine 자체가 안전하다고 가정하지 않고 §5 projection을 동일하게 적용

계약별 의미는 다르다.

- A: known scanner의 train slide에서 scanner-coordinate field를 fit할 수 있다.
- B: 새 scanner의 spatially dispersed paired calibration이 있어야 field를 fit할 수 있다. Patch 수뿐 아니라
  WSI 위치 coverage가 필요하다.
- C: source-only 영상에서 tissue density와 scanner shading을 분리하기 어려우므로 primary spatial-affine
  claim에서 제외한다. Reference-free flat-field 가정을 별도로 검증한 경우에만 exploratory arm으로 둔다.

Paired evaluation patch마다 자유 coefficient map을 fit하거나 query와 같은 slide의 paired AT2로 field를
추정하는 것은 leakage이므로 금지한다.

### 4.3 Layer 2 — bounded deep residual

Layer 2는 안전하고 유효한 L1/L1.5를 **대체하거나 구조 실패를 구조하는 모델이 아니다.** Best safe
analytic base가 이미 존재하고, 그 이후에도 slide-cross-fitted 예측 가능한 잔차가 남을 때만 작은 추가
모델로 진입한다.

- 저주파 band에만 작용
- zero-initialized residual
- smoothness와 magnitude regularization
- saturating `tanh` rail을 correction의 유일한 안전장치로 사용하지 않음
- 최종 composite projection은 analytic model과 동일하게 적용
- pre-activation, residual distribution, gradient norm을 학습 전반에 로깅
- context off 시 output이 raw가 아니라 **analytic base**로 회귀하도록 zero anchor를 구조적으로 강제

---

## 5. Range safety — 첫 번째 기술 문제

### 5.1 현재 baseline의 실제 문제

기존 scanner-specific affine은 효과가 강하지만 clipping을 일으킨다.

| Scanner | Mean clipping | Per-image max clipping |
|---|---:|---:|
| GT450 | `0.534%` | `55.0%` |
| VERSA | `0.036%` | `0.621%` |
| S60 | `0.135%` | `20.9%` |

따라서 `79.7–93.8%` 개선은 range-safe efficacy가 아니다. Range를 보장했을 때 correction이 얼마나
남는지가 Stage 1의 첫 질문이다.

### 5.2 왜 low band만 bound하면 충분하지 않은가

Pyramid reconstruction을 단순화해 쓰면

\[
x = R(l,b), \qquad y = R(l',b)
\]

이다. `l`은 input low coefficient, `l'`은 corrected low, `b`는 그대로 복사한 detail coefficients다.
`l'` 자체가 bounded여도 `R(l',b)`는 range를 넘을 수 있다. 최종 RGB를 clamp하면 `b`가 더는 output의
실제 detail coefficient가 아니므로 high-band preservation이 깨진다.

### 5.3 첫 번째 range-safe baseline: identity 방향 scalar projection

가장 단순한 안전 경로를 먼저 사용한다.

\[
d = R(l'-l,0), \qquad y(\alpha)=x+\alpha d, \quad 0\le\alpha\le1
\]

모든 pixel/channel에서 `[-1,1]` 안에 남는 최대 `α`를 analytic line search로 선택하고, 구현 검사는
`1e-6` numerical tolerance를 둔다. 각 원소 `i`에 대해 `d_i>0`이면 `(1-x_i)/d_i`, `d_i<0`이면
`(-1-x_i)/d_i`가 상한이며, 이 상한들과 `1`의 최솟값을 `[0,1]`로 clamp한다. 원본 `x`가 range 안이면
`α=0`은 항상 feasible하다. 결과는

\[
y(\alpha)=R(l+\alpha(l'-l),b)
\]

이므로 copied `b`를 직접 바꾸지 않는다.

장점은 range와 copied-band identity를 구조적으로 보장하고 구현이 단순하다는 점이다. 불가능한 지점은
극단 pixel 하나가 전체 이미지의 `α`를 거의 0으로 만들 수 있다는 것이다. 이는 버그가 아니라
**exact range + exact copied band + 큰 global correction을 동시에 요구할 때 생기는 실제 Pareto 경계**다.

반드시 `α`의 mean/median/q05와 `α≈0` 이미지 비율을 scanner별로 보고한다. 추가로 `α`의 **binding
원인을 분해**한다: `α`를 결정한 pixel에서 `|x|≈1`(copied detail이 만든 사전 존재 극단값) 때문인지,
`|d|`가 큰(보정 크기) 때문인지를 구분해 보고한다. `d`는 low-band 유래라 매끄러운 반면 극단 pixel은
copied high band에서 오므로, `α`가 사전 극단값에 걸리면 range-safe efficacy의 scanner 간 비교가 저주파
correctability가 아니라 **detail 포화도에 confound**된다. 따라서 `α`를 image별 `‖d‖`(보정 크기)와
`frac(|x|>0.98)`(사전 극단 비율)에 대해 각각 회귀/상관으로 어느 쪽이 더 잘 설명하는지 보고한다. Scalar
projection이 지나치게 보수적이면 두 번째 단계로 low-coefficient-space convex projection/QP를 검토한다.
Full-resolution output projection이나 post-hoc clamp는 허용하지 않는다.

### 5.4 Stage 1 range baseline 순서

1. Identity
2. 원래 range-unsafe affine — efficacy ceiling 비교용
3. Scalar-projected affine — 첫 safe baseline
4. Ridge/coefficient-regularized affine + scalar projection
5. Affine↔identity fixed conservative interpolation
6. OD diagonal / 3×3 + gamma + projection
7. Reinhard/Macenko 계열 + projection
8. Spatial-affine + projection
9. 필요할 때만 low-coefficient convex projection

모든 방법에서 최종 output을 다시 pyramid decomposition하여 copied coefficient error를 측정한다.
`pre-clipping copied error=0`만으로 통과시키지 않는다.

---

## 6. Exp-06 붕괴에서 가져갈 교훈

Exp-06 residual이 정확히 `-0.12(=-residual_scale)`였다는 사실은 tanh의 음의 rail saturation과 강하게
일치한다. Affine base가 없어 평균적으로 밝은 source를 균일하게 어둡게 하는 해가 싸게 선택됐다는 해석도
데이터와 정합적이다.

다만 이를 유일한 인과라고 단정하지 않는다. 확인되지 않은 것은 pre-tanh logit, gradient norm, optimizer
trajectory다. 따라서 Layer 2가 실제로 열릴 경우 다음을 기록한다.

- pre-activation histogram과 rail 근접 비율
- output/residual의 scanner·channel·image·spatial variance
- context shuffle에 대한 residual 변화
- layer별 gradient norm
- epoch 0부터의 trajectory

Idempotence는 hard gate가 아니다. Full-rank affine `T(x)=Ax+b`가 image-only idempotent이면
`A²=A`, `Ab=0`에서 `A=I`가 된다. 반복 적용이 배포 요구라면 normalized-state flag 또는 별도 상태
계약으로 해결한다.

---

## 7. 정보 보존과 harmonization efficacy를 분리해 평가

### 7.1 Architectural safety

- 최종 output range 초과 없음 (`1e-6` numerical tolerance)
- 최종 output 재분해 copied coefficient max error `<1e-6`
- clamp로 바뀐 pixel 수 `0`
- correction bandwidth와 power spectrum 보고

이는 "모델이 허용하지 않은 주파수를 직접 바꾸지 않았다"는 강한 증거지만, 진단 정보 전체의 보존을
증명하지는 않는다. Low-frequency color도 downstream model이나 사람의 판독에 영향을 줄 수 있다.

### 7.2 StarDist morphology proxy

StarDist는 stain/tone에 민감하므로 candidate transform의 결과 분포를 보고 candidate가 자기 threshold를
정하게 해서는 안 된다. Threshold는 **development data와 사전 정의 control**로만 정하고 final test와
Akoya 전에 잠근다.

- Numerical control: no-op/round-trip
- Expected-safe controls: candidate fit과 독립적으로 사전 생성한 mild, range-safe color-only scale/offset.
  크기는 실제 harmonization correction 범위 안으로 calibrate하고(과소하면 무의미, 과대하면 실제로
  unsafe) 그 선택 규칙을 Stage 0에서 고정한다. StarDist가 color에 민감하다는 사실 때문에 이 control이
  자동으로 safe하지 않으므로, 크기별로 여러 지점을 두어 metric의 민감도 곡선을 함께 본다.
- Harmful controls: blur, 강한 clipping, Exp-06식 dark collapse
- 측정: count, centroid-matched F1, area/eccentricity, unmatched instance fraction

Metric이 harmful control을 expected-safe control과 구분하지 못하면 safety gate로 사용할 수 없다.
Threshold가 scanner마다 달라야 한다면 그 이유와 선택 규칙을 test 전에 고정한다.

### 7.3 추가 proxy와 주장 한계

- focus/edge와 copied-detail energy/coherence
- frozen UNI의 output↔input drift와 paired AT2 consistency — diagnostic only
- low/high/full scanner probe — harmonization diagnostic

현재 repo에는 구체적인 diagnostic task label과 frozen task model 계약이 없다. 따라서 Exp-07만으로
"task information이 보존됐다"고 주장하지 않는다. 가능한 표현은 **architectural high-band preservation과
StarDist/UNI 등 사전 지정 proxy에서 허용 범위를 지켰다**까지다. 실제 downstream task가 추가되기 전에는
task performance를 hard gate나 성공 문장에 넣지 않는다.

### 7.4 AT2 paired distance의 올바른 역할

Paired low-MAE/SSIM을 버리지 않는다. 이것은 AT2 anchor에 대한 저주파 fidelity를 재는 유용한 endpoint다.
다만 full-resolution AT2 pixel MAE 또는 low-MAE 하나만으로 성공을 선언하지 않는다.

성공 판정은 다음 축을 함께 사용한다.

1. paired low-MAE/SSIM 개선
2. range와 copied-band safety
3. morphology/embedding proxy 안정성
4. slide 및 scanner generalization
5. low-band scanner leakage 변화

### 7.5 paired low-distance와 low-band scanner leakage는 decouple할 수 있다 (co-primary)

이 프로젝트의 실제 배포 가치는 **invariance**이지 AT2와의 픽셀 근접이 아니다. 그런데 affine이 AT2의
low-band **평균/색**을 맞추면 paired low-MAE는 크게 줄지만, low-band의 **고차 통계**(영역별 covariance,
low-band 질감)에 남은 scanner 판별성은 affine이 건드리지 않는다. 프로젝트 내부에 이 decoupling의 선례가
있다:

- Exp-05: 매우 낮은 band `low_sigma16`에서도 scanner BACC가 GT450 `0.701`, VERSA `0.558`,
  Akoya `0.725`로 남았다.
- 이전 실험: distance/cosine/silhouette이 붕괴해도 residual linear scanner-acc `0.59`(chance `0.25`)가
  생존했다.

따라서 **low-band frozen-probe scanner BACC 감소를 paired low-MAE와 함께 co-primary efficacy 축으로**
둔다. 그리고 test 전에 다음 해석 규칙을 **사전 약정**한다.

> paired low-MAE는 개선했으나 low-band scanner BACC가 raw 대비 사전 등록 마진(`Δ_bacc`)만큼 감소하지
> 않으면, 그 계약/방법은 "**AT2 rendering에 더 가까워졌으나 저주파 invariance는 개선하지 못함**"으로
> 보고한다. 이는 부분 성공이며, paired 개선만으로 harmonization 성공을 주장하지 않는다.

`Δ_bacc`는 `δ`와 함께 Stage 0에서 development data로 고정하며 candidate 결과를 보고 정하지 않는다.
low-band probe 입력은 raw/output에 동일 fixed filter/upsampling을 적용해 filter artifact가 label
shortcut이 되지 않게 한다.

---

## 8. AT2 bypass와 normalized state

계약별로 가능한 bypass가 다르다.

- **A:** scanner/reference metadata가 있으므로 AT2에 `is_reference`를 전달해 bitwise bypass 가능
- **B:** calibration workflow가 source와 AT2 reference를 구분하므로 동일하게 가능
- **C0 — strict input-only:** normalized-state metadata도 허용하지 않는다. Bitwise bypass를 hard gate로
  요구하지 않고, raw AT2를 입력했을 때의 drift와 false correction을 별도 hard gate로 둔다. AT2 proximity
  detector를 쓰면 development data에서 고정하고 false-positive/negative를 보고한다.
- **C1 — state flag 허용:** scanner label은 없지만 `already_normalized` flag는 허용한다. 이 경우 bitwise
  bypass 가능하지만 완전 metadata-free라고 부르지 않는다.
- **O:** 평가 diagnostic이므로 AT2 reference를 명시적으로 안다.

외부에서 "비-AT2 입력에만 적용"하려면 누군가 AT2 여부를 알고 있어야 하므로 C0가 아니다. State flag를
사용하면서 metadata-free라고 부르지 않는다. Exp-07의 primary zero-shot claim은 더 어려운 **C0**로 둔다.

---

## 9. 통계 설계와 leakage 방지

### 9.1 Slide split과 nested CV

고정 test slide `8-12_14`, `2-8_2`는 최종 internal test로 한 번만 사용한다. 나머지 8개 slide를
development set으로 사용한다.

- Development: 기존 train 6 + val 2 slide
- Outer development CV: 4-fold grouped-by-slide, fold당 2 slide
- Inner CV: 나머지 6 development slide 안에서 transform family, ridge, polynomial degree, threshold 선택
- Final internal test: configuration을 고정하고 8 development slide로 refit한 뒤 test 2 slide에 한 번 적용

7개 이상의 baseline과 hyperparameter를 같은 outer 결과로 선택·평가하지 않는다. Inner CV가 선택을,
outer CV와 final test가 평가를 담당한다.

### 9.2 Scanner holdout은 slide LOSO와 구분한다

기존 문서의 LOSO가 leave-one-slide-out을 의미한 적이 있으므로 여기서는 **scanner-LOO**라고 부른다.

- C: GT450/VERSA/S60 중 하나를 통째로 holdout하고 나머지 두 scanner의 development data로 estimator와
  hyperparameter를 학습한다. Held-out scanner의 coefficient/statistic은 사용하지 않는다. 세 방향 수행.
- B: held-out scanner의 사전 등록 calibration slide/pair만 coefficient fit에 사용한다. Evaluation slide는
  calibration과 완전히 분리한다. Transform family와 regularization은 다른 scanner/inner CV에서 고정한다.
- A: known scanner 계약이므로 scanner-LOO claim을 하지 않는다. Train/development slide에서 fit하고
  held-out slide로만 일반화한다.
- O: scanner-LOO가 아니라 non-deployable ceiling이다.

### 9.3 분석 단위와 효과 추정

Patch 수천 개를 독립 표본으로 취급하지 않는다.

- Primary unit: `slide × scanner`
- Comparison: 동일 tuple의 paired method difference
- 보고: fold/slide effect, median, range, slide-clustered bootstrap interval
- Gate: 사전 등록한 최소 효과 `δ` 초과 + development outer fold 다수에서 동일 방향 + scanner별 큰 harm 없음

`δ`는 candidate test 결과를 보고 정하지 않는다. Development pair의 registration repeatability, 기존
affine residual과 control transform 분포로 Stage 0에서 고정한다. `δ`의 **스케일도 명시**한다: 1차
endpoint는 `slide×scanner`당 **절대 low-MAE**(입력 `[-1,1]` 스케일)이며, 감소율(%)은 보조로만 보고한다.
low-band BACC의 마진 `Δ_bacc`(절대 BACC 변화)와 residual gate의 `δ_res`도 같은 방식으로 스케일과 함께
Stage 0에서 고정한다. 10-slide 결과는 proof-of-concept이며
population 수준 주장은 추가 cohort 없이는 하지 않는다.

S60는 internal GT450/VERSA와 lattice가 다르므로 scanner 사이 tuple join을 하지 않는다. "동일 tuple 비교"는
각 scanner store 안에서 **같은 source–AT2 tuple에 모든 method를 적용**한다는 뜻이다.

---

## 10. Go/no-go — 모순 없는 세 갈래 판정

### 10.1 공통 hard safety gate

- Composite output range 초과 없음 (`1e-6` tolerance)
- Post-projection/post-reconstruction copied coefficient max error `<1e-6`
- Clamp-changed pixel `0`
- StarDist control-calibrated 허용 범위 통과
- Unsupported high-band energy/edge 생성 없음
- C0를 제외한 metadata-aware 계약: AT2 bitwise bypass
- C0: raw AT2 drift와 false-correction gate 통과

Akoya는 이 threshold를 정하는 데 사용하지 않는다.

### 10.2 계약별 efficacy gate

- **A:** 각 scanner에서 raw보다 개선하고 best safe A baseline 대비 선택된 방법의 효과를 보고
- **B:** scanner-LOO 세 방향에서 calibration/evaluation slide 분리, budget 증가에 따른 learning curve와
  raw/C 대비 개선
- **C:** scanner-LOO 세 방향에서 raw보다 개선; 한 held-out scanner의 큰 악화를 pooled 평균으로 숨기지 않음
- **O:** ceiling만 보고하며 go/no-go 또는 배포 성공으로 계산하지 않음

모든 계약에서 paired low-distance만 보지 않고 §7의 safety/proxy를 동시에 만족해야 한다. 특히 §7.5에
따라 **low-band scanner BACC 감소를 paired low-MAE와 co-primary**로 평가하며, paired만 개선하고 BACC가
`Δ_bacc`만큼 내려가지 않으면 "rendering 근접, invariance 미개선"의 부분 성공으로 사전 약정한 대로 보고한다.

### 10.3 Layer 2 진입 조건

다음을 **모두** 만족해야 한다.

1. 선택된 L1 또는 L1+L1.5가 공통 safety gate를 통과하고 raw보다 의미 있게 개선한다.
2. Best safe analytic base 이후 잔차를 slide-cross-fitted predictor가 zero-residual보다 `δ_res` 이상 개선한다.
3. 추가 효과가 patch가 아니라 development slide/fold 단위에서 반복된다.
4. Spatial-affine이 그 예측 가능한 잔차를 이미 흡수하지 못한다.
5. Low-gradient 영역에서도 잔차 구조가 관찰된다.
6. 잔차가 `q_reg`, residual shift, edge/deformation과 강하게 연관되지 않는다.
7. C용 deep model이라면 scanner-LOO에서 query/context shortcut 없이 일반화된다.

### 10.4 Gate 결과 해석

| 관찰 | 결론 | 다음 단계 |
|---|---|---|
| Nontrivial safe analytic base 자체가 없음 | Exact range+band 보존 아래 image-level correction no-go | Layer 2 금지, downstream 인계 |
| Safe analytic base가 유효하고 추가 잔차가 예측 불가 | 이 데이터/계약에서 analytic transform으로 충분 | Layer 2 불필요 |
| Safe analytic base가 유효하고 추가 잔차가 재현 가능 | Deep residual 가설이 정당화됨 | Layer 2 파일럿 |
| 잔차 신호는 있으나 slide/scanner 수로 판정 불가 | Inconclusive | 추가 cohort/scanner 전까지 주장 보류 |

Gate 실패를 자동으로 "affine으로 충분"이라고 해석하지 않는다. Safety failure와 residual absence,
statistical power 부족은 서로 다른 결론이다.

---

## 11. Layer 2의 context 사용 검증

Canonical context로 바꾸는 것만으로 context 사용이 보장되지 않는다. Query만 보고 scanner correction을
외울 수 있기 때문이다. 다음을 함께 사용한다.

- Zero-anchor: `r(q,c)=g(q,c)-g(q,c_canonical)` 또는 동등한 구조
- Context off: residual `0`, output이 analytic base로 정확히 회귀
- Shuffled context: 올바른 context보다 성능 저하
- No-query-stat: context-only contribution 측정
- Query-only: context 없이 scanner memorization 가능한지 측정
- Scanner-label probe on context code: diagnostic, model selection에는 제한적으로 사용

Zero-anchor도 완전한 보장은 아니다. `g`의 두 항이 context를 무시하면서 같아지거나 query shortcut을 쓸 수
있으므로 ablation 결과가 함께 필요하다.

---

## 12. 실행 순서

### Stage 0 — 계약·metric·최소 효과 확정

1. A/B/C0/C1/O 이름과 inference input을 config/schema에 고정
2. B calibration budget과 slide separation을 index 수준에서 고정
3. StarDist expected-safe/harmful control 실행, threshold 고정
4. `δ`, `δ_res`와 scanner harm 기준을 development data에서 고정
5. Scalar composite projection과 post-projection band/range unit test

### Stage 1A — analytic baseline과 range Pareto

1. §5.4 baseline을 동일 tuple에서 평가
2. `α` distribution과 efficacy 손실을 scanner/slide별 보고
3. Nested development slide CV로 best safe transform family 선택
4. A의 slide-CV, B/C의 scanner-LOO와 B calibration learning curve 실행
5. Spatial-affine는 A/B에서만 primary 평가

### Stage 1B — residual explainability gate

1. Best safe global/spatial analytic base의 residual 저장
2. Slide-cross-fitted low-capacity predictor와 zero-residual 비교
3. q_reg/deformation/edge/gradient dependence 분석
4. §10.3에 따라 Layer 2 진입 또는 종료

### Stage 2 — Layer 2 파일럿 (조건부)

1. Logit/gradient/context diagnostics를 포함한 one-batch smoke
2. One-slide overfit은 optimization sanity만 확인하며 generalization 증거로 사용하지 않음
3. Nested development 평가와 analytic base 대비 paired effect
4. Context off/shuffle/query-only ablation
5. 최종 pipeline을 고정하고 internal test 2 slide에 한 번 평가

Layer 2가 열리지 않으면 Stage 1의 best analytic pipeline을 refit하여 internal final test에 적용한다.

### Stage 3 — locked Akoya challenge

Akoya는 **Layer 2 진입 여부와 무관하게 최종 pipeline 선택 후 한 번** 평가한다.

- C0: zero-shot 적용
- C1: state flag 사용 여부를 명시
- B: 사전 등록 budget으로 development Akoya calibration slide에서 fit하고 final Akoya test slide에서 평가
- O: evaluation-paired ceiling, 배포 성능에 포함하지 않음

Akoya는 Exp-02–05에서 이미 여러 번 분석됐다. 따라서 pristine external validation이 아니라
**Exp-07 학습과 선택에서 withheld된, 역사적으로 노출된 scanner challenge**로 부른다. Akoya 결과를 본 뒤
transform, bandwidth, threshold 또는 calibration budget을 다시 선택하지 않는다.

---

## 13. 데이터 계약

- Internal `AT2–GT450/VERSA/Akoya` store와 external `AT2–S60` store는 별도 lattice이므로 join하지 않는다.
- 한 evaluator가 scanner pair를 균형 있게 선택하되 각 store 안의 paired tuple만 읽는다.
- S60 index는 main slide split을 non-destructive view로 사용한다.
- Registration quality는 scanner별 임의 loss weight가 아니라 per-sample `q_reg`와 별도 sensitivity analysis로
  처리한다.
- Akoya tensor/statistic은 final challenge 전 train, threshold, transform selection에 사용하지 않는다.

| 역할 | Slides |
|---|---|
| Development | `12.5_4`, `12.5_17`, `2-8_1`, `2-8_14`, `8-12_5`, `8-12_9`, `12.5_27`, `2-8_3` |
| Final internal test | `8-12_14`, `2-8_2` |

B calibration에서는 development slide 중 일부를 calibration으로 사용하고 그 draw의 evaluation fold에서는
완전히 제외한다. Final internal test slide는 calibration에 사용하지 않는다.

---

## 14. 명시적 anti-goals

- Full-image CycleGAN/unpaired translation
- 고주파 neural style transfer 또는 unsupported detail 생성
- A의 숫자로 C의 실현 가능성을 주장
- O를 deployable performance로 보고
- Calibration/evaluation slide 또는 tuple leakage
- Full-resolution AT2 pixel MAE나 paired low-MAE 하나만으로 성공 선언
- Image-only idempotence를 hard gate로 사용
- Clamp로 range safety를 만들고 copied-band preservation을 주장
- StarDist proxy만으로 diagnostic information 보존 주장
- Akoya를 pristine external validation이라고 주장
- 보편적 불가능성이나 well-posedness를 정리처럼 주장

---

## 15. 최종 판정 문장 템플릿

### A 강, B calibration-dependent, C 약

> Exp-07은 저주파 scanner harmonization의 계약별 한계를 측정했다. Known-scanner supervised
> affine(A)은 held-out slide에서 강했고 composite projection 아래의 안전한 efficacy를 정량화했다.
> Unseen scanner few-shot(B)은 calibration budget에 따라 성능이 증가했지만, strict zero-shot(C0)은
> scanner-LOO에서 일관된 일반화가 지지되지 않았다. 남는 고주파 batch effect는 band별로 보고하여
> downstream/feature-level로 인계한다.

### Safe analytic transform으로 충분

> Range-safe global/spatial analytic transform 이후 잔차는 slide-cross-validation에서 재현 가능하게
> 예측되지 않았다. 따라서 이 데이터와 배포 계약에서는 deep image-level residual의 추가 복잡성이
> 정당화되지 않는다.

### Image-level no-go

> Nontrivial correction은 exact composite range와 copied-band safety를 동시에 만족하지 못했다. 이는
> affine이 충분하다는 뜻이 아니라, 현재의 exact safety contract 아래 image-level 저주파 correction
> 자체가 성립하지 않았다는 뜻이다.

### Deep residual 진입 및 성공

> Best safe analytic base 이후에도 registration 지표와 독립적인 slide-cross-fitted 잔차가 남았고,
> bounded residual이 analytic base 대비 사전 등록 효과 크기를 넘는 추가 개선을 보였다. 이 결론은
> architectural band preservation과 지정된 morphology/embedding proxy 범위에 한정되며, 별도 downstream
> task 정보 보존을 의미하지 않는다.

### Inconclusive

> 잔차 신호는 관찰됐으나 scanner domain과 slide 수가 작아 계약 간 차이 또는 deep residual의 추가
> 이득을 안정적으로 판정하지 못했다. 추가 scanner/cohort 없이 일반화 주장을 하지 않는다.

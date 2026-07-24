# Exp-02 실행 기록

작성일: 2026-07-15 (마지막 검증: 2026-07-23)
현재 판정: **Stage 1B sample-identity 수정 및 lattice별 matched probe 재산출 완료**
선행 설계는 완료 후 정리했으며 핵심 계약은 이 실행 기록에 보존한다.

> **2026-07-24 정리:** Stage-0/local projection과 fingerprint-driven
> zero-shot CLI는 종료하고 identity-fixed evaluator, pairwise transform과
> HF trajectory만 active code로 유지한다. Historical path와 무효화 사유는
> [`retired_attempts.md`](retired_attempts.md), 현재 과학적 해석은
> [`research_synthesis.md`](research_synthesis.md)에 정리했다.

> **2026-07-23 정정:** §6.3, §8.3, §8.5의 기존 joint scanner-probe 수치는
> internal-v3와 external-S60 lattice를 `(slide_id, tuple_id)`만으로 합친 오류가 있어
> 최종 근거로 사용하지 않는다. Paired cosine처럼 각 source와 자기 AT2 pair만 비교한 값은
> 참고 가능하지만, scanner 간 비교는 아래 §9의 lattice별 matched 결과로 대체한다.
> §8.1–8.2의 3-scanner fingerprint 분류와 그 예측에 의존한 closed-set/LOSO 수치도
> lattice confounding을 제거하기 전까지 exploratory로만 취급한다.

## 1. 구현 산출물

- `src/prenorm/exp02/projection.py`: §5.3 scalar identity-direction composite range projection
  (`scalar_range_alpha`, `project_correction`, `low_correction_delta`)
- `tests/test_exp02.py`: range safety / copied-band identity / α maximality·linearity / identity-input 계약 테스트
- `src/verify_exp02_stage0.py`: 이미 fit된 Exp-01 affine(contract A)에 projection 적용, range-safe
  efficacy·α·binding·safety 감사
- `scripts/exp02/stage0_cpu.sbatch`: CPU 감사 launcher

## 2. Projection 계약 검증 (SLURM `13516062`, COMPLETED)

`tests/test_exp02.py` **4 passed**. `y(α)=x+αd=R(l+α(l'−l),b)` 선형 항등식, 원소별 α 상한, α
maximality(binding image가 경계에 접함), identity-input(α=1) 모두 통과. Architectural safety는 실데이터에서도
전 scanner **max clip `0.0000%`, copied-detail MAE `≤4.9e-8`** — composite range와 high-band identity가
구조적으로 보장됨(계획 §6.1의 미해결 문제를 scalar projection이 실제로 해결).

## 3. Range-safe affine efficacy 실측 (held-out test, n=512/scanner)

Contract A(scanner-known, train-fit) affine을 test tuple에 적용하고 §5.3 projection으로 range를 강제한
뒤 coarse low-band MAE(입력 `[-1,1]` 스케일)를 측정했다.

| Scanner | raw MAE | unsafe affine | safe projected | unsafe 감소 | safe 감소 | **efficacy survival** |
|---|---:|---:|---:|---:|---:|---:|
| GT450 | 0.333 | 0.020 | 0.057 | 94.0% | 83.0% | **88.3%** |
| VERSA | 0.160 | 0.033 | 0.077 | 79.5% | 52.3% | **65.8%** |
| S60 | 0.199 | 0.015 | 0.120 | 92.5% | 39.9% | **43.1%** |

- raw/unsafe 수치가 Exp-01 baseline(93.8/79.7/92.3%)과 일치 → affine 로딩·파이프라인 신뢰 가능.
- **핵심 결과:** range-safe projection 후에도 correction이 0으로 붕괴하지 않는다(계획 §3의 "correction이
  거의 0이 될 수 있음" 최악 시나리오는 발생 안 함). GT450은 88% 생존으로 강하고, VERSA 66%, **S60은
  43%로 절반 이상 손실**.

## 4. α 분포와 binding 분해 (계획 §5.3, review issue 2)

| Scanner | α mean/median/q05 | frac<0.05 | binding 비율 | \|x\|_bind (median) | corr(α, ‖d‖∞) | corr(α, frac_extreme) |
|---|---|---:|---:|---:|---:|---:|
| GT450 | 0.86 / 0.88 / 0.61 | 0.0% | 58.6% | 0.529 | **−0.82** | +0.63 |
| VERSA | 0.64 / 0.60 / 0.21 | 1.2% | 68.2% | 0.922 | +0.26 | +0.31 |
| S60 | 0.44 / 0.31 / 0.00 | **21.5%** | 74.4% | **0.945** | −0.39 | −0.06 |

해석:

- **GT450:** α가 **보정 크기 ‖d‖에 걸린다**(corr −0.82), binding pixel이 mid-tone(|x|=0.53). 즉 range가
  아니라 correction 자체 크기가 제약. 이 경우 scalar projection이 적절하고 efficacy 손실이 작다.
- **VERSA·S60:** binding pixel이 **|x|≈0.92–0.95의 near-saturated 극단값** — copied high band가 만든
  사전 극단 pixel이 global α를 choke한다. 특히 **S60는 21.5% 이미지에서 α<0.05, q05=0**(최소 5%
  이미지는 correction 완전 억제). 이것이 S60 efficacy가 43%로 무너지는 직접 원인.
- ⟹ review issue 2(scalar-α의 detail-포화 confound)가 **VERSA/S60에서 실증됨.** S60의 낮은 survival은
  저주파 correctability가 낮아서가 아니라 **한 개의 near-saturated pixel이 이미지 전체 α를 죽이는 global
  scalar의 구조적 한계** 때문일 가능성이 크다.

## 5. 함의와 다음 단계

1. **scalar projection = valid, safe한 첫 range baseline.** 계약(§10.1)을 완전히 만족하고 GT450에서
   강함.
2. **S60(그리고 부분적으로 VERSA)는 §5.3 "두 번째 단계 = low-coefficient convex projection/QP"가 필요.**
   per-pixel/local 제약이면 saturated pixel 근처에서만 correction을 줄이고 나머지는 보존 → S60 survival이
   43%보다 오를 것으로 예상. 이것이 다음 결정적 실험.
3. 남은 Stage 1A baseline(ridge/coeff-reg affine, OD diag/3×3+gamma, Reinhard/Macenko, spatial-affine)과
   δ/Δ_bacc 고정, low-band scanner-BACC(§7.5 co-primary) 측정은 미실행.

> **갱신(§6):** 위 1·2·3(local projection·baseline·scanner-BACC)은 Stage-1에서 병렬 실측 완료.
> 예상(S60는 local projection으로 회복)이 확인됨 → §6.1~6.4 참조.

## 6. Stage-1 (2026-07-15, SLURM 13522378/79/80, 병렬 실행)

Stage-0의 세 후속 질문을 동시에 실측했다. 산출물:
`src/verify_exp02_local_projection.py`, `src/eval_exp02_baselines.py`,
`src/eval_exp02_scanner_bacc.py`; `prenorm/exp02/projection.py`에
`local_range_project` 추가(`tests/test_exp02.py` 6 passed).

### 6.1 Piece 1 — local(coefficient-wise) range projection (§5.3 두 번째 단계)

핵심 재정식화: `reconstruct(c, bands)`는 *어떤* coarse `c`에 대해서도 copied detail을 정확히
보존하므로(bands 재사용) 고주파 identity는 `c`를 구속하지 않는다. 따라서 문제는 순수 coarse-공간 QP
`min ‖c − l'‖²  s.t.  R(c,bands) ∈ [−1,1]`. penalty-projected GD(autograd가 정확한 synthesis
adjoint 제공) + 마지막 scalar backstop(α=0=무보정이 항상 feasible → 엄격 range 안전 구조적 보장).

| Scanner | scalar survival | **local survival** | gain | scalar ᾱ → local ᾱ | feasible pre-backstop |
|---|---:|---:|---:|---:|---:|
| GT450 | 88.3% | **100.5%** | +12.2 | 0.86 → 0.99 | 42.0% |
| VERSA | 65.8% | **97.2%** | +31.4 | 0.64 → 0.96 | 32.6% |
| S60 | 43.1% | **85.5%** | **+42.4** | 0.44 → 0.76 | 26.2% |

- **결론: S60의 43% 천장은 근본 한계가 아니라 global-scalar의 detail-포화 아티팩트였다** (review issue 2
  확정). local QP가 S60를 43→85.5%로 거의 2배 회복시키면서 range-safe·high-band-exact 유지.
- GT450 survival 100.5%(>100%)는 정상: range-safe local 출력이 *un-projected* affine보다도 AT2에 더
  가깝다 — affine 자체가 range 밖으로 밀려 clip으로 손해 보던 영역을 QP가 feasible하게 재분배하기 때문.
- backstop은 이제 가볍게만 작동(local ᾱ가 0.76~0.99). penalty solve만으로 ~1/3 이미지가 strict
  feasible, 나머지는 median 위반 ~0.002를 backstop이 정리. QP가 실제로 문제를 푸는 중.

### 6.2 Piece 2 — Stage-1A 고전 baseline (동일 튜플·동일 range-safe scalar 파이프라인)

| Scanner | affine | **spatial_affine** | od_affine | ridge | reinhard | macenko |
|---|---:|---:|---:|---:|---:|---:|
| GT450 safe.red | 83.0% | **83.0%** | 83.3% | 83.0% | 1.5% | 46.3%¹ |
| VERSA safe.red | 52.3% | **52.3%** | 52.4% | 52.3% | 1.6% | 32.7%¹ |
| S60 safe.red | 39.9% | **39.8%** | 35.8% | 39.9% | 3.2% | 26.4%¹ |

- **spatial_affine(quadratic field) = global affine (소수 4자리까지 동일).** global affine의 strict
  superset인데 이득 ~0 → **저주파 스캐너 차이는 순수 global**(coarse 스케일에서 vignetting/uneven
  illumination 무의미) → **deep residual의 저주파 구조적 여지 없음.** exp01 deep no-go를 새 각도로 강화.
- **ridge == affine:** 정규화로 clipping 안 줄음 → clipping은 계수 축소가 아니라 projection으로만 해결
  (Piece 1이 옳은 레버).
- **Reinhard/Macenko:** method_reduction 음수(Reinhard −22~−48%, Macenko −15~−74%) — coarse band에서
  AT2 거리를 *오히려 벌림*. 프로젝트 출발 근거("stain-norm이 이 문제를 못 푼다") 정량 재확인.
  ¹Macenko safe.red 양수는 scalar projection이 나쁜 방향을 feasibility로 당긴 부수효과일 뿐, method 자체는
  해로움(16×16 coarse에서 stain 추정 불안정).

### 6.3 Piece 3 — low-band scanner-BACC (invariance co-primary, §7.5)

scalar-projected corrected로 스캐너 분류 balanced accuracy 측정(slide split, 재학습 adversary=P1).

| task | chance | raw BACC | corrected BACC (P1) | transfer (P2) |
|---|---:|---:|---:|---:|
| 3-class {gt450,versa,s60} | 0.333 | 0.827 | **0.640** | 0.507 |
| 4-class {+at2} | 0.250 | 0.706 | **0.521** | 0.387 |

**교정 후 per-class recall (누가 아직 식별되나):**

| scanner | raw recall | corrected recall (P1) |
|---|---:|---:|
| gt450 | 0.94 | **0.44** (signature 대부분 제거) |
| versa | 0.64 | 0.58 |
| **s60** | 0.91 | **0.90** (거의 그대로 식별) |

- **MAE만 내려간 게 아니라 signature가 실제로 상당 부분 제거됨**(순수 "렌더링만 가까워짐" 귀무가설 기각) —
  진짜 부분 invariance. 단 chance까지는 못 감(재학습 분류기 잔여 분리성).
- **교정 후 잔여 분리성은 사실상 S60 전담**(recall 0.90). 이 Piece 3의 corrected는 **scalar** projection —
  즉 S60는 교정의 43%만 적용됨(6.1). ⟹ **S60 signature가 안 지워진 건 S60 교정 자체가 눌려서**라는 가설과
  정합. **예측: local projection(6.1)으로 Piece 3를 재실행하면 S60 잔여 BACC가 떨어져야 한다** (미실행).

### 6.4 종합 판정

세 갈래가 한 이야기로 수렴: (a) 저주파 교정은 global affine이 천장이고 clipping이 유일한 병목 →
(b) local QP projection이 그 병목을 풀어 전 scanner를 range-safe하게 85~100% efficacy로 회복 →
(c) 그 교정은 진짜 부분 invariance를 주지만, scalar 하에서 억눌린 S60의 잔여 signature가 남음.
**다음 결정 실험 = local projection으로 Piece 3 재실행**(S60 BACC 예측 검증) + Stage-1 CV/LOSO.

## 7. SLURM 기록

| Job | 역할 | 결과 |
|---:|---|---|
| 13516018 | Stage-0 (잘못된 base env) | FAILED — `No module named pytest` (env=base, cpath 아님) |
| 13516062 | Stage-0 (cpath env) | COMPLETED, 4 tests passed + efficacy 감사 |
| 13522378 | Stage-1 Piece 1 (local projection) | COMPLETED, 6 tests + local vs scalar survival |
| 13522379 | Stage-1 Piece 2 (baselines) | COMPLETED, 7 methods × 3 scanner |
| 13522380 | Stage-1 Piece 3 (scanner-BACC) | COMPLETED, raw/corrected BACC + confusion |
| 13527178 | Stage-1B params (fingerprint→scanner) | COMPLETED |
| 13527179/209 | Stage-1B viz (2-track panel) | COMPLETED (209 = 정규화 재렌더) |
| 13527180 | Stage-1B UNI embedding | COMPLETED (GPU, HF offline) |
| 14262927 | Identity-fixed low-band BACC | COMPLETED, lattice별 matched task |
| 14262930 | Identity-fixed UNI embedding | COMPLETED, lattice별 matched task |
| 14262931 | Identity-fixed UNI band audit | COMPLETED, full oracle + pre-clamp range |
| 14262946 | 전체 regression suite | COMPLETED, 64 passed |

## 8. Stage-1B (2026-07-15, SLURM 13527178/180/209): zero-shot fingerprint + UNI + viz

사용자 아이디어(feature bank + 슬라이드 단위 projection; blur/focus로 스캐너 파라미터 추정)를 실측.
산출물: `src/eval_exp02_scanner_params.py`, `src/eval_exp02_uni.py`, `src/viz_exp02_correction.py`,
`outputs/exp02_stage1/{scanner_params,uni_embedding,estimated_affines}.json`,
`outputs/exp02_stage1/viz/correction_panels.png`.

### 8.1 스캐너 파라미터 설명력 (fingerprint) — 핵심 결정

split 슬라이드는 모든 스캐너로 촬영(registered)되므로 슬라이드별 fingerprint의 스캐너 차이는
**조직 통제된 순수 스캐너 신호**.

| fingerprint feature | 스캐너 식별 BACC (leave-slide-out, chance 0.33) |
|---|---:|
| **sharpness만** (finest 2 band energy) | **0.767** |
| color만 (저주파 mean/std) | 0.633 |
| 전체 | 0.733 |

- **sharpness > color** → 사용자 가설 확인: 고주파/초점 통계가 tissue-robust 스캐너 지문. (단 0.77, 완벽 아님.)
- oracle affine 슬라이드 간 안정성(rel std): gt450 0.49, s60 0.59, **versa 1.98**(versa affine은 슬라이드마다 크게 흔들림).

### 8.2 fingerprint-driven 교정 성능 유지

| | gt450 | versa | s60 |
|---|---:|---:|---:|
| **closed-set** (지문→known 식별→그 affine) | 87% | 91% | 55% |
| **LOSO** (진짜 unseen, exploratory·N=2 train) | 30% | −5% | 32% |

- closed-set(=known 중 하나): gt450/versa 교정력 ~90% 유지, s60 55%.
- LOSO(unseen): 붕괴(versa는 오히려 악화 −5%) = 일반화 벽. **단 3개뿐이라 구조적 underpowered — "증거 없음"이지 "불가 증명" 아님.**
- 저주파 scanner-BACC 유지: oracle-corrected 0.471 ≈ closedset-corrected 0.486 → 지문으로 골라도 oracle과 같은 invariance.

### 8.3 UNI 임베딩 — **가장 중요한(그리고 냉정한) 결과**

| 지표 | raw | local(oracle) | est_local |
|---|---:|---:|---:|
| paired cosine→AT2 gt450 | 0.837 | **0.752** | 0.752 |
| paired cosine→AT2 versa | 0.876 | 0.872 | 0.872 |
| paired cosine→AT2 s60 | 0.756 | **0.736** | 0.736 |
| scanner-probe 4class (chance .25) | 0.998 | 0.999 | 0.999 |
| scanner-probe 3class (chance .33) | 0.999 | 0.999 | 0.999 |

- **저주파 색 교정은 UNI 임베딩 단위에서 아무 것도 안 한다.** 스캐너 분리성 0.999→0.999(불변), AT2까지의
  cosine은 오히려 소폭 **감소**(gt450 0.84→0.75). est_local==local → 지문 추정 교정도 동일(=효과 없음이 유지).
- **원인:** UNI 임베딩은 우리가 일부러 보존하는 **고주파 스캐너 signature에 지배**됨(8.1의 sharpness=지문과 정합).
  저주파만 바꾸고 고주파(source texture)를 그대로 두면, UNI엔 "source 텍스처+shifted color" 키메라 →
  AT2로 가까워지지 않음.
- 트리의 학습형 canonicalizer도 UNI scanner-acc ~0.99→~0.99(Explore 확인) — 이미지 단 교정 전반이 PFM
  분리성을 못 줄임. (메모리의 0.96→0.65는 현 트리에서 재현 안 되는 옛 run.)

### 8.4 Stage-1B 종합 판정 (범위의 재정의)

이미지 단 저주파 교정은 **안전하고(고주파 정확 보존·range-safe), 저주파 특징 단위 invariance를 주고,
시각적으로 AT2에 정규화**한다. 그러나 **고주파-지배 PFM(UNI)에는 거의 무효** — PFM이 읽는 배치효과는
우리가 (정당하게) 보존하는 고주파에 있다. ⟹ **"feature-level = 별도 연구"라는 프로젝트 범위 결정을
데이터가 정당화**한다. zero-shot은 **closed-set(지문→known scanner) 타당 / open-set(unseen) 벽**(현 4개 규모).
가치 제안은 "PFM invariance 개선"이 아니라 **"이미지 단에서 무엇이 교정 가능/불가한지의 정확한 경계 + 안전한
저주파 정규화 + 정직한 평가"** 로 프레이밍해야 한다.

### 8.5 고주파 오라클 probe — "고주파 교정을 도전할 가치가 있나"의 결정적 답 (SLURM 13527315)

registered AT2 pair로 **진짜 AT2 detail 밴드를 이식**(hallucination 0) = 오라클 고주파 교정. `src/eval_exp02_uni_bands.py`.

| variant | UNI 4class | 3class | meanCos→AT2 |
|---|---:|---:|---:|
| raw | 0.998 | 0.999 | 0.823 |
| low_corr (우리 저주파 교정) | 0.999 | 0.999 | 0.787 |
| low_oracle (AT2 저주파 완벽 이식 + src 고주파) | 0.999 | 0.999 | 0.800 |
| **hi_oracle (src 저주파 + AT2 고주파 완벽 이식)** | **0.964** | 0.976 | 0.911 |
| **both_oracle (우리 저주파 + AT2 고주파)** | **0.670** | 0.721 | 0.965 |
| atten0.75 / 0.50 / 0.25 | 0.998 / 0.998 / 0.995 | ~0.99 | 0.747 / 0.667 / 0.507 |
| atten0.00 (순수 저주파, 최대 블러) | 0.763 | 0.660 | 0.022 |

- **저주파는 레버가 아님(오라클로 확정):** AT2 저주파를 **완벽 이식**해도 0.999. low_corr도 0.999.
- **고주파 단독 교정은 천장이 낮다:** hallucination 0의 **완벽한 AT2 고주파 이식(hi_oracle)조차 0.998→0.964**.
  생성 모델은 오라클보다 나쁠 수밖에 없으니 **고주파-only는 ~0.96 아래로 못 감** → 도전 가치 낮음.
- **고주파는 부분 교정에 강건:** atten 0.25~0.75(고주파 25~75% 감쇠)는 분리성 거의 불변(0.99+); λ=0(완전 제거)만
  0.66~0.76로 떨어지나 이는 정보 파괴 + AT2에서 오히려 멀어짐(cos 0.022). = 교훈 2의 정량 확인, 절벽형.
- **UNI를 움직이는 유일한 길은 두 밴드 동시 교정(both_oracle 0.67)** — 즉 타일을 사실상 AT2로 재생성. 그마저
  0.67(chance 0.25/0.33에 못 미침)은 우리 저주파 low_c의 잔차 탓; low까지 완벽하면 AT2 자체 → chance.
  = UNI 배치효과는 어느 밴드든 남으면 읽힌다(밴드 간 **interaction**).

**판정 (고주파 도전):** **가치 낮음.** 오라클 천장이 고주파-only는 0.96에서 막힘을 증명. UNI 불변의 유일한
경로는 "두 밴드 동시 near-AT2 재생성" = 이미 no-go였던 최대-hallucination 전면 translation(pix2pix no-go,
canonicalizer 0.99). 이는 별도 feature-level 영역. ⟹ **경계-규명(A) 논문을 더 강하게 뒷받침**: 이미지 단
교정이 왜 PFM 불변을 못 사는지를 **주파수 분해 오라클로 정량 증명**.

## 9. Sample identity 정정 및 재검증 (2026-07-23)

### 9.1 오류와 수정

Internal AT2/GT450/VERSA와 external AT2/S60는 서로 다른 물리 sampling
lattice다. 그러나 기존 Exp-02 probe는 위치 key를 `(slide_id, tuple_id)`로만
만들어 서로 다른 grid의 같은 정수 tuple을 같은 위치로 취급했다.

실데이터 감사 결과 숫자 key는 train/val/test에서 각각 4,526/988/1,516개
겹쳤지만, 해당 key의 `(x,y)`가 실제로 같은 경우는 **모든 split에서 0개**였다.
따라서 joint 3/4-class probe는 scanner effect와 lattice/content sampling을
분리할 수 없다.

수정한 계약:

- 모든 Exp-01/02 record에 `lattice_id`를 명시
- location identity를 `(lattice_id, slide_id, tuple_id)`로 고정
- internal-v3 `{AT2, GT450, VERSA}`와 external-S60 `{AT2, S60}`를 별도 task로 평가
- 각 task에서 scanner들이 공유하는 물리 location의 교집합만 동일 개수로 선택
- 이전 JSON은 `archived/2026-07-23_pre_identity_fix_exp02/`에 보존

관련 구현은 `src/prenorm/data/identity.py`와
`src/eval_exp02_{scanner_bacc,uni,uni_bands}.py`에 반영했다.
핵심 테스트 14개와 전체 regression 64개가 모두 통과했다.

### 9.2 Low-band scanner separability 재산출

SLURM `14262927`에서 scalar range-safe affine correction을 동일 location에
적용했다. Train+val은 lattice별 1,600 location, test는 800 location이며 각
class count가 정확히 같다.

| lattice task | chance | raw BACC | corrected BACC (P1) | transfer (P2) |
|---|---:|---:|---:|---:|
| internal-v3 `{AT2, GT450, VERSA}` | 0.333 | 0.813 | **0.556** | 0.396 |
| external-S60 `{AT2, S60}` | 0.500 | 0.928 | **0.780** | 0.770 |

저주파 affine correction은 두 lattice 모두 scanner separability를 실제로
낮추지만 chance까지 제거하지는 않는다. Internal에서는 감소 폭이 0.257로
크고, external S60에서는 0.148에 그친다. 이 S60 결과는 local projection이
아닌 scalar projection이므로 §6.1에서 확인한 correction suppression을
포함한 보수적 수치다.

### 9.3 UNI embedding 재산출

SLURM `14262930`에서 test location을 lattice별 300개로 맞추어 재평가했다.
Scanner 간 sampling content가 같아졌지만 기존 paired-cosine 결론은 거의
변하지 않았다.

| scanner | raw cosine→AT2 | local low correction | estimated local |
|---|---:|---:|---:|
| GT450 | 0.841 | **0.759** | 0.759 |
| VERSA | 0.874 | 0.871 | 0.871 |
| S60 | 0.763 | **0.743** | 0.743 |

| lattice task | chance | raw | local | estimated local |
|---|---:|---:|---:|---:|
| internal-v3 `{AT2, GT450, VERSA}` | 0.333 | 0.998 | 0.998 | 0.998 |
| external-S60 `{AT2, S60}` | 0.500 | 0.998 | 1.000 | 1.000 |

따라서 **저주파 교정은 UNI representation alignment를 개선하지 않는다**는
핵심 결과는 identity 정정 후에도 유지된다. GT450과 S60에서는 오히려 자기
AT2 pair와의 cosine이 낮아진다. 이 결과는 저주파 이미지 correction의
실패가 아니라, 저주파 시각 정규화와 PFM invariance가 서로 다른 목적임을
보여준다.

### 9.4 Band manipulation, exact oracle, range audit

SLURM `14262931`에서 기존 `both_oracle`을
`corr_low_at2_high`로 정확히 다시 명명하고, paired AT2 원본 자체인
`full_oracle`을 positive control로 추가했다.

| variant | internal BACC | external BACC | mean cosine→AT2 | 입력 range |
|---|---:|---:|---:|---|
| raw | 0.998 | 0.998 | 0.826 | valid |
| low_corr | 0.998 | 1.000 | 0.791 | valid |
| low_oracle | 0.998 | 1.000 | 0.803 | **invalid 일부** |
| hi_oracle | 0.979 | 0.985 | 0.910 | **invalid 다수** |
| corr_low_at2_high | 0.691 | 0.860 | 0.966 | **invalid 일부** |
| **full_oracle** | **0.333** | **0.500** | **1.000** | valid |
| atten0.75 | 0.997 | 1.000 | 0.750 | valid |
| atten0.50 | 0.998 | 0.998 | 0.670 | valid |
| atten0.25 | 0.994 | 1.000 | 0.510 | valid |
| atten0.00 | 0.837 | 1.000 | 0.023 | valid |

Exact full oracle은 두 task에서 각각 chance와 정확히 같고 cosine 1.0이므로
새 grouping과 probe가 올바르게 작동한다. 동시에 pre-embedding range
audit가 기존 band-oracle 해석의 중요한 한계를 드러냈다.

- `hi_oracle`은 scanner별 이미지의 99.7–100%가 range를 벗어나며 평균
  7.99–13.27% pixel이 범위 밖이다.
- `corr_low_at2_high`도 95.3–100% 이미지가 범위를 벗어난다. 평균 위반
  pixel은 GT450 0.034%, VERSA 0.249%, S60 2.051%다.
- UNI evaluator는 입력을 방어적으로 clamp하므로 이 두 score는
  “정확한 paired coefficient + incompatible band composition + clipping”의
  결과다. 이를 clean high-frequency causal oracle로 부르면 안 된다.
- 반대로 attenuation frontier는 전부 range-valid다. Detail을 75→0%로
  줄이면 AT2 cosine은 0.750→0.023으로 붕괴하지만 scanner BACC는 거의
  그대로다. 특히 external S60는 detail을 완전히 제거해도 BACC 1.000이다.

정정된 결론은 “고주파만이 scanner signature의 유일한 원인”이 아니다.
**UNI는 저주파 또는 고주파 중 어느 한쪽에 scanner-specific residual이
남아 있어도 거의 완벽히 분리하며, 단순 blur는 content만 파괴하고
alignment를 만들지 못한다.** Paired high band를 다른 low band와 조합하면
물리 범위 자체가 쉽게 깨진다는 사실은 고주파 복원의 hallucination/validity
문제를 직접 보여준다.

### 9.5 남은 identity debt

`eval_exp02_scanner_params.py`는 slide-level fingerprint를 GT450/VERSA/S60
3-class로 분류한다. 직접 location join을 하지는 않지만 S60만 다른 grid에서
sampling되므로 scanner와 lattice를 분리할 수 없다. 따라서 §8.1의
sharpness/color BACC와 이를 사용한 §8.2의 predicted-affine 결과는 core
finding에서 제외한다. 다음 재설계는 internal GT450-vs-VERSA와 external
AT2-vs-S60를 별도 matched task로 만들고, feature-level correction baseline과
함께 평가해야 한다.

## 10. Pairwise scanner rendering과 UNI/blur frontier (2026-07-23)

> **Superseded contract:** §10.1–10.3은 strict-safe sensitivity와 sharp-target
> blur 비교를 사용한 contract-v2 기록이다. 현재 분석에서는 strict-safe를
> 활성 extractor/figure에서 제거했고, Akoya와 matched-blur reference를
> 포함한 contract-v3가 이를 대체한다. 아래 수치는 과거 provenance로만
> 남기며 현재 결론의 근거로 사용하지 않는다.

### 10.1 계약과 실행

`src/prenorm/exp02/pairwise.py`에 각 lattice 안의 모든 directed scanner
pair를 train split에서 fit하는 affine과 source detail-retention frontier를
추가했다. Test split은 방향마다 동일한 registered location 160개를 사용했다.

- internal: `AT2 ↔ GT450 ↔ VERSA` 6방향
- external: `AT2 ↔ S60` 2방향
- identity: `(lattice_id, slide_id, tuple_id)`
- primary correction: `clamp(R(affine(low), source_detail), -1, 1)`
- sensitivity correction: coefficient-wise range projection + scalar fallback
- detail frontier: source high band retention `{1.00, 0.75, 0.50, 0.25, 0.00}`
- embedding 직전 모든 operational variant의 out-of-range pixel fraction: **0**

초기 strict-safe 분석은 `archived/2026-07-23_pre_clipped_primary_exp02_visualization/`
에 보존했다. SLURM `14267071`에서 전체 test suite **68 passed**,
`14267072`에서 clipped-primary와 strict-safe sensitivity UNI를 함께
재추출했다. 산출물은 1,280-row pairwise table, 6,400-row clipped blur
table, 6,400-row safe blur table, exemplar와 summary다.

### 10.2 방향별 low-frequency correction과 UNI response

| direction | LF MAE raw→clip / safe | source↔clip UNI | Δ target cosine clip / safe | safe fallback |
|---|---:|---:|---:|---:|
| AT2→GT450 | 0.333→0.018 / 0.313 | 0.817 | -0.104 / -0.065 | 43.1% |
| AT2→VERSA | 0.161→0.036 / 0.163 | 0.984 | +0.004 / -0.005 | 81.3% |
| GT450→AT2 | 0.333→0.023 / 0.017 | 0.797 | -0.091 / -0.088 | 0.0% |
| GT450→VERSA | 0.183→0.040 / 0.057 | 0.841 | -0.067 / -0.048 | 26.9% |
| VERSA→AT2 | 0.161→0.034 / 0.038 | 0.979 | +0.001 / -0.004 | 1.3% |
| VERSA→GT450 | 0.183→0.029 / 0.212 | 0.863 | -0.085 / -0.045 | 46.3% |
| AT2→S60 | 0.193→0.028 / 0.182 | 0.960 | -0.006 / -0.014 | 84.4% |
| S60→AT2 | 0.193→0.023 / 0.044 | 0.955 | +0.008 / -0.019 | 21.9% |

사용자 지적대로 strict-safe의 전체 fallback rate는 **38.1%**였고,
source↔output cosine도 safe `0.927` 대 clipped `0.900`이었다. 따라서
기존 safe 결과는 UNI movement를 축소하며 “embedding이 거의 움직이지
않는다”는 표현을 과장했다.

그러나 operational clipping 후에도 target alignment 결론은 뒤집히지
않았다. 전체 평균 Δ target cosine은 clipped **-0.0426**, safe `-0.0360`이고,
AT2 anchor 세 방향 평균도 clipped **-0.0275**, safe `-0.0373`이다. 8방향
중 3방향만 `+0.001–0.008`의 작은 양수이며 5방향은 감소한다. Pilot test
split이 slide 2개뿐이므로 이 작은 양수는 confirmatory evidence가 아니다.

> 저주파 교정은 UNI embedding을 움직일 수 있지만, 그 이동은 paired target
> scanner embedding으로의 alignment가 아니며 scanner separability도
> 해소하지 않는다.

#### Figure-1 visual contract 정정

초기 Figure 1은 affine-transformed low band와 range-safe full composite를
모두 “transformed”로 부르면서 두 단계를 혼동했다. Representative tile을
분해해 다시 감사하면, 제약 전 affine low band는 8방향 모두 target LF MAE를
`0.005–0.021`까지 낮춘다. 반면 이를 source detail과 합성한 full image는
방향에 따라 최대 11.4% pixel이 `[-1,1]` 범위를 벗어났다.

특히 exact source-detail preservation과 hard range를 동시에 요구하면
source의 saturated pixel 하나가 마지막 scalar backstop의 `alpha=0`을 만들
수 있다. AT2→GT450/VERSA/S60 exemplar에서 이 현상이 확인됐다. 따라서
range-safe composite가 target처럼 보이지 않는 것을 affine low-frequency
mapping 실패로 해석하면 안 된다. 동시에 현재 penalty solver와 scalar
backstop이 feasible correction을 과도하게 억제하는 수치적 가능성도 남는다.

Figure 1은 이에 맞춰 다음 일곱 열로 재구성했다.

`source image | source LF | affine LF | target LF | clipped composite | range-safe composite | target image`

이 그림이 지지하는 두 주장은 분리한다.

1. **Low-band statement:** 단순 affine만으로 registered target low band에
   매우 가깝게 갈 수 있다.
2. **Full-image statement:** target low band와 source high band를 그대로
   조합하는 것은 항상 valid RGB image를 만들지 않으며, strict safety가
   실제 적용 가능한 correction을 제한할 수 있다.

Main Figure 1은 결과를 보고 성공 방향을 고르는 대신, 프로젝트에서 사전에
canonical anchor로 지정했던 `GT450/VERSA/S60 → AT2` 세 방향만 사용한다.
전체 8방향은
`figure_s1_all_pairwise_feasibility`에 보존해 선택되지 않은 실패 방향을
숨기지 않는다.

Supplementary figure에는 `clamp(unsafe, -1, 1)` 결과도 추가했다. Clipping은
실용적으로 target appearance에 가까운 이미지를 만들지만, pyramid를 다시
분해하면 source detail coefficient가 변한다. 따라서 다음처럼 사용한다.

- practical image-normalization baseline: 사용 가능;
- “source high-frequency information unchanged” causal probe: 사용 불가;
- clipped pixel fraction과 detail-coefficient MAE를 항상 함께 보고.

### 10.3 Source-detail attenuation

| blur severity | UNI cosine→source | UNI cosine→paired target | image SSIM→target |
|---:|---:|---:|---:|
| 0.00 | 0.900 | 0.773 | 0.458 |
| 0.25 | 0.876 | 0.757 | 0.460 |
| 0.50 | 0.799 | 0.704 | 0.434 |
| 0.75 | 0.603 | 0.559 | 0.365 |
| 1.00 | 0.019 | 0.023 | 0.244 |

Clipped-primary 평균은 blur severity가 증가할수록 source와 paired target
UNI cosine이 함께 감소한다. 방향별로는 6/8이 단조 감소하고 2방향은 약한
초기 비단조성이 있지만, 어느 방향도 detail 제거를 통해 target에 지속적으로
수렴하지 않는다. 심한 blur는 scanner alignment가 아니라 representation과
morphology를 함께 붕괴시키는 harmful control이다.

최종 시각화는 다음 네 세트다.

- `figures/figure1_pairwise_image_transform.{png,pdf}`
- `figures/figure2_uni_low_frequency_response.{png,pdf}`
- `figures/figure3_blur_embedding_frontier.{png,pdf}`
- `figures/figure_s1_all_pairwise_feasibility.{png,pdf}`

### 10.4 Contract-v3: Akoya와 rotating matched-blur reference

현재 실행 계약은 다음과 같다.

- internal lattice: `AT2/GT450/VERSA/Akoya` 모든 12개 directed pair
- S60 lattice: `AT2/S60` 2개 directed pair
- S60 표기: 독립적으로 구축된 `S60 lattice`; S60 train slide로 affine을
  fit하므로 unseen-scanner external validation으로 부르지 않는다.
- image policy: `clip(R(A_source→target(L_source), λH_source), -1, 1)`
- matched reference:
  `clip(R(L_target, λH_target), -1, 1)`; target-space affine은 identity다.
- raw-blur control:
  `clip(R(L_source, λH_source), -1, 1)`
- primary blur statistic:
  `cos(transformed_source_λ, target_λ) -
  cos(raw_source_λ, target_λ)`

즉 Figure 3은 sharp target과 blur image를 직접 비교하지 않는다. 모든
source/target 방향을 정량 분석하고, qualitative panel은 각 lattice의
reference scanner를 한 번씩 순회한다. 이 설계는 LF affine 효과와 양쪽
이미지에 동일하게 가한 detail 손상을 분리한다.

#### Contract-v3 pilot 결과

GPU job `14297678`은 160 locations/lattice에서 2,240 pair rows와 11,200
matched-blur rows를 생성했다. 전체 regression은 job `14297747`에서
**69 passed**였다.

- LF MAE: raw `0.2531` → requested affine `0.0297` → operational clipped
  `0.0366`
- full-image SSIM: raw `0.2810` → clipped `0.4329`
- UNI source↔clipped: `0.8303`
- sharp paired-target UNI gain: 전체 `-0.0202`; 14방향 중 7 positive /
  7 negative
- 기존 non-Akoya 8방향 gain: `-0.0426`
- Akoya-related 6방향 gain: `+0.0096`
- 14방향 중 6방향은 두 test slide 사이에서 gain 부호가 바뀜

따라서 “LF correction이 PFM embedding을 움직이지 않는다” 또는
“항상 target alignment에 실패한다”는 표현은 지지되지 않는다. 현재
pilot이 지지하는 범위는 다음과 같다.

> LF affine은 시각/저주파 공간에서는 강하게 작동하지만, sharp PFM
> alignment의 방향과 크기는 scanner, target, slide에 의존하며 공통된
> 교정 효과로 재현되지 않는다.

Matched-blur 결과는 두 현상을 분리한다.

| detail retention | transformed↔matched target UNI | raw-blur↔matched target UNI | LF-affine gain | blurred target↔sharp target UNI |
|---:|---:|---:|---:|---:|
| 1.00 | 0.6864 | 0.7066 | -0.0202 | 1.0000 |
| 0.75 | 0.7150 | 0.7345 | -0.0194 | 0.9500 |
| 0.50 | 0.7606 | 0.7740 | -0.0134 | 0.8260 |
| 0.25 | 0.8256 | 0.8166 | +0.0090 | 0.6196 |
| 0.00 | 0.9898 | 0.9490 | +0.0408 | 0.0447 |

Detail을 완전히 제거하면 14/14 방향에서 LF-affine gain이 양수가 되고
matched target cosine은 `0.9898`에 도달한다. 그러나 같은 시점의 target
blur embedding은 자기 sharp target과 `0.0447`밖에 남지 않는다. 이는
“LF만 남기면 scanner appearance/alignment를 맞출 수 있다”는 양성
대조군인 동시에, 그 결과가 morphology/representation 파괴를 대가로 한
것임을 보여주는 harmful control이다.

### 10.5 Baseline gate

Classical evaluator도 strict-safe projection을 제거하고
`requested LF → source-detail reconstruction → explicit clipping → re-decomposed
operational LF` 순서로 수정했다. Job `14297720`의 32-train/16-test smoke는
GT450/VERSA/Akoya/S60에서 identity, RGB affine, ridge affine, OD affine,
spatial affine, RGB Reinhard, Macenko를 모두 실행했다. 이 결과는 구현
검증용이며 효과 크기 추론에는 사용하지 않는다.

다음 full baseline은 아래 순서로 잠근다.

1. identity, RGB/OD affine, Reinhard, Macenko, low-frequency/Fourier transfer
2. paired-target oracle와 feature-level CORAL/Procrustes positive control
3. paired Pix2Pix
4. CycleGAN은 unpaired hallucination stress test로만 사용

모든 방법은 같은 test locations, clipping audit, matched-blur UNI, morphology
damage, scanner probe를 공유해야 한다. Classical baseline이 이 공통
evaluator에 연결되기 전에는 deep baseline training을 시작하지 않는다.

## 11. Scanner × high-frequency embedding trajectory (2026-07-23)

### 11.1 계약

Embedding 결과를 보지 않고 internal test lattice에서 최소 registration QC가
높은 세 위치를 선택했다. Qualitative exemplar에 pen-marker/비정상 청색
배경이 들어가는 문제를 막기 위해 tissue density는 `[0.50,0.90]`, 강한
blue-dominant pixel fraction은 scanner별 최대 `0.80`, slide당 최대 두 위치,
같은 slide 안의 좌표 거리는 최소 2,048로 제한했다. 이 screen은 image
color와 metadata만 사용하며 UNI embedding을 사용하지 않는다. 최종 위치는
`8-12_14/680`, `2-8_2/620`, `8-12_14/325`이고, 각 위치에서
AT2/GT450/VERSA/Akoya 네 scanner를 모두 사용했다.

각 raw image에 다음 16 states를 적용해 총 192 UNI embeddings를 추출했다.

- raw
- blur severity `{0.25,0.50,0.75,1.00}`
- sharpen gain increment `{0.25,0.50,1.00}`
- registered leave-one-scanner-out HF mean mix
  `{0.25,0.50,0.75,1.00}`
- 1,024 train images에서 얻은 global HF mean mix
  `{0.25,0.50,0.75,1.00}`

모든 image는 explicit clipping을 사용했다. UMAP은 PCA-reduced 전체
embedding에 한 번만 fit하고 row별 zoom만 달리했다. 거리, centroid, cosine,
각도와 content margin은 모두 normalized UNI 원공간에서 계산했다. 각도는
raw embedding 위치의 unit-sphere tangent plane에서 측정했다.

### 11.2 세-patch descriptive 결과

Train global HF mean의 RMS는 `0.00430`, 개별 image HF RMS는 `0.10084`로,
평균 HF는 원래 energy의 **4.27%**만 남았다. 따라서 global coefficient
mean은 실제로 거의 low-pass negative control이었다.

| endpoint | self UNI | Δ LOO patch-consensus UNI | patch centroid radius | content margin | cross-scanner retrieval |
|---|---:|---:|---:|---:|---:|
| raw | 1.000 | 0.000 | 0.095 | 0.374 | 1.00 |
| blur 1.00 | 0.065 | -0.781 | 0.040 | -0.000 | 0.67 |
| sharpen 1.00 | 0.828 | -0.125 | 0.120 | 0.302 | 1.00 |
| registered HF mean 0.25 | 0.957 | **+0.028** | 0.069 | **0.384** | 1.00 |
| registered HF mean 1.00 | 0.757 | -0.015 | 0.091 | 0.377 | 1.00 |
| global HF mean 1.00 | 0.078 | -0.759 | 0.040 | 0.007 | 1.00 |

해석은 다음과 같다.

1. Blur/global mean은 scanner/patch centroid를 모으지만 content와 원 embedding을
   함께 붕괴시킨다. Centroid contraction만으로 normalization success를
   주장하면 안 된다.
2. Sharpening은 patch centroid radius를 `0.095→0.120`으로 증가시킨다. 즉
   고주파 증폭은 scanner disagreement를 강화한다.
3. Registered HF mean은 유일하게 약한 positive consensus gain과 완전한
   retrieval을 동시에 보였다. 특히 25% mix가 가장 좋은 Pareto point였다.
4. 따라서 “모든 mean HF reference가 무의미하다”는 너무 강하다. 실현
   불가능한 same-location oracle mean은 일부 의미가 있지만, train global
   mean은 사실상 blur이며 virtual pre-normalization reference로 쓸 수 없다.

`registered/global HF mean`은 UNI feature의 평균이 아니다. 전자는 각
source scanner에 대해 같은 위치의 나머지 세 scanner Laplacian HF band를
평균한 image-space reference이고, 후자는 1,024 train image의
coefficient-wise HF band 평균이다. Figure 4c의 virtual centroid는 각
reference mix 이후 네 UNI embedding의 spherical mean이다.

Centroid-only 분석에서도 이 차이가 명확했다. Registered mean 25%는 평균
centroid shift가 `0.017`이면서 scanner radius를 `0.095→0.069`로 줄였다.
반면 global mean 100%는 radius를 `0.040`으로 줄이지만 raw centroid와의
cosine distance가 `0.921`까지 커졌다. 즉 “모임” 자체는 normalization
evidence가 아니며, centroid preservation과 함께 봐야 한다.

이 결과는 세 위치의 mechanistic visualization이며 통계적 population
estimate가 아니다. SLURM GPU `14298821`, 최종 CPU render `14299836`,
회귀 test `14299837` (`72 passed`)로 재생성했다. 다음 단계에서는 동일
지표를 사전 고정된 test location 전체와 복수 PFM에 적용한다.

산출물:

- `outputs/exp02_hf_trajectory/figures/figure4a_blur_umap`
- `outputs/exp02_hf_trajectory/figures/figure4b_sharpen_umap`
- `outputs/exp02_hf_trajectory/figures/figure4c_reference_centroid_umap`
- `outputs/exp02_hf_trajectory/figures/figure5_hf_trajectory_metrics`
- `outputs/exp02_hf_trajectory/figures/pairs/blur/` (3 pair figures)
- `outputs/exp02_hf_trajectory/figures/pairs/sharpen/` (3 pair figures)
- `outputs/exp02_hf_trajectory/figures/pairs/reference_mixing/` (3 pair figures)
- `outputs/exp02_hf_trajectory/trajectory_metrics.csv`
- `outputs/exp02_hf_trajectory/reference_centroid_trajectories.csv`
- `outputs/exp02_hf_trajectory/analysis_summary.json`

## 12. LF affine 이후 scanner × HF trajectory (2026-07-23)

### 12.1 실험 계약

Section 11과 같은 artifact-screened test 위치 세 개
(`8-12_14/680`, `2-8_2/620`, `8-12_14/325`)를 사용했다. AT2를 고정
reference로 두지 않고 AT2/GT450/VERSA/Akoya를 차례로 LF reference로
사용했다. 모든 directed scanner pair의 RGB affine은 test image나 UNI
embedding을 보지 않고 train split의 coarse Laplacian band로만 적합했다.

각 source image는 다음 순서로 생성했다.

1. source의 low-frequency band를 선택한 scanner reference로 RGB affine 변환
2. source의 원래 high-frequency band를 그대로 결합
3. HF gain으로 blur `{0.25,0.50,0.75,1.00}` 또는 sharpen
   `{0.25,0.50,1.00}` 적용
4. 최종 image에서 한 번만 explicit clipping하고 clip fraction 기록

`aligned_raw`를 포함해 위치 3개 × LF reference 4개 × source scanner 4개 ×
state 8개, 총 384 UNI embeddings를 추출했다. 비교용 unaligned trajectory
96개와 합친 480개 행에 UMAP을 한 번만 적합했다. UMAP 화살표는
qualitative visualization이고, 방향 cosine과 scanner radius는 normalized
UNI 원공간에서 계산했다.

### 12.2 LF 교정 및 clipping audit

Identity pair를 제외하면 reference와의 low-band MAE는 raw
`0.161–0.339`에서 affine 적용 후 `0.012–0.048`, operational image의
clipping 후 `0.023–0.055`로 감소했다. 따라서 affine 자체는 요청한
low-frequency appearance를 실제로 상당히 맞췄다.

다만 target LF와 source HF를 독립적으로 결합하면 일부 조합은 RGB gamut을
벗어났다. 특히 Akoya를 LF reference로 사용한 aligned baseline에서
AT2/GT450/VERSA의 pre-clipping pixel fraction은 각각
`17.1%/9.2%/18.5%`였다. GT450 reference에서도 AT2와 VERSA가 약 `11%`
clipping되었다. Endpoint image의 Akoya 관련 green/magenta edge는 이
incompatibility를 그대로 보여준다. 따라서 강한 sharpen 결과는 scanner의
고유 HF뿐 아니라 LF-HF incompatibility와 clipping interaction도 포함하며,
낮은 gain 결과와 clip audit를 함께 해석해야 한다.

### 12.3 UNI 원공간 결과

LF affine 자체는 일관된 scanner alignment를 만들지 못했다. 개입 강도
0에서 세 patch의 평균 scanner radius는 unaligned `0.0948`이었고,
LF→AT2 `0.1080`, LF→GT450 `0.1145`, LF→VERSA `0.1110`으로 오히려
커졌다. LF→Akoya만 `0.0873`으로 소폭 감소했다. 즉 시각적인 LF matching과
UNI scanner alignment는 같은 현상이 아니다.

Complete blur는 LF reference별 radius를 `0.0047–0.0122`까지 붕괴시켰다.
그러나 Section 11에서 확인했듯 이 상태는 원 embedding과 content separation도
함께 잃으므로 normalization success가 아니라 정보 제거에 의한 contraction이다.

Sharpen 1.0은 unaligned radius를 `0.0948→0.1195`로 늘렸다. LF→AT2,
LF→VERSA, LF→Akoya에서도 각각 `0.1080→0.1239`,
`0.1110→0.1231`, `0.0873→0.1109`로 늘었다. 반면 LF→GT450에서는
`0.1145→0.1085`로 약간 감소했다. 같은 HF 개입의 효과가 LF reference에
따라 달라지는 명확한 scanner × reference interaction이다.

Sharpen endpoint 방향의 scanner 간 cosine도 단일한 공통 방향을 보이지
않았다. Strength 1.0에서 Akoya-to-others 평균 cosine은 unaligned
`0.415`, LF→AT2 `0.302`, LF→GT450 `0.371`, LF→VERSA `0.309`,
LF→Akoya `0.284`였다. 그러나 UMAP에서 보인 것처럼 Akoya가 항상 반대
방향인 것은 아니다. 원공간 cosine은 모두 양수였고, non-Akoya LF
reference에서는 GT450이 가장 낮은 평균 agreement를 보였다. Akoya는
unaligned와 LF→Akoya endpoint에서만 가장 불일치했다. 따라서 올바른
결론은 “Akoya만 반대”가 아니라 **scanner × LF reference × HF intervention
interaction이 크다**는 것이다. 이 패턴은 weak sharpen에서도 유지되어
strength 1 clipping만의 산물은 아니지만, 세 patch의 descriptive 결과라는
한계가 있다.

이 결과는 “모든 scanner를 동시에 만족하는 image-level common space”에
대한 직접적인 부정 증명은 아니다. 하지만 (1) LF target을 바꾸면 baseline
UNI alignment가 달라지고, (2) 같은 blur/sharpen의 방향과 크기도
reference-dependent하며, (3) 일부 LF-HF 조합은 물리적으로 out-of-gamut이
된다는 세 가지 failure mode를 제공한다. 다음 confirmatory 실험은 사전
고정된 test set 전체에서 reference별 worst-case alignment, cycle
consistency, content preservation을 함께 평가해야 한다.

GPU extraction은 SLURM `14301631`, 최종 CPU render는 `14302342`,
전체 회귀 test는 `14301632` (`73 passed`)에서 완료했다.

산출물:

- `outputs/exp02_lf_aligned_hf_trajectory/figures/figure6a_lf_aligned_blur_umap`
- `outputs/exp02_lf_aligned_hf_trajectory/figures/figure6b_lf_aligned_sharpen_umap`
- `outputs/exp02_lf_aligned_hf_trajectory/figures/figure6c_sharpen_direction_cosine`
- `outputs/exp02_lf_aligned_hf_trajectory/figures/figure6d_sharpen_direction_by_strength`
- `outputs/exp02_lf_aligned_hf_trajectory/figures/figure6e_scanner_radius_after_lf_alignment`
- `outputs/exp02_lf_aligned_hf_trajectory/figures/pairs/`
- `outputs/exp02_lf_aligned_hf_trajectory/figures/umap_pairs/{blur,sharpen}/`
- `outputs/exp02_lf_aligned_hf_trajectory/direction_cosines.csv`
- `outputs/exp02_lf_aligned_hf_trajectory/scanner_radius_metrics.csv`
- `outputs/exp02_lf_aligned_hf_trajectory/low_frequency_alignment_summary.csv`
- `outputs/exp02_lf_aligned_hf_trajectory/analysis_summary.json`

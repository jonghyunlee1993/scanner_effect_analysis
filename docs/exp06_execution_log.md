# Exp-06 실행 기록

작성일: 2026-07-15  
현재 판정: **LF16 hard no-go; LF8/LF4·Akoya·CV 미실행**

## 1. 계획 검토에서 확정한 구현 계약

- Internal `AT2–GT450/VERSA`와 external `AT2–S60`는 서로 다른 lattice이므로 물리적으로 join하지
  않는다. 하나의 pair sampler가 scanner를 `1:1:1`로 선택하고 각 store 안에서만 paired tuple을 읽는다.
- S60 index의 기존 `test` 표시는 원본을 변경하지 않고 main 6/2/2 slide split을 적용한 별도 parquet
  view로 저장한다.
- LF16은 256 px 입력을 4회 reduce한 16 px coarse tensor만 수정한다. 더 미세한 네 coefficient tensor는
  입력에서 그대로 synthesis에 재사용한다.
- “copied coefficient identity”와 출력 재분해 coefficient는 구분한다. 전자는 architecture contract이고,
  후자는 clipping/range 변화까지 포함하는 diagnostic이다.
- float32 synthesis round-off가 clipping으로 오인되지 않도록 range 초과는 `1e-6` tolerance 뒤 측정한다.

## 2. 구현 산출물

- `src/prenorm/exp06/`: fixed pyramid, balanced pair data contract, minimal LF residual model, Lightning loss
- `src/train_exp06.py`: epoch 10/30 milestone을 포함한 학습 CLI
- `src/audit_exp06_stage0.py`: scanner별 pair/registration/range/band audit
- `src/overfit_exp06.py`: 고정 one-slide, three-scanner gradient/idempotence audit
- `src/eval_exp06.py`: held-out low/detail/clipping/idempotence evaluator
- `src/eval_exp06_baselines.py`: train-only paired affine와 fixed-target RGB Reinhard LF16 baseline
- `configs/experiments/exp06{a,b,c}_*.yaml`: LF16/LF8/LF4 사전 고정 bandwidth
- `scripts/exp06/`: CPU array와 GPU smoke/overfit/train/eval launchers
- `tests/test_exp06.py`: pyramid round-trip, AT2 bitwise bypass, finite backward

## 3. 데이터 및 filter audit

### 3.1 Pair 수와 registration

| Scanner | Train pairs | Val pairs | Test pairs | q_reg train/val/test mean | Test residual shift mean |
|---|---:|---:|---:|---:|---:|
| GT450 | 8,024 | 1,988 | 2,781 | 0.989 / 0.980 / 0.992 | 0.120 |
| VERSA | 7,411 | 1,798 | 2,686 | 0.588 / 0.638 / 0.602 | 0.296 |
| S60 | 4,646 | 993 | 1,525 | 0.991 / 0.898 / 0.993 | 0.118 |

S60의 AT2 기준 matching은 train/test에서 양호하다. S60 validation에는 `q_reg=0`인 하위 표본이 있어
계획대로 per-sample quality weighting을 유지한다. 가장 어려운 registration source는 S60가 아니라 VERSA다.

### 3.2 Frequency contract

- LF16 pyramid round-trip max absolute error: scanner/sample 전체 최대 `2.38e-7`
- 계약 threshold: `1e-6`
- AT2 hard bypass: `torch.equal(output, input)` unit/GPU smoke 통과
- 전체 regression suite: `83 passed`

## 4. GPU gate 결과

### 4.1 Smoke

- SLURM job `13512620`, A100-40G, `COMPLETED (0:0)`
- one train batch + one validation batch의 bf16 forward/backward 통과
- model size: 76.6K trainable parameters

### 4.2 One-slide efficacy와 idempotence

Slide `12.5_17`의 scanner별 고정 batch를 반복 최적화했다. 원래 idempotence weight `0.5`는 low pair
MAE를 GT450 39%, S60 41%, VERSA 75% 줄였지만, 두 번째 적용 MAE가 약 `0.12`로 hard threshold
`1/255 = 0.00392`를 크게 넘었다.

Static weight `6/7/8`, fractional boundary `7.2/7.4/7.6`, 그리고 weight `0.5` warm-up 뒤
`7.4/8/10`으로 전환하는 curriculum을 검사했다. 결과는 연속적인 Pareto trade-off가 아니라 다음 두 해로
갈라졌다.

1. correction 해: pair MAE는 크게 감소하지만 하나 이상의 scanner에서 반복 correction이 남음
2. identity 해: idempotence는 통과하지만 pair MAE 감소가 5% efficacy gate에 미달

예를 들어 warm-up 뒤 weight 7.4는 S60/VERSA idempotence를 만족했지만 GT450이 `0.12`로 실패했다.
weight 8 이상은 세 scanner 모두 correction이 사실상 0으로 붕괴했다. 따라서 scanner별 loss weight를
도입하지 않는 현재 계약 아래 단일 objective가 efficacy와 idempotence를 함께 만족한다는 증거가 없다.

## 5. Classical LF16 baseline

Train slide에서만 2,048 pair/scanner로 전역 coarse RGB affine을 fit하고 전체 test pair에서 평가했다.

| Scanner | Identity low MAE | Affine low MAE | 감소율 | Affine idempotence MAE | Affine clipping |
|---|---:|---:|---:|---:|---:|
| GT450 | 0.3280 | 0.0204 | 93.8% | 0.5757 | 0.534% |
| VERSA | 0.1601 | 0.0326 | 79.7% | 0.1496 | 0.036% |
| S60 | 0.1975 | 0.0152 | 92.3% | 0.2060 | 0.135% |

Paired affine은 두 test slide 모두에서 매우 강한 efficacy를 보였지만 image-only idempotence를 위반했다.
Fixed-target RGB Reinhard는 idempotent였지만 GT450 개선은 23.5%에 그쳤고 VERSA/S60 low MAE를 각각
46.3%, 23.9% 악화시켰으며 clipping도 1.19–3.05%였다.

이 결과는 기존 hard gate가 단순히 learned model의 최적화 실패를 잡은 것이 아니라 비자명한 invertible color
correction과 구조적으로 충돌함을 보인다. affine `T(x)=Ax+b`가 image-only idempotent이려면 `A²=A`,
`Ab=0`이어야 하고, full-rank transform에서는 `A=I`만 가능하다. 따라서 idempotence는 diagnostic으로 내리고,
반복 실행이 필요한 deployment에서는 normalized-state metadata/bypass 또는 별도 projection을 요구한다.

## 6. LF16 full pilot

Job `13513721`에서 30 epoch를 학습했고 exact epoch 10/30 checkpoint를 Job `13513863`에서 같은 balanced
test sampler로 평가했다. 중복 tuple 제거 후 GT450 367, VERSA 369, S60 354개, 총 1,090 pair다.

### 6.1 Held-out low-band 결과

Epoch 10과 30 결과는 표시 정밀도뿐 아니라 per-image 수준에서 동일했다.

| Scanner | Raw low MAE | Learned low MAE | 감소율 | Raw→learned low SSIM | 두 test slide 평균 개선 |
|---|---:|---:|---:|---:|---|
| GT450 | 0.33395 | 0.21758 | 34.9% | 0.5732→0.6259 | 모두 개선 |
| VERSA | 0.16054 | 0.05598 | 65.1% | 0.7181→0.8848 | 모두 개선 |
| S60 | 0.19683 | 0.11065 | 43.8% | 0.6699→0.8016 | 모두 개선 |

세 scanner 모두 5% efficacy threshold와 slide-direction criterion은 만족했다. 그러나 learned 결과는 paired
affine의 감소율 GT450 93.8%, VERSA 79.7%, S60 92.3%보다 일관되게 낮다. Hard gate가 먼저 실패했으므로
low-band ResNet18 scanner probe와 full-image sealed observer는 실행하지 않았다.

### 6.2 Architecture 및 range safety

- AT2 bitwise bypass: 통과
- pre-clipping copied coefficient max error: `0`
- GT450 clipping mean: `0%`
- VERSA clipping mean: `0.0145%`
- S60 clipping mean: `0.1982%` — 사전 기준 `0.1%` 실패
- S60 single-image clipping max: `19.08%`
- S60 copied-detail energy ratio range: `0.99584–1.00730`; 상한 `1.005` 실패
- S60 copied-detail coherence minimum: `0.96391`; clipping outlier에서 high-detail contract 실패

가장 큰 S60 outlier는 test slide `2-8_2`, tuple `526`이었다. `q_reg=0.966`으로 registration failure가
아니며, output low MAE도 raw `0.0472`에서 `0.1392`로 악화됐다. Test를 본 뒤 threshold를 완화하지 않았다.

### 6.3 Model-collapse audit

Epoch 30 residual을 scanner, channel, image, spatial position별로 검사한 결과 모두 정확히 `-0.12`였다.
표준편차는 0이고 다른 scanner의 context로 바꿔도 max 변화가 0이었다. 즉 76.6K model은 context-conditioned
harmonization을 학습하지 않고 모든 coarse RGB를 residual cap만큼 어둡게 하는 공통 shift로 붕괴했다.
Epoch 10/30 동일성도 이 조기 포화와 일치한다. Validation best인 epoch 0도 residual mean이
`-0.11995–-0.12000`으로 이미 거의 포화되어, 더 이른 checkpoint를 선택해도 결론이 달라지지 않는다.

## 7. 최종 go/no-go 결정

LF16은 paired low metric 자체는 개선했지만 다음 독립적인 이유로 **no-go**다.

1. S60 clipping pixel fraction과 copied-detail safety threshold 실패
2. context를 전혀 사용하지 않는 constant `-0.12` residual collapse
3. 모든 scanner에서 paired affine baseline보다 낮은 low-band efficacy

따라서 다음 작업은 실행하지 않았다.

- LF8/LF4 bandwidth 확장
- Akoya challenge
- grouped CV

StarDist/ResNet18/UNI/Akoya를 추가 실행해도 위 architecture/range failure를 되돌릴 수 없으므로 계산을 중단했다.
다음 설계는 deep spatial residual보다 train-only paired affine을 출발점으로 삼되, analytic range constraint와
명시적 normalized-state contract를 먼저 해결해야 한다.

## 8. SLURM 기록

| Job | 역할 | 결과 |
|---:|---|---|
| 13512608 | 전체 CPU regression test | COMPLETED, 83 passed |
| 13512609 | GT450/VERSA/S60 Stage-0 CPU array | COMPLETED |
| 13512620 | LF16 GPU fast-dev smoke | COMPLETED |
| 13512690 | initial one-slide overfit | COMPLETED, idempotence fail |
| 13512801 / 13512850 | weight 5 / 10 audit | COMPLETED |
| 13512964 | static weight 6/7/8 array | COMPLETED |
| 13513268 | static weight 7.2/7.4/7.6 array | COMPLETED |
| 13513479 | warm-up → 7.4/8/10 array | COMPLETED |
| 13513646 | classical baseline CPU array | COMPLETED |
| 13513721 | LF16 30-epoch full pilot | COMPLETED, epoch 10/30 saved |
| 13513781 | 수정 후 전체 CPU regression test | COMPLETED, 83 passed |
| 13513820 / 13513839 | GPU milestone eval | drained-node capacity로 취소 |
| 13513863 | CPU milestone eval fallback | COMPLETED, 1,090 unique pairs/epoch |

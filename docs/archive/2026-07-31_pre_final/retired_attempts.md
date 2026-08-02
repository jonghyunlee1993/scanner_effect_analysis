# 종료된 실험과 코드 정리 기록

> Archive snapshot. 종료된 시도의 상세 provenance로만 유지한다.

최종 갱신: 2026-07-24

이 문서는 active repository에서 제거한 실험 코드와 생성물을 결과와 함께
참조하기 위한 기록이다. 기존 `archived/` 내용은 이번 정리에서 변경하지
않았다. 제거된 tracked source는 cleanup commit 이전 Git history에서
복구할 수 있다.

## 1. Legacy canonicalizer와 report stack

목적:

- 하나의 RGB canonical space를 생성하는 context-conditioned canonicalizer
- Reconstruction, scanner invariance와 detail preservation을 하나의 학습
  objective로 만족
- Internal 및 S60 model report 생성

판정:

- Fidelity–invariance trade-off가 반복됨
- Source HF를 보존하면 scanner signature가 남고, 억제하면 morphology가 손상
- 현재 연구 질문은 image generator 개발이 아니라 frequency-resolved
  boundary characterization으로 변경

조치:

- `src/prenorm/models/`, legacy Lightning module/loss/callback/checkpoint
- legacy `eval_report`, validation/report renderer와 external-S60 model evaluator
- 해당 unit test

를 active tree에서 제거했다. 완전한 modeling-v4 artifact는 기존
`archived/2026-07-23_modeling_v4/`에 보존되어 있다.

## 2. Exp-01 learned LF residual

목적:

- Fixed pyramid의 LF16만 학습형 residual로 바꾸고 HF coefficient를 복사
- Pair efficacy, idempotence와 range safety를 동시에 만족

시도:

- 76.6K parameter LF residual model
- Static 및 warm-up idempotence weight sweep
- Epoch 10/30 held-out evaluation
- LF8/LF4와 Akoya는 hard gate 이후 실행하지 않음

결과:

- Held-out LF MAE는 GT450 34.9%, VERSA 65.1%, S60 43.8% 감소
- Train-paired affine의 79.7–93.8% 감소보다 모두 낮음
- Residual이 scanner, image와 spatial context에 무관한 정확한 `-0.12`
  constant shift로 붕괴
- S60 clipping/detail-safety gate 실패
- Efficacy와 image-only idempotence는 correction 해와 identity 해로 갈림

판정:

- LF16 learned model no-go
- 이후 연구는 train-only analytic affine와 explicit clipping을 기준으로 진행

조치:

- Exp-01 model/module/train/overfit/eval code, training config와 launcher
- Checkpoint, overfit model과 milestone output

를 제거했다. 상세 수치와 SLURM job은
[`exp01_execution_log.md`](exp01_execution_log.md)에 남아 있다.

## 3. Exp-02 strict range-safe projection 경로

목적:

- Source HF coefficient를 정확히 유지하면서 corrected LF를 feasible range
  안으로 투영

시도:

- Image-wide scalar alpha
- Coefficient-wise local projection
- Ridge/OD/spatial affine 및 classical stain normalization

결과:

- Scalar projection은 GT450/S60에서 correction을 크게 억제
- Local projection은 scalar efficacy를 회복했으나 operational image
  normalization의 주 계약으로는 지나치게 보수적
- Strict-safe fallback이 38.1%에서 raw로 돌아가 UNI movement를 축소
- 실제 pipeline은 requested LF + source detail reconstruction + explicit
  clipping으로 변경하고 clip distortion을 보고

판정:

- Projection utility와 identity tests는 causal/range audit용으로 유지
- Stage-0/local diagnostic CLI와 launcher는 종료

상세 결과는 [`exp02_execution_log.md`](exp02_execution_log.md)의 §2–6과
§10에 남아 있다.

## 4. Slide fingerprint 기반 zero-shot affine

목적:

- Source slide의 color/sharpness fingerprint로 scanner 또는 affine을 추정
- Paired target이나 test-time scanner metadata 없는 zero-shot correction

초기 결과:

- Closed-set known-scanner lookup은 일부 scanner에서 oracle efficacy를 유지
- Scanner-LOO는 VERSA를 포함해 불안정

무효화 사유:

- Internal lattice와 S60 lattice의 동일 숫자 tuple을 같은 physical location으로
  취급한 sample identity 오류를 발견
- S60이 다른 sampling grid이므로 slide fingerprint scanner BACC가 lattice와
  scanner를 분리하지 못함

판정:

- Fingerprint BACC와 predicted-affine 결과는 core finding에서 제외
- Identity-fixed low-band BACC와 UNI 분석만 유지

조치:

- `eval_exp02_scanner_params.py`, correction visualization과 launcher 제거
- 이전 잘못된 JSON/figure는 이미
  `archived/2026-07-23_pre_identity_fix_exp02/`에 보존

## 5. Superseded pairwise visualization contracts

종료된 계약:

- Raw fallback이 포함된 strict-safe comparison
- Sharp target과 blurred transformed image를 직접 비교
- Akoya가 빠진 fixed-AT2 reference figure
- Pen-marker artifact가 포함된 첫 exemplar

대체 계약:

- 모든 operational image에 explicit clipping
- Akoya 포함 internal four-scanner rotating reference
- Source와 target에 같은 HF retention을 적용한 matched-blur comparison
- Embedding을 보지 않은 artifact-screened three-location trajectory

이전 figure는 기존 dated archive에 보존되어 있으며 active output에서는
smoke와 superseded artifact를 제거했다.

## 6. 제거한 생성물

문서화 후 active workspace에서 제거한 disposable artifact:

- `logs/*`: SLURM stdout과 임시 diagnostic script
- `outputs/exp01a_lf16_10slide/`: no-go learned LF checkpoints와 overfit sweep
- `outputs/exp02_smoke/`: superseded pairwise smoke contract
- `outputs/exp02_baseline_smoke/`: implementation-only 32/16 smoke
- `outputs/exp02_stage0/`: superseded scalar range-safe diagnostic
- Python/pytest cache

유지한 결과:

- `outputs/exp01_stage0/`, `outputs/exp01_baselines/`
- `outputs/exp02_stage1/`
- `outputs/exp02_visualization/`
- `outputs/exp02_hf_trajectory/`
- `outputs/exp02_lf_aligned_hf_trajectory/`

대용량 output과 data는 Git으로 추적하지 않으며 active analysis를 다시
실행하는 데 필요한 source와 launcher만 유지한다.

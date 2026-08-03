# RF1 improvement handoff and cleanup checkpoint

**Snapshot:** 2026-08-03 16:18 EDT  
**Workspace:** `/mnt/isilon/oldridge_lab/leej/prenorm`  
**Base Git HEAD before snapshot commit:** `36be3a5`  
**Purpose:** 다음 세션에서 RF1 analytic improvement를 이어가기 위한 실행·결과·해석·정리 기록

## 1. 현재 의사결정 상태

- PanNormal E0--E7 core와 원래 RF1은 이미 result-locked 상태이다.
- learned generator인 RF2는 별도 논문으로 분리한다.
- 현재 논문에서는 RF1 내부의 analytic improvement 네 가지를 image-only pilot으로
  검토한다.
- 새 variant의 PFM embedding, scanner-radius, tissue endpoint는 아직 열지 않았다.
- 네 branch를 하나로 합치거나 최종 RF1을 교체하는 결정도 아직 하지 않았다.
- 다음 단계는 실행 중인 image-only job을 완결하고, 결과를 비교한 뒤 단 하나의
  executable candidate contract를 먼저 동결하는 것이다.

선택·실행 경계는
[`e5_rf1_improvement_pilot_contract.md`](e5_rf1_improvement_pilot_contract.md)에 기록했다.

## 2. 이미 잠긴 연구 기반

### PanNormal core

E0--E7은 2026-08-03에 완료·잠금되었다.

- common native-AA grid: 109 slides × 6 scanners × 100 locations = 65,400 rows
- four PFM: ResNet50, UNI v1, CONCH v1, Virchow2
- raw PFM population: 436 shards, 261,600 embeddings
- E4 controls: 2,354,400 embeddings
- E5 comparators: 1,308,000 embeddings
- eight result locks, six main figures, 130 audited artifacts
- top-level lock: `outputs/pannormal_core_results_lock/summary.json`
- lock gate: `core_result_lock_pass=true`

상세 근거는 [`storyline_completion_audit.md`](storyline_completion_audit.md)에 있다.

### Locked RF1

현재 잠긴 RF1은 다음 순서다.

1. outer-training fold의 scanner별 CIE Lab 평균·표준편차로 source를 AT2에 Reinhard
   정규화한다.
2. post-Reinhard mean OD의 72-bin radial power와 raw AT2 radial power에서
   `sqrt(P_AT2/P_source)` gain을 구한다.
3. 0.03--0.10 cycles/µm anchor의 geometric mean을 1로 맞추고 log-gain을 smoothing한다.
4. 0.10 cycles/µm 미만은 identity로 두고 panel-wide gain cap 1.25를 적용한다.
5. Fourier correction으로 얻은 scalar mean-OD residual을 세 OD channel에 동일하게
   더해 chromatic OD difference를 보존한다.
6. residual을 RGB8-supporting OD interval로 정확히 projection한다.

Locked RF1 결과:

| PFM | Reinhard RR | RF1 RR | RF1 incremental RR |
|---|---:|---:|---:|
| ResNet50 | 27.56% | 29.71% | +2.15 pp |
| UNI v1 | 6.04% | 6.66% | +0.62 pp |
| CONCH v1 | 11.90% | 14.65% | +2.75 pp |
| Virchow2 | 3.68% | 4.66% | +0.98 pp |

네 PFM 모두 fidelity-safe이고 Reinhard보다 radius가 유의하게 감소했다. 다만 image
spectrum은 15개 FOV×scanner cell 중 12개만 개선했고, S360 세 cell은 모두 약 29.6%
악화했다. 이 S360 counterexample이 현재 improvement pilot의 직접적인 동기다.

## 3. 완료된 시각화

Source scanner에서 paired AT2 target으로의 correction과 fixed joint UMAP을 생성했다.

- `outputs/e5_rf1_visual_comparison/source_to_at2_method_comparison.png`
- `outputs/e5_rf1_visual_comparison/joint_umap_all_pfms.png`
- `outputs/e5_rf1_visual_comparison/umap_scanner_centroid_trajectories.png`
- 설명: [`e5_rf1_visual_comparison.md`](e5_rf1_visual_comparison.md)

비교 영상은 raw, Reinhard, paired OD affine, frequency-only, Macenko, RF1을 포함한다.
RF1과 Reinhard의 육안 차이가 작은 것은 설계상 정상이다. RF1은 저주파 색을 바꾸지
않고 cap 1.25 이내의 shared-OD 고주파 residual만 더하기 때문이다. UMAP은 각 PFM
내에서 여섯 condition을 한 번에 fit했으며 정성적 시각화다. 정량 판정은 원래 feature
space의 locked RR와 fidelity gate를 사용한다.

실행 job `16088758`과 `16088759`는 모두 `COMPLETED 0:0`; 당시 전체 test는
`137 passed`였다.

## 4. RF1 improvement branch별 기록

### Branch 1 — scanner-specific no-harm strength

#### 검증하려는 것

하나의 panel-wide cap 대신 scanner/FOV별 correction strength를 image-only nested
selection하면 S360처럼 이미 Reinhard가 충분히 가까운 scanner에는 identity를 선택하고,
headroom이 있는 scanner에는 더 강한 보정을 줄 수 있는지 검증한다.

#### 구체적으로 한 것

- identity cap 1.0과 기존 13개 cap 후보를 사용했다.
- exploratory meta-cross-fit에서는 candidate가 모든 selection fold에서 spectrum
  non-worse이고 frozen gamut gate를 통과해야 eligible하게 했다.
- strict nested 구현에서는 outer fold `h`와 inner validation fold `j`를 모두 training에서
  제외한다. 한 unordered pair를 한 번 fit해 두 방향 inner validation을 생성하므로 30개
  GPU task가 60개 directional audit을 만든다.
- strict selector는 contract의 one-standard-error rule을 따른다.
  - fold별 `delta = candidate RMSE - identity RMSE`
  - 네 inner fold 모두 non-worse + gamut-safe인 candidate만 eligible
  - best candidate delta의 `SE = sd(delta, ddof=1)/sqrt(4)`
  - `mean delta <= best mean delta + SE`인 가장 작은 cap 선택

#### 현재 결과와 해석

저비용 exploratory 결과는 15/15 cell이 spectrum-nonworse/gamut-safe였다.

- S360: 15/15 fold에서 exact identity
- GT450: cap 2.0
- VERSA: 주로 1.04--1.05
- AKOYA: cap 1.25
- S60: 주로 1.5
- 평균 cell spectrum reduction: 기존 cap 1.25의 7.21% → 18.51%

이 결과는 방향성이 강하지만 `strict_nested_training_exclusion=false`인 탐색 결과라 최종
수치로 사용할 수 없다. strict rerender가 이를 확인해야 한다.

#### 실행 중 / 해야 할 것

- live array: `16125877`
- output: `outputs/rf1_improvement_pilot/noharm_nested/inner/`
- snapshot 당시 60개 directional summary 중 19개 생성
- 완료 후 `scripts/rf1_noharm_nested_aggregate.sbatch`를 `afterok:16125877`로 제출
- strict result의 75/75 outer fold×scanner×FOV non-worse, gamut pass, S360 identity 빈도를
  확인
- strict gate가 통과하기 전에는 PFM re-encoding 금지

관련 코드:

- `src/analyze_rf1_noharm_gain.py`
- `src/build_rf1_noharm_nested_cell.py`
- `src/analyze_rf1_noharm_nested.py`
- `tests/test_rf1_noharm_gain.py`

### Branch 2 — paired robust spectrum estimator

#### 검증하려는 것

현재 pooled source/target power ratio가 동일 위치의 paired acquisition 정보를 충분히
사용하지 못한다는 가설을 검증한다. anatomy energy를 patch-pair 안에서 상쇄하고,
scanner response의 근거가 약한 frequency를 identity 쪽으로 줄이는 것이 목적이다.

#### 구체적으로 한 것

paired patch `i`에 대해

`ell_i(f) = 0.5 * [log P_AT2,i(f) - log P_source,i(f)]`

를 계산하고 각 patch의 0.03--0.10 cycles/µm anchor mean을 제거했다. 100 patches의
median으로 slide curve를 만든 뒤, training slides에 coordinate-wise Huber location
(`c=1.345`)을 적용했다. reliability는

`r = max(0, |theta| - 1.96*robust_SE) / (|theta| + 1e-12)`

이며 `robust_SE = 1.4826*MAD/sqrt(n_slides)`이다. current pooled, paired robust,
paired reliable 세 방법을 동일 cap 1.25와 기존 gamut projection으로 held-out 비교한다.

#### 현재 부분 결과와 해석

FOV224 fold 0--3, 88/109 slides의 부분 결과:

| Scanner | Current pooled | Paired robust | Reliable shrinkage |
|---|---:|---:|---:|
| GT450 | −22.80% | −23.05% | −22.64% |
| VERSA | −1.90% | −20.98% | −15.45% |
| AKOYA | −21.47% | −21.07% | −21.15% |
| S60 | −21.27% | −24.58% | −20.78% |
| S360 | +29.95% | +13.52% | +10.72% |

음수는 Reinhard 대비 spectrum RMSE 감소다. paired estimator는 VERSA에서 크게
개선하고 S360 악화를 줄였지만, 현재 부분 결과에서는 S360 부호를 음수로 바꾸지 못했다.
reliability는 uncertainty shrinkage이지 no-harm selector가 아니므로 이 결과는 예상 가능한
한계다. 완료 전 efficacy 결론을 내리면 안 된다.

#### 실행 중 / 해야 할 것

- task 0: job `16125442_0`, 완료
- live array tasks 1--14: `16125464`
- snapshot 당시 총 15 fold 중 5개 summary 생성
- 완료 후 `scripts/rf1_paired_robust_aggregate.sbatch` 제출
- full 15-cell에서 S360 sign, current 대비 우위 cell 수, reliability/gamut을 확인
- output figure `heldout_spectrum_comparison.png` 검수

관련 코드:

- `src/analyze_rf1_paired_robust.py`
- `tests/test_rf1_paired_robust.py`

### Branch 3 — local multiscale Laplacian correction

#### 검증하려는 것

72-bin global radial FFT보다 적은 수의 localized spatial-frequency band가 boundary와
texture를 더 안정적으로 보정하고 S360 reversal을 해결하는지 검증한다.

#### 구체적으로 한 것

- post-Reinhard mean OD를 Gaussian/Laplacian pyramid로 분해
- sigma 1, 2, 4 px의 세 additive band 사용
- sigma-4 residual low-pass base는 그대로 보존
- scanner/fold별 band gain을 target/source band-energy ratio로 fit하고 cap 1.25 적용
- shared scalar OD residual과 기존 exact gamut projection 유지
- 현재 estimator를 유지해 transform representation만 radial에서 multiscale로 바꾼
  isolation pilot

#### 완료된 FOV256 결과와 해석

109 slides, 54,500 paired source patches에서 다섯 scanner 모두 Reinhard보다 spectrum이
개선했다.

| Scanner | Reinhard RMSE | Radial RF1 | Multiscale | Multiscale change |
|---|---:|---:|---:|---:|
| GT450 | 1.5209 | 1.1805 | 1.1074 | −27.19% |
| VERSA | 0.3169 | 0.3124 | 0.1707 | −46.12% |
| AKOYA | 1.2581 | 0.9940 | 1.0007 | −20.46% |
| S60 | 1.1350 | 0.9009 | 0.7079 | −37.63% |
| S360 | 0.2489 | 0.3224 | 0.1711 | −31.26% |

scanner 평균 spectrum improvement는 radial RF1 7.17%에서 multiscale 32.53%로
커졌다. paired RGB/mean-OD/gradient MAE도 Reinhard 대비 각각 평균 3.11%, 6.36%,
9.14% 감소했다. final clamp는 0이고 모든 기존 gamut threshold를 통과했다.

현재 네 branch 중 가장 유망한 image-domain signal이다. 특히 S360 reversal을 별도
identity gate 없이도 해결했다. 다만 native pyramid training energy를 직접 누적하지 않고
locked 72-bin radial statistics에서 band energy를 근사한 pilot이라는 한계가 있다.

시각화:

- `outputs/rf1_improvement_pilot/multiscale/visual/paired_patch_multiscale_comparison.png`
- job `16126806`, `COMPLETED 0:0`, `visual_gate_pass=true`

#### 실행 중 / 해야 할 것

- FOV224+512 live array: `16126095`
- all-15-cell dependent aggregate: `16126118`
- 완료 후 45 spectrum rows, 45 paired rows, 15 gamut rows, 75 gain rows와 총 163,500
  paired source patches를 검증
- 모든 FOV에서 S360 sign과 gamut gate가 재현되는지 확인
- confirmatory version에서는 native pyramid band energy를 training patch에서 직접 누적

관련 코드:

- `src/analyze_rf1_multiscale.py`
- `tests/test_rf1_multiscale.py`

### Branch 4 — gamut-safe Reinhard

#### 검증하려는 것

현재 Reinhard의 independent RGB clipping을 source-to-proposal correction vector를 보존하는
gamut projection으로 바꾸면 색 왜곡과 구조 error를 줄일 수 있는지 검증한다.

#### 구체적으로 한 것

Reinhard Lab proposal을 RGB로 변환한 값이 `p`, 원래 valid source RGB가 `x`일 때,

`x_safe = x + beta * (p - x)`

로 두고 세 channel 모두가 `[0,1]`에 머무는 최대 `beta in [0,1]`를 픽셀별 closed-form으로
구했다. independent channel clipping과 달리 세 channel correction의 방향을 보존한다.

5-fold held-out 전체 109 slides × 100 locations × 5 source scanners × 3 FOV = 163,500
patch에서 paired RGB, mean-OD gradient, radial spectrum과 gamut을 비교했다.

#### 결과와 해석

- exact final clamp error: `1.98e-10` 이하
- paired RGB point improvement: 12/15 cells; CI-level improvement: 10/15
- paired mean-OD gradient improvement: 15/15
- spectrum improvement: 12/15
- AKOYA의 세 FOV에서 paired RGB와 spectrum이 소폭 악화

대표 FOV256 spectrum 변화:

- GT450: 1.5209 → 1.3966
- VERSA: 0.3169 → 0.2982
- AKOYA: 1.2581 → 1.2960
- S60: 1.1350 → 0.9324
- S360: 0.2489 → 0.2226

따라서 hard clipping 제거는 구조 gradient에는 일관되게 유리하지만 모든 scanner의 색·
spectrum을 동시에 개선하지 않는다. 현재 source-ray candidate를 그대로 RF1에 넣는 것은
권장하지 않는다. scanner-specific application 또는 hue/chroma-preserving alternative를
새 image-only contract에서 비교해야 한다.

시각화:

- `outputs/rf1_improvement_pilot/gamut_reinhard/gamut_reinhard_visual_audit.png`
- current Reinhard와 source-ray의 차이는 작지만 ×50 residual과 intervention mask에서
  포화 경계·진한 핵·밝은 background 경계에 집중됨을 확인

관련 코드:

- `src/rf1_gamut_reinhard.py`
- `src/analyze_rf1_gamut_reinhard.py`
- `src/aggregate_rf1_gamut_reinhard.py`
- `src/render_rf1_gamut_reinhard_audit.py`
- `tests/test_rf1_gamut_reinhard.py`

## 5. 현재 종합 해석

| Branch | 현재 신호 | 주요 장점 | 현재 제한 | 잠정 판단 |
|---|---|---|---|---|
| 1. No-harm | 강함, strict pending | S360 identity, scanner별 headroom | exploratory result에 leakage 가능 | strict 결과 필요 |
| 2. Paired robust | 혼합, full panel pending | paired anatomy cancellation, VERSA 개선 | S360 부호 미해결 가능 | 단독 최종안은 불확실 |
| 3. Multiscale | 가장 강함, FOV 확장 pending | S360 포함 5/5 개선, local transform | band energy 근사, FOV256만 완료 | 현재 1순위 |
| 4. Gamut-safe Reinhard | 구조에는 일관됨 | hard clipping 제거, gradient 개선 | AKOYA RGB/spectrum 악화 | 그대로 포함하지 않음 |

현재 가장 합리적인 예상 최종 후보는 multiscale correction에 strict no-harm selector를
결합하는 것이다. paired robust estimator는 full result가 multiscale에 추가 이득을 줄
근거를 보일 때만 결합한다. gamut-safe Reinhard는 AKOYA no-harm 해결 전까지 독립적인
방법 개선 근거로만 남긴다. 이 조합은 아직 결정 사항이 아니며, 새 PFM access 전에
사용자가 확인하고 executable contract를 동결해야 한다.

## 6. 다음 세션의 정확한 실행 순서

### A. live jobs 확인

먼저 controller를 확인한다.

```bash
getent hosts respublica-01 && scontrol ping
squeue -j 16125877,16125464,16126095,16126118 -o '%.18i %.24j %.8T %.10M %R'
```

실행 중인 job은 중복 제출하지 않는다.

### B. Branch 1 strict aggregate

`16125877` 완료 후 60 NPZ + 60 summary와 모든 exit code를 확인하고:

```bash
sbatch --dependency=afterok:16125877 \
  --export=ALL,JOB_CONDA_PREFIX=/home/leej70/miniconda3/envs/cpath,JOB_WORKDIR="$PWD" \
  scripts/rf1_noharm_nested_aggregate.sbatch
```

결과의 75/75 outer non-worse와 one-SE 선택 안정성을 검토한다.

### C. Branch 2 aggregate

`16125464` 완료 후 15 fold output을 확인하고
`scripts/rf1_paired_robust_aggregate.sbatch`를 CPU SLURM으로 제출한다. full-panel S360
부호와 paired reliable의 current 대비 우위를 확인한다.

### D. Branch 3 all-FOV aggregate

`16126095`와 dependent `16126118`의 완료·exit code를 확인한다.
`outputs/rf1_improvement_pilot/multiscale/panel_summary.json`의 population/gamut gate와
FOV224/512 재현성을 확인한다.

### E. 단일 후보 결정 및 동결

image-only 결과를 비교해 다음 중 하나를 고른다.

1. multiscale only;
2. multiscale + strict scanner-specific no-harm;
3. 위 방법 + paired robust band estimator.

선택한 뒤 native band-energy accumulation, exact coefficients, nested selection,
identity/gamut gates, feature population과 endpoint rule을 새 contract로 동결한다. 그 후에만
4-PFM encoding과 기존 invariance/content/collapse evaluation을 한 번 실행한다.

### F. 검증

- 새/변경 코드 `py_compile`
- focused tests와 full tests를 CPU SLURM으로 실행
- `git diff --check`
- image artifacts 직접 시각 검수
- 새 PFM endpoint 접근 여부를 summary에 명시

## 7. 실행 히스토리

| 시점 | 작업 | 상태/핵심 결과 |
|---|---|---|
| 2026-08-03 오전 | PanNormal E0--E7 finalization | core lock pass, 8 locks/6 figures/130 artifacts |
| 2026-08-03 | E5-RF1 analytic extension | cap 1.25, four-PFM safe+improved, S360 spectrum counterexample |
| 2026-08-03 14:xx | source-to-AT2 + joint UMAP | jobs 16088758/16088759 complete; 15,696 UMAP rows |
| 2026-08-03 15:xx | RF1 improvement pilot contract | four image-only branches, PFM sealed |
| 2026-08-03 15:xx | no-harm exploratory | 15/15 non-worse, S360 identity, strict rerender 승인 |
| 2026-08-03 15:xx | gamut-safe Reinhard | 163,500 patch 완료; gradient 15/15, AKOYA trade-off |
| 2026-08-03 15:xx | multiscale FOV256 | five scanners 개선, mean spectrum −32.53% |
| 2026-08-03 16:18 | pause snapshot | three GPU pipelines remain live; no new PFM access |
| 2026-08-03 16:32 | selective cleanup | 33 smoke/trial/preview/partial directories와 1,819 old logs 삭제; active/final artifacts 유지 |
| 2026-08-03 16:39 | documentation cleanup | archive/manuscript/superseded E0 문서와 orphan helper 제거; canonical docs 26개 유지 |

주요 SLURM provenance:

- `16088758`: source-to-target image panel, complete
- `16088759`: joint UMAP, complete
- `16125439`: gamut-safe Reinhard population, complete
- `16125531`: gamut-safe aggregate, complete
- `16125629`: gamut-safe visual, complete
- `16125485`: multiscale FOV256 folds, complete
- `16125532`: multiscale FOV256 aggregate, complete
- `16126806`: multiscale visual, complete
- `16125877`: strict no-harm nested, live at snapshot
- `16125464`: paired robust remaining folds, live at snapshot
- `16126095`: multiscale FOV224/512, pending/live pipeline
- `16126118`: multiscale panel aggregate, dependent

## 8. Git snapshot과 cleanup 주의사항

### Repository permission boundary

원래 repository에서 snapshot commit을 시도했지만 `.git/index.lock`을 생성할 수 없는
read-only filesystem 정책 때문에 실패했다. 따라서 원래 `main` branch는 이 문서 작성
시점에도 base HEAD `36be3a5`를 가리킨다.

대신 full Git history를 임시 writable clone으로 복제하고 현재 working tree의 모든
non-ignored 파일을 그 clone에서 commit한 뒤 다음 bundle로 내보내는 fallback recovery
snapshot을 사용한다.

- `checkpoints/prenorm_precleanup_20260803.bundle`

이 bundle은 clone/fetch 가능한 실제 Git commit과 이전 history를 포함하지만, 원래
repository branch를 직접 이동시키지는 않는다. `.git` write 권한이 복구되면 bundle의
snapshot commit을 원래 repository로 가져와 branch에 적용해야 한다. 아래 generated
outputs/logs는 bundle에도 포함되지 않는다.

`outputs/`와 `logs/`는 `.gitignore` 대상이다. snapshot 시점 크기는:

- `outputs/`: 약 156 GB
- `logs/`: 약 137 MB, 약 2,818 files

따라서 Git commit은 코드, tests, SLURM scripts, contracts, narrative와 이 handoff 문서를
복구하지만 generated image/features/results/log 자체를 복구하지 않는다. `outputs/`를
삭제하면 result-lock JSON의 hash가 남더라도 실제 artifact는 Git에서 되살릴 수 없고,
원자료에서 재실행해야 한다.

### Cleanup inventory

삭제 범위를 이름만으로 추측하지 않도록 snapshot 시점의 결과를 다음처럼 분류한다.

| 분류 | 대표 경로 | 크기/상태 | 삭제 시 의미 |
|---|---|---:|---|
| 현재 실행 중 | `outputs/rf1_improvement_pilot/` | 148 MB, 계속 기록 중 | jobs 16125877/16125464/16126095/16126118 종료 전 삭제 금지 |
| canonical locks | `outputs/*_results_lock/`, `outputs/pannormal_core_results_lock/` | 각 80--152 KB | 논문 결과의 audit entry point 소실 |
| final audit/figures | `outputs/e0_e3_main_figures/`, `outputs/e5_rf1_visual_comparison/` 등 | 수 MB--40 MB | 그림과 사람이 검수한 artifact 소실 |
| 대형 재생성 가능 intermediate | `e0_valis_from_scratch_v2` 60 GB, `e0_native_aa_grid` 51 GB, `e0_valis_from_scratch_affine` 12 GB, `e4_control_features` 9 GB | 합계의 대부분 | 원 WSI와 코드에서 재실행 필요; 재계산 비용이 큼 |
| feature populations | `e0_pfm_contract` 4.9 GB, E5/E6 feature·harmonization 경로 약 1.1--3.1 GB씩 | 약 20 GB 이상 | PFM 추출과 endpoint 분석 재실행 필요 |
| 명시적 disposable candidates | 이름에 `_smoke`, `_trial_`, `_preview`, `_partial`이 있는 경로 | 대부분 KB--수백 MB | 최종 lock이 아닌 개발·부분 실행 결과 |
| historical tracked docs | `docs/archive/2026-07-31_pre_final/` | 304 KB, 7 files | active tree에서 삭제; snapshot bundle에서 복구 가능 |
| scheduler logs | `logs/` | 137 MB, 2,818 files | 실행 provenance와 failure diagnosis 소실 |

즉, 공간 회수 효과는 smoke/trial보다 60 GB VALIS v2와 51 GB native-AA grid에서
대부분 나온다. 반면 이 둘은 재계산 비용도 가장 크므로 단순히 "legacy"라는 이름으로
분류하지 않는다. 실제 cleanup 전에는 live job 종료 확인, canonical lock/figure 보존 여부,
대형 intermediate 재생성 의사를 함께 확정해야 한다.

검토한 cleanup 범위는 다음 세 가지였다.

1. **권장 cleanup:** smoke/trial/preview/partial output과 오래된 logs만 삭제하고,
   `*_results_lock`, final geometry, main figures, RF1 pilot summary/figures는 유지한다.
2. **Generated-result purge:** `outputs/`와 `logs/` 전체 삭제. 약 156 GB를 회수하지만 모든
   figure, feature population, lock artifact와 현재 pilot output을 재실행해야 한다.
3. **Full legacy purge:** 2번에 더해 이전 experiment source/scripts/docs도 snapshot commit
   이후 working tree에서 삭제한다. 복구는 Git commit에서만 가능하다.

### Executed selective cleanup

사용자는 2026-08-03에 1번 권장 cleanup을 선택했다. 16:32 EDT에 다음의 명시적 기준만
적용했다.

- `outputs/` 바로 아래 디렉터리 중 basename에 `smoke`, `trial`, `preview`, `partial`이
  포함된 33개 삭제: 삭제 전 합계 약 200 MB
- `logs/` 바로 아래 파일 중 수정 시각이 `2026-08-03 00:00:00` 이전인 1,819개 삭제:
  삭제 전 할당량 합계 약 112 MB
- 삭제 후 `logs/`: 약 25 MB, 1,007 files

삭제 후 같은 이름 조건에 맞는 top-level output directory가 0개임을 확인했다. 다음은
명시적으로 보존했다.

- live `outputs/rf1_improvement_pilot/`와 2026-08-03 로그
- 모든 result lock, final figure/audit, feature population
- `e0_valis_from_scratch_v2`, `e0_native_aa_grid` 등 대형 intermediate
- canonical source/scripts/tests와 active documentation
- pre-cleanup recovery bundle

cleanup 중 SLURM job을 취소하지 않았고, active RF1 output path는 삭제 조건과 일치하지
않았다. 따라서 `16125877`, `16125464`, `16126095`, `16126118`은 그대로 계속된다.

cleanup 실행 기록까지 포함한 후속 Git bundle은
`checkpoints/prenorm_postcleanup_20260803.bundle`에 보존한다.

### Executed documentation cleanup

사용자 승인에 따라 archive와 manuscript를 포함한 superseded documentation을 active
tree에서 제거했다. 삭제 대상은 archive 7개, manuscript 2개, 최종 cohort contract로
대체된 E0 sentinel/cohort/alias/pilot 문서 4개였다. 삭제한 manuscript만을 생성하던 Python
및 SLURM helper 2개도 함께 제거했다. 남은 `docs/`는 storyline, protocol, handoff,
frozen contracts와 locked-results 문서로만 구성하며 [`README.md`](README.md)를 canonical
index로 사용한다. 삭제된 15개 파일은 post-cleanup Git bundle commit `06dcf72`에서
복구할 수 있다. 정리 후 `docs/`는 26 files, 약 896 KB이며 모든 Markdown local link가
존재함을 확인했다.

documentation cleanup 이후 canonical tree는
`checkpoints/prenorm_post_doc_cleanup_20260803.bundle`에 별도 보존한다.

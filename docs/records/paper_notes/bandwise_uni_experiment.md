# UNI의 주파수 band별 민감도: 탐색적 추가 분석

## 질문과 설계

동일한 크기의 영상 변화에서 UNI 표현이 high band에 더 크게 반응하는지, 그리고 그 반응이 실제 paired scanner target으로의 이동인지 분리해 평가했다. PanNormal의 103개 물리 슬라이드에서 기존 UNI 평가 위치 중 슬라이드당 고정된 3개를 사용했다. AT2 영상을 기존 5-fold training slide로 추정한 Reinhard 파라미터로 다섯 target scanner의 색 공간에 맞췄다. 이 영상의 평균 optical density에 low–mid, mid, high Fourier band를 한 번에 하나씩 증감했다. Band 경계는 기존 원고의 0.10–0.30, 0.30–0.60, 0.60–0.90 cycles/µm와 같다.

각 영상에서 세 band component RMS의 최솟값을 구하고, 그 값의 25% 및 50%를 목표 OD RMS로 정했다. 모든 band와 증감 방향에서 최종 RGB 영상의 *실제* OD RMS가 해당 목표값과 같도록 계수를 조정했다. 이 목표는 target 영상이나 UNI embedding을 보지 않고 정했다. 기존 training-fold frequency gain의 부호를 `predicted target direction`으로 사용하고 반대 부호도 평가했다. 각 위치·scanner·band·dose에서 양·음 두 조작을 수행해 총 18,540개의 조작 영상을 UNI-v1으로 평가했다. 신뢰구간은 물리 슬라이드 103개를 단위로 2,000회 bootstrap했다.

## 핵심 결과

| 목표 dose | Low–mid UNI 이동 | Mid UNI 이동 | High UNI 이동 | High − low–mid (95% CI) | High − mid (95% CI) |
|---|---:|---:|---:|---:|---:|
| 0.25 | 0.004321 | 0.004294 | 0.005635 | 0.001314 (0.000926–0.001690) | 0.001342 (0.001125–0.001579) |
| 0.50 | 0.017049 | 0.016941 | 0.021022 | 0.003973 (0.002520–0.005401) | 0.004081 (0.003345–0.004872) |

UNI 이동은 색 보정된 원본과 조작 영상 사이의 cosine distance다. 각 band의 영상 변화량은 같은 위치에서 짝지어 비교했다. 실제 dose의 상대 오차 중앙값은 모든 조건에서 0.001% 미만이었다. Dose 0.25에서 high band의 이동은 low–mid보다 약 30%, mid보다 약 31% 컸다. 다섯 scanner 각각에서도 두 차이가 양수였다. 다만 dose 0.50의 *attenuation*만 따로 보면 high − low–mid의 신뢰구간은 0을 포함했다. 따라서 모든 강도와 방향에서 high band가 우세하다는 주장은 맞지 않는다.

| Target scanner | High band의 predicted target 방향 UNI gain, dose 0.25 (95% CI) | Target 방향 − 반대 방향 gain (95% CI) |
|---|---:|---:|
| VERSA | −0.002009 (−0.002971–−0.001029) | +0.002201 (+0.000203–+0.004152) |
| AKOYA | +0.000459 (−0.000623–+0.001534) | +0.014119 (+0.011218–+0.017170) |
| GT450 | −0.000105 (−0.001396–+0.001102) | +0.008390 (+0.006124–+0.010607) |
| S360 | +0.000085 (−0.000822–+0.000969) | +0.006730 (+0.005040–+0.008509) |
| S60 | −0.001785 (−0.002750–−0.000872) | +0.003034 (+0.001178–+0.004873) |

UNI target gain은 색 보정된 원본의 paired target distance에서 조작 후 distance를 뺀 값이다. High band의 target 방향은 반대 방향보다 다섯 scanner 모두에서 유리했으나, 원본 색 보정 상태보다 실제 target에 가까워진다는 근거는 없다. 다섯 scanner 평균 high band gain은 dose 0.25에서 −0.000671 (95% CI −0.001145–−0.000228), dose 0.50에서 −0.008167 (−0.009438–−0.006974)이었다. AKOYA의 high band 단독 gain도 dose 0.25에서는 0을 포함한다. 원고의 full radial frequency matching에서 보인 AKOYA 개선을 high band 단독 효과로 돌릴 수 없다.

## 해석 범위

이 자료는 **색 보정된 PanNormal 입력에서 UNI-v1의 embedding이 동일 OD RMS의 low–mid/mid band 조작보다 high band 조작에 평균적으로 더 크게 반응한다**는 주장에 근거를 준다. High band amplitude가 scanner batch effect의 주된 원인이라는 주장이나 high band 단독 교정이 paired target alignment를 개선한다는 주장은 뒷받침하지 않는다. 다른 PFM으로 일반화하는 실험도 아직 없다. 주파수 band의 resampling·Nyquist 검증은 이번 분석 범위에서 제외했다.

## 산출물과 실행 확인

- 재현 코드: `scripts/frequency/bandwise_uni_experiment.py`, `scripts/frequency/bandwise_uni_experiment.sbatch`, `scripts/frequency/plot_bandwise_uni.py`
- 원자료 결과: `outputs/bandwise_uni_experiment/fold_0.csv.gz`부터 `fold_4.csv.gz`
- 요약: `outputs/bandwise_uni_experiment/summary.csv`, `sensitivity_contrasts.csv`, `dose_audit.csv`
- 그림: `outputs/bandwise_uni_experiment/bandwise_uni.png`, `00_manuscript/figures/fig_uni_band_sensitivity.pdf`
- 올바른 방향으로 실행한 Slurm array job `23559093`의 다섯 task가 모두 exit code 0으로 완료했다. 초기 job `23559032`는 파라미터 방향을 잘못 읽은 것을 발견해 취소했으며, 최종 분석에서는 사용하지 않았다. 최종 결과는 103개 슬라이드·18,540개 조작 행을 확인했고, 기존 저장 UNI baseline/target embedding과의 일치 검사를 각 슬라이드에서 통과했다.

# 스캐너 스펙트럼은 병리 foundation model의 불변성과 생물학적 충실도 사이의 경계를 드러낸다

영문 제목 후보: Scanner spectra reveal content-dependent limits of image-space harmonization for pathology foundation models

투고 목표: Medical Image Analysis

원고 상태: 한글 working draft v0.1 (2026-07-31)

저자: [저자명 및 순서 확정 필요]

소속: [소속 및 교신저자 정보 입력]

> 내부 작업 초안. 109-slide scanner-spectrum 및 background 분석은 완료 결과이며, multi-PFM population experiment, biological-fidelity endpoint와 PLISM external validation은 확정 프로토콜 또는 자리표시자다. 미완료 결과를 완료된 결과처럼 인용하지 않는다.

<!-- PAGEBREAK -->

## Highlights

- 동일한 109개 물리 슬라이드를 6개 스캐너로 취득하여 scanner effect를 biological variation과 분리했다.
- 스캐너 차이는 전역적인 style이 아니라 주파수와 조직 내용에 의존하는 scanner × content response였다.
- 저주파 외관 교정은 강력했지만 UNI representation alignment를 보장하지 않았다.
- scanner separability 감소만으로는 정보 보존과 representation collapse를 구별할 수 없었다.
- scanner invariance와 biological fidelity를 함께 평가하는 frequency-resolved framework를 제시한다.

## 초록

디지털 병리 영상의 스캐너 차이는 색, 광학적 해상도, 초점, 샤프닝 및 압축을 동시에 변화시키며, pathology foundation model(PFM)의 표현 안정성을 저해할 수 있다. 기존 연구는 여러 PFM이 스캐너 정체성을 인코딩한다는 사실을 보여주었지만, 관측된 shift가 교정 가능한 외관 차이인지, 스캐너가 보존하거나 손실한 구조 정보인지, 또는 조직 내용과의 상호작용인지는 충분히 분리하지 못했다. 본 연구는 scanner invariance와 biological fidelity를 동시에 평가하는 주파수 분해 framework를 제안한다.

동일한 109개 물리 슬라이드(37개 조직 유형)를 6개 whole-slide scanner로 취득하고, slide–scanner당 동일 조직 위치 100개를 분석하였다. AT2를 상대적 기준으로 두고 optical-density 영상의 방사형 power spectrum에서 effective relative transfer를 계산하였다. 패치 100개를 서로 겹치지 않는 20-patch replicate 5개로 분할하여 scanner main effect, scanner × tissue, scanner × slide 및 sampling variation을 분해하였다. 추가로 scanner-pair 저주파 affine correction, 고주파 attenuation·boost·oracle intervention, frozen UNI embedding 및 65,400개 white-space patch의 background 특성화를 평가하였다.

고주파 대역(0.60–0.90 cycles/µm)의 AT2 대비 cohort transfer는 GT450 1.595, VERSA 1.011, AKOYA 0.297, S60 0.994, S360 0.784였다. 균형화된 기술적 분해에서 scanner main effect는 고주파 변이의 85.5%를 차지했으나, 이 비율은 장비 구성에 민감했으며 scanner × tissue 2.5%, scanner × slide 11.2%, sampling 0.8%가 남았다. 15개 scanner–band 조합 중 8개에서 tissue-level scanner slope variation이 다중검정 보정 후 유의하였다. Mechanistic pilot에서 저주파 affine은 MAE를 0.2531에서 0.0297로 낮췄지만, UNI scanner balanced accuracy는 사실상 변하지 않았다. 완전한 고주파 제거는 scanner centroid를 수축시켰으나 self-cosine 0.065와 content-margin collapse를 동반했다. 반면 registered same-location HF mean 25%는 self-cosine 0.957과 retrieval 1.0을 유지하면서 scanner radius를 0.095에서 0.069로 낮추었다. White-space background feature는 15개 모델 모두에서 BH 보정 후 유의한 증분 설명력을 제공하지 않았다.

이 결과는 스캐너 효과가 분리 가능한 하나의 전역 style layer가 아니라 저주파 외관 차이, 고주파 정보 전달 및 조직·슬라이드 내용의 상호작용임을 보여준다. 따라서 appearance alignment 또는 scanner 분류 정확도의 감소만으로 PFM invariance를 주장할 수 없으며, 모든 normalization은 paired content preservation과 biological fidelity를 함께 입증해야 한다. [최종 multi-PFM 및 PLISM 결과 입력 후 초록 수치와 결론을 갱신한다.]

키워드: digital pathology; whole-slide scanner; pathology foundation model; domain shift; spectral analysis; harmonization; biological fidelity

## 1. 서론

Whole-slide imaging은 병리 슬라이드를 고해상도 디지털 영상으로 변환하여 정량 분석, 원격 판독 및 인공지능 기반 의사결정을 가능하게 한다. 그러나 디지털화 과정은 중립적인 복사가 아니다. 스캐너의 광원과 센서, 광학계, 초점 알고리즘, sampling pitch, sharpening 및 압축 pipeline은 동일한 물리 조직에서 서로 다른 색과 texture를 생성한다. 이러한 차이는 기관 간 domain shift의 주요 원인이며, 사람의 시각에는 미미해 보이는 차이도 대규모 표현 모델에는 안정적인 장비 지문으로 남을 수 있다 [1–3].

최근 pathology foundation model(PFM)은 대규모 사전학습을 통해 다양한 병리 task에 재사용 가능한 representation을 제공한다 [4–6]. 그러나 모델 규모와 학습 데이터의 증가가 acquisition invariance를 자동으로 보장하지는 않는다. Paired multi-scanner 연구들은 여러 최신 PFM의 embedding이 scanner에 따라 분리되고, downstream prediction의 calibration 또는 agreement가 달라질 수 있음을 보고하였다 [7–10]. 이 연구들은 PFM scanner sensitivity의 존재를 명확히 했지만, 왜 scanner shift가 발생하는지와 어떤 교정이 실제 정보 보존인지에 대해서는 여전히 중요한 공백이 남아 있다.

병리 영상의 scanner effect를 흔히 ‘style’로 부르지만, 이 표현은 서로 다른 현상을 한데 묶는다. 색조, white balance 및 넓은 조명 변화와 같은 저주파 외관은 전역 color transform으로 상당 부분 맞출 수 있다. 반면 blur, optical transfer, sampling, sharpening 및 compression은 고주파 구조와 phase를 변화시킨다. 후자는 이미 손실된 핵 경계나 chromatin texture를 단순한 전역 변환으로 복원할 수 없으며, 동일한 광학 전달 특성도 핵 밀도, gland boundary 및 stroma texture가 다른 조직에서 서로 다른 관측 차이를 만든다. 따라서 scanner effect는 평균적인 장비 경향과 content-dependent response를 함께 포함할 가능성이 높다.

또 다른 핵심 문제는 scanner invariance와 biological fidelity가 동치가 아니라는 점이다. 강한 blur는 scanner 간 embedding을 가깝게 만들 수 있지만 동시에 조직 구별력을 제거한다. 반대로 source morphology를 보존하는 correction은 scanner separability를 충분히 줄이지 못할 수 있다. 그러므로 scanner classifier accuracy 또는 scanner centroid distance만으로 normalization 성공을 판정하면 representation collapse를 진정한 alignment로 오인할 위험이 있다. 같은 이유로 픽셀 수준 색상 오차의 감소가 frozen PFM의 표현 정렬을 보장하지 않는다.

본 연구는 동일 조직의 paired acquisition을 사용하여 scanner sensitivity를 세 층으로 분해한다. 첫째, optical-density spectrum에서 scanner별 effective relative transfer를 연속적인 물리 좌표로 정의한다. 둘째, scanner main effect와 scanner × tissue 및 scanner × slide response를 분리한다. 셋째, 저주파 교정과 고주파 intervention이 PFM invariance와 content fidelity 사이에서 만드는 frontier를 측정한다. 외부 검증에는 동일한 biological TMA를 여러 scanner와 stain condition으로 구성한 PLISM을 사용한다 [11].

본 연구의 구체적 기여는 다음과 같다.

1. 109개 물리 슬라이드, 37개 조직 유형 및 6개 스캐너의 paired tissue spectrum을 이용해 scanner difference를 범주가 아닌 주파수별 연속 transfer로 표현한다.
2. 균형화된 replicate 설계와 계층적 모델로 scanner main, scanner × tissue, scanner × slide 및 patch-sampling variation을 분리한다.
3. 저주파 appearance calibration과 고주파 information loss를 분리하고, image-space intervention이 PFM별 invariance–fidelity frontier에 미치는 영향을 평가한다.
4. scanner separability를 content retrieval, embedding rank, nuclear morphology 및 spatial neighborhood preservation과 짝지어 collapse를 명시적으로 진단한다.
5. 내부 데이터에서 동결한 protocol을 PLISM에 적용하여 scanner-only 효과와 scanner × stain interaction을 외부에서 검증한다.

## 2. 재료 및 방법

### 2.1 연구 설계와 분석 단위

본 연구는 동일한 물리 슬라이드를 여러 whole-slide scanner로 반복 취득한 paired acquisition study다. 내부 cohort는 109개 물리 슬라이드와 37개 tissue type으로 구성되며, 각 슬라이드는 AT2, GT450, VERSA, AKOYA, S60 및 S360의 6개 scanner로 취득되었다. AT2를 상대적 transfer reference로 사용하였다. [각 장비의 제조사, 정확한 모델명, firmware, objective NA, nominal magnification, MPP 및 compression 설정을 표 S1에 입력한다.]

분석과 통계 추론의 기본 biological unit은 physical slide다. 개별 patch는 관측치이지만 독립적인 biological replicate로 취급하지 않았다. Tissue type은 slide를 묶는 상위 biological grouping으로 두었다. 모든 train/test split, bootstrap 및 cross-validation은 physical slide를 block으로 사용한다.

윤리 및 데이터 거버넌스 정보는 최종 원고에 다음 형식으로 추가한다: [IRB 기관, 승인번호, 동의면제 여부, 비식별화 절차, 인체유래물의 연구목적 사용 근거].

### 2.2 등록과 패치 표본추출

스캐너 간 동일 tissue field를 비교하기 위해 AT2 좌표계를 기준으로 paired registration을 수행하였다. 각 slide–scanner 조합에서 조직 patch 100개를 선택하였으며, 각 patch는 256 × 256 pixel이었다. Scanner별 영상 크기와 registration lattice의 일치 여부를 검사하고, local normalized cross-correlation을 이용하여 제한된 반경 안에서 residual translation을 보정하였다. [최종 protocol의 registration algorithm, search radius, NCC threshold, 실패율 및 수동 QC 절차를 상세 입력한다.]

Patch-sampling uncertainty를 별도로 추정하기 위해 100개 patch를 서로 겹치지 않는 20-patch replicate 5개로 분할하였다. 동일한 replicate index는 모든 scanner에서 같은 위치 집합을 사용하였다. Internal lattice와 독립적으로 생성된 scanner lattice의 정수 tuple ID는 결합하지 않았으며, 모든 sample identity는 lattice ID, slide ID 및 위치 ID를 함께 사용하였다.

### 2.3 주파수별 effective relative transfer

RGB patch를 optical-density 형태로 변환한 뒤 세 채널 평균을 취하고 patch 평균을 제거하였다. 경계 artifact를 줄이기 위해 2차원 Hann window를 적용하고 fast Fourier transform으로 power spectrum을 계산하였다. Scanner별 2차원 power를 100개 patch에서 평균한 뒤 공간주파수의 방사형 bin으로 집계하였다.

Scanner s와 AT2 기준 r의 상대 transfer는 power ratio의 제곱근으로 정의하였다.

T_s(f) = sqrt(P_s(f) / P_r(f)) / C_s,

여기서 C_s는 0.03–0.10 cycles/µm 구간에서 T_s(f)의 기하평균이 1이 되도록 하는 정규화 상수다. 이 정규화는 전체 밝기 또는 저주파 에너지 차이와 frequency-dependent shape을 분리한다. Band summary는 기하평균으로 계산하였으며 사전 정의한 구간은 low–mid 0.10–0.30, mid 0.30–0.60, high 0.60–0.90 cycles/µm였다.

T_s(f)는 조직 영상에서 얻은 effective relative transfer다. 이 값은 optical blur뿐 아니라 scanner-side sharpening, sampling, compression 및 noise의 영향을 포함하므로 detector의 absolute modulation transfer function(MTF)으로 해석하지 않았다. 절대적인 광학 성능 주장을 위해서는 slanted-edge 또는 resolution target과 통제된 detector 측정이 추가로 필요하다.

### 2.4 Scanner, tissue, slide 및 sampling variation의 분해

각 scanner–band 조합에서 5개의 disjoint replicate 값을 사용하여 다음 계층적 모델을 적합하였다.

y_tsr = μ_s + u_ts + v_tsl,s + ε_tslr,

여기서 μ_s는 연구 대상 scanner의 fixed mean, u_ts는 tissue type에 대한 scanner contrast의 random slope, v_tsl,s는 tissue 안에 중첩된 slide에 대한 scanner contrast의 random slope, ε는 20-patch replicate sampling error다. 분산 성분은 profiled restricted maximum likelihood로 추정하였다. Tissue-level variance의 검정은 경계 모수에 대한 mixture likelihood-ratio test를 사용하고 15개 scanner–band family에서 Benjamini–Hochberg 방법으로 보정하였다.

별도로, 다섯 non-reference scanner를 동일 가중치로 놓은 balanced marginal descriptive variance를 계산하였다. 이 분해의 scanner main percentage는 특정 scanner 집합에 대한 기술통계이며 장비 모집단의 random-effect variance가 아니다. 특히 극단적인 transfer를 가진 AKOYA의 포함 여부에 대한 scanner-subset sensitivity를 함께 보고하였다.

### 2.5 저주파 appearance correction

Scanner-pair 저주파 correction은 train slide의 paired coarse-band RGB 값으로 전역 affine transform을 적합하였다. 각 source patch의 Laplacian pyramid에서 저주파 coefficient를 변환하고 source high-frequency coefficient와 재결합하였다. Operational image는 유효 RGB 범위로 clipping하였으며, requested transform의 성능과 clipping 이후 성능을 분리하여 보고하였다. 주요 image endpoint는 LF mean absolute error(MAE), structural similarity index(SSIM), out-of-gamut image 비율, clipping pixel fraction 및 재분해 후 detail-coefficient 변화였다.

Reinhard, Macenko, RGB/OD affine, ridge affine 및 spatial affine을 classical baseline으로 비교한다. [최종 population experiment에서 Fourier transfer와 paired image translation baseline 결과 추가].

### 2.6 PFM panel과 representation endpoint

현재 완료된 mechanistic pilot는 frozen UNI encoder를 사용하였다. 최종 confirmatory experiment에서는 학습 paradigm과 전처리가 다른 6–8개 모델을 outcome 확인 전에 동결한다. 후보는 UNI 또는 UNI2-h, Virchow2, Prov-GigaPath, H-Optimus 계열, CONCH, Phikon 또는 Hibou, stain-normalized pathology model 및 DINOv2/ImageNet natural-image baseline이다. 각 checkpoint, input size, physical field of view, color preprocessing 및 feature layer를 표 S2에 기록한다.

Representation endpoint는 paired cosine similarity, leave-one-scanner-out consensus gain, scanner balanced accuracy, cross-scanner same-location retrieval, scanner centroid distance/radius, biological content margin 및 neighborhood preservation을 포함한다. Collapse diagnostic은 embedding variance, effective rank와 pairwise distance를 사용한다. Scanner invariance endpoint는 적어도 하나의 content-fidelity endpoint와 함께 해석한다.

### 2.7 고주파 intervention과 invariance–fidelity frontier

모든 모델에 동일한 physical field of view와 intervention grid를 적용한다. High-frequency attenuation은 retention 0.75, 0.50, 0.25 및 0으로, boost는 gain 1.25, 1.50 및 2.00으로 정의한다. Registered same-location leave-one-scanner-out HF mean의 25% mixing은 attainable image-space positive control로 사용한다. 전체 train patch의 coefficient-wise global HF mean은 phase cancellation에 대한 negative control이다. 모든 재결합 image에서 clipping fraction을 기록한다.

Oracle은 실용적인 deployable normalization method가 아니라 정보 보존의 상한을 확인하는 control이다. 특히 서로 다른 LF와 paired HF coefficient를 결합하면 RGB gamut을 벗어날 수 있으므로, paired coefficient를 사용했다는 이유만으로 clean causal oracle이라고 부르지 않는다.

### 2.8 Biological fidelity

최종 biological endpoint는 tissue-type classification/retrieval, nucleus segmentation agreement, nucleus count, area, eccentricity와 boundary fidelity를 포함한다. 가능할 경우 cell graph 또는 spatial-neighborhood preservation을 추가한다. [사용할 nucleus segmentation model, human-review subset, ground truth 또는 equivalence margin을 protocol freeze 단계에서 입력].

Primary endpoint는 paired embedding consensus gain, cross-scanner same-location retrieval, tissue content margin 또는 tissue retrieval, nuclear boundary/count fidelity로 제한한다. Scanner balanced accuracy와 spectral association은 secondary mechanistic endpoint다.

### 2.9 White-space background 특성화

109 slides × 6 scanners × 100 patches, 총 65,400개의 raw white-space patch를 추출하였다. Background에서 radial noise-power proxy, spectral tilt, optical-density luminance 및 두 개의 chromatic axis를 계산하였다. Tissue transfer model M0에 5개의 scanner-centered background covariate를 추가한 M1, 그리고 background patch acceptance sensitivity를 추가한 M2를 비교하였다. Fixed-effect model comparison은 maximum likelihood, variance component는 REML을 사용하였다. 일반화 성능은 tissue BLUP을 train fold 안에서 추정하는 5-fold grouped held-out-slide cross-validation으로 평가하였다.

이 분석은 white space를 진정한 detector NPS 또는 DQE 측정으로 해석하지 않는다. 통제된 균일 노출과 입사 광자 통계가 없으므로 background feature는 scanner/noise characterization, acquisition QC 및 tissue-spectrum noise-floor sensitivity에 한정한다.

### 2.10 PLISM 외부 검증

PLISM은 동일한 biological TMA를 13개 H&E staining condition과 13개 imaging device로 취득한 registered multi-domain dataset이다 [11]. 본 연구는 WSI scanner 조건 중 내부 cohort와 겹치는 장비 및 새로운 장비를 사용하여 내부에서 동결한 spectrum–representation 관계를 외부에서 검증한다. Primary analysis는 stain condition 하나를 고정한 scanner-only contrast이며, scanner × stain은 secondary analysis다.

내부 band boundary, LF transform family, intervention grid, endpoint 및 성공 기준은 PLISM 결과를 보기 전에 동결한다. PLISM tile은 registration과 resampling을 거쳤으므로 HF 분석에 앞서 interpolation audit를 수행한다. Native image를 확보하지 못하면 결과를 scanner MTF가 아니라 ‘PLISM preprocessing 이후의 effective transfer’로 표현한다. PLISM의 반복 조건은 독립 환자 cohort가 아니므로 external mechanistic validation으로만 해석한다.

### 2.11 통계 분석

최종 multi-PFM outcome의 개념적 model은 다음과 같다.

outcome ~ scanner × intervention × PFM × band
        + (1 + scanner + intervention | tissue_type)
        + (1 + scanner + intervention | slide_id)

Scanner, intervention, PFM 및 band는 fixed effect다. Tissue type과 slide가 scanner/intervention에 다르게 반응하도록 random slope 또는 Bayesian partial pooling을 사용한다. Singular fit가 발생하면 사전 정의한 순서로 random-effect structure를 단순화하거나 band별 model로 분리한다. 모든 confidence interval과 bootstrap은 slide-blocked 방식으로 계산하고, 효과크기와 CI를 p-value보다 우선하여 보고한다. Related endpoint family별 multiple testing에는 BH correction을 적용한다.

### 2.12 구현과 재현성

Spectrum, tissue random-slope 및 background 분석 코드는 각각 `analyze_exp05_spectral_cohort.py`, `analyze_exp05_tissue_random_slopes.py`, `analyze_exp08_background_mixed_effects.py`에 구현하였다. Pairwise LF/HF intervention은 `prenorm/exp02/pairwise.py`와 `prenorm/exp02/trajectory.py`에 구현하였다. [공개 repository URL, commit hash, software version, hardware 및 random seed 입력].

## 3. 결과

### 3.1 Scanner는 재현 가능한 주파수별 population signature를 보였다

109개 슬라이드 모두에서 동일한 6개 scanner와 100개 paired tissue patch가 확보되었다. AT2-normalized spectrum은 scanner별로 뚜렷하고 반복적인 frequency-dependent shape을 보였다(그림 1). GT450은 고주파로 갈수록 상대 transfer가 증가했고, AKOYA는 반대로 고주파 transfer가 크게 감소하였다. VERSA와 S60의 cohort mean은 AT2에 가까웠지만 slide별 분산은 남았으며, S360은 고주파에서 중간 정도의 감소를 보였다.

High band의 cohort fold transfer는 GT450 1.595, VERSA 1.011, AKOYA 0.297, S60 0.994, S360 0.784였다(표 1). GT450, AKOYA 및 S360의 평균은 AT2 대비 통계적으로 뚜렷했지만, VERSA와 S60의 population mean은 AT2와 구별되지 않았다. 그러나 mean이 1에 가깝다는 사실은 개별 slide 또는 tissue에서 scanner effect가 없다는 뜻이 아니다.

| Scanner | Low–mid transfer | Mid transfer | High transfer | High-band 해석 |
|---|---:|---:|---:|---|
| GT450 | 1.020 | 1.189 | 1.595 | 고주파 증폭/샤프닝 방향 |
| VERSA | 1.040 | 1.016 | 1.011 | 평균은 AT2와 유사, slide variation 존재 |
| AKOYA | 0.676 | 0.392 | 0.297 | 넓은 대역의 강한 attenuation |
| S60 | 1.020 | 1.039 | 0.994 | 평균은 AT2와 유사, tissue interaction 존재 |
| S360 | 0.977 | 0.918 | 0.784 | 중·고주파 attenuation |

표 1. AT2 대비 scanner별 effective relative transfer. 이는 tissue spectrum에서 얻은 상대값이며 absolute MTF가 아니다.

[[FIGURE:outputs/exp05_spectral_cohort_109/figure1_cohort_spectral_overview.png|그림 1. 109개 물리 슬라이드의 paired-scanner effective transfer. (A) scanner별 cohort median과 slide 10–90% 범위. (B) high-band scanner mean과 slide spread. (C) 109개 slide의 high-band signature. AT2에 대한 상대 transfer이며 absolute MTF로 해석하지 않는다.]]

### 3.2 Scanner effect의 크기는 tissue와 physical slide에 따라 달라졌다

다섯 non-reference scanner를 균형 있게 놓은 기술적 분해에서 scanner main effect는 low–mid, mid 및 high band variation의 각각 71.1%, 85.5%, 85.5%를 차지하였다. Scanner × tissue는 각각 6.6%, 3.2%, 2.5%, scanner × slide는 21.2%, 10.7%, 11.2%, 100-patch sampling은 1.0%, 0.5%, 0.8%였다(표 2). 이 결과는 scanner가 population level에서 강한 경향을 갖지만, 관측 spectrum의 약 11% 정도가 high band에서 slide-specific response로 남음을 보여준다.

| Band | Scanner main | Scanner × tissue | Scanner × slide | Sampling |
|---|---:|---:|---:|---:|
| Low–mid | 71.1% | 6.6% | 21.2% | 1.0% |
| Mid | 85.5% | 3.2% | 10.7% | 0.5% |
| High | 85.5% | 2.5% | 11.2% | 0.8% |

표 2. 균형화된 marginal descriptive variance decomposition. Scanner main percentage는 현재 scanner 집합에 의존하는 기술통계다.

Scanner main 비율은 장비 subset에 민감했다. 특히 AKOYA를 제외하면 scanner main fraction은 low–mid, mid 및 high band에서 약 5%, 28%, 65%로 낮아졌다. 따라서 ‘scanner가 보편적으로 전체 변이의 85.5%를 설명한다’고 일반화할 수 없다.

Tissue type이 between-slide scanner-slope variance에서 차지하는 비율도 scanner별로 달랐다(그림 2). High band에서 tissue share는 GT450 17%, VERSA 11%, AKOYA 28%, S60 27%, S360 3%였다. 15개 scanner–band 조합 중 8개에서 tissue-level variance가 BH 보정 후 유의하였다. S60은 AT2와 population mean이 유사했지만 tissue share가 컸고, S360은 tissue share가 낮은 대신 slide-within-tissue variation이 컸다. 이는 scanner effect가 하나의 tissue-independent style vector가 아니라 scanner × content response임을 직접 지지한다.

[[FIGURE:outputs/exp05_spectral_cohort_109/figure4_tissue_random_slopes.png|그림 2. 37개 tissue type과 109개 slide에서 분해한 content-dependent scanner response. (A) high-band scanner × tissue, scanner × slide-within-tissue 및 sampling variance. (B) scanner별 between-slide variance 중 tissue share. 별표는 tissue-variance LRT의 BH q<0.05를 나타낸다. (C) tissue-specific high-band scanner slopes의 BLUP.]]

[[FIGURE:outputs/exp05_spectral_cohort_109/figure2_mixed_effects.png|그림 S1. Scanner fixed effect, high-band slide variation 및 scanner-slope correlation. 이 그림의 scanner × slide 분해는 tissue annotation을 추가하기 전의 slide-level model이며, 본문의 최종 계층적 해석은 그림 2를 따른다.]]

### 3.3 저주파 appearance alignment는 PFM representation alignment와 분리되었다

Paired mechanistic pilot에서 scanner-specific RGB affine은 LF MAE를 raw 0.2531에서 requested transform 0.0297로 낮추었다. 이는 약 88%의 error reduction이다. Source high-frequency coefficient와 재결합하고 RGB range로 clipping한 operational image의 LF MAE는 0.0366이었으며, full-image SSIM은 0.281에서 0.433으로 증가하였다. 따라서 known-scanner paired calibration 조건에서 저주파 외관의 상당 부분은 단순한 전역 affine으로 교정 가능했다.

그러나 저주파 image-space improvement는 frozen UNI의 scanner invariance로 이어지지 않았다. Sample identity를 lattice별로 바로잡은 matched task에서 low-band scanner balanced accuracy는 internal lattice에서 0.813에서 0.556, S60 lattice에서 0.928에서 0.780으로 감소했다. 반면 UNI scanner balanced accuracy는 internal 0.998에서 0.998, S60 0.998에서 1.000으로 사실상 변하지 않았다. Paired target cosine gain도 14개 scanner direction 평균 -0.0202였고 7개 direction은 증가, 7개 direction은 감소하였다. 14개 중 6개는 두 test slide 사이에서 효과 방향이 바뀌었다.

이 pilot는 population estimate가 아니지만 두 중요한 failure mode를 보여준다. 첫째, 시각적 또는 저주파 feature 수준의 scanner separability 감소는 PFM representation alignment와 동일하지 않다. 둘째, 같은 LF transform도 scanner direction과 slide content에 따라 PFM 이동 방향이 달라질 수 있다. [109-slide multi-PFM 결과 및 slide-bootstrap CI 입력].

### 3.4 고주파 감소로 얻은 scanner convergence는 collapse와 구별되어야 했다

세 개의 사전 선택된 matched location에서 blur, sharpening, registered HF mean 및 global HF mean trajectory를 평가하였다. 1,024개 train image의 coefficient-wise global HF mean은 개별 image HF RMS의 4.27%만 보존하였다. 이는 phase cancellation으로 인해 사실상 low-pass negative control로 작동하였다.

Detail을 완전히 제거하면 scanner radius는 0.095에서 0.040으로 감소했지만 self-cosine은 0.065, consensus gain은 -0.781, content margin은 거의 0, cross-scanner retrieval은 0.67이었다. 즉 scanner embedding이 모인 이유는 alignment가 아니라 정보 제거였다. Global HF mean 100%도 scanner radius를 0.040으로 줄였지만 self-cosine 0.078과 consensus gain -0.759를 보여 유사한 collapse를 일으켰다.

반대로 sharpening 1.0은 self-cosine 0.828과 retrieval 1.0을 유지했지만 scanner radius를 0.095에서 0.120으로 증가시켰다. Registered same-location leave-one-scanner-out HF mean 25%는 self-cosine 0.957과 retrieval 1.0을 유지하면서 scanner radius를 0.069로 낮추고 consensus gain +0.028 및 content margin 0.384를 보였다. 이는 실용적인 method가 아닌 oracle positive control이지만, scanner convergence가 content preservation과 동시에 가능한 영역이 있음을 보여준다.

[[FIGURE:outputs/exp05_spectral_cohort_109/figure3_blur_sharpen_bridge.png|그림 3. Cohort scanner spectrum과 mechanistic HF intervention의 연결. (A) scanner curve와 blur/sharpen control. (B) 하나의 HF-gain axis만으로 scanner variation을 설명할 수 있는지 평가. (C) 각 slide의 nearest intervention gain과 남는 off-axis spectrum error. Scanner 차이는 단일 blur/sharpen scalar로 완전히 환원되지 않는다.]]

LF reference를 AT2, GT450, VERSA 및 AKOYA로 회전했을 때 같은 HF intervention의 방향과 scanner radius 변화가 달라졌다. 일부 LF–HF 조합은 out-of-gamut clipping을 유발했으며, AKOYA LF reference에서는 non-AKOYA source pixel의 9–19%가 clipping 대상이었다. 이는 high-frequency reconstruction이 단순히 source texture에 scanner-specific gain을 곱하는 문제가 아니며, LF context와의 물리적 호환성까지 요구함을 보여준다.

### 3.5 White-space background는 tissue-domain transfer를 실질적으로 설명하지 못했다

109 × 6 × 100개의 raw white-space patch에서 background spectrum과 optical-density feature를 계산하였다. Tissue 및 slide random slope를 포함한 baseline model에 5개의 background covariate를 추가했을 때 15개 scanner–band model 중 AIC는 3개에서만 background model을 선호했고 BIC는 어느 model에서도 선호하지 않았다. Likelihood-ratio test는 15개 모두 BH 보정 후 유의하지 않았다.

Held-out-slide prediction은 2/15 조합에서만 개선되었으며 RMSE improvement는 +0.69%와 +0.09%였다. 나머지 13개에서는 악화하였다. 75개 background coefficient 중 5개만 nominally significant였고 BH 보정 후 유의한 coefficient는 없었다. Background-patch acceptance sensitivity 또한 15개 모두 nominally significant하지 않았다.

[[FIGURE:outputs/exp08_background_mixed_effects/figure1_background_mixed_model_comparison.png|그림 S2. Scanner-centered background feature가 tissue transfer를 설명하는지 평가한 mixed-model comparison. Background feature는 일부 in-sample variance를 바꾸었지만, 정보기준과 held-out-slide 예측에서 일관된 증분 설명력을 보이지 않았다.]]

이 결과는 background가 무의미하다는 뜻이 아니다. White space는 dark/white balance, noise fingerprint, compression artifact 및 failed-scan QC에 유용하다. 다만 tissue가 없는 영역만으로 tissue-dependent optical transfer를 식별하거나 global tissue transform을 학습하는 근거는 현재 데이터에서 지지되지 않았다.

### 3.6 Multi-PFM population 결과

> [필수 결과 자리표시자] 6–8개 PFM에서 raw, LF affine, HF attenuation/boost, registered oracle, global mean 및 feature-space correction을 109개 slide 전체에 적용한 결과를 입력한다. 최소 보고 항목은 paired consensus gain, same-location retrieval, scanner BACC, content margin, effective rank, tissue retrieval 및 PFM × scanner × intervention interaction이다.

예상 표 구조:

| PFM | Best image-space condition | Δ consensus | Retrieval | Scanner BACC | Content margin | Effective rank | 판정 |
|---|---|---:|---:|---:|---:|---:|---|
| UNI/UNI2-h | [입력] | [입력] | [입력] | [입력] | [입력] | [입력] | [입력] |
| Virchow2 | [입력] | [입력] | [입력] | [입력] | [입력] | [입력] | [입력] |
| Prov-GigaPath | [입력] | [입력] | [입력] | [입력] | [입력] | [입력] | [입력] |
| H-Optimus | [입력] | [입력] | [입력] | [입력] | [입력] | [입력] | [입력] |
| CONCH | [입력] | [입력] | [입력] | [입력] | [입력] | [입력] | [입력] |
| Natural-image baseline | [입력] | [입력] | [입력] | [입력] | [입력] | [입력] | [입력] |

### 3.7 Biological fidelity와 PLISM external validation

> [필수 결과 자리표시자] Nuclear boundary/count fidelity, tissue retrieval 및 spatial-neighborhood preservation 결과를 intervention별로 입력한다. Scanner invariance improvement가 equivalence margin 안의 biological fidelity와 함께 나타나는지 판정한다.

> [필수 결과 자리표시자] PLISM의 frozen scanner-only primary analysis와 scanner × stain secondary analysis를 입력한다. Internal–external effect direction, effect size와 CI를 나란히 보고하고, interpolation audit 결과를 먼저 제시한다.

## 4. 논의

본 연구는 동일한 물리 조직을 여러 scanner로 취득한 paired design을 이용해 scanner effect를 주파수, content interaction 및 representation fidelity의 세 층으로 분해하였다. 현재 완료된 결과의 핵심은 네 가지다. 첫째, scanner는 109개 slide에서 재현 가능한 population-level spectral signature를 보였다. 둘째, 그 signature의 크기는 tissue type과 physical slide에 의해 수정되었다. 셋째, 저주파 appearance correction은 강력했지만 UNI representation alignment를 보장하지 않았다. 넷째, scanner separability 또는 centroid radius의 감소는 content 보존과 함께 평가하지 않으면 collapse를 normalization으로 오인할 수 있었다.

### 4.1 Scanner style은 population tendency이지만 전역적으로 분리 가능한 layer는 아니다

GT450과 AKOYA는 평균적으로 매우 다른 high-frequency transfer를 보였으며, 이는 scanner identity가 population level에서 안정적인 signature를 가진다는 것을 보여준다. 그러나 VERSA와 S60처럼 AT2와 평균이 유사한 scanner도 tissue 및 slide slope variation을 보였다. 같은 장비 contrast가 tissue에 따라 크기와 때로는 방향이 달라졌다는 점은 ‘scanner style’을 하나의 고정 vector 또는 global transform으로 표현하는 접근의 한계를 드러낸다.

이 결과는 scanner main effect와 content interaction 중 어느 하나만 강조해서는 안 됨을 뜻한다. 전체 scanner 집합에서는 main effect가 컸지만, AKOYA를 제외한 sensitivity analysis에서 특히 low–mid와 mid band의 main fraction이 크게 낮아졌다. 장비 선택에 의존하는 descriptive percentage를 scanner 모집단 전체의 보편적인 분산 설명률로 해석해서는 안 된다. 더 적절한 표현은 scanner style이 population tendency로 존재하지만 실제 관측 shift는 scanner × content response라는 것이다.

### 4.2 색상 또는 저주파 정규화는 필요한 calibration이지만 충분조건이 아니다

Train-paired affine이 저주파 MAE의 대부분을 제거한 결과는 color calibration의 실용적 가치를 보여준다. 동시에 UNI scanner BACC와 paired target alignment가 개선되지 않은 결과는 appearance alignment와 representation alignment가 서로 다른 목표임을 명확히 한다. PFM은 color-normalized image에서도 source scanner의 high-frequency texture, residual low-frequency cue 또는 두 band의 interaction을 사용할 수 있다.

따라서 LF correction을 실패로만 규정하는 것도 부정확하다. 이 교정은 시각적 일관성, low-frequency feature 및 특정 downstream pipeline에 유용할 수 있다. 다만 PFM invariance를 주장하려면 embedding과 biological endpoint에서 별도의 검증이 필요하다. 최종 multi-PFM 분석은 모델의 pretraining paradigm과 input preprocessing에 따라 이 분리가 얼마나 달라지는지 정량화해야 한다.

### 4.3 Invariance–fidelity frontier가 normalization 성공의 중심 평가가 되어야 한다

Complete blur와 global HF mean은 scanner radius를 줄였지만 원래 embedding과 content structure를 함께 제거했다. 이는 낮은 scanner classification accuracy가 항상 좋은 결과가 아님을 보여주는 직접적인 harmful control이다. 반대로 registered same-location HF mean의 약한 mixing은 content를 유지하면서 scanner radius를 줄일 수 있는 oracle 영역을 제시했다. 중요한 비교는 ‘얼마나 scanner가 덜 보이는가’가 아니라 ‘같은 biological content를 얼마나 보존한 채 scanner effect를 줄였는가’다.

이 관점은 최근 multi-scanner PFM 연구와 상보적이다. 선행 연구들은 여러 PFM이 scanner를 인코딩하고 downstream calibration 또는 prediction agreement에 영향을 줄 수 있음을 대규모로 보였다 [7–10]. 본 연구는 scanner difference를 paired tissue spectrum과 intervention trajectory에 연결하고, scanner convergence가 collapse인지 fidelity-preserving alignment인지 구별하는 것을 목표로 한다.

### 4.4 Image-space와 feature-space correction의 역할은 다를 수 있다

이미지 수준 correction은 사람과 여러 model이 공유하는 하나의 입력을 제공한다는 장점이 있다. 그러나 손실된 고주파 구조를 복원하려면 입력에 없는 정보를 추정해야 하며, LF–HF recombination 자체가 valid RGB range를 위반할 수 있다. 이러한 문제는 sharp-to-blur와 blur-to-sharp 방향에서 비대칭적이다. 후자는 target scanner가 보았을 detail을 source에서 식별할 수 없기 때문에 hallucination 위험이 더 크다.

Feature-space correction은 frozen embedding에서 scanner nuisance를 직접 줄이므로 물리적으로 그럴듯한 RGB image를 생성할 필요가 없다. CORAL, Procrustes, ComBat 및 FEATMAP과 같은 접근은 image-space transform의 positive comparator가 될 수 있다 [10,12]. 그러나 feature harmonization도 biological geometry를 왜곡할 수 있으므로 tissue retrieval, neighborhood preservation 및 downstream calibration을 함께 검증해야 한다. 최종 결론은 image-space와 feature-space 중 하나를 선험적으로 선택하는 것이 아니라, 새로운 tissue, slide 및 scanner에서의 invariance–fidelity frontier를 비교해 내려야 한다.

### 4.5 Background physics의 제한된 역할

White-space patch는 scanner의 dark/white balance와 noise/compression fingerprint를 관찰하는 저비용 source다. 그러나 현재 분석에서 background feature는 tissue-domain transfer의 held-out prediction을 일관되게 개선하지 못했다. 이는 광학계가 tissue morphology와 상호작용하여 만드는 frequency response를 tissue가 없는 background만으로 식별하기 어렵기 때문이다. 따라서 background는 acquisition QC, failed-scan detection 및 noise-floor sensitivity에 사용하되, tissue normalization의 핵심 학습 signal 또는 absolute detector characterization으로 과장하지 않는 것이 타당하다.

### 4.6 외부 타당성과 임상적 의미

PLISM은 scanner와 stain을 factorial하게 분리할 수 있어 내부 기전을 검증하기에 적합하다. 그러나 동일 TMA의 반복 acquisition이므로 독립 환자 cohort의 clinical validation은 아니다. 최종 연구는 PLISM에서 scanner-only effect의 방향과 크기가 재현되는지 먼저 확인하고, scanner × stain interaction을 secondary analysis로 다뤄야 한다. 실제 임상 deployment 주장을 위해서는 독립 기관, 환자-level downstream task 및 calibration consistency가 추가로 필요하다.

본 연구의 실용적 함의는 scanner harmonization을 하나의 성공/실패 문제로 보지 않는 데 있다. 색상 calibration은 일부 workflow에서 안전하고 유용할 수 있고, 고주파 손실은 scanner와 task에 따라 별도 관리가 필요하다. PFM 배포 전에는 scanner robustness를 단일 평균 성능뿐 아니라 paired representation stability, calibration, content retrieval 및 morphology preservation으로 평가해야 한다.

### 4.7 한계

첫째, effective relative transfer는 tissue power spectrum 기반이며 absolute MTF가 아니다. Noise, sampling, sharpening 및 compression을 완전히 분리하지 못한다. 둘째, 현재 완료된 PFM mechanistic result는 주로 UNI와 소수 location에 기반하므로 population 또는 model-general conclusion으로 확대할 수 없다. 셋째, scanner main percentage는 현재 선택한 장비 집합에 민감하다. 넷째, 일부 paired LF–HF recombination은 clipping을 유발하여 clean band intervention의 해석을 제한한다. 다섯째, 37개 tissue type 중 일부는 slide 수가 1–2개로 tissue-level variance의 불확실성이 크다. 여섯째, background patch는 통제된 detector exposure가 아니므로 NPS/DQE를 직접 측정하지 않는다. 일곱째, PLISM의 registered tile은 interpolation에 의해 고주파 spectrum이 변할 수 있다.

이 한계를 줄이기 위해 최종 연구는 multi-PFM population analysis, slide-blocked uncertainty, biological equivalence endpoint, scanner-subset sensitivity, interpolation audit 및 frozen external validation을 포함한다.

## 5. 결론

병리 scanner effect는 하나의 전역적인 style layer가 아니라 교정 가능한 저주파 외관 차이와 scanner가 보존·손실하는 고주파 정보가 tissue 및 physical slide content와 상호작용한 결과다. 109-slide paired spectrum은 강한 scanner population signature와 동시에 tissue- 및 slide-specific response를 보여주었다. 저주파 affine은 appearance를 크게 맞췄지만 UNI alignment를 보장하지 않았고, 강한 고주파 제거는 scanner convergence와 함께 representation collapse를 일으켰다. 따라서 scanner invariance는 biological fidelity와 분리해 평가할 수 없다. 최종 multi-PFM 및 PLISM 분석은 어떤 image-space 또는 feature-space correction이 이 공동 기준을 만족하는지 결정할 것이다.

## 선언

### 윤리 승인

[IRB 기관, 승인번호, 동의면제 및 비식별화 절차 입력]

### 데이터 및 코드 가용성

내부 whole-slide image는 [접근 제한 사유 및 신청 절차 입력]에 따라 제공된다. PLISM은 CC BY 4.0으로 공개된 외부 dataset이다 [11]. 분석 코드와 frozen configuration은 [repository URL 및 release DOI 입력]에서 공개할 예정이다.

### 이해상충

[저자별 이해상충 입력. 없으면 “저자들은 이해상충이 없음을 선언한다.”]

### 연구비

[연구비 기관, 과제번호 및 funder role 입력]

### 저자 기여

[CRediT taxonomy에 따라 Conceptualization, Data curation, Formal analysis, Investigation, Methodology, Software, Supervision, Validation, Visualization, Writing–original draft, Writing–review & editing 입력]

### 생성형 AI 사용 고지

[투고 시점의 Elsevier 정책을 재확인한다. 본 한글 working draft의 언어 구성과 Word formatting에 OpenAI Codex가 사용되었고, 과학적 주장·수치·인용의 최종 검토와 책임은 저자에게 있다는 형태의 disclosure 필요 여부를 결정한다.]

## 참고문헌

1. Clarke EL, Treanor D. Colour in digital pathology: a review. Histopathology. 2017;70:153–163. https://doi.org/10.1111/his.13079.
2. Tellez D, Litjens G, Bándi P, et al. Quantifying the effects of data augmentation and stain color normalization in convolutional neural networks for computational pathology. Medical Image Analysis. 2019;58:101544. https://doi.org/10.1016/j.media.2019.101544.
3. Macenko M, Niethammer M, Marron JS, et al. A method for normalizing histology slides for quantitative analysis. IEEE ISBI. 2009:1107–1110. https://doi.org/10.1109/ISBI.2009.5193250.
4. Chen RJ, Ding T, Lu MY, et al. Towards a general-purpose foundation model for computational pathology. Nature Medicine. 2024;30:850–862. https://doi.org/10.1038/s41591-024-02857-3.
5. Xu H, Usuyama N, Bagga J, et al. A whole-slide foundation model for digital pathology from real-world data. Nature. 2024;630:181–188. https://doi.org/10.1038/s41586-024-07441-w.
6. Lu MY, Chen B, Williamson DFK, et al. A visual-language foundation model for computational pathology. Nature Medicine. 2024;30:863–874. https://doi.org/10.1038/s41591-024-02856-4.
7. Carloni G, Brattoli B, Keum S, et al. Pathology foundation models are scanner sensitive: benchmark and mitigation with contrastive ScanGen loss. arXiv. 2025. https://doi.org/10.48550/arXiv.2507.22092.
8. Thiringer E, Gustafsson FK, Ledesma Eriksson K, Rantalainen M. Scanner-induced domain shifts undermine the robustness of pathology foundation models. arXiv. 2026. https://doi.org/10.48550/arXiv.2601.04163.
9. Henriksen AL, Skrede OJ, van der Schee L, et al. Enabling clinical use of foundation models in histopathology. arXiv. 2026. https://doi.org/10.48550/arXiv.2602.22347.
10. FEATMAP authors. FEATMAP: Targeted correction of acquisition signatures harmonizes medical foundation model embeddings and enables robust task generalization. bioRxiv. 2026. https://doi.org/10.64898/2026.07.02.736184. [최종 저자 목록과 version 확인]
11. Ochi M, Komura D, Onoyama T, et al. Registered multi-device/staining histology image dataset for domain-agnostic machine learning models. Scientific Data. 2024;11:330. https://doi.org/10.1038/s41597-024-03122-5.
12. Sun B, Saenko K. Deep CORAL: correlation alignment for deep domain adaptation. ECCV Workshops. 2016:443–450. https://doi.org/10.1007/978-3-319-49409-8_35.

## 표 및 그림 완성 계획

| 항목 | 현재 상태 | 최종 원고에서 필요한 작업 |
|---|---|---|
| 그림 1 Scanner spectrum | 완료 | 축·폰트 journal style 통일, 대표 patch 추가 검토 |
| 그림 2 Content interaction | 완료 | tissue label 가독성 개선, CI 또는 uncertainty 보강 |
| 그림 3 LF calibration across PFMs | 미완료 | image metric과 PFM metric을 같은 panel에 배치 |
| 그림 4 HF invariance–fidelity frontier | pilot 완료 | 109 slides × multi-PFM population CI 추가 |
| 그림 5 PFM × scanner × tissue | 미완료 | task-band susceptibility와 biology endpoint 통합 |
| 그림 6 PLISM validation | 미완료 | frozen internal prediction과 external effect 나란히 표시 |
| 표 1 Cohort/scanner details | 부분 완료 | 모델명, MPP, optics, compression, firmware 입력 |
| 표 2 Spectral effects | 완료 | fold estimate, 95% CI, q-value 포함 |
| 표 3 Multi-PFM outcomes | 미완료 | primary endpoint와 collapse diagnostic 입력 |
| 표 4 External validation | 미완료 | PLISM scanner/stain 조건 및 effect 입력 |

## 투고 전 필수 체크리스트

- PFM panel, checkpoint, input FOV 및 preprocessing을 결과 확인 전에 동결한다.
- 109-slide multi-PFM primary endpoint와 slide-blocked CI를 완성한다.
- Nuclear morphology 또는 동등한 biological-fidelity endpoint를 최소 하나 완성한다.
- PLISM 한 stain scanner-only primary replication과 interpolation audit를 완성한다.
- Scanner subset, background, clipping 및 singular-fit sensitivity를 Supplement에 포함한다.
- Ethics, scanner acquisition metadata, code/data availability 및 CRediT를 채운다.
- 초록의 모든 수치를 final locked table과 대조한다.
- “MTF”, “clinical validation”, “universal impossibility”와 같은 과도한 표현을 제거한다.
- Medical Image Analysis 최신 Guide for Authors에서 abstract, highlights, figure 및 AI disclosure 요건을 재확인한다.

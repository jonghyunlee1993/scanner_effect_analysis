# Scanner Spectrum 연구 종합

최종 갱신: 2026-07-24

## 1. 연구 질문

이 프로젝트는 새로운 image canonicalizer를 제안하는 연구가 아니다. 동일한
조직을 여러 scanner로 취득한 registered image를 이용해 다음 세 목표가
서로 같은지 검증한다.

1. **Appearance alignment:** scanner 간 색과 저주파 외관이 유사해지는가?
2. **Representation alignment:** frozen pathology foundation model(PFM)의
   embedding이 동일 조직 위치에서 가까워지는가?
3. **Biological fidelity:** 미세 형태, content geometry와 물리적으로 valid한
   RGB image가 보존되는가?

현재 결과는 이 세 목표가 서로 동치가 아니며, post-hoc pixel-space
harmonization에는 appearance–representation–biology trade-off가 있음을
지지한다.

## 2. 주파수별 해석

현재 frequency contract는 256×256 RGB tile을 fixed Laplacian pyramid로
분해한 연산적 정의다. Coarsest 16×16 tensor를 low frequency(LF), 나머지
coefficient를 detail/high frequency(HF)로 부른다. 이는 생물학적 정보의
절대적인 구분이 아니다.

| band | 포함 가능한 정보 | scanner component | 교정 위험 |
|---|---|---|---|
| LF | 밝기, RGB/OD 색조, stain density, 배경, 넓은 조직 구조 | spectral response, white balance, contrast, offset, 저주파 조명 | tissue/stain composition과 scanner color를 혼동 |
| HF | 핵 경계, chromatin, 세포질 texture, 미세 구조 | focus, MTF, optical blur, scanner sharpening | morphology와 scanner signature가 같은 band에 공존 |
| 최고주파/artifact | sensor noise, compression, ringing, pen marker | device-specific noise와 processing | 증폭 또는 생성 시 hallucination |

실험 결과는 PFM response가 LF와 HF의 단순 합이 아님을 보여준다. 같은
blur/sharpen도 어떤 scanner의 LF를 reference로 사용했는지에 따라 UNI
이동 방향이 달라졌다.

## 3. 실험으로 확인된 사실

### 3.1 LF appearance는 단순 affine으로 대부분 설명된다

- Scanner-specific train-paired RGB affine은 held-out LF MAE를
  GT450 93.8%, VERSA 79.7%, S60 92.3% 줄였다.
- Rotating-reference audit에서 non-identity direction의 LF MAE는 raw
  `0.161–0.339`에서 requested affine `0.012–0.048`, operational clipping
  후 `0.023–0.055`로 줄었다.
- Ridge affine과 quadratic spatial affine은 global affine보다 의미 있게
  낫지 않았다.
- Reinhard와 Macenko는 coarse-band target error를 대체로 악화시켰다.

따라서 known-scanner paired calibration 조건에서는 복잡한 image model보다
global color transform이 강한 baseline이다. 이 결과는 unseen scanner의
source image만 보고 동일한 transform을 식별할 수 있다는 뜻은 아니다.

### 3.2 LF correction은 frozen UNI alignment가 아니다

- 동일 location으로 identity를 바로잡은 뒤 LF scanner BACC는 internal
  `0.813→0.556`, S60 lattice `0.928→0.780`으로 감소했다.
- 반면 UNI scanner BACC는 internal `0.998→0.998`, S60 lattice
  `0.998→1.000`으로 개선되지 않았다.
- Pairwise clipped-primary에서 LF correction은 embedding을 움직였지만
  sharp paired-target cosine gain은 14 direction 평균 `-0.0202`였고,
  7 direction positive / 7 direction negative였다.
- 14 direction 중 6 direction은 두 test slide 사이에서 gain 부호가 바뀌었다.

올바른 결론은 “LF correction이 UNI를 움직이지 않는다”가 아니다. LF
correction은 UNI를 움직이지만, 그 방향이 target scanner alignment로
일관되지 않는다.

### 3.3 Blur에 의한 convergence는 information destruction일 수 있다

- Complete blur에서 raw self-cosine은 `0.065`, content margin은 거의 0,
  cross-scanner retrieval은 0.67까지 떨어졌다.
- Scanner centroid radius는 줄지만 consensus gain은 `-0.781`이었다.
- Matched-blur 실험에서 detail을 완전히 제거하면 transformed-target cosine은
  `0.9898`까지 증가하지만 blurred target과 자기 sharp target cosine은
  `0.0447`만 남았다.

따라서 scanner distance 감소만으로 normalization success를 주장할 수
없다. Content geometry와 centroid preservation을 동시에 평가해야 한다.

### 3.4 Sharpen은 scanner disagreement를 대체로 증폭한다

- Sharpen 1.0에서 raw self-cosine은 `0.828`, content retrieval은 1.0을
  유지했지만 scanner radius는 `0.095→0.120`으로 증가했다.
- Sharpen trajectory는 strength에 대해 비교적 일관되지만 scanner 간
  direction agreement는 낮다.
- LF reference를 바꾸면 동일 sharpen의 방향과 scanner radius 변화가
  달라졌다. LF→GT450만 radius가 소폭 감소하는 예외를 보였다.

이는 scanner style이 단일 전역 vector가 아니라
scanner × LF context × HF intervention interaction임을 지지한다.

### 3.5 평균 HF reference는 종류에 따라 전혀 다르다

- 1,024 train image의 coefficient-wise global HF mean RMS는 개별 image
  HF RMS의 4.27%였다. Phase cancellation 때문에 사실상 blur reference다.
- Global HF mean mixing의 UNI 방향은 약한–중간 strength에서 blur와 거의
  동일했다.
- 동일 위치의 나머지 scanner HF를 평균한 registered oracle은 blur와 다른
  방향을 보였고, 25% mixing에서 scanner radius `0.095→0.069`,
  consensus gain `+0.028`, retrieval 1.0을 달성했다.

따라서 “평균 HF가 모두 무의미하다”는 틀리다. 배포 불가능한 same-location
oracle consensus에는 신호가 있지만, population global mean은 거의
low-pass negative control이다.

### 3.6 LF–HF 재조합은 물리적 validity 문제를 만든다

Target LF와 source HF를 독립적으로 합치면 일부 scanner direction에서
RGB gamut을 벗어난다. Akoya LF reference에서는 non-Akoya source의
aligned baseline부터 약 9–19% pixel이 clipping 대상이었다. Paired target의
진짜 HF coefficient를 사용해도 incompatible source LF와 결합하면 다수
image가 range를 벗어났다.

따라서 target-like HF generation은 두 문제를 함께 해결해야 한다.

1. Source morphology에서 target scanner가 관찰했을 HF를 식별할 것
2. 그 HF가 LF와 결합되어 valid RGB image를 만들 것

Clipping 후 image는 operational input으로 사용할 수 있지만 clean HF-only
causal intervention으로 해석할 수 없다.

## 4. 외부 경험적 관찰

별도 프로젝트에서 stain-normalized image로 사전학습한 EXAONEPath
embedding에도 명확한 scanner separation이 관찰됐다. 이 결과는 현재
repository의 정식 실험 결과가 아니며 external empirical observation으로만
다룬다.

이 관찰이 지지하는 범위는 “stain-normalized pretraining이 scanner
invariance에 충분하지 않다”까지다. Stain normalization이 scanner effect를
전혀 줄이지 않았다는 결론에는 non-normalized matched control이 필요하다.
향후 정식 multi-PFM 실험에서는 EXAONEPath을 stain-normalized PFM control로
포함한다.

## 5. 논문의 방어 가능한 주장

Primary claim:

> Frozen PFM을 위한 post-hoc pixel-space scanner harmonization에는
> appearance alignment, representation alignment, biological fidelity
> 사이의 구조적인 trade-off가 있다.

Supporting claims:

1. Known-scanner LF appearance의 큰 부분은 단순 affine으로 교정 가능하다.
2. LF appearance alignment는 PFM representation alignment를 보장하지 않는다.
3. HF suppression으로 얻은 scanner convergence는 representation collapse와
   구분해야 한다.
4. HF amplification과 LF-reference 교체는 scanner-specific embedding
   response를 드러낸다.
5. Target-like HF synthesis는 morphology identifiability와 LF–HF physical
   compatibility 문제를 동시에 가진다.
6. 따라서 calibrated feature-space correction은 universal pixel-space
   canonicalization보다 better-posed한 대안일 수 있으나, biological
   geometry preservation은 별도로 검증해야 한다.

현재 주장하지 않는 것:

- 모든 image-level correction의 수학적 불가능성
- 고주파만이 scanner signature의 유일한 원인이라는 주장
- 모든 PFM이 UNI와 동일하다는 주장
- 공통 image space가 존재하지 않는다는 직접적인 부정 증명
- feature-level correction이 이미 검증됐다는 주장
- image translation에는 반드시 대규모 paired data가 필요하다는 주장

## 6. Pix2Pix falsification gate

LF-affine composite에서 actual paired target으로 가는 Pix2Pix는 image-level
correction의 가장 강한 반론을 시험하는 falsification experiment다. 처음부터
모든 scanner pair로 확장하지 않고 두 reciprocal direction으로 gate한다.

각 direction에서 다음 네 조건을 비교한다.

1. Raw source → target, deterministic L1 regression
2. LF-affine composite → target, deterministic L1 regression
3. Raw source → target, Pix2Pix
4. LF-affine composite → target, Pix2Pix

평가는 held-out slide의 RGB/LF/HF paired error, HF coherence, PFM alignment,
scanner probe, nucleus morphology, clipping과 unsupported edge generation으로
한다. Sharp→blur와 blur→sharp direction을 분리한다. 후자는 source에 없는
detail을 target에서 요구하므로 hallucination 위험이 더 크다.

LF-conditioned Pix2Pix가 raw translator와 linear HF transfer를 모두 이기고
biology를 보존할 때만 전체 scanner pair와 sample-size learning curve로
확장한다. 그렇지 않으면 음성 falsification baseline으로 남기고 중단한다.

## 7. 투고 전략과 요구 실험량

### Primary target: Medical Image Analysis

Biomedical image representation, feature extraction와 spectral analysis에
가장 직접적으로 맞는다. 필요한 최소 package:

- UNI, EXAONEPath, Virchow2, Prov-GigaPath, vision-language PFM,
  ImageNet ResNet 계열
- Patch가 아닌 physical slide를 단위로 한 population confirmatory analysis
- RGB/OD affine, Reinhard, Macenko, Fourier/LF transfer
- 최소 Pix2Pix falsification gate
- CORAL/Procrustes/ComBat/feature affine positive controls
- Nucleus morphology와 content-neighbourhood preservation
- Hierarchical bootstrap과 scanner × PFM × intervention interaction

### Stretch target: Nature Communications

위 package에 independent cohort/institution, multiple tissue types,
downstream prediction 및 calibration consistency, stronger pathology review,
full pixel-vs-feature Pareto analysis가 추가되어야 한다. 이 경우 paired
nonlinear image translator는 사실상 필수다.

### 현재 우선순위가 낮은 target

- npj Digital Medicine: clinical application과 external deployment 결과가
  없는 mechanistic study에는 fit이 낮다.
- IEEE TMI: 새로운 correction method 또는 formal framework가 추가될 때
  적합하다.
- Nature Biomedical Engineering / Nature Machine Intelligence: 현재
  pathology-only characterization보다 훨씬 넓은 기술적·임상적 advance가
  필요하다.

## 8. 다음 confirmatory 순서

1. Multi-PFM extractor와 동일 metric contract 고정
2. 전체 registered test slide에서 LF/HF intervention 재실행
3. Classical image baseline과 feature positive control 실행
4. Biology-preservation evaluator 고정
5. 두-direction Pix2Pix gate
6. Reference rotation, cycle consistency와 common-space minimax test
7. Gate 결과에 따라 MedIA 또는 Nature Communications 볼륨 결정

세부 실행 provenance는
[`exp01_execution_log.md`](exp01_execution_log.md)와
[`exp02_execution_log.md`](exp02_execution_log.md)에 남긴다.

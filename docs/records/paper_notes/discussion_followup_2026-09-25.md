# Discussion follow-up experiments (103-slide manuscript cohort)

## Scope and shared evaluation

PanNormal uses the locked five physical-slide folds (103 slides). Feature transforms are fitted on 40 paired raw UNI-v1 positions per training slide and evaluated on the manuscript's fixed 20 held-out positions per slide. No held-out slide enters fitting. PLISM applies the five PanNormal fold maps without refitting and averages the three shared scanner directions across 13 sections. Positive paired UNI target gain means movement toward the actual same-location target embedding. Results below are for the AT2-to-scanner direction unless stated otherwise.

The affine conditions implement unregularized least-squares mapping (the core FEATMAP fitting rule) and inner-validation-selected ridge affine mapping; they are described as FEATMAP-style rather than an official FEATMAP software reproduction. ComBat uses the scanner target as reference batch. Orthogonal Procrustes is a paired, norm-preserving comparator. Harmony was omitted from the held-out transfer comparison because the installed implementation does not provide a frozen out-of-sample mapping; fitting it on PLISM would change the transfer question.

## Confirmed results

| Method | PanNormal gain, five directions | PLISM gain, three shared directions |
| --- | ---: | ---: |
| Raw | 0 | 0 |
| Reinhard image correction | +0.0339 | -0.0041 |
| Reinhard + frequency image correction | +0.0343 | -0.0111 |
| Affine OLS | +0.0552 (95% CI 0.0480–0.0621) | -0.0882 (95% CI -0.1032–-0.0751) |
| Ridge affine | +0.0733 (95% CI 0.0664–0.0799) | -0.0620 (95% CI -0.0752–-0.0507) |
| Orthogonal Procrustes | +0.0615 (95% CI 0.0560–0.0669) | -0.0474 (95% CI -0.0551–-0.0406) |
| ComBat | +0.0591 (95% CI 0.0563–0.0618) | -0.0146 (95% CI -0.0201–-0.0094) |

The PanNormal ridge-affine minus combined-image gain is +0.0390 (paired slide-bootstrap 95% CI 0.0330–0.0448). It is a cohort-level result, not a scanner-independent rule: for AKOYA the difference is +0.0064 (95% CI -0.0029–0.0156), and for VERSA +0.0024 (95% CI -0.0035–0.0082). ComBat is worse than the combined image correction for AKOYA by -0.0383 (95% CI -0.0460–-0.0304).

Coarse tissue-type retrieval does not support an unrestricted “safe feature correction” claim. Scanner-equal macro recall is approximately 64.9% for raw AT2, 63.0% for ridge affine, and 62.8% for real targets. Ridge affine also reduces slide-embedding variance trace to approximately 91% of the real-target trace. These are descriptive content checks, not a diagnostic-task validation.

Training sample count matters. With only 20 training positions per slide, the affine OLS mean gain is -0.0461; with 40 it is +0.0552. Thus the smaller-sample failure does not establish a limitation of affine correction itself.

The PLISM failure persists in both directions: scanner-to-AT2 maps fitted on 40 PanNormal positions yield pooled gain -0.0100 (ComBat), -0.0533 (ridge affine), -0.0503 (Procrustes), and -0.0831 (affine OLS). The same scanner model is not a sufficient guarantee of transferable embedding mapping across these acquisition contexts.

Existing GT450 bidirectional learned-image results show marked directional asymmetry. Pix2Pix gain is -0.0420 for AT2-to-GT450 and -0.1214 for GT450-to-AT2; CycleGAN gain is +0.0200 and -0.2139 respectively. The paired slide-bootstrap forward-minus-reverse contrasts are +0.0794 (95% CI 0.0655–0.0936) and +0.2339 (95% CI 0.2158–0.2517). This establishes directional difficulty for these trained methods; it does not measure a scanner's physical information limit.

Reinhard does not consistently eliminate the measured low–mid frequency residual. For example, AKOYA's mean absolute low–mid residual falls from 0.574 to 0.481 while its high-band residual rises from 2.113 to 2.379, even though its paired UNI target gain is +0.1325. The controlled UNI perturbation in the manuscript also gives a nonzero low–mid response (0.0043 versus 0.0056 for high at dose 0.25). The proposed claim that stain normalization fails because low frequencies do not matter to PFM is unsupported.

## Structure-loss ablation

The five-fold AT2-to-GT450 Pix2Pix comparison used the original architecture, seeds, 200-pass budget, image-only checkpoint selection, and evaluation positions. The added condition penalized the difference between source and generated grayscale gradients (weight 100); the baseline had no explicit gradient loss. All five training, prediction, image-evaluation and UNI jobs completed successfully. Differences below are regularized minus baseline, paired over 103 held-out slides.

| Endpoint | Baseline | Gradient constraint | Paired difference (95% CI) |
| --- | ---: | ---: | ---: |
| Target SSIM | 0.6931 | 0.7111 | +0.0181 (0.0114–0.0249) |
| Source-gradient NCC | 0.6228 | 0.6502 | +0.0274 (0.0114–0.0432) |
| Target-gradient NCC | 0.4557 | 0.4781 | +0.0224 (0.0128–0.0321) |
| Invented-edge fraction | 0.00539 | 0.00421 | -0.00118 (-0.00191–-0.00048) |
| UNI target gain | -0.0420 | -0.0484 | -0.00645 (-0.0170–0.00423) |

The real source–target gradient NCC is 0.5650, so even the improved target-gradient NCC remains below the unmodified source's match. Source-edge deletion barely changes (0.0768 to 0.0757; CI for the difference includes zero), and invented edges still exceed the existing 0.001 image safety threshold. Slide-level nearest-tissue macro recall is 0.6810 baseline, 0.6833 regularized, and 0.6579 for real targets; these are descriptive values.

In three fixed positions per slide, the generated high-band amplitude relative to the 0.03–0.10 cycles/µm anchor is close to the target in both arms. The generated/target relative-high ratio is 0.962 baseline versus 0.952 regularized; their paired difference is -0.0103 (95% CI -0.0524–0.0311). The added constraint therefore improves some image and boundary metrics without a clear high-band advantage or UNI alignment gain. A classifier still distinguishes generated from real target UNI embeddings at balanced accuracy about 0.998 in both arms. This tests one explicit structure constraint; it does not establish that every unconstrained generator converges to blur.

## Manuscript placement

1. Add a short **Feature-level correction** paragraph to Methods, after image-level correction, specifying the five-fold maps, 40 training/20 evaluation positions, PLISM frozen transfer, and tissue/variance checks.
2. The short Results subsection after color–frequency correction refers to Supplementary Table S3 for the image–feature comparison. This table now includes raw and corrected UNI distances and the percentage reduction from raw. Scanner-specific examples, affine training-size sensitivity, and tissue retrieval stay in Supplementary Results.
3. Results 3.4 briefly reports that the GT450 gradient constraint improved image structure without improving UNI alignment. The remaining image, high-band, and tissue details are described in Supplementary Results without another table.
4. Discussion interprets feature correction as a more direct way to align UNI in PanNormal while retaining the content checks and PLISM result. Physical high-frequency information limits and universal scanner-ID conditional generation are not presented as established conclusions.

The color–frequency comparison moved to main Table 4. The raw scanner-distance, band-response, and scanner-specific incremental frequency tables were archived after their key results were retained in Results, Supplementary Results, or figures. The remaining supplementary tables are S1 (full scanner–tissue variance), S2 (augmentation oracle), and S3 (image versus feature UNI distance). Figure numbers still follow the previously requested Figure 1 placement in section 3.5, so they do not follow source order.

Suggested Results wording, subject to manuscript tone edit:

> 동일한 held-out 위치에서 UNI-v1 feature를 직접 교정하면 PanNormal의 paired target distance는 결합 image 교정보다 평균적으로 더 크게 줄었다. 그러나 scanner별 이득은 달랐고, PanNormal에서 학습한 feature 변환은 PLISM의 세 공통 scanner 방향에서 평균 이득을 재현하지 못했다. 따라서 feature-space alignment의 내부 이득도 획득 맥락을 넘어 일반화된다고 해석할 수 없다.

> GT450 방향에서 Pix2Pix에 원본 경계 보존 제약을 추가하면 SSIM과 source·target 경계 일치가 개선되고 새로 생긴 edge가 줄었다. 그러나 UNI target distance의 추가 개선은 확인되지 않았다. 구조 보존은 생성 영상의 충실도를 평가할 때 중요하지만, 이 제약 하나로 PFM 표현 정렬이 해결되지는 않았다.

Source outputs: `outputs/discussion_followup_2026-09-25/feature40_forward/`, `feature40_reverse/`, `feature40_external_forward/`, `feature40_external_reverse/`, `image20_compare/`, and `existing_review/`. The existing image methods were reevaluated on the exact same 20 held-out positions for direct internal comparisons. The Korean manuscript now contains the short feature comparison, the GT450 structure result, and a revised Discussion.

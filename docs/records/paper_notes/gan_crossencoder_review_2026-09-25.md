# Frozen encoder review of Pix2Pix and CycleGAN images

## Scope and input audit

This is a PanNormal-only review. Four frozen image encoders—UNI-v1, UNI2-h,
Virchow2, and H-optimus-1—evaluated the existing AT2-to-target Pix2Pix and
CycleGAN predictions. No generator was retrained or selected using an encoder.
The evaluation uses the manuscript's 103 physical slides, five scanner
directions, and fixed 20 held-out locations per slide. The paired target is the
real same-location scan. The prediction audit passed for all 515
slide–direction pairs; both methods have the same location index and source and
target images at audited sentinel locations. The evaluator also checked source
and target equality at every one of the 20 used locations before embedding.

Positive gain is raw source–target cosine distance minus generated–target
cosine distance. Each location was averaged within slide and scanner; scanner
directions then received equal weight within each slide. Confidence intervals
resample the 103 physical slides 4,000 times. Absolute distances should not be
ranked across encoders; the within-encoder paired gain is the comparison.

## Pooled paired-target results

| Encoder | Raw distance | Pix2Pix gain (95% CI) | CycleGAN gain (95% CI) |
| --- | ---: | ---: | ---: |
| UNI-v1 | 0.2137 | −0.0567 (−0.0661, −0.0471) | +0.0122 (+0.0058, +0.0184) |
| UNI2-h | 0.2795 | +0.0184 (+0.0094, +0.0278) | +0.0902 (+0.0812, +0.0990) |
| Virchow2 | 0.0928 | −0.1441 (−0.1537, −0.1347) | −0.0285 (−0.0365, −0.0217) |
| H-optimus-1 | 0.1434 | −0.0481 (−0.0575, −0.0390) | −0.0185 (−0.0253, −0.0120) |

CycleGAN gain exceeded Pix2Pix gain for all four encoders when scanner
directions were averaged, though some scanner-specific comparisons reversed.
For the same AT2-to-GT450 generated images, Pix2Pix gain was −0.0420 in UNI-v1,
+0.1070 in UNI2-h, −0.1312 in Virchow2, and +0.0467 in H-optimus-1. AKOYA
had positive CycleGAN gain in all four encoders, while VERSA generally reduced
target alignment.

The UNI-v1 reevaluation is close to manuscript Table 3: raw distance 0.2137
versus 0.2136, Pix2Pix 0.2704 versus 0.2716, and CycleGAN 0.2015 versus
0.2020. It reproduces the method directions, though it is not bitwise identical
to the earlier embedding run.

## Coarse tissue-information check

Each query is a slide-level mean embedding, matched against other physical
slides' raw AT2 means in the same encoder. The macro recall covers 36 tissue
types and excludes the singleton type. Values are descriptive, not diagnostic
task performance.

| Encoder | Real target | Pix2Pix | CycleGAN |
| --- | ---: | ---: | ---: |
| UNI-v1 | 63.3% | 60.9% | 62.2% |
| UNI2-h | 66.0% | 67.2% | 67.3% |
| Virchow2 | 63.0% | 55.8% | 61.0% |
| H-optimus-1 | 65.5% | 65.7% | 66.6% |

Virchow2's Pix2Pix target-distance loss coincided with lower tissue retrieval.
H-optimus-1 target distances increased for both methods while its coarse tissue
retrieval stayed close to the real target. Paired target alignment and tissue
retrieval therefore remain separate endpoints.

## Manuscript interpretation

The current Table 3 is a UNI-v1 benchmark, not a cross-PFM method ranking. The
cross-encoder review supports a short Results statement that the image-to-PFM
relationship depends on the encoder, with one compact supplementary gain figure
if this analysis is included. The existing image similarity metrics already
describe the generated images and need not be recalculated for each PFM.
External PLISM image outputs were not retained in the same form for this
four-encoder analysis, so this review establishes internal encoder dependence
only. It does not establish cross-encoder external transfer.

Review artifacts: `outputs/gan_encoder_review_2026-09-25/audit.json`,
`gan_target_gain_summary.csv`, `gan_tissue_retrieval.csv`, and
`gan_crossencoder_gain_review.png`. The Korean manuscript has not been edited.

# Feature correction across four pathology image encoders

## Design

The same 103 PanNormal physical slides, five AT2-to-target scanner directions,
and fixed 20 held-out locations per slide were used for UNI-v1, UNI2-h,
Virchow2, and H-optimus-1. For each outer physical-slide fold, the ridge affine
map and target-reference ComBat were fitted with 40 paired locations per
training slide. Ridge strength was chosen on a separate inner validation fold.
No tissue labels entered either fit. The trained maps were applied without
refitting to 13 PLISM sections in the three shared scanner directions. All
comparisons use same-location real target embeddings from the same encoder.
Positive target gain means lower cosine distance to that real target.

The 309 PanNormal and 39 PLISM raw feature files passed count and shape audits.
All 15 fitting jobs completed. The 15 internal outputs and nine external outputs
passed checks for full slide/section coverage, finite values, and arithmetic
identity of the reported gains. UNI-v1 features and fitted maps come from the
existing manuscript analysis. The other encoder features were extracted anew
from the same physical view and locked locations. Pix2Pix and CycleGAN results
are the prior cross-encoder evaluation of unchanged generated images on those
same held-out locations. UNI-v1 GAN embeddings were re-extracted and differ
slightly from the older manuscript table; method directions agree.

## PanNormal target gain

Each value is the mean of scanner directions within each slide, then of 103
slides. Intervals bootstrap the physical slides 4,000 times. Distances are on
different embedding spaces and should be compared only within an encoder.

| Encoder | Raw distance | Pix2Pix gain | CycleGAN gain | Ridge gain | ComBat gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| UNI-v1 | 0.2136 | −0.0567 | +0.0122 | +0.0733 (0.0664, 0.0800) | +0.0591 (0.0563, 0.0618) |
| UNI2-h | 0.2795 | +0.0184 | +0.0902 | +0.1020 (0.0879, 0.1146) | +0.0588 (0.0549, 0.0625) |
| Virchow2 | 0.0928 | −0.1441 | −0.0285 | +0.0184 (0.0145, 0.0223) | +0.0249 (0.0232, 0.0265) |
| H-optimus-1 | 0.1434 | −0.0481 | −0.0185 | +0.0314 (0.0235, 0.0390) | +0.0359 (0.0326, 0.0392) |

Both feature methods improved pooled alignment in all four encoders. This is a
within-PanNormal result, not a universal ranking. Ridge is worse than raw for
Virchow2 in VERSA and S360 and for H-optimus-1 in S360. In UNI2-h, CycleGAN
gain (+0.0902) exceeds ComBat (+0.0588); ridge exceeds CycleGAN by only
+0.0118 (slide bootstrap CI +0.0017 to +0.0213). In UNI-v1 AKOYA, CycleGAN
exceeds ridge. For the same GT450 Pix2Pix outputs, target gain changes sign
across encoders, demonstrating that an image correction has no single PFM
effect. Feature methods fit the evaluated embedding directly, whereas the GAN
checkpoints were selected without PFM feedback; their gain comparison does not
isolate an intrinsic advantage of the correction level.

The coarse slide-level tissue retrieval check is compatible with partial
content retention but does not prove it. Averaged across directions, raw,
ridge, ComBat and real-target macro recall respectively were UNI-v1
64.9/63.0/62.4/62.8%; UNI2-h 68.9/66.8/67.5/67.4%; Virchow2
60.9/62.3/62.0/61.4%; H-optimus-1 65.4/65.2/64.3/65.0%.

## Frozen transfer to PLISM

Each value averages five PanNormal-fitted maps, three common scanner directions
and 13 physical sections equally. Intervals bootstrap sections 4,000 times.

| Encoder | Ridge gain | ComBat gain |
| --- | ---: | ---: |
| UNI-v1 | −0.0620 (−0.0752, −0.0508) | −0.0146 (−0.0202, −0.0095) |
| UNI2-h | −0.0767 (−0.1087, −0.0556) | +0.0094 (+0.0034, +0.0141) |
| Virchow2 | −0.0462 (−0.0544, −0.0401) | +0.0038 (+0.0020, +0.0056) |
| H-optimus-1 | −0.0683 (−0.0864, −0.0549) | −0.0011 (−0.0077, +0.0043) |

The paired affine map failed to transfer in all four encoders. ComBat was much
closer to zero and had small positive pooled gains in UNI2-h and Virchow2, but
the direction depended on the scanner: all four ComBat encoders benefited in
GT450, while all four lost alignment in S360. Thus even the simpler feature
correction did not yield a consistently transferable scanner map.

## Frequency and color context

On the separate three-location-per-slide audit, Reinhard reduced paired target
distance in all four encoders (gain UNI-v1 +0.0345, UNI2-h +0.0326, Virchow2
+0.0163, H-optimus-1 +0.0100). Frequency matching added to Reinhard showed
mean gains of +0.0007, +0.0156, +0.0038, and −0.0011, respectively. This is
not the locked 20-location image/feature benchmark and should be reported as
a distinct exploratory check. The OD band perturbation on the same
three-location audit found high minus low–mid embedding displacement of
+0.00131, +0.00348, −0.00058, and −0.00159, respectively. OD perturbation
size was matched at the 256-pixel physical-view image before each encoder's
input resize; a model-input dose audit is needed before claiming a causal
encoder frequency preference from these cross-model contrasts.

## Manuscript implication

The robust primary claim is that scanner image phenotype and correction
effects are multidimensional, scanner/context dependent, and encoder
dependent. PanNormal paired feature correction aligns the target in all four
encoders, often more than the tested image translations, while affine maps fail
on PLISM. A blanket claim that feature correction is superior or externally
stable is contradicted by UNI2-h CycleGAN versus ComBat, individual scanner
directions, and the external results. UNI-v1's high-band response remains a
valid within-model finding. The cross-model frequency contrast can be mentioned
only as exploratory until effective input dose is audited.

Review outputs are under `outputs/feature_crossencoder_review_2026-09-25/summary/`.

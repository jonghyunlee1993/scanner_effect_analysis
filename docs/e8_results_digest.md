# E8 results digest

**Status:** post-core exploratory extension, 2026-08-06. Nothing here enters the
locked E5 five-method ranking; the locked outputs were not recomputed.

Raw material for the report rewrite. Every number carries the artifact it comes
from. Conditions follow `docs/e8_condition_registry.md`, so each was scored by
the same two tools regardless of which arm produced it.

## 0. The pipeline reproduces the locked results

The new evaluator was checked against the frozen numbers before anything new was
read from it.

| Locked value | Report | E8 rerun | Artifact |
|---|---:|---:|---|
| CORAL relative radius reduction, ResNet50 | +34.9% | +35.1% | `e8_condition_frontier` |
| Orthogonal Procrustes, ResNet50 | +43.9% | +44.3% | `e8_condition_frontier` |
| RF1U toward GT450, ResNet50 | +32.85% | +33.3% | `e8_rf1u_dual_anchor` |
| RF1U toward AT2, UNI content delta | −0.0338 | −0.0338 | `e8_rf1u_dual_anchor` |
| RF1U toward AT2, Virchow2 content delta | −0.0366 | −0.0366 | `e8_rf1u_dual_anchor` |
| RF1U toward AT2, UNI GT450 variance trace | 0.775 | 0.7752 | `e8_rf1u_dual_anchor` |
| CORAL scanner probe, CONCH | 0.101 | 0.1007 | `e8_target_probe` |

Content is anchored on the destination's own raw acquisition, the one
generalization the frozen contract allows (`analyze_rf1u_frontier.py`), and the
gate is the pooled delta's bootstrap CI lower bound, as in the locked script.
An AT2-anchored column is reported beside it: that anchor turns negative for any
correction aimed away from AT2 whether or not content was lost, so it is not
usable for comparing destinations.

## 1. The image-space ceiling is a property of the domain

Linear scanner probe, slide-blocked, chance 0.167. `e8_control_probe`.

| Intervention | ResNet50 | UNI v1 | CONCH v1 | Virchow2 |
|---|---:|---:|---:|---:|
| raw | 0.970 | 0.997 | 0.940 | 0.996 |
| HF ×2.0 sharpen | 0.952 | 0.994 | 0.934 | 0.995 |
| HF ×0.25 blur | 0.902 | 0.987 | 0.894 | 0.995 |
| **HF ×0.00 — destroys the representation** | **0.808** | **0.931** | **0.906** | **0.950** |
| **paired oracle — not deployable** | **0.941** | **0.992** | **0.928** | **0.989** |
| global HF mean, negative control | 0.806 | 0.935 | 0.905 | 0.951 |
| *CORAL, feature space* | *0.103* | *0.077* | *0.101* | *0.084* |

The paired oracle substitutes the true registered high-frequency content of
another scanner and moves the probe by 0.005–0.029. Deleting the high band
outright — the intervention that costs −0.28 to −0.74 of content margin and
collapses the UMAP — leaves it at 0.81 or above. Feature correction moves the
same probe by 0.87.

The negative control tracks the destruction control to three decimals
(0.806/0.808, 0.935/0.931, 0.905/0.906, 0.951/0.950), so 0.81–0.95 is the floor
for "an image with its detail removed", not an artefact of one intervention.

**Reading.** The ceiling is not a capacity limit of the corrections tested. An
oracle with ground-truth paired information cannot pass it either.

### 1a. Destroying content can *raise* scanner retrieval

Cosine k-NN probe, same conditions. CONCH 0.624 → **0.719** and Virchow2 0.750 →
**0.798** under complete high-frequency removal. When similarity has no tissue
left to rest on, what remains is the instrument. This is the invariance–fidelity
degeneracy in its purest form and it is visible without reference to a threshold.

## 2. The signature is redundant across image statistics

RF1U already corrects both axes — Reinhard for colour, band gains for frequency —
and still leaves the probe at 0.885 or above. The controls show why: removing all
detail leaves 0.81–0.95 because the colour base still carries the scanner, and
removing colour leaves 0.89–0.99 because detail does.

An image correction can only match the moments it explicitly models, six Lab
moments and three band powers here. A feature-space affine matches the whole
second-moment structure of the representation. The oracle result shows this is
not a question of which moments were chosen.

## 3. The moment ladder dissociates by probe

Toward GT450, fitted leave-one-physical-slide-out. `e8_moment_ladder`,
`e8_ladder_probe`, `e8_condition_frontier`.

| Rung | RR ResNet50 | RR UNI | RR CONCH | RR Virchow2 | linear | k-NN |
|---|---:|---:|---:|---:|---:|---:|
| raw | — | — | — | — | 0.970 | 0.831 |
| mean shift, first moment | +26.6% | +14.9% | +20.8% | +13.8% | 0.148 | **0.643** |
| + per-dimension scale | +34.4% | +14.2% | +21.3% | +14.2% | 0.118 | 0.545 |
| CORAL, full covariance | +35.0% | +15.8% | +24.1% | +16.7% | 0.102 | **0.276** |

All rungs safe and improved under the frozen gates.

**Reading, and it is not the obvious one.** A single per-scanner mean vector
recovers 76–94% of CORAL's radius gain and takes the linear probe from 0.970 to
0.148 — so the linearly decodable part of the scanner is a translation. But the
k-NN probe separates the rungs sharply: mean shift only reaches 0.643 where CORAL
reaches 0.276. The first moment carries the linearly decodable part; the second
moment carries the neighbourhood structure. The scanner is **not** simply a
translation, and a deployment that cares about retrieval needs the covariance.

## 4. Composition: image correction is complementary, except to the linear probe

CORAL applied on top of an image correction, all toward GT450.

| Input to CORAL | RR ResNet50 | RR UNI | RR CONCH | RR Virchow2 | linear | k-NN |
|---|---:|---:|---:|---:|---:|---:|
| raw | +35.0% | +15.8% | +24.1% | +16.7% | 0.102 | 0.276 |
| Reinhard first | +43.1% | +17.8% | +29.0% | +16.7% | 0.110 | 0.242 |
| **RF1U first** | **+45.5%** | **+18.8%** | **+31.4%** | **+18.3%** | 0.109 | **0.230** |

RF1U before CORAL is the best configuration in all four models on radius and on
k-NN, and is safe. It changes nothing on the linear probe. Image and feature
correction are complementary in geometry and in neighbourhood structure, and
redundant only in the linear direction.

## 5. Feature-space correction is destination-invariant

`e8_target_harmonization`, `e8_target_probe`, `e8_condition_frontier`. Virchow2's
three-destination probe timed out at 12 h and was resubmitted; radius and content
are complete for all four.

| | AT2 | GT450 | S60 |
|---|---:|---:|---:|
| CORAL RR, ResNet50 | +35.1% | +35.0% | +34.1% |
| CORAL linear probe, ResNet50 | 0.103 | 0.102 | 0.102 |
| Procrustes RR, ResNet50 | +44.3% | +42.9% | +45.5% |
| Procrustes linear probe, CONCH | 0.147 | 0.148 | 0.151 |

Every cell safe and improved. The destination effect that inverts the outcome in
image space (7 of 8 versus 0 of 4) does not exist in feature space. That
asymmetry is evidence for the mechanism: in pixel space a destination fixes
colour and detail power together, in feature space the correction aligns
distributions and the anchor is immaterial.

This also closes the fairness question about the report's headline contrast. The
image and feature arms were compared across different destinations, but the
feature arms give the same answer at any destination, so the comparison stands.

## 6. Destination sweep: the optimum is a real scanner, not an extremum

`e8_lambda_energy`, `e8_lambda_features`, `e8_lambda_frontier`, `e8_lambda_probe`.
A synthetic destination is GT450's colour with its post-Reinhard band power
scaled by λ, so colour is fixed and only detail power moves. λ = 1 reproduces the
real GT450 destination.

| λ | ResNet50 | UNI v1 | CONCH v1 | Virchow2 |
|---:|---:|---:|---:|---:|
| 0.5 | +30.8% | +5.75% | +17.4% | +0.84% (not improved) |
| **1.0, the real GT450** | **+33.3%** | **+6.96%** | **+20.1%** | **+4.33%** |
| 1.5 | +32.9% | +6.74% | — | — |
| 2.0 | +32.0% | +6.10% ⚠ | — | — |
| 3.0 | +29.5% | +4.54% ⚠ | +15.6% | +2.10% ⚠ |

⚠ fails the frozen fidelity gates.

λ = 1 is the maximum in all four models. Softer loses, sharper loses
monotonically, and past λ = 2 the gates start failing. The linear probe is flat
across the whole sweep (ResNet50 0.882–0.915, UNI 0.988–0.991), so no destination
choice touches the ceiling.

**Reading.** "More detail power is better" is rejected. λ = 1 is the only point in
the sweep that corresponds to a physically realisable acquisition; the others
synthesise image statistics no scanner produces. ResNet50, pretrained on natural
images, peaks in the same place as the three pathology models, which argues for an
on-manifold explanation rather than a pathology-pretraining prior.

Method saturation is recorded rather than hidden: at λ = 3.0, 6.7% of
scanner-band-fold cells hit the frozen hard cap of 4.0, and mean shrinkage α rises
from 0.84 to 0.95 across the sweep — the reliability brake loosens exactly as the
correction is pushed harder, because α divides an unchanged standard error by a
growing gain.

## 7. The scanner is a small, consistent, low-dimensional offset

Crossed balanced decomposition of L2-normalised raw embeddings.
`e8_variance_components`.

| Model | scanner | content | slide | residual |
|---|---:|---:|---:|---:|
| CONCH v1 | 3.9% | 46.2% | 42.3% | 7.6% |
| Virchow2 | 4.4% | 54.8% | 27.8% | 13.1% |
| UNI v1 | 7.3% | 52.3% | 21.7% | 18.8% |
| ResNet50 | 10.6% | 49.5% | 28.7% | 11.2% |

The scanner main effect is 4–11% of embedding variance while a linear probe names
it at 0.94–0.997. Small in energy, consistent in direction, and therefore almost
perfectly decodable — which is why radius moves little for the pathology models
while the probe stays pinned, and why a mean vector removes the linear part.

RF1U removes most of the main effect in variance terms (ResNet50 0.106 → 0.025,
CONCH 0.039 → 0.016) and the probe still reads 0.89. Fractions after correction
are relative to that condition's own total and should not be compared as if the
denominator were fixed.

## 8. The sub-chance CORAL probe is a fold artefact, and now it is diagnosed

Two checks, both in `e8_target_probe`.

- **Permutation null.** Labels permuted within fold, linear probe refitted, 20
  replicates: 0.1652–0.1676 across every condition and model. Chance is calibrated
  correctly at 0.167, so CORAL's 0.077–0.103 is genuinely below what no signal
  produces.
- **Confusion matrix.** After CORAL the diagonal is the smallest entry in almost
  every row (0.09–0.13 against off-diagonals of 0.15–0.32) and every row except
  AKOYA's predicts AKOYA. The classifier actively avoids the true class.

So the correct statement is that feature-space correction takes the linear probe
**to chance**, and the sub-chance excess is a leave-one-slide-out artefact, not
additional invariance. Reporting it as "at or below chance" overstates the result
in one direction and understates the diagnosis in the other.

Paired with §3 and §5, the honest summary is: the scanner is removed from every
linear direction but remains in neighbourhood structure at 0.20–0.34 against a
chance of 0.167.

## 9. Macenko fails the gates in three of four models

Locked supplement, `e5_macenko_supplement`.

| Model | RR | content delta | verdict |
|---|---:|---:|---|
| ResNet50 | +10.1% | −0.006 | safe and improved |
| UNI v1 | −1.1% | −0.103 | unsafe |
| CONCH v1 | −14.7% | −0.039 | unsafe |
| Virchow2 | −19.7% | −0.120 | unsafe |

The second most used stain normalizer in the field makes scanner radius worse in
three of four models and breaks content non-inferiority in the same three. This
answers "where is Macenko" and shows the frozen gates doing real work.

## 10. Acquisition provenance closes the resampling question

`e8_acquisition_provenance`, header-only over all 654 native files.

| Scanner | native MPP | reduction to 0.5052 | JPEG Q | high-band transfer |
|---|---:|---:|---|---:|
| GT450 | 0.2624 | **1.93×** | 91 | **1.478** |
| VERSA | 0.2742 | 1.84× | 75 | 0.938 |
| S60 | 0.4426 | 1.14× | — | 1.044 |
| S360 | 0.4603 | 1.10× | — | 0.842 |
| AKOYA | 0.4999 | 1.01× | 70 | **0.335** |
| AT2 | 0.5052 | 1.00× | 70 | 1.000 |

Resampling only removes high-frequency content, so if the reduction drove the
ordering the most heavily reduced scanner would be the softest. It is the
sharpest. Among the four scanners reduced by 1.15× or less the transfer still
spans 3.12×, so the render chain cannot be the driver.

Two honest caveats belong in the same table. GT450 is the only scanner encoded at
JPEG Q91 against Q70–75 elsewhere, so part of its high-band advantage may be less
compression smoothing. AKOYA's header carries a 10× objective string with a 20×
scan profile at 0.5 µm/px, which is a plausible optical account of its position.

All 654 acquisition timestamps were recovered and are kept in `scan_schedule.csv`.

## 11. Not run, and why

**Vahadane.** The frozen environment has no reference implementation
(`staintools`, `torchstain` and `spams` are all absent). A sparse-NMF stain
estimator built on scikit-learn's positive dictionary learning converges — the
stain vector agrees to three decimals between 30 iterations at stride 8 and 200
iterations at stride 2 — but converges to `H = [0.465, 0.625, 0.627]` where the
Ruifrok reference is `[0.650, 0.704, 0.286]`, because scikit-learn's objective is
not the one Vahadane specifies. It also costs 20–68 s per patch against
milliseconds for Macenko's SVD, which is 1,860 CPU-hours at cohort scale.

A home-rolled version that performed badly would document our implementation, not
the method. Macenko represents the stain-matrix family and is reported instead,
with this omission stated.

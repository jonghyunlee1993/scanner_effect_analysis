# Exp-03: Common image-space feasibility

## Claim boundary

An experiment cannot prove that a common scanner image space does not exist in
an unrestricted mathematical sense. Exp-03 can test the narrower and useful
claim:

> No method in a pre-specified, capacity-spanning family reaches a common
> scanner-invariant image space while simultaneously satisfying registered
> content preservation, valid image range, and multi-PFM consistency on
> held-out slides.

The conclusion must therefore be “no feasible common space was found under the
tested method and safety classes,” not universal non-existence.

## Feasibility gates

A candidate normalization is feasible only if its slide-bootstrap confidence
interval passes every gate:

1. **Image validity:** zero out-of-range pixels before evaluator clamps.
2. **Registered content preservation:** nuclei/edge/topology and paired
   high-frequency coherence remain inside pre-registered equivalence margins.
3. **Scanner invariance:** a grouped adversary is equivalent to chance within
   each physical lattice.
4. **Paired alignment:** same-location cross-scanner distances decrease.
5. **PFM consistency:** gates 3–4 hold for multiple pathology FMs and a
   natural-image control, not only for one representation.
6. **Target robustness:** success does not depend on selecting one favorable
   target scanner after inspecting test results.

## Experiments

### E3.1 Pairwise transform graph and path dependence

Fit every directed transform on development slides:

`AT2 ↔ GT450 ↔ VERSA ↔ Akoya` within the internal lattice and `AT2 ↔ S60`
within the independently built S60 lattice. On held-out registered locations
measure:

- direct-vs-composed error, e.g. `T_GT→V(x)` versus
  `T_AT2→V(T_GT→AT2(x))`;
- cycle error, e.g. `T_V→GT(T_GT→V(x)) - x`;
- scanner-, slide-, and frequency-specific residuals.

A simple shared canonical space predicts approximate path independence and
small cycles. Stable, content-dependent violations reject that transform
family, although they do not reject all possible spaces.

### E3.2 Target-choice sensitivity

Repeat normalization using each internal scanner as the canonical target and
using a train-only barycenter target. Lock all choices before test evaluation.
Measure whether image, morphology, and PFM conclusions change sign or rank with
the target. Strong target dependence is evidence against a unique operational
common space.

### E3.3 Capacity-spanning method family

Pre-register one representative from each class:

- identity;
- RGB/OD affine and Reinhard/Macenko;
- Fourier/low-frequency transfer;
- pointwise learned color mapping;
- paired image translation;
- one unpaired translation stress test;
- frequency-separated transform;
- exact paired-target oracle.

Compare the full Pareto frontier rather than selecting methods independently
for scanner invariance and morphology.

### E3.4 Multi-PFM intersection test

For each candidate output, run identical location-grouped probes in UNI, at
least two additional pathology FMs, and ResNet50. Report layerwise results when
available. Define the feasible set for each model and test whether their
intersection is non-empty under the same image-validity and morphology gates.

If different PFMs require incompatible image changes, a single pre-normalizer
is not operationally universal even if each PFM can be satisfied separately.

### E3.5 Feature-space positive control

On frozen raw embeddings compare:

- centering/scaling;
- CORAL whitening/recoloring;
- paired ridge or orthogonal Procrustes;
- scanner-nuisance subspace removal.

Use train-slide fitting and held-out-slide evaluation. A feature method that
passes invariance and task-preservation gates while every image method misses
the joint Pareto region supports feature-level correction as the practical
alternative.

## Data and statistics

- Expand the confirmatory analysis to the 109-slide cohort.
- Never join internal and S60-lattice tuple identifiers; all groups include
  `lattice_id`.
- Split and bootstrap at the slide level.
- Use paired estimates at registered locations and hierarchical summaries by
  slide and scanner.
- Pre-register equivalence margins for morphology and chance-equivalence
  margins for scanner probes.
- Keep the exact paired target as a positive control and destructive blur,
  clipping, and phase/noise perturbations as harmful controls.

## Decision language

- **Feasible:** at least one locked method passes every gate on held-out data.
- **PFM-specific:** per-model feasible methods exist, but their intersection is
  empty.
- **No feasible tested common space:** the simultaneous confidence region is
  empty for all pre-registered method families.
- **Inconclusive:** confidence intervals are too wide or controls fail.

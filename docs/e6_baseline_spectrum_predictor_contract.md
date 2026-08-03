# E6 secondary baseline-spectrum predictor contract

**Status:** FROZEN after the E6 variance/winner result was locked, but before any
baseline-predictor association was computed, on 2026-08-03  
**Placement:** descriptive secondary analysis; it cannot alter E5/E6 primary decisions

The storyline asks which baseline spectrum or morphology predicts correction response. No
independent morphology annotation exists, so this analysis is deliberately limited to six
pre-existing spectral/content-texture summaries. It must not be described as a morphology
model.

## Frozen predictors

All predictors come from the locked native-AA E1 population and are shared across PFMs:

1. raw AT2 mean log2 radial power in 0.10--0.30 cycles/µm;
2. raw AT2 mean log2 radial power in 0.30--0.60 cycles/µm;
3. raw AT2 mean log2 radial power in 0.60--0.90 cycles/µm;
4. across-source-scanner SD of raw log2 transfer in each of those same three bands.

No outcome-driven feature screening, interactions, polynomial terms or tissue one-hot
predictors are added.

## Prediction target and validation

- Target: E5 LOSO slide-level raw-minus-method scanner centroid RMS, separately for every
  PFM × actual method (20 cells).
- Model: ridge regression with fixed `alpha=1.0`; predictors are standardized using training
  slides only and the intercept is the training response mean.
- Validation: exact 37-fold leave-one-tissue-type-out. The null prediction is the same
  training-fold response mean. All slides of a held-out tissue leave together.
- Report tissue-balanced predictive R² relative to the cross-fitted null, absolute-error
  reduction, Pearson correlation and standardized full-population ridge coefficients.
- Uncertainty uses 5,000 hierarchical tissue→slide bootstrap replicates with seed `20260805`
  on the already cross-fitted predictions. Repeat after excluding tissue types with fewer than
  three slides.

There is no success threshold, predictor-selection claim or multiplicity-based declaration.
Negative predictive R² is retained. Association does not imply a causal optical or biological
mechanism.


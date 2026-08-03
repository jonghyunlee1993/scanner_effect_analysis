# E5 Macenko Supplement-only execution contract

**Status:** FROZEN before Macenko PFM outcome access on 2026-08-03  
**Parent decisions:** [`e4_e7_decision_record.md`](e4_e7_decision_record.md) Decision 4 and
[`e5_comparator_execution_contract.md`](e5_comparator_execution_contract.md)

Macenko is included only as the approved stain-normalization supplement. It cannot enter the
five-method primary benchmark, select a primary winner or override a failed primary PFM.

- The same 109-fold leave-one-physical-slide-out split, raw AT2 target, native-AA grid,
  model-specific FOVs and four frozen PFMs are used.
- A fixed 8-pixel lattice supplies OD samples. For each slide, AT2 stain vectors and q99 stain
  concentrations are estimated with the standard Macenko OD-plane procedure (`beta=0.15`,
  angular percentiles 1 and 99). For held-out slide `l`, the target stain vectors are the
  normalized arithmetic mean and target concentrations are the geometric mean of the other
  108 AT2 slide estimates.
- Each held-out source patch estimates its own stain vectors and q99 concentrations from its
  sampled pixels without using AT2, another location or a label. This is an inference-time
  unsupervised operation, not train-fold leakage. Negative fitted concentrations are clipped
  to zero before reconstruction.
- Invalid patches with fewer than 50 eligible OD samples or a failed finite solve are returned
  unchanged and flagged. AT2 is copied from the audited raw feature shard exactly.
- RGB clipping fraction, clipping MAE and fallback fraction are reported. The frozen E4/E5
  radius, content-margin, every-scanner collapse and slide-bootstrap rules are reused.
- Results appear in a Supplement-only table/figure. The primary five-method result lock does
  not wait for or incorporate the Macenko outcome, but the final manuscript reports it before
  PanNormal completion.


# E5-RF1 source-to-target image and joint-UMAP visualization

## Scope

This post-lock visualization compares image-level correction from each non-AT2 source
scanner to the paired AT2 acquisition at the same physical location. It does not alter the
frozen E5 or E5-RF1 estimands, method selection, or result locks.

The image panel includes methods that produce pixels:

1. raw source,
2. LOSO Reinhard,
3. paired OD affine,
4. frequency calibration,
5. LOSO Macenko as a Supplement comparator, and
6. E5-RF1, our cross-fitted Reinhard plus residual-frequency correction.

CORAL and orthogonal Procrustes are not included in the image panel because they transform
representations rather than images. They remain in the locked E5 feature-comparator result.

## Source-to-AT2 image panel

- [PNG](../outputs/e5_rf1_visual_comparison/source_to_at2_method_comparison.png)
- [PDF](../outputs/e5_rf1_visual_comparison/source_to_at2_method_comparison.pdf)
- [Per-patch metrics](../outputs/e5_rf1_visual_comparison/source_to_at2_patch_metrics.csv)
- [Artifact manifest](../outputs/e5_rf1_visual_comparison/source_to_at2_summary.json)

The five rows are the frozen, deterministic, outcome-blind examples from
`outputs/e5_rf1_visual_audit/sample_manifest.csv`: one example each for GT450, VERSA,
AKOYA, S60, and S360. Every column uses the same 256-pixel field of view at the same physical
location. Primary methods use the exact held-out-scanner LOSO parameters; RF1 uses its frozen
five-fold cross-fit.

Qualitatively, Reinhard and RF1 most consistently move the global appearance toward the paired
AT2 image. RF1 intentionally remains close to Reinhard because its added operation is a capped
residual-frequency correction rather than a new global recoloring. Frequency calibration alone
largely retains the raw stain/color offset. Macenko is variable and shows conspicuous color
excursions in the AKOYA and S60 examples.

For a limited numerical check, the mean pixel-space RGB MAE to paired AT2 across these five
selected examples was 0.1024 for raw, 0.0670 for Reinhard, 0.0618 for paired OD affine, 0.1022
for frequency calibration, 0.1115 for Macenko, and 0.0656 for RF1. These five rows are an
illustration audit, not a population estimator, so no superiority claim is based on these MAEs.

## Joint pre/post UMAP

- [All four PFMs, PNG](../outputs/e5_rf1_visual_comparison/joint_umap_all_pfms.png)
- [All four PFMs, PDF](../outputs/e5_rf1_visual_comparison/joint_umap_all_pfms.pdf)
- Individual PNGs:
  [ResNet50](../outputs/e5_rf1_visual_comparison/joint_umap_resnet50.png),
  [UNI v1](../outputs/e5_rf1_visual_comparison/joint_umap_uni_v1.png),
  [CONCH v1](../outputs/e5_rf1_visual_comparison/joint_umap_conch_v1.png), and
  [Virchow2](../outputs/e5_rf1_visual_comparison/joint_umap_virchow2.png)
- [Scanner-centroid trajectories](../outputs/e5_rf1_visual_comparison/umap_scanner_centroid_trajectories.png)
- [Coordinates](../outputs/e5_rf1_visual_comparison/joint_umap_coordinates.csv)
- [Artifact manifest](../outputs/e5_rf1_visual_comparison/joint_umap_summary.json)

UMAP is fit once per PFM across all displayed conditions, rather than refit separately for each
panel. This makes pre/post positions comparable within a PFM. The construction is:

1. L2-normalize each patch embedding;
2. average patches within each physical slide and scanner;
3. L2-normalize that slide-scanner centroid;
4. reduce to 50 principal components; and
5. fit cosine UMAP with 30 neighbors, minimum distance 0.15, and seed 20260803.

The plotted population contains 109 physical slides, six scanners, and six conditions for each
of four PFMs, for 15,696 coordinate rows. Corrected AT2 is not an independently transformed
duplicate: its exact raw-AT2 coordinate is reused. Small points are physical slides and the large
markers are scanner grand centroids. Axes are shared across conditions within a PFM but are not
comparable across PFMs.

The visual pattern agrees with the locked original-space analysis:

- frequency calibration produces little population-level movement;
- Macenko can move the representation substantially, but is fidelity-unsafe in UNI v1, CONCH
  v1, and Virchow2;
- paired OD affine is fidelity-unsafe in UNI v1 and Virchow2 despite apparent scanner movement;
- Reinhard is safe in all four PFMs; and
- RF1 remains safe in all four PFMs and gives the largest safe image-level scanner-radius
  reduction in this displayed comparison.

The title of every UMAP panel reports the locked scanner-radius reduction (`RR`) and frozen
safe/unsafe decision from the original feature space. Those locked values, not geometry or
distances measured after nonlinear UMAP projection, support the quantitative claims. RF1 RR is
+29.7% for ResNet50, +6.7% for UNI v1, +14.7% for CONCH v1, and +4.7% for Virchow2.

## Reproducibility and validation

The renderers are `src/render_e5_rf1_method_comparison.py` and
`src/render_e5_rf1_joint_umap.py`, with their SLURM entry points under `scripts/`. Both jobs
completed successfully. The image manifest passes `visual_gate_pass`; the UMAP manifest passes
`umap_gate_pass`. The UMAP coordinate audit confirms 109 observations in every
PFM-condition-scanner cell, finite coordinates throughout, and exact coordinate reuse for all
2,180 corrected-AT2 duplicate rows.

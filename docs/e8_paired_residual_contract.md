# E8 paired residual-regression execution contract

**Status:** DRAFT — not frozen. Nothing here may run against a PFM endpoint until this
document is dated, frozen and its image-only gates are declared closed.
**Parent boundary:** [`e5_reinhard_residual_frequency_contract.md`](e5_reinhard_residual_frequency_contract.md)
§ "Learned-generator boundary", which requires this document to exist and fix outer folds,
image-only model selection, frequency/structure losses and hallucination checks **before** any
PFM or tissue endpoint is opened.
**Precedes:** [`e5_rf1u_multitarget_results.md`](e5_rf1u_multitarget_results.md)

## 1. Why this condition exists

E8 is a **baseline**, not a proposed normalizer. It exists to close one specific hole in the
report's argument.

The report concludes that the image domain caps out and the headroom is in feature space. Its
evidence for the ceiling is a 7×-gain variant blocked by the no-harm gate, a slide-adaptive
estimator, and three destinations — all variants **inside the same analytic per-band scalar-gain
family**. The family has never been left.

That matters because the perfect image-space map provably exists in this cohort: it is the
target scanner's own acquisition of the same location. So "the image domain caps out" cannot be
an information-theoretic statement. It is necessarily a statement about **estimability**, and an
estimability statement is only supported by actually running a strong estimator. E8 runs one.

A second hole is structural. In the E5 frontier the paired image cell holds only
`paired_od_affine`, a 3-channel linear map that cannot represent the scanner × tissue
interaction E6 measures, while the paired feature cell holds orthogonal Procrustes, the winner
in all four PFMs. The headline contrast is therefore confounded with fit-needs and with model
capacity at the same time. E8 puts a nonlinear, spatially varying method in the paired image
cell so the contrast is matched on both axes.

Unpaired adversarial translation (CycleGAN) is **excluded and the exclusion is recorded**: this
cohort's defining asset is pixel registration, an unpaired objective discards it, and for the
ceiling question an unpaired learned method is dominated by a paired one. See §11.

## 2. Population, folds and targets

Unchanged from RF1U. Reusing them exactly is what makes E8 directly comparable.

- The 109 physical slides, six scanners `("at2", "gt450", "versa", "akoya", "s60", "s360")`,
  100 registered locations, native-AA grid at 0.5052 µm/px.
  Source store: `outputs/e0_native_aa_grid/shards/<slide_id>.h5`, dataset `rgb`, shape
  `(6, 100, 512, 512, 3)`, uint8.
- **Outer folds:** the five outcome-blind physical-slide folds from
  `SHA256("e5_rf1_fold_v1:" + slide_id)`, via `fold_assignments` in
  `src/e5_reinhard_residual_frequency.py`. No new fold scheme is introduced and no deviation
  from the locked protocol is claimed: RF1U already fits on ~87 slides under this scheme.
- **Inner validation:** for outer test fold `f`, the inner validation fold is `(f + 1) % 5`.
  Training uses the remaining three folds. Train ≈ 65 slides, inner-val ≈ 22, test ≈ 22.
  The assignment is deterministic and carries no free choice.
- **Targets:** `AT2`, `GT450`, `S60`, frozen as in RF1U. Execution order is GT450 first (the
  destination where RF1U succeeded), then AT2 (where it failed), then S60. Running the two
  poles first is deliberate: the destination question is answered before the third target is
  spent.
- For each target `T` the five non-`T` scanners are sources. `T`'s own patches are never
  rendered or re-encoded; its audited raw embeddings are copied into the condition, exactly as
  in E5 and RF1U.
- Training pairs per target: 5 sources × 109 slides × 100 locations = 54,500, of which ≈32,500
  are in any one outer training set.

## 3. What the network predicts — the target definition

This is the design decision the rest of the contract hangs on, so it is stated in full.

### 3.1 The base is RF1U, not raw

For source `s`, outer fold `f` and target `T`, the **base** image `z = B_{f,T,s}(x_s)` is the
frozen RF1U output: Reinhard toward `T` on the fold's pooled Lab moments, then the shrunk
three-band shared-OD correction with exact gamut projection. The base is computed with the
existing torch operators (`reinhard_lab`, `laplacian_pyramid`, `shared_od_multiscale`) using the
statistics already materialized under `outputs/rf1u_multitarget/energy` and
`outputs/e5_image_statistics` for that fold. It is not re-fitted here.

The network predicts a correction **on top of the base**. Switching the network off returns
RF1U bit-exactly. This preserves the property that makes the analytic method defensible and
makes "does learning add anything beyond the analytic gain?" a directly measurable quantity
rather than a comparison across two unrelated pipelines.

### 3.2 The regression target is the paired acquisition, read through band statistics

The supervision signal is `x_T(slide, location)` — the raw target-scanner acquisition of the
**same physical location**. It is never used at inference and never used to fit the base.

It is *not* consumed as a per-pixel L1 target on raw RGB. The geometry gate admits integer
shifts up to 8 px with a residual phase shift, so a per-pixel loss on raw RGB pays for
misregistration, and the cheapest way for a network to pay less is to blur. Blurring is exactly
the failure mode E4 and the AT2 destination result identify as fatal. A loss that rewards it
would produce a baseline that fails for a reason having nothing to do with the question being
asked.

Two mechanisms remove this:

1. **Precomputed residual alignment.** Before training, for every `(slide, location, source)`
   triple, the integer shift `τ* ∈ [-3, 3]²` minimizing mean-OD absolute difference between the
   base image and the target acquisition is computed once and cached to
   `outputs/e8_residual/alignment/<target>.h5`. It is image-only, outcome-blind and fixed for
   the whole study. All losses are evaluated on the shifted target. Patches whose `τ*` sits on
   the search boundary are recorded and excluded from the pixel term only.
2. **The primary loss is a local band statistic, not a pixel difference** (§5), which is robust
   to the sub-pixel residual that survives (1).

### 3.3 Two arms, with different guarantees

| Arm | Predicts | Guarantee | Role |
|---|---|---|---|
| **A** `gainfield` | per-band spatially varying log-gain `u ∈ R^{N×3×H×W}` | see §3.4 — the residual is a strictly positive pointwise rescaling of the base's own band maps, hue preserved, same shared-OD gamut projection | the deployable method; identity gain returns RF1U exactly, and a constant gain field is another RF1U-family correction applied on top |
| **B** `free` | free 3-channel OD residual `r ∈ R^{N×3×H×W}` | none beyond a per-channel OD clamp to `[0, log 256]` | the **ceiling probe**; this is pix2pix's generator, and it is the arm that earns or breaks the §12 claim |
| **C** `free_gan` | B plus a 70×70 PatchGAN discriminator | none | conditional ablation, run only if B is interesting (§9); tests whether adversarial realism buys probe reduction at fidelity cost |

Arm A answers "is the scanner × tissue interaction worth modelling?". Arm B answers "is the
image domain out of room?". They are different questions and must not be collapsed into one
model.

Arm B intentionally abandons hue preservation and the shared-OD constraint. That is the point:
a ceiling probe that inherits the analytic method's restrictions is not a ceiling probe. Its
loss of guarantees is why it is a baseline and not a proposal, and why it carries the
hallucination audit in §8.

### 3.4 What arm A actually guarantees, measured rather than asserted

The first draft of this contract claimed arm A preserves band sign pixelwise and leaves phase
untouched. Measured on the implementation, one half of that is exact and the other half is not,
and the distinction is recorded here because the guarantee is the whole reason arm A exists.

**Exact, and verified numerically** (`tests/test_e8_paired_residual.py`):

1. Wherever the gamut projection does not bind, `m_out − m_base = Σ_k (G_k − 1) ⊙ b_k` to
   floating-point tolerance — measured maximum deviation `5e-7`. The residual therefore lies
   pointwise in the span of the base's **own** band maps. No spatial pattern absent from the
   base can appear in the output: nothing is synthesized.
2. `sign(G_k ⊙ b_k) = sign(b_k)` at every pixel and band, because `G_k = exp(clip(u, ±log 4))`
   is strictly positive. No band's zero crossings move, so edge positions are preserved. Zero
   violations observed under gain fields as aggressive as `G ∈ [0.25, 4]`.
3. `G_k ≡ 1` returns the base bit-exactly, which is what makes the zero-initialised head start
   training at RF1U.

**Not exact, and weaker than RF1U:** a spatially varying gain is a multiplication in the spatial
domain and therefore a convolution in frequency, so arm A is **not** a globally zero-phase
filter the way constant-gain RF1U is. Re-extracting the Laplacian bands from the output and
comparing their sign with the base's gives 6.2 % changed pixels at `G ∈ [0.6, 2]` and 12.8 % at
`G ∈ [0.25, 4]`. This statistic is **not** a defect measure on its own — constant-gain RF1U
itself scores 1.5 % on it, so it is nonzero for the locked method too — but the increase is real
and is reported for both arms in §8 rather than claimed away.

The gamut projection binds on roughly 2 % of pixels of tissue-like input even at constant gain,
which is why `L_gamut` and the projection diagnostics apply to both arms.

## 4. Architecture

- **Backbone:** U-Net, 4 down / 4 up, base width 48, GroupNorm(8), SiLU, bilinear upsample +
  3×3 conv (no transposed convolution — checkerboard artefacts are indistinguishable from the
  high-frequency signal being measured). ≈ 4.1 M parameters. Fully convolutional.
- **Input:** the base image `z` in OD, 3 channels, concatenated with its three Laplacian bands
  and low-pass base of mean OD (4 channels) → 7 input channels. Handing the network the same
  decomposition the analytic method uses removes the need for it to rediscover it.
- **Scanner conditioning:** the source scanner index (5 values per target) enters as a learned
  16-d embedding broadcast through FiLM (per-channel scale and shift) at every decoder stage.
  One model serves all five sources of a target. Three targets × five outer folds = 15 runs per
  arm, not 75.
- **Output head:** zero-initialized final convolution, so at step 0 the model is **exactly
  RF1U** for arm A (`u = 0 ⇒ G = 1`) and **exactly RF1U** for arm B (`r = 0`). Training starts
  at the analytic solution and can only be reported as an improvement over it.
- **FOV:** trained fully convolutionally on 256 px crops and applied at each model's native FOV
  (224 / 256 / 512). Because the base already carries the FOV-specific fitted statistics, this
  is an approximation, and it is declared: the §7 image-only gate is evaluated at all three FOVs
  on inner-val, and a 512-specific model is trained only if the gate fails there.

## 5. Losses

The base transform and the network run on the full centred 256 px FOV crop, which is exactly the
image the frozen fov-256 RF1U statistics were fitted on, so the reflect padding at the border is
the same one deployment uses. Losses are then evaluated on the 250 px window that the residual
alignment leaves valid for a shift of at most 3 px. `b_k` denotes the Laplacian bands of mean OD
at σ = (1, 2, 4) px and `y` the residual-aligned target.

`ε = 1e-6`, not `RF1M_ENERGY_FLOOR`. The latter is `1e-20`, a division guard; used as a loss
floor it would let empty background — whose local band energy runs around `1e-8` — dominate a
log-ratio loss. `1e-6` sits two orders below the weakest band energy tissue carries and four
below the typical one, so background is damped and real structure is untouched.

1. **`L_band` — local band-power match (primary).** `E_k` is the mean square of band `k` over a
   non-overlapping 16 px window grid, computed for output and for `y`:
   `L_band = Σ_k mean | log(E_k^out + ε) − log(E_k^y + ε) |`.
   This is the report's own measurement, made local and made differentiable. It is
   shift-insensitive by construction, which is why it carries the primary weight.

   The band maps are cropped to the residual-aligned window *first* — the shift is defined in
   full-resolution pixels — and pooled after. A box window replaced the σ = 8 px Gaussian
   originally specified here: it estimates the same local band power, gives independent cells,
   and removes six 65-tap separable convolutions per step. The two agree to a log-space
   correlation above 0.95 (`tests/test_e8_paired_residual.py`). The Gaussian form is retained
   for the audit, where cost does not matter.
2. **`L_pix` — residual-aligned pixel term.** Mean absolute difference on mean OD (arm A) or on
   all three OD channels (arm B), evaluated at the cached `τ*`. Weight `λ_pix` small; its job is
   to anchor the low-frequency base, not to drive the fit.
3. **`L_id` — identity prior (arm A only).**
   `L_id = Σ_k mean (log G_k − log g_k^RF1U)²`, pulling the gain field toward the frozen RF1U
   scalar. As `λ_id → ∞` the arm reduces to RF1U exactly, so the prior is a principled knob
   whose limit is a known-safe method.
4. **`L_gamut`** — mean pre-projection material out-of-range fraction, thresholded at
   `MATERIAL_EXCURSION = 1/255`. Prevents the network from outsourcing its dynamic range to the
   projection step.

`L = L_band + λ_pix·L_pix + λ_id·L_id + λ_gamut·L_gamut`.

`λ_pix ∈ {0.1, 0.3}`, `λ_id ∈ {0, 0.03, 0.3}` (arm A), `λ_gamut = 10` fixed. The grid is
declared here and is the **only** hyperparameter search permitted; it is resolved on inner-val
image metrics alone (§7), on outer fold 0 only, and the winner is then applied unchanged to
folds 1–4.

**Augmentation is geometric only** — the four-element flip subgroup (identity, reverse rows,
reverse columns, both). Rotations are excluded, not for principle but for correctness: the
cached alignment shift must transform with the image, and under the flip subgroup it is exactly
`(t_y, t_x) → (±t_y, ±t_x)`, whereas a rotation also swaps the axes. Four-fold augmentation is
ample at 32,500 training pairs and the swap is one more thing to get silently wrong.

No photometric, blur, sharpen, noise or JPEG augmentation of any kind, because every one of
those perturbs the exact quantity under measurement.

## 6. Lightning structure

`torch` 2.5.1 is pinned by the PFM contract. `lightning >= 2.0` is a new dependency and its
version is recorded in the run manifest; it touches training only and never the encoding pass,
so it does not reopen the runtime-equivalence amendment.

```python
# src/e8_paired_residual.py

class PairedScannerDataModule(pl.LightningDataModule):
    """(slide, location, source) triples for one target and one outer fold.

    Reads uint8 from the E0 grid shards; returns source and target crops at the
    same coordinates plus the source index and the cached residual shift.  The
    base transform is NOT applied here — it runs on GPU inside the module so the
    frozen torch operators are reused verbatim and the CPU workers stay light.
    """
    def __init__(self, target, outer_fold, crop=320, batch_size=32, workers=8): ...
    def setup(self, stage): ...          # fold_assignments -> train/val/test slide ids
    def train_dataloader(self): ...      # shuffle, drop_last, persistent_workers
    def val_dataloader(self): ...        # inner-val fold, deterministic order
    def test_dataloader(self): ...       # outer test fold, centred FOV crop, no aug


class FrozenBase(nn.Module):
    """RF1U toward T for one outer fold.  Buffers only, no parameters, eval-only.

    Reinhard moments and shrunk band gains are loaded per (fold, target, source)
    from the materialized RF1U statistics and registered as buffers.
    """
    def forward(self, rgb01, source_index) -> dict: ...   # 'output', gamut diagnostics


class ResidualCorrector(pl.LightningModule):
    """arm in {'gainfield', 'free', 'free_gan'}."""
    def __init__(self, arm, target, outer_fold, lambdas, lr=2e-4): ...
    def forward(self, rgb01, source_index): ...           # base -> unet -> compose -> project
    def training_step(self, batch, idx): ...
    def validation_step(self, batch, idx): ...            # logs the §7 gate metrics only
    def configure_optimizers(self): ...                   # AdamW, cosine, 500-step warmup
```

Frozen `Trainer` settings:

```python
pl.seed_everything(20260803 + task_index, workers=True)
pl.Trainer(
    accelerator="gpu", devices=1,
    precision="bf16-mixed",          # NOT 16-mixed: fp16 overflows on this pipeline
    max_epochs=14, gradient_clip_val=1.0,
    deterministic=False, log_every_n_steps=50,
    callbacks=[
        ModelCheckpoint(monitor="val/band_rmse", mode="min", save_top_k=1),
        EarlyStopping(monitor="val/band_rmse", mode="min", patience=4),
    ],
)
```

Autocast is disabled around the frozen base, the composition and the losses, so every `log`,
`exp` and band energy is computed in fp32; only the U-Net runs in bf16. Mixing the two would put
band energies of order `1e-8` through a bf16 logarithm.

**Was arm B trained enough?** This is the first question a negative ceiling result has to answer,
so the evidence is recorded rather than asserted. On the grid fold both arm-B configurations ran
all 14 epochs without early stopping, which by itself would be a fair objection. The validation
curve says otherwise: `paired_mae` reached its minimum at **epoch 10** and the remaining three
epochs did not improve on it (0.04641 → 0.04643, 0.04660, 0.04664), so the run met the epoch cap
exactly one epoch before `patience = 4` would have fired, and the saved checkpoint is epoch 10,
not epoch 13. The curve is flat from epoch 6 onward — 0.0470 to 0.0464, about 1 %. Training
converged; the budget did not truncate it. Arm A early-stops at epoch 5–6, so the cap never
approached binding there.

`deterministic=False` is deliberate and is not a relaxation of anything real:
`reflection_pad2d_backward_cuda` has no deterministic implementation, so `deterministic="warn"`
never produced determinism here — it logged that fact once per run and selected slower kernels
for it. Runs are seeded, the seed is recorded in every run summary, and no endpoint in this
study depends on bitwise reproducibility of a training run; the endpoints depend on the frozen
folds, the frozen gates and the encoded features, all of which are deterministic.

Arm C uses `automatic_optimization = False` with separate generator and discriminator
optimizers; arms A and B use Lightning's automatic loop.

Each run writes `outputs/e8_residual/<arm>/<target>/fold<k>/` containing the checkpoint, the
resolved config, the inner-val gate table, the package version manifest and the git SHA.

## 7. Image-only model selection and the no-harm gate

Selection touches **no PFM embedding, no scanner radius, no content margin and no tissue
label**. It uses these inner-val quantities:

- `paired_mae` — residual-aligned mean absolute OD error against the paired target acquisition,
  over all three channels;
- `band_rmse` — RMSE between the corrected and the raw-target three-band power profile,
  computed exactly as in the RF1U image-domain audit, per source scanner and FOV;
- `projection_fraction` and `material_range_fraction` from the gamut report;
- for arm A only, `min G` and `max G` over the fold.

The checkpoint is the inner-val **`paired_mae`** minimum, and early stopping watches the same
quantity.

The **grid winner** is then the best *gate-eligible* configuration, ranked by `paired_mae`
within the eligible set, ties broken toward the larger identity weight — that is, toward RF1U.
Eligibility leads because the winner is carried into the other four outer folds, and a
configuration that cannot clear the frozen gate on the grid fold has no business being carried
anywhere. This ordering was adopted after the grid fold ran and before any PFM endpoint existed,
because ranking on `paired_mae` alone selected a configuration that beat the base overall but in
only three of five source scanners — which the gate then rejects. The margins involved make the
point: the top four configurations spanned `0.05055` to `0.05063` in `paired_mae`, a 0.14 %
spread that is effectively noise, while `sources_improved` differed 3 against 4. Selecting on
the noisy coordinate and gating on the informative one is a rule that discards its own best
candidates.

**Why `paired_mae` and not `band_rmse`, decided before any run completed.** The first draft made
`band_rmse` the selection metric, which is wrong twice over. RF1U's gains are fitted to match
pooled band power, so `band_rmse` is close to the quantity the base was optimised for and is a
weak discriminator. More seriously, arm B is free to move colour, and `band_rmse` cannot see
colour at all — the ceiling probe, which is the entire reason this condition exists, would have
been judged on a statistic blind to most of what it does. `paired_mae` is the direct measure of
agreement with the paired acquisition, is identical for both arms, and is what a paired
regression baseline should be judged by. `band_rmse` is retained as the non-inferiority leg
below so the spectral property cannot silently degrade.

> **No-harm gate.** An arm proceeds to encoding only if, in **at least 4 of 5 outer folds**, its
> selected checkpoint (a) beats the RF1U base on inner-val `paired_mae` overall and in **no
> fewer than 4 of the 5 source scanners**, (b) does not worsen `band_rmse` against the base by
> more than 5 % relative, and (c) needs no more gamut projection than the base — measured as
> `material_range_fraction` no more than 5 % above the base's own, plus `1e-4` absolute slack.

Leg (c) originally read "records `material_range_fraction = 0`", which was wrong and would have
failed every run. It confused two different quantities. What RF1U reports as exactly zero is the
**final clamp after projection**, and that is zero by construction for both E8 arms too: the
shared-OD projection already bounds the residual into the representable range, and arm B clamps
OD before conversion, so the closing `[0, 1]` clamp is a floating-point safeguard that never
fires. Pre-projection *material excursion* is a different number and is nonzero for the locked
method as well — the first validation epoch measured 6.6 % for an arm-A run. The defensible bar
is therefore relative to RF1U's own excursion, computed by the same operator on the same images,
which is what leg (c) now states.

If the gate fails, the arm is declared an image-domain no-go and **no embedding is computed** —
the same rule that closed RF1M, applied to a learned method for the same reason.

## 8. Hallucination audit — required before any encoding

Run on the outer test folds, image-only, and reported regardless of outcome.

- **Arm A (structural, expected to pass by construction).** Verify that
  `sign(G_k ⊙ b_k) == sign(b_k)` at every pixel and band and that
  `output − base − Σ_k (G_k − 1)·b_k` is zero to floating-point tolerance wherever the
  projection does not bind. Both are the §3.4 exact properties; a single violation is a defect,
  not a finding. Report the `G` distribution per band and source, and the projection fraction.
- **Both arms** additionally report the two §3.4 approximate statistics — re-extracted band sign
  change rate and phase correlation — beside the constant-gain RF1U value, so arm A's departure
  from a zero-phase filter is quantified rather than assumed away.
- **Arm B (real audit).**
  1. *Phase preservation:* per-patch correlation between the FFT phase of output mean OD and of
     base mean OD, over the band above 0.10 cyc/µm. Report the distribution.
  2. *Invented structure:* fraction of pixels where `|∇m_out| > τ` while both `|∇m_base| < τ/4`
     and `|∇m_y| < τ/4` — structure present in neither the source nor the paired target. `τ` is
     the cohort median gradient magnitude, computed once on training folds and frozen.
     Declared budget: **≤ 0.1 % of pixels**. Exceeding it does not stop the arm — arm B is a
     ceiling probe and is allowed to be unsafe — but the number is reported beside every one of
     its endpoints, and no arm-B result may be quoted without it.
- Both arms: a fixed 24-patch qualitative strip per target, rendered with the same
  ×5 difference amplification used in the report gallery.

## 9. Endpoints

Every definition, threshold, bootstrap seed `20260803` and 5,000 slide-resampled replicates are
reused **unchanged** from the E4–E7 decision record and the RF1U contract, including RF1U's
generalization of the content anchor to the condition's own target `T`.

- **Invariance:** scanner centroid RMS over the six scanners; relative reduction from raw.
- **Content:** per source scanner, mean cosine to raw `T` at the same location minus the 95th
  percentile of raw `T` unmatched self-similarity. Margin `−0.02`.
- **Collapse:** variance trace, entropy effective rank, median pairwise distance as ratios to
  raw, for **every** source scanner. Point `0.90`, CI lower `0.85`.
- **Scanner probe:** slide-blocked linear probe, balanced accuracy, chance `0.167`, reported
  beside CORAL and Procrustes. Secondary, never read alone.
- Four frozen PFMs: ResNet50, UNI v1, CONCH v1, Virchow2.

Three primary comparisons, all as paired per-slide differences with 5,000-replicate CIs:
vs **raw** (does it qualify at all), vs **Reinhard_T** (placement in the report's table), and vs
**RF1U_T** (*does learning add anything beyond the analytic gain* — the question arm A exists to
answer).

## 10. Pre-registered reading of the result

Declared before any embedding exists, because a negative result is only worth what its
pre-registration is worth.

- **The §12 ceiling claim is supported** if arm B, having passed content non-inferiority and the
  collapse gate, leaves the scanner probe **above 0.50 in at least 3 of 4 PFMs** while the
  feature-space methods reach ≤ 0.17. The report's wording then stands and gains a much stronger
  basis than the analytic-family variants currently provide.
- **The §12 ceiling claim is falsified** if arm B passes content and collapse and drops the
  probe **below 0.30 in at least 2 PFMs**. In that case §12 step 4 and §13's "fully optimized
  image correction" clause are rewritten before submission, and arm B is promoted from baseline
  to result.
- **The degeneracy reading.** If arm B lowers the probe substantially but **fails** content or
  collapse, that is neither support nor falsification: it is the invariance–fidelity degeneracy
  appearing for a third time — after the physical controls in E4 and the AT2 destination in
  RF1U — and this time in a modern learned method. It strengthens the report's central negative
  result and is written up as such.
- **Arm A's outcome is independent of all three.** If arm A is safe and improved over RF1U, the
  scanner × tissue interaction is worth modelling and the report gains a deployable extension.
  If it is not, the analytic three-number gain is confirmed as sufficient within its family.

No other reading is admissible, and no threshold in this section may be revised after an
endpoint is seen.

## 11. Boundaries and exclusions

- E8 is a post-core condition. It cannot enter the frozen five-method E5 primary ranking, cannot
  reopen the E4–E7 decision record, and cannot alter any locked artifact.
- **CycleGAN is excluded, deliberately.** An unpaired adversarial objective discards this
  cohort's pixel registration, which is its defining asset; for the ceiling question it is
  dominated by the paired arm B; and its predictable failure mode — hallucination caught by the
  content and collapse gates — would read as a strawman rather than a measurement. The
  deployable-unpaired lane is already occupied by Reinhard, Macenko and RF1U in image space and
  by CORAL in feature space. This exclusion is stated in the manuscript rather than left silent,
  because CycleGAN and pix2pix both appear in the 66-laboratory benchmark the report cites.
- Better image spectrum, lower scanner radius or a lower probe is not evidence of biological
  fidelity. This cohort carries no molecular or clinical endpoint.
- External validation on PLISM remains outstanding for E8 exactly as it does for RF1U, and is
  subject to the same interpolation audit.

## 12. Execution order

1. Cache residual alignments; materialize per-fold base statistics. CPU array, `defq`.
2. Arm A and arm B, target GT450, five outer folds. `gpuq`, 10 runs.
3. Image-only gate (§7) and hallucination audit (§8). **Encoding is blocked until both close.**
4. Encode six conditions × four PFMs, reusing the `extract_rf1u_features.py` pass.
5. Endpoints, bootstrap, report against §10.
6. Target AT2 — the destination test — then S60.
7. Arm C only if arm B moved the probe by more than 0.10 from RF1U.


## 13. Amendment 1 — scope of the outstanding PLISM external validation (2026-08-23)

§11 records external validation on PLISM as outstanding. This amendment scopes it so that the
gap is a decision rather than a vague debt. **Nothing here is executed and no endpoint is
opened.** Thresholds are written before the data is touched, as in §10.

### 13.1 What makes this test unusual

Arm B is a **ceiling probe, not a deployable method**. So the external question is not "does it
perform well elsewhere" but "does the ceiling hold in a cohort it never saw". A successful
external result is therefore the probe **staying high** while the fidelity gates pass — the
opposite of how an external test of RF1U would read. This has to be stated before any number is
seen, or the result will be reported with whichever sign flatters it.

### 13.2 The scanner overlap is the binding constraint

Arm B is scanner-conditioned on the six PanNormal instruments
(`at2, gt450, versa, akoya, s60, s360`). PLISM has seven (`AT2, GT450, S210, S360, S60,
Philips, SQ`). The intersection is **four**: `AT2, GT450, S60, S360`. `versa` and `akoya` are
PanNormal-only; `S210`, `Philips` and `SQ` have no conditioning vector.

Two routes follow, and only the first is external validation:

- **Zero-shot, four shared scanners.** Restrict to the intersection. The target `GT450` exists in
  both cohorts, so the trained destination is preserved and three sources remain
  (`AT2, S60, S360`). Scanner-probe chance becomes **0.25**, not PanNormal's 0.167 or PLISM's
  0.143 — every threshold below is stated against 0.25.
- **Few-shot, fitting new conditioning vectors for the three unseen scanners.** This is a
  different experiment and must not be reported as external validation. If run at all it is a
  separate condition with its own pre-registration.

### 13.3 What is already compatible, and what is not

Compatible: the render chain — PLISM core tiles are produced at 0.5052 µm/px, matching the E8
cache contract, at the same model-native FOV (256 px for the core panel). The interpolation and
alignment audit PLISM requires is already closed (E9 §1: 700,986 non-reference measurements,
median residual 0.091--0.113 µm, 99.57 % clearing the 1 µm gate), so the frequency claim is
licensed on this cohort.

Not compatible without a decision: **the encoder panels do not match.** Arm B's features exist
for the frozen core four (ResNet50, UNI v1, CONCH v1, Virchow2); E9 runs UNI2-h, CONCHv1.5 and
H-optimus-1. A result on the modern panel alone would confound "the ceiling does not transfer"
with "the ceiling is panel-specific". **Both panels must be extracted on the shared-scanner
subset**, which is the single largest cost in this test and the reason it has not been run
casually.

### 13.4 Endpoints and pre-registered reading

Same endpoints as §9, on the frozen PLISM evaluation grid (2,000 locations per section,
13 sections, ~26,000 locations per encoder), blocked by section rather than by physical slide —
leave-one-section-out is the LOSO analogue, giving 13 folds.

Declared now, against chance 0.25:

- **The ceiling transfers** if arm B clears content non-inferiority and the collapse gate and
  leaves the linear scanner probe **above 0.50 in at least 3 of 4 core-panel encoders**, while
  CORAL on the same embeddings reaches **≤ 0.30**. The report's §12 wording then holds in a
  cohort it never saw.
- **The ceiling is cohort-specific** if arm B passes the gates and drops the probe **below 0.35
  in at least 2 encoders**. §12 and §13 are rewritten before submission and the PanNormal result
  is restated as a property of that cohort's six instruments.
- **Degeneracy reading.** Arm B lowering the probe while *failing* content or collapse is
  neither: it is the invariance–fidelity degeneracy again, and is written up as such.
- **The nonlinear probe is primary here, not secondary.** The MLP result on PanNormal
  (Procrustes 0.614--0.778 against CORAL 0.074--0.269) showed the linear probe alone can rank two
  feature methods as equivalent when they are not. Both probes are reported for every condition,
  and no ceiling claim rests on the linear probe alone.

### 13.5 Cost, and why it is not free

Correcting three sources toward `GT450` over the frozen grid is ~78,000 tiles per encoder panel,
plus the target's own ~26,000 for reference — rendered, corrected through arm B on GPU, and
encoded twice (core panel and E9 panel). The fused read → render → correct → encode pattern
established for the E9 conditions applies directly and avoids materializing intermediates.

**Recommendation.** Run it only after a labeled downstream endpoint is in hand, or alongside it.
The ceiling result is already pre-registered and supported on 109 slides; a second cohort
strengthens it, but the gap that changes what this work *means* is the missing biological
endpoint, not the missing second cohort.

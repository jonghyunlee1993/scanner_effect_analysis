# E8 condition registry

**Status:** interface frozen 2026-08-05. Post-core exploratory extension; nothing
here enters the locked E5 five-method ranking.

Any new correction arm — a learned residual model, another stain normalizer, a
synthetic destination — needs the same six numbers as every arm already
measured: scanner radius, content margin, three collapse statistics and a
scanner probe. Before this registry each arm carried its own copy of the loading
and scoring code, so adding one meant rewriting the evaluation. This document
defines what an arm must produce; everything downstream is then shared.

## The contract

An arm writes **features**, one HDF5 shard per physical slide:

```
<root>/<encoder_id>/shards/<slide_id>.h5
    condition   (C,)              utf-8 names, one per variant the arm produces
    features    (C, 6, 100, D)    condition, scanner, location, dimension
```

- `encoder_id` is the frozen name: `resnet50`, `uni_v1`, `conch_v1`, `virchow2`.
- Scanner order is `at2, gt450, versa, akoya, s60, s360` — the frozen `SCANNERS`.
- Location order is the shard's own, matching `outputs/e0_pfm_features`.
- `D` must equal the encoder's `feature_dim` in the PFM contract.
- 109 shards, one per physical slide, named by slide id.
- The destination scanner's own row should pass through uncorrected, so its
  radius contribution is the untouched acquisition.

Optional but carried through when present: `scanner`, `location_id`,
`replicate_id`, `canonical_center_x`, `canonical_center_y`, and attributes
`analysis`, `encoder_id`, `slide_id`, `target`.

An arm that corrects **images** rather than features does not need its own
encoder. Write corrected patches in the layout of
`outputs/e0_native_aa_grid/shards` and encode them through
`src/extract_rf1u_features.py`, which accepts any destination name whose
band-energy file passes its gate; the features then land in this layout
automatically.

## What you get for free

```bash
# invariance and fidelity, slide-blocked bootstrap, frozen gates
python src/analyze_e8_condition_frontier.py --encoder-index N \
    --source myarm=outputs/e8_myarm --output outputs/e8_myarm_frontier

# scanner probe, three estimators, same folds and stride as the locked probe
python src/analyze_e8_condition_probe.py --encoder-index N \
    --source myarm=outputs/e8_myarm --output outputs/e8_myarm_probe
```

Both take `--source LABEL=ROOT` repeatably and report every condition found
under every root, always beside `raw`. Conditions are reported as
`LABEL:condition`, so two arms can share a condition name without colliding.

SLURM wrappers take the same arguments and fan one array task per encoder:
`scripts/e8_condition_frontier.sbatch`, `scripts/e8_condition_probe.sbatch`.

## Reading rules that come with it

The probe is **invariance-only and degenerate read alone**: a constant
representation reaches chance while destroying every patch. The frontier is what
distinguishes alignment from destruction, and an arm is described as safe only
when it clears content non-inferiority and per-scanner collapse — every source
scanner, not the average.

Report the linear probe next to the k-NN probe. The measured gap is large and
systematic: feature-space affine corrections drive the linear probe to 0.08–0.16
while cosine k-NN still reads 0.20–0.34 against a chance level of 0.167. A
linear-only claim of "at chance" is not supported by the k-NN column, and a
correction fitted in the same function class the probe uses will always flatter
itself.

## Arms currently registered

| Root | Conditions | Owner |
|---|---|---|
| `outputs/e0_pfm_features` | `raw` (implicit in both tools) | E0 locked |
| `outputs/e4_control_features` | 9 physical controls incl. the paired oracle | E4 locked |
| `outputs/e5_feature_harmonization` | `coral`, `orthogonal_procrustes` toward AT2 | E5 locked |
| `outputs/rf1u_multitarget/features/<target>` | `reinhard_<t>`, `reinhard_unpaired_band_<t>` | E5 locked |
| `outputs/e8_target_harmonization/<target>` | same two, toward GT450 and S60 | E8 |
| `outputs/e8_moment_ladder/<label>` | `mean_shift`, `diag_scale`, `coral` | E8 |
| `outputs/e8_lambda_features/<target>` | swept synthetic destinations | E8 |
| `outputs/e8_residual/features` | learned paired residual | E8, separate session |

# Superseded records, removed 2026-08-24

**Recovery:** `checkpoints/prenorm_postcleanup_20260824.bundle` records the complete history
including every file below. Clone it to a separate path to read one; do not restore anything
here into the active tree, per the rule in [`README.md`](README.md).

This follows the 2026-08-03 cleanup, whose own bundle is
`checkpoints/prenorm_postcleanup_20260803.bundle`.

## What was removed, and where its content now lives

### `presentations/pannormal_plism_2026-08-10/scanner_batch_effect.html` (14 MB)

An earlier build of the same document as
`presentations/pannormal_plism_2026-08-23/index.html` — same title, same sections 1–19.
It is not merely older: it carries four claims the current build corrects, so leaving both in
the tree meant a reader could open the superseded one and not know.

| Claim in the 08-10 build | Status |
|---|---|
| feature correction reaches "at or below the chance level" | **wrong** — the sub-chance excess is a leave-one-slide-out fold artefact; the correct statement is "to chance" |
| the ordering ends on "Procrustes above CORAL" | **not established** — every PLISM probe is a linear logistic regression, which cannot adjudicate the two feature methods |
| PLISM proposed as future work, "before PLISM is opened" | **done** — Part IV of the same document reports it |
| a learned residual "would need its own frozen contract" | **done** — E8, contract frozen and reading pre-registered |

Superseded by the 2026-08-23 build in full.

### `presentations/plism_registration_2026-08-08/index.html` (5.5 MB)

*"Aligning PLISM: what the seven scanners agree on, and where this cohort differs from ours."*
Alignment QC for the **sparse** PLISM arm: registration quality by scanner, the fitted scale as
an independent MPP audit, per-section and per-staining-condition breakdowns, and a comparison
against the PanNormal cohort.

Superseded by the core-grid rebuild (Amendment 3), which re-derived the alignment on a
patch-pitch lattice. The numbers that replace it are in
[`e9_plism_core_results.md`](e9_plism_core_results.md) §1: 700,986 non-reference measurements,
median residual 0.091–0.113 µm per scanner, 99.57 % clearing the 1 µm gate, the Hamamatsu pitch
recovered independently on every section at 0.2188–0.2219 µm/unit, rotation within 3.25° of
zero, mask Dice 0.82–0.86.

The locked sparse-arm outputs it was built from are untouched and still on disk under their
original names; Amendment 3 writes the core-grid results beside them rather than over them.
Its generator, `src/build_plism_registration_report.py`, is kept and still defaults to this
path, so the artifact can be regenerated on demand — which is part of why removing the rendered
copy costs nothing.

### `presentations/scanner_drift_2026-07-28/index.html` (128 KB)

*"The Myth of the Standard Image."* An early framing talk, pre-E0. Kept here as a record
because **two of its central assertions were later refuted by this study's own results**, and
that is worth knowing rather than losing:

- *"Shared biology, detachable scanner style."* The scanner effect is **not** a detachable
  global style. Its magnitude is a scanner × content response — the high-band tissue variance
  fraction ranges from 3.3 % to 31.3 % and differs by instrument, and a single scalar
  blur–sharpen family was rejected for all five scanners. This is section 3.2 of the current
  report, listed there as a central negative result.
- *"Feature-space correction does not scale gracefully."* It does. The apparent failure on
  high-dimensional encoders was an **estimation artefact**, not a property of the method:
  refitting CORAL at growing sample sizes puts break-even at 6.5 samples per dimension for the
  768-dimensional encoder and 32.5 for both 1536-dimensional ones, so the requirement is not a
  constant per parameter and has to be measured per encoder.

Its remaining framing — registration-supervised canonicalization, normalize once and reuse
across models — belongs to the pre-normalization design strand, not to this manuscript.

### `docs/2026-08-03_rf1_improvement_handoff.md` (502 lines)

A coordination snapshot: current results, live SLURM jobs, next execution order, cleanup
history. Transient by construction, and every durable statement in it has since been absorbed
into [`storyline.md`](storyline.md), [`final_study_protocol.md`](final_study_protocol.md) and
the locked RF1/RF1U result documents.

### Unreachable modules

Nine modules in `src/` that no script invokes, no test imports, no other module imports, and
whose outputs no document cites. Reachability was computed from the import graph rooted at
every sbatch entry point, test and document citation, then intersected with a check on the
`outputs/` directories each module writes — because documents cite results by output directory,
not by module name, and a name-only sweep would have deleted live code.

`analyze_exp06_nps_interaction`, `analyze_plism_feature_correction`, `build_e8_transfer_figure`,
`build_exp07_background_manifest`, `build_plism_features_readme`, `precompute_stardist_masks`,
`smoke_e8_paired_residual`, `verify_store`, `viz_store_inputs`.

## What was deliberately kept

- **Every locked-result and contract document.** These are the reproduction record for the
  frozen E0–E7 results and the post-core extensions; superseded *analysis* does not make a
  locked *result* disposable.
- **The `exp0*`-named modules.** The names are legacy but the code is not: `run_exp05_spectral_pilot`
  alone is imported by the E0, E1, E3 and E0d analysis chains. Deleting by name prefix would
  have broken reproduction of the locked results.
- **`presentations/rf1u_multitarget_2026-08-04/index.html`.** It is the source
  `build_pannormal_plism_report.py` reads, so it cannot be removed. It is no longer a faithful
  record of what was presented on 2026-08-04 — it has been edited since, and was already
  modified before this cleanup. Splitting the living source from the frozen presentation is
  outstanding.

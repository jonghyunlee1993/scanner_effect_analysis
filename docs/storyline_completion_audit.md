# PanNormal storyline completion audit

**Status:** PanNormal core E0--E7 complete and independently result-locked on 2026-08-03.
Post-core E8 and E9 executed 2026-08-06 and 2026-08-22/23; they are audited here under their
own contracts and touch no core lock.  
**Scope:** requirement-by-requirement evidence for [`storyline.md`](storyline.md)  
**Top-level lock:** `outputs/pannormal_core_results_lock/summary.json`

## Experimental requirements

| Requirement | Authoritative evidence | Observed gate | Verdict |
|---|---|---|---|
| E0a common physical grid | `outputs/e0_native_geometry_final/summary.json` | 65,400/65,400 scanner-location rows, 654/654 cells, 10,900/10,900 six-scanner tuples | Complete |
| E0b registration recovery | native-geometry final audit and route provenance | outcome-blind preserved/targeted VALIS comparison; final native bounds and identity pass | Complete |
| Constant/affine slide-prior diagnostic | `outputs/e0_registration_cohort_109/prior_fits.csv` | 2,725 five-fold scanner fits; 1,544 affine and 1,181 constant selections | Complete |
| E0c resampling/alias validity | `outputs/e0_alias_audit_final_geometry/summary.json` | explicit-AA q05/q50/q95 gate passed on final transforms | Complete |
| E0d anchor/background sensitivity | E0--E3 lock and same-chain background outputs | 65,400 glass patches; 0 negative bins among 47,088; high-band ERT median change ≤0.0039 log2 | Complete |
| E1 paired scanner spectrum | `outputs/e0_e3_locked_results/summary.json` | 109-slide locked spectrum and CI rows | Complete |
| E2 tissue/slide decomposition | same E0--E3 lock | 37-tissue nested REML and sampling decomposition | Complete |
| E3 scalar reducibility | same E0--E3 lock | five scanner LOSO residual CIs above zero | Complete |
| Four-PFM extraction | `outputs/e0_pfm_features/audit/summary.json` | 436/436 shards and 261,600/261,600 embeddings | Complete |
| E4 control-bounded frontier | `outputs/e4_control_results_lock/summary.json` | 2,354,400 embeddings; destructive and paired-oracle controls; Figure 4 | Complete |
| E5 actual comparator frontier | `outputs/e5_comparator_results_lock/summary.json` | 1,308,000 embeddings; five-method LOSO grid; Figure 5 | Complete |
| Macenko supplement | `outputs/e5_macenko_supplement_lock/summary.json` | 261,600 embeddings and four-PFM supplement result | Complete |
| E6 correction heterogeneity | `outputs/e6_heterogeneity_results_lock/summary.json` | 240 converged full/sensitivity fits; radius 18/20 and content 71/100 full gates | Complete |
| E6 baseline-spectrum predictor | `outputs/e6_baseline_spectrum_predictor_lock/summary.json` | exact tissue-held-out ridge; all 20 cells reported | Complete |
| E6 unseen-tissue transfer | `outputs/e6_loto_results_lock/summary.json` | exact 37-fold LOTO, 1,308,000 embeddings, 0/20 decision changes | Complete |
| E7 coarse tissue evidence | `outputs/e7_tissue_probe_results_lock/summary.json` | 36 tissues/108 slides, 5,000 hierarchical bootstrap, full/minimum-three results, Figure 6 | Complete |

## Post-core extensions

These are not E0--E7 requirements. They are audited to the same standard because two of them
carry claims the manuscript now rests on.

| Requirement | Authoritative evidence | Observed gate | Verdict |
|---|---|---|---|
| E8 pre-registered ceiling reading | `outputs/e8_residual/reading/gt450/summary.json` | verdict `supported`; arm B probe floor 0.759 against a §10 support floor of 0.50 and a falsification ceiling of 0.30 | Complete |
| E8 image-only arm gate | `outputs/e8_residual/audit/gt450/summary.json` | arm A 3/5 folds, gate needs 4 — encoding blocked; arm B 5/5 folds, all 5 sources improved | Complete |
| E8 arm B feature population | `outputs/e8_residual/features/gt450/*/manifest.json` | 436 shards, 261,600 embeddings, 109/109 slides encoded, 0 skipped, 6 scanners | Complete |
| E8 frontier and probe | `outputs/e8_residual/frontier/gt450/`, `outputs/e8_residual/probe/` | arm B safe + improved 4/4; linear/MLP/k-NN probes on frozen RF1 slide folds | Complete |
| E8 hallucination audit | same audit summary | both arms inside the 1e-3 invented-fraction budget; arm A exact properties hold; arm B phase correlation 0.694 recorded as a caveat, not hidden | Complete |
| E8 arm B external validation | — | not run; named as the remaining gap in contract §11 | **Outstanding** |
| E9 alignment/interpolation audit | `outputs/plism_core_registration`, `e9_plism_core_results.md` §1 | 700,986 non-reference measurements, median residual 0.091--0.113 µm, 99.57% clearing the 1 µm gate, mask Dice 0.82--0.86 | Complete |
| E9 core-grid cohort | `outputs/plism_core_*` | 116,831 locations, 817,817 scanner-to-reference measurements, 88.2% tissue coverage, 46 named tissues, 13 stain × 7 scanner | Complete |
| E9 condition integrity | `scripts/_verify_conditions.py` | 2,177,672 condition vectors checked against the checkpoint contract, finiteness and the frozen evaluation grid; PROBLEMS 0 | Complete |
| E9 external replication | `e9_plism_core_results.md`, report Part IV | 5 of 8 claims replicated; 2 corrections to the previous edition recorded rather than silently swapped | Complete |
| E9 core-panel reproduction | — | E9 uses a 2026 encoder panel (UNI2-h, CONCHv1.5, H-optimus-1), not the frozen core four; reproduction on the core panel is not run | **Outstanding** |

## Cross-cutting contracts

| Contract | Evidence | Verdict |
|---|---|---|
| Outcome-before-threshold separation | frozen `e4_e7_decision_record.md` and downstream stored hashes | Satisfied |
| Core PFM panel | ResNet50, UNI v1, CONCH v1 and Virchow2 checkpoint/preprocessing manifest | Satisfied |
| Same biological centers with model-native FOV | raw feature-population audit and location identity | Satisfied |
| Physical-slide inference | E4/E5 slide bootstrap; E6 nested slide model; E7 tissue→slide bootstrap | Satisfied |
| Fidelity before invariance ranking | E4/E5 result locks recompute content/collapse and safe+improved decisions | Satisfied |
| No post-outcome comparator expansion | primary methods match the frozen decision record; Macenko remains Supplement-only | Satisfied |
| Limited biology claim | only tissue type is used; no nucleus, morphology, biological non-inferiority or clinical claim | Satisfied |
| PLISM separation | deferred until after the core locks, then executed as E9 on outputs written beside the locked ones (`plism_core_*`) rather than over them; no core lock recomputed | Satisfied |
| E8/E9 pre-registration | E8's support/falsification thresholds fixed in contract §10 before any embedding existed and unrevised after; E9's estimator, bands, thresholds and bootstrap unchanged under Amendment 3 | Satisfied |
| Post-core claims kept out of the primary ranking | neither E8 nor E9 enters the frozen five-method E5 ranking; both are labelled post-core wherever quoted | Satisfied |

## Figures, reproducibility and current boundaries

Main Figure 1--6 exist as PNG/PDF and are covered by the top-level audit. That audit verified
eight result locks, six main figures and 130 unique artifacts totaling 828,758,274 bytes.
The final full test run (`16056716`) passed 130 tests; the only messages were 14 existing
Matplotlib/PyParsing deprecation warnings. Markdown local-link checks found zero missing links,
and the manuscript DOCX contains all 11 referenced main/supplementary images.

The scientific core has no remaining planned experiment. Three post-core gaps are open and
named above rather than left implicit: E8 arm B has no external validation, E9 has no
core-panel reproduction, and no labeled downstream endpoint exists in either cohort — the last
being the one that would move the work from representation geometry to changed decisions. Items still requiring author or
institutional input are author order/affiliations, IRB wording, funding, conflicts, CRediT,
restricted-data access language, public repository/release identifiers and scanner fields not
present in native headers. Journal-portal formatting must be rechecked at submission. Additional
PFMs, independent labeled outcomes and PLISM are post-core extensions rather than missing E0--E7
analyses.

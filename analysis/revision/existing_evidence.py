#!/usr/bin/env python3
"""RV06: consolidate existing, not-yet-reported results with verified design and provenance.

No analysis is re-run. The script reads frozen outputs, checks their integrity (recorded
hashes), their design against the locked cohort, folds and location sets, and re-derives
each reported summary from the frozen per-unit file it was aggregated from (same seeds and
estimators) to confirm the summary is internally consistent. It writes

  results/existing_evidence/existing_evidence.csv   one row per reported value
  results/existing_evidence/verification_checks.csv one row per check (pass/fail/note)

Sources: (a) symmetric low-pass filtering, (b) PLISM Pix2Pix/CycleGAN in four PFMs,
(c) feature correction scanner-to-AT2 fitted on 40 locations, and (d) the GT450<->AT2
bidirectional GAN runs. Code identity with the run-time byte code is established by
`restore_code_identity.py` (results/provenance_restoration/code_identity_cp*.csv); the
summary cites it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent / "results/existing_evidence"
STUDY = PROJECT / "outputs/scanner_batch_effect_analysis_2026-09-17"
COHORT = STUDY / "00_contract/cohort.csv"
FOLDS = PROJECT / "outputs/augmentation_ood_v1/00_contract/slide_folds.csv"
SET20 = PROJECT / "analysis/revision/results/location_sets/set20.csv"
SET40 = PROJECT / "analysis/revision/results/location_sets/set40.csv"
MATCHED = STUDY / "12_manuscript_completion/04_frequency_mechanism/02_matched_information_removal"
FREQ_CONTRACT = STUDY / "12_manuscript_completion/04_frequency_mechanism/00_contract"
GAN_PFM = PROJECT / "outputs/plism_gan_crossencoder_2026-09-25"
PLISM_LEARNED = STUDY / "12_manuscript_completion/06_plism_learned_external"
FOLLOWUP = PROJECT / "outputs/discussion_followup_2026-09-25"
UNI_SHARDS = STUDY / "03_uni/shards"
PIX2PIX_BI = STUDY / "06_learned_baselines/09_bidirectional_full_training"
CYCLEGAN_BI = STUDY / "06_learned_baselines/11_cyclegan_full_training"
TABLE2 = PROJECT / "analysis/paper/table2_baseline_summary.csv"
TARGETS = ("versa", "akoya", "gt450", "s360", "s60")

rows: list[dict] = []
checks: list[dict] = []


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(PROJECT))


def check(item: str, check_id: str, description: str, passed: bool | None, detail: str = "") -> None:
    status = "note" if passed is None else ("pass" if passed else "FAIL")
    checks.append({"item": item, "check_id": check_id, "description": description,
                   "status": status, "detail": detail})


def add(item: str, source: Path, script: str, design: str, *, dataset: str, pfm: str,
        direction: str, scanner: str, method: str, metric: str, value, ci_low=None,
        ci_high=None, n_units=None, unit: str = "", bootstrap: str = "") -> None:
    rows.append({
        "item": item, "source_path": rel(source), "producer_script": script,
        "design_notes": design, "dataset": dataset, "pfm": pfm, "direction": direction,
        "scanner": scanner, "method_or_condition": method, "metric": metric,
        "value": None if value is None else float(value),
        "ci_low": None if ci_low is None or pd.isna(ci_low) else float(ci_low),
        "ci_high": None if ci_high is None or pd.isna(ci_high) else float(ci_high),
        "n_units": n_units, "inference_unit": unit, "bootstrap": bootstrap,
    })


def max_abs(a, b) -> float:
    return float(np.nanmax(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def locked_design() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    cohort = pd.read_csv(COHORT, dtype={"slide_id": str})
    folds = pd.read_csv(FOLDS, dtype={"slide_id": str}).set_index("slide_id")["fold"]
    set20 = pd.read_csv(SET20, dtype={"slide_id": str})
    set40 = pd.read_csv(SET40, dtype={"slide_id": str})
    return cohort, folds, set20, set40


# ---------------------------------------------------------------------------
# (a) Symmetric low-pass filtering
# ---------------------------------------------------------------------------

def item_a(cohort: pd.DataFrame, folds: pd.Series, set20: pd.DataFrame) -> None:
    import manuscript_completion.frequency_aggregate as fa

    item = "a_symmetric_lowpass"
    manifest = json.loads((MATCHED / "manifest.json").read_text())
    for name, expected in manifest["outputs"].items():
        check(item, f"a_output_hash:{name}", "output hash recorded in manifest.json",
              sha256(MATCHED / name) == expected)
    contract = MATCHED / "matched_filter_contract_v2.json"
    check(item, "a_contract", "frozen v2 contract hash and status",
          sha256(contract) == manifest["contract_sha256"]
          and json.loads(contract.read_text())["status"] == "frozen_before_uni_outcomes",
          "status frozen_before_uni_outcomes; supersedes v1 before any UNI outcome")
    implementation = FREQ_CONTRACT / "frequency_implementation_manifest_v3.json"
    check(item, "a_impl_manifest", "implementation manifest v3 hash as recorded",
          sha256(implementation) == manifest["implementation_manifest_sha256"])
    impl = json.loads(implementation.read_text())
    differing = [f for f, h in impl["files"].items()
                 if not (PROJECT / f).is_file() or sha256(PROJECT / f) != h]
    check(item, "a_impl_code_current", "current code equals the v3 code that ran (22827864/22827866)",
          None, "differs now: " + ", ".join(differing) + ". frequency_aggregate.py and "
          "perturbation.py were changed on 2026-09-18 22:18-22:23 EDT (manifests v4/v5, "
          "'aggregation only' bugfixes for the band-sensitivity REML scopes) after the matched "
          "aggregate finished at 18:43 EDT; the v3 sources are not on disk. A cp313 byte-code "
          "cache compiled from the 2026-09-17 23:56 EDT sources (5 min before the v3 freeze) differs "
          "from the current files only in aggregate_sensitivity and verify_implementation_manifest "
          "(the documented v4/v5 fixes; the latter only adds an accepted manifest status); "
          "aggregate_matched and its estimation helpers are byte-code identical "
          "(restore_code_identity.py). Summary re-derivation below uses the current functions.")

    # Shards: integrity, cohort, folds, locations = set20.
    s20 = set20.groupby("slide_id")
    bad_hash, bad_design, bad_locations = [], [], []
    for summary_path in sorted((MATCHED / "shards").glob("*.summary.json")):
        summary = json.loads(summary_path.read_text())
        shard = summary_path.with_name(summary_path.name.replace(".summary.json", ".h5"))
        if sha256(shard) != summary["output_sha256"]:
            bad_hash.append(shard.name)
        with h5py.File(shard, "r") as store:
            slide = str(store.attrs["slide_id"])
            fold = int(store.attrs["fold"])
            location = np.asarray(store["location_index"], dtype=int)
            source = np.asarray(store["source_index"], dtype=int)
            replicate = np.asarray(store["replicate_id"], dtype=int)
        if slide not in folds.index or fold != int(folds[slide]):
            bad_design.append(slide)
            continue
        expected = s20.get_group(slide).sort_values("location_index")
        order = np.argsort(location)
        if not (np.array_equal(location[order], expected.location_index.to_numpy())
                and np.array_equal(source[order], expected.source_index.to_numpy())
                and np.array_equal(replicate[order], expected.replicate_group.to_numpy())):
            bad_locations.append(slide)
    n_shards = len(list((MATCHED / "shards").glob("*.h5")))
    check(item, "a_shard_hash", "103 shard hashes equal their summary records",
          n_shards == 103 and not bad_hash, f"shards={n_shards}; mismatched={bad_hash}")
    check(item, "a_shard_folds", "shard fold attributes equal locked slide_folds.csv",
          not bad_design, f"mismatched={bad_design}")
    check(item, "a_shard_set20", "shard locations (index, source index, replicate) equal set20",
          not bad_locations, f"mismatched={bad_locations}; 20 per slide, 2,060 total")

    slide = pd.read_csv(MATCHED / "matched_filter_slide_summary.csv", dtype={"slide_id": str})
    check(item, "a_slide_summary_design", "103 slides x 5 target scanners x 9 conditions; folds locked",
          set(slide.slide_id) == set(cohort.slide_id) and len(slide) == 103 * 5 * 9
          and (slide.fold == slide.slide_id.map(folds)).all(),
          f"rows={len(slide)}; tissue types={slide.tissue_type.nunique()}")
    probes = pd.read_csv(MATCHED / "information_probe_predictions.csv.gz",
                         dtype={"slide_id": str, "truth": str, "predicted": str})
    check(item, "a_probe_folds", "probe outer folds equal locked slide folds",
          bool((probes.fold == probes.slide_id.map(folds)).all()),
          "slide x scanner mean embeddings (618 rows); 5 outer folds; inner C in {0.01,0.1,1}")
    tissue_rows = probes[(probes.probe == "tissue_type") & (probes.condition == "raw")]
    excluded = sorted(set(cohort.slide_id) - set(tissue_rows.slide_id))
    check(item, "a_probe_tissue_scope", "tissue probe scope", None,
          f"{tissue_rows.slide_id.nunique()} slides, {tissue_rows.truth.nunique()} tissue types; "
          f"excluded (class unseen in training folds): {excluded}")

    # Re-derive the distance and probe summaries with the frozen estimators and seeds.
    retention = pd.read_csv(MATCHED / "information_retention_summary.csv")
    conditions = manifest["conditions"]
    pooled = slide.groupby(["slide_id", "condition"], as_index=False)["scanner_cosine_distance"].mean()
    worst = 0.0
    for condition in conditions:
        subset = pooled[pooled.condition == condition]
        est = fa._bootstrap_mean(subset, "scanner_cosine_distance",
                                 fa.BOOTSTRAP_SEED + conditions.index(condition))
        delta = (0.0, 0.0, 0.0) if condition == "raw" else fa._paired_bootstrap(
            pooled, condition, "scanner_cosine_distance",
            fa.BOOTSTRAP_SEED + 100 + conditions.index(condition))
        frozen = retention[(retention.condition == condition)
                           & (retention.metric == "pooled_scanner_cosine_distance")].iloc[0]
        worst = max(worst, max_abs(est + delta, frozen[["estimate", "ci_low", "ci_high", "delta_from_raw",
                                                         "delta_ci_low", "delta_ci_high"]]))
    check(item, "a_rederive_distance", "pooled scanner distance, CIs and paired deltas re-derived "
          "from matched_filter_slide_summary.csv (20,000 slide bootstraps, frozen seeds)",
          worst < 1e-12, f"max |difference| = {worst:.3g}")
    worst = 0.0
    for condition in conditions:
        for probe in ("scanner", "tissue_type"):
            est = fa._balanced_accuracy_inference(
                probes, probe, condition, fa._stable_seed(fa.BOOTSTRAP_SEED, "probe", probe, condition))
            frozen = retention[(retention.condition == condition)
                               & (retention.metric == f"{probe}_balanced_accuracy")].iloc[0]
            worst = max(worst, max_abs([est[k] for k in ("estimate", "ci_low", "ci_high", "delta_from_raw",
                                                         "delta_ci_low", "delta_ci_high")],
                                       frozen[["estimate", "ci_low", "ci_high", "delta_from_raw",
                                               "delta_ci_low", "delta_ci_high"]]))
    check(item, "a_rederive_probes", "scanner and tissue balanced accuracy, CIs and deltas re-derived "
          "from information_probe_predictions.csv.gz", worst < 1e-12, f"max |difference| = {worst:.3g}")

    reported = {("raw", "pooled_scanner_cosine_distance", "estimate"): 0.2136,
                ("raw", "tissue_type_balanced_accuracy", "estimate"): 0.671,
                ("raw", "scanner_balanced_accuracy", "estimate"): 0.997,
                ("lowpass_0p18", "pooled_scanner_cosine_distance", "estimate"): 0.148,
                ("lowpass_0p18", "tissue_type_balanced_accuracy", "estimate"): 0.616,
                ("lowpass_0p18", "tissue_type_balanced_accuracy", "delta_from_raw"): -0.055,
                ("lowpass_0p18", "tissue_type_balanced_accuracy", "delta_ci_low"): -0.073,
                ("lowpass_0p18", "tissue_type_balanced_accuracy", "delta_ci_high"): -0.037,
                ("lowpass_0p18", "scanner_balanced_accuracy", "estimate"): 0.994}
    details, ok = [], True
    for (condition, metric, column), value in reported.items():
        observed = float(retention[(retention.condition == condition)
                                   & (retention.metric == metric)][column].iloc[0])
        decimals = len(str(value).split(".")[1])
        ok &= round(observed, decimals) == value
        details.append(f"{condition}/{metric}/{column}: {observed:.5f} (reported {value})")
    check(item, "a_reported_values", "values quoted in the RV06 brief reproduce at their rounding",
          ok, "; ".join(details))
    table2 = pd.read_csv(TABLE2).set_index(["dataset", "method"])
    raw_t2 = float(table2.loc[("PanNormal", "raw"), "uni_distance_mean"])
    raw_here = float(retention[(retention.condition == "raw")
                               & (retention.metric == "pooled_scanner_cosine_distance")].estimate.iloc[0])
    check(item, "a_cross_table2", "raw AT2-target UNI distance equals the Table 2 raw value", None,
          f"{raw_here:.7f} here vs {raw_t2:.7f} in table2_baseline_summary.csv "
          f"(|diff| {abs(raw_here - raw_t2):.1e}; separate UNI extraction of the same set20 images)")

    design = ("Same filter applied to all six scanner images at a location (patch-mean-removed OD, "
              "reflect pad 32 px, taper 0.015 cyc/px); frozen UNI v1; 103 slides, set20 (2,060 "
              "locations). Distance: AT2 vs each of 5 targets, slide mean, directions equal weight; "
              "20,000 slide bootstraps (protocol default 2,000). Probes: slide x scanner mean "
              "embeddings, L2-normalised; logistic regression, locked 5 outer folds, inner C "
              "selection; balanced accuracy = macro recall over classes seen in training; tissue "
              "probe 102 slides/36 classes; CIs by slide cluster bootstrap (tissue: within class).")
    script = ("src/manuscript_completion/perturbation.py (matched shards, job 22827864, impl. "
              "manifest v3) + src/manuscript_completion/frequency_aggregate.py::aggregate_matched "
              "(job 22827866); v3 source versions not on disk, current versions are v5")
    for record in retention.itertuples(index=False):
        n = None if pd.isna(record.evaluated_slides) else int(record.evaluated_slides)
        unit = {"pooled_scanner_cosine_distance": "physical slide",
                "scanner_balanced_accuracy": "physical slide (cluster)",
                "tissue_type_balanced_accuracy": "physical slide within tissue"}.get(record.metric, "")
        boot = "20,000 slide" if unit else ""
        add(item, MATCHED / "information_retention_summary.csv", script, design,
            dataset="PanNormal", pfm="UNI v1", direction="AT2 vs target (symmetric filter)",
            scanner="pooled (5 targets)" if record.metric == "pooled_scanner_cosine_distance" else "all 6",
            method=record.condition, metric=record.metric, value=record.estimate,
            ci_low=getattr(record, "ci_low"), ci_high=getattr(record, "ci_high"),
            n_units=n if n is not None else (103 if unit else None), unit=unit, bootstrap=boot)
        if record.condition != "raw" and not pd.isna(record.delta_from_raw):
            add(item, MATCHED / "information_retention_summary.csv", script, design,
                dataset="PanNormal", pfm="UNI v1", direction="AT2 vs target (symmetric filter)",
                scanner="pooled (5 targets)" if record.metric == "pooled_scanner_cosine_distance" else "all 6",
                method=record.condition, metric=f"{record.metric}_minus_raw",
                value=record.delta_from_raw, ci_low=record.delta_ci_low, ci_high=record.delta_ci_high,
                n_units=n if n is not None else 103, unit=unit, bootstrap="20,000 (paired)")


# ---------------------------------------------------------------------------
# (b) PLISM Pix2Pix/CycleGAN in four PFMs
# ---------------------------------------------------------------------------

def item_b() -> None:
    item = "b_plism_gan_four_pfms"
    selected = pd.read_csv(PLISM_LEARNED / "pix2pix/00_contract/selected_locations.csv")
    selected_keys = set(zip(selected.stain, selected.location))
    other = pd.read_csv(PIX2PIX_BI / "10_plism_external_bidirectional/00_contract/selected_locations.csv")
    check(item, "b_plism_index", "PLISM evaluation index: 2,387 locations in 13 sections, identical "
          "to the GT450 bidirectional PLISM index", len(selected) == 2387 and selected.stain.nunique() == 13
          and selected_keys == set(zip(other.stain, other.location)))
    frames = []
    for index in range(18):
        frames.append(pd.read_csv(GAN_PFM / "shards" / f"part_{index:02d}.csv.gz"))
    data = pd.concat(frames, ignore_index=True)
    per_task = data.groupby(["model", "scanner", "method", "fold"]).size()
    keys_ok = all(set(zip(g.section, g.location_index)) == selected_keys
                  for _, g in data.groupby(["model", "scanner", "method", "fold"]))
    check(item, "b_shard_design", "18 tasks x 5 fold generators x 2,387 locations; locations equal the index",
          len(data) == 214830 and per_task.eq(2387).all() and keys_ok and data.fold.nunique() == 5,
          f"rows={len(data)}; models={sorted(data.model.unique())}")
    raw_spread = data.groupby(["model", "scanner", "section", "location_index"]).raw_distance.agg(np.ptp).max()
    check(item, "b_raw_consistent", "raw distance identical across fold generators and methods",
          raw_spread < 1e-6, f"max spread {raw_spread:.2g}")
    check(item, "b_gain_definition", "gain = raw - corrected at every location",
          max_abs(data.raw_distance - data.corrected_distance, data.gain) < 1e-8)
    audit = json.loads((GAN_PFM / "aggregate_audit.json").read_text())
    check(item, "b_raw_vs_table4", "raw distances agree with the Table 4 cross-PFM raw values (audit)",
          audit["max_abs_raw_difference_from_table4"] < 1e-3,
          f"max |diff| {audit['max_abs_raw_difference_from_table4']:.2g}")
    summary = pd.read_csv(GAN_PFM / "gan_gain_summary.csv")
    compact = pd.read_csv(GAN_PFM / "proposed_s5_gan_gains.csv").set_index("model")
    pooled = summary[summary.scanner == "all"].pivot(index="model", columns="method", values="gain")
    table2 = pd.read_csv(TABLE2).set_index(["dataset", "method"])
    uni = {m: float(table2.loc[("PLISM", "raw"), "uni_distance_mean"]
                    - table2.loc[("PLISM", m), "uni_distance_mean"]) for m in ("pix2pix", "cyclegan")}
    consistent = (max_abs(compact.loc[pooled.index, "pix2pix_gain"], pooled["pix2pix"]) < 1e-12
                  and max_abs(compact.loc[pooled.index, "cyclegan_gain"], pooled["cyclegan"]) < 1e-12
                  and abs(compact.loc["uni_v1", "pix2pix_gain"] - uni["pix2pix"]) < 1e-12
                  and abs(compact.loc["uni_v1", "cyclegan_gain"] - uni["cyclegan"]) < 1e-12)
    check(item, "b_compact_consistent", "proposed_s5_gan_gains.csv equals pooled summary rows and the "
          "Table 2 UNI v1 PLISM values", consistent)
    negative_pooled = bool((compact[["pix2pix_gain", "cyclegan_gain"]] < 0).all().all())
    per_scanner = summary[summary.scanner != "all"]
    positive = per_scanner[per_scanner.gain > 0]
    check(item, "b_claim_all_negative", "reported 'gains all negative'", negative_pooled,
          "true for the pooled (three-direction) gain of all 4 PFMs x 2 methods; per direction, "
          + "; ".join(f"{r.model} {r.scanner} {r.method} gain {r.gain:+.4f} (CI {r.ci_low:+.4f} to "
                      f"{r.ci_high:+.4f})" for r in positive.itertuples())
          + " are positive point estimates with CIs including 0")
    check(item, "b_producer", "producer located", None,
          ".Trash/2026-09-29_paper_code_cleanup/scripts/review_plism_gan_crossencoder.py (+ _aggregate.py, "
          ".sbatch); run as array 23670688 (0-17) and job 23670935 on 2026-09-25; logs in "
          ".Trash/2026-09-26_paper_refactor/generated/outputs/plism_gan_crossencoder_2026-09-25/logs; "
          "restored and re-run under RV-P0d (results/provenance_restoration/plism_gan_crossencoder)")

    design = ("Frozen PLISM renders (13 sections, 2,387 locations, <=4 per section and core); AT2 "
              "input translated by each of the 5 PanNormal fold generators (GT450: 09/11 bidirectional "
              "runs trained on training-fold slides; S360/S60: 12_manuscript_completion/02 baseline "
              "runs), no refitting on PLISM. Distance to the real same-location target in each PFM "
              "(UNI2-h bf16, Virchow2 bf16 CLS+mean-patch, H-optimus-1 fp16). Aggregation location -> "
              "core -> section x fold -> mean over folds -> section; 'all' = mean over the 3 directions "
              "per section. CI: 4,000 section bootstraps (protocol default 2,000). UNI v1 values come "
              "from table2_baseline_summary.csv (same hierarchy, no CI).")
    script = ("review_plism_gan_crossencoder.py + review_plism_gan_crossencoder_aggregate.py "
              "(recovered from .Trash/2026-09-29_paper_code_cleanup/scripts; restored as "
              "analysis/revision/restore_plism_gan_crossencoder.py)")
    names = {"uni2": "UNI2-h", "virchow2": "Virchow2", "hoptimus1": "H-optimus-1"}
    for record in summary.itertuples(index=False):
        for metric in ("raw_distance", "corrected_distance", "gain"):
            add(item, GAN_PFM / "gan_gain_summary.csv", script, design, dataset="PLISM",
                pfm=names[record.model], direction="AT2 -> target", scanner=record.scanner,
                method=record.method, metric=f"target_{metric}" if metric == "gain" else metric,
                value=getattr(record, metric),
                ci_low=record.ci_low if metric == "gain" else None,
                ci_high=record.ci_high if metric == "gain" else None,
                n_units=int(record.sections), unit="PLISM section",
                bootstrap="4,000 section" if metric == "gain" else "")
    for method in ("pix2pix", "cyclegan"):
        for metric, value in (("raw_distance", table2.loc[("PLISM", "raw"), "uni_distance_mean"]),
                              ("corrected_distance", table2.loc[("PLISM", method), "uni_distance_mean"]),
                              ("target_gain", uni[method])):
            add(item, TABLE2, "analysis/paper/table2_benchmark.py (Table 2)", design, dataset="PLISM",
                pfm="UNI v1", direction="AT2 -> target", scanner="all", method=method, metric=metric,
                value=value, n_units=13, unit="PLISM section")


# ---------------------------------------------------------------------------
# (c) Feature correction, scanner -> AT2, fitted on 40 locations
# ---------------------------------------------------------------------------

def bootstrap_rows(pooled: pd.DataFrame, draws: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(20260925)
    sampled = rng.integers(0, len(pooled), size=(draws, len(pooled)))
    out = []
    for method in pooled.columns:
        values = pooled[method].to_numpy()
        boot = values[sampled].mean(axis=1)
        out.append({"method": method, "mean_target_gain": float(values.mean()),
                    "ci_low": float(np.quantile(boot, 0.025)),
                    "ci_high": float(np.quantile(boot, 0.975))})
    return pd.DataFrame(out)


def item_c(cohort: pd.DataFrame, folds: pd.Series, set20: pd.DataFrame, set40: pd.DataFrame) -> None:
    from manuscript_completion.feature_panel40 import feature_dir

    item = "c_feature_reverse_40"
    internal, external = FOLLOWUP / "feature40_reverse", FOLLOWUP / "feature40_external_reverse"
    design_json = json.loads((internal / "design.json").read_text())
    check(item, "c_design_json", "design.json: scanner_to_at2, 40 training / 20 evaluation locations",
          design_json == {"direction": "scanner_to_at2", "training_locations_per_slide": 40,
                          "evaluation_locations_per_slide": 20,
                          "outer_folds": "locked physical-slide 5-fold"})
    per_slide = pd.read_csv(internal / "per_slide.csv", dtype={"slide_id": str})
    check(item, "c_cohort_folds", "103 slides; per-slide folds equal locked slide_folds.csv",
          set(per_slide.slide_id) == set(cohort.slide_id)
          and bool((per_slide.fold == per_slide.slide_id.map(folds)).all()),
          f"rows={len(per_slide)} (103 x 5 scanners x 5 methods)")
    audit = pd.read_csv(internal / "input_audit.csv", dtype={"slide_id": str})
    bad_hash, bad_loc = [], []
    s20 = set20.groupby("slide_id")
    for record in audit.itertuples(index=False):
        path = UNI_SHARDS / f"{record.slide_id}.h5"
        if sha256(path) != record.h5_sha256:
            bad_hash.append(record.slide_id)
        with h5py.File(path, "r") as store:
            location = np.sort(np.asarray(store["location_index"], dtype=int))
        if not np.array_equal(location, np.sort(s20.get_group(record.slide_id).location_index.to_numpy())):
            bad_loc.append(record.slide_id)
    check(item, "c_eval_inputs", "evaluation embeddings: 03_uni shard hashes as audited at run time; "
          "locations equal set20", not bad_hash and not bad_loc and len(audit) == 103,
          f"hash mismatches={bad_hash}; location mismatches={bad_loc}")
    s40 = set40.groupby("slide_id")
    bad40 = []
    for scanner in TARGETS:
        root = PROJECT / feature_dir(scanner)
        for slide_id in cohort.slide_id:
            with h5py.File(root / f"{slide_id}.h5", "r") as store:
                location = np.sort(np.asarray(store["location_index"], dtype=int))
            if not np.array_equal(location, np.sort(s40.get_group(slide_id).location_index.to_numpy())):
                bad40.append(f"{scanner}/{slide_id}")
    check(item, "c_train_panel_set40", "40-location training embeddings (raw_source/real_target of the "
          "Pix2Pix UNI feature shards) are at set40 locations", not bad40, f"mismatches={bad40[:10]}")

    scanner_summary = pd.read_csv(internal / "per_scanner_summary.csv")
    rederived = per_slide.groupby(["scanner", "method"], sort=True)[
        ["raw_distance", "corrected_distance", "target_gain"]].mean().reset_index()
    pooled_frozen = pd.read_csv(internal / "pooled_summary.csv")
    pooled_new = bootstrap_rows(per_slide.groupby(["slide_id", "method"], sort=True)["target_gain"]
                                .mean().unstack())
    check(item, "c_rederive_internal", "per-scanner and pooled summaries (5,000 slide bootstraps, seed "
          "20260925) re-derived from per_slide.csv",
          max_abs(rederived[["raw_distance", "corrected_distance", "target_gain"]],
                  scanner_summary[["raw_distance", "corrected_distance", "target_gain"]]) < 1e-12
          and max_abs(pooled_new[["mean_target_gain", "ci_low", "ci_high"]],
                      pooled_frozen[["mean_target_gain", "ci_low", "ci_high"]]) < 1e-12)
    fold_rows = pd.read_csv(external / "per_section_fold.csv")
    section = fold_rows.groupby(["scanner", "section", "method"], sort=True)[
        ["raw_distance", "corrected_distance", "target_gain"]].mean().reset_index()
    section_frozen = pd.read_csv(external / "per_section.csv")
    ext_pooled_new = bootstrap_rows(section.groupby(["section", "method"], sort=True)["target_gain"]
                                    .mean().unstack())
    ext_pooled = pd.read_csv(external / "pooled_summary.csv")
    locations = fold_rows[fold_rows.method == "raw"].groupby(["scanner", "fold"]).locations.sum()
    check(item, "c_rederive_external", "PLISM section, scanner and pooled summaries (5,000 section "
          "bootstraps) re-derived from per_section_fold.csv; 2,387 locations per scanner and fold",
          max_abs(section[["raw_distance", "corrected_distance", "target_gain"]],
                  section_frozen[["raw_distance", "corrected_distance", "target_gain"]]) < 1e-12
          and max_abs(ext_pooled_new[["mean_target_gain", "ci_low", "ci_high"]],
                      ext_pooled[["mean_target_gain", "ci_low", "ci_high"]]) < 1e-12
          and bool(locations.eq(2387).all()) and fold_rows.section.nunique() == 13)
    ridge = pd.read_csv(internal / "ridge_selection.csv")
    ridge20 = pd.read_csv(FOLLOWUP / "feature_reverse/ridge_selection.csv")
    shared = ridge.merge(ridge20, on=["scanner", "outer_fold"], suffixes=("_40", "_20"))
    shared = shared[shared.scanner.isin(["gt450", "s360", "s60"])]
    check(item, "c_ridge_alpha", "ridge alphas used by the PLISM transfer", None,
          f"feature40_reverse chosen alphas: {sorted(ridge.chosen_relative_alpha.unique())}; "
          f"identical to feature_reverse for the PLISM scanners: "
          f"{bool((shared.chosen_relative_alpha_40 == shared.chosen_relative_alpha_20).all())} "
          "(so the --internal-output choice cannot change the external result)")
    wrong = pd.read_csv(FOLLOWUP / "feature_reverse/pooled_summary.csv").set_index("method")
    right = pooled_frozen.set_index("method")
    check(item, "c_not_20_location_run", "values taken from feature40_* (40 training locations), not "
          "feature_reverse (20)", None,
          "pooled internal gain 40 vs 20 training locations: " + "; ".join(
              f"{m} {right.loc[m, 'mean_target_gain']:+.4f} vs {wrong.loc[m, 'mean_target_gain']:+.4f}"
              for m in right.index if m != "raw"))
    reported = {"combat": -0.0100, "featmap_ridge": -0.0533, "procrustes": -0.0503, "featmap_ols": -0.0831}
    ext = ext_pooled.set_index("method")
    check(item, "c_reported_values", "archive note's PLISM reverse gains reproduce at 4 decimals",
          all(round(float(ext.loc[m, "mean_target_gain"]), 4) == v for m, v in reported.items()),
          "; ".join(f"{m} {ext.loc[m, 'mean_target_gain']:+.5f}" for m in reported))
    check(item, "c_run_command", "run command (reconstructed)", None,
          "sbatch --export=ALL,JOB_CONDA_PREFIX=/home/leej70/miniconda3/envs/cpath,JOB_WORKDIR="
          "/mnt/isilon/oldridge_lab/leej/prenorm scripts/discussion_feature40.sbatch (array job 23600634; "
          "task 1 ended 2026-09-25 12:13:22 = per_slide.csv mtime) running `python -m "
          "manuscript_completion.discussion_feature_compare --direction scanner_to_at2 --train-panel 40 "
          "--output outputs/discussion_followup_2026-09-25/feature40_reverse`; then `sbatch "
          "--dependency=afterok:23600634 ... scripts/discussion_feature_external40.sbatch` (23600681, task 1 "
          "ended 12:17:55 = output mtime) running `python -m manuscript_completion.discussion_feature_external "
          "--direction scanner_to_at2 --train-panel 40 --internal-output .../feature40_reverse --output "
          ".../feature40_external_reverse`. Submit lines from sacct; the two sbatch files themselves are "
          "not preserved (flags inferred from design.json and module arguments). Modules were renamed "
          "feature_uni_v1 / feature_uni_external; current wrappers scripts/features/feature_uni_v1_*.sbatch.")

    design_int = ("Feature maps fitted per scanner and outer fold on set40 raw UNI v1 embeddings of the 4 "
                  "training folds (Pix2Pix UNI feature shards), mapping target scanner -> AT2; evaluated "
                  "on set20 embeddings (03_uni shards) of held-out slides. Ridge alpha chosen on the "
                  "validation fold ((k+1) mod 5). Slide mean over 20 locations; pooled = mean over 5 "
                  "directions per slide; 5,000 slide bootstraps (protocol default 2,000). Training and "
                  "evaluation embeddings come from different UNI extractions (see RV-P0c).")
    design_ext = ("The 5 PanNormal fold maps (scanner -> AT2) applied without refitting to PLISM UNI v1 "
                  "embeddings (GT450, S360, S60 -> AT2; 13 sections, 2,387 locations); averaged over "
                  "folds within section; pooled = mean over the 3 directions per section; 5,000 section "
                  "bootstraps.")
    script_int = ("src/manuscript_completion/feature_uni_v1.py (ran as discussion_feature_compare; "
                  "byte code of main identical) + feature_panel40.py + prenorm/feature_correction.py")
    script_ext = "src/manuscript_completion/feature_uni_external.py (ran as discussion_feature_external)"
    for record in pooled_frozen.itertuples(index=False):
        add(item, internal / "pooled_summary.csv", script_int, design_int, dataset="PanNormal",
            pfm="UNI v1", direction="target -> AT2", scanner="pooled (5)", method=record.method,
            metric="target_gain", value=record.mean_target_gain, ci_low=record.ci_low,
            ci_high=record.ci_high, n_units=103, unit="physical slide", bootstrap="5,000 slide")
    for record in scanner_summary.itertuples(index=False):
        for metric in ("raw_distance", "corrected_distance", "target_gain"):
            add(item, internal / "per_scanner_summary.csv", script_int, design_int, dataset="PanNormal",
                pfm="UNI v1", direction="target -> AT2", scanner=record.scanner, method=record.method,
                metric=metric, value=getattr(record, metric), n_units=103, unit="physical slide")
    content = pd.read_csv(internal / "content_summary.csv")
    for record in content.itertuples(index=False):
        for metric in ("macro_tissue_retrieval", "target_self_retrieval", "variance_trace_ratio"):
            add(item, internal / "content_summary.csv", script_int, design_int, dataset="PanNormal",
                pfm="UNI v1", direction="target -> AT2", scanner=record.scanner, method=record.method,
                metric=metric, value=getattr(record, metric), n_units=103, unit="physical slide")
    for record in ext_pooled.itertuples(index=False):
        add(item, external / "pooled_summary.csv", script_ext, design_ext, dataset="PLISM",
            pfm="UNI v1", direction="target -> AT2", scanner="pooled (3)", method=record.method,
            metric="target_gain", value=record.mean_target_gain, ci_low=record.ci_low,
            ci_high=record.ci_high, n_units=13, unit="PLISM section", bootstrap="5,000 section")
    ext_scanner = pd.read_csv(external / "per_scanner_summary.csv")
    for record in ext_scanner.itertuples(index=False):
        for metric in ("raw_distance", "corrected_distance", "target_gain"):
            add(item, external / "per_scanner_summary.csv", script_ext, design_ext, dataset="PLISM",
                pfm="UNI v1", direction="target -> AT2", scanner=record.scanner, method=record.method,
                metric=metric, value=getattr(record, metric), n_units=13, unit="PLISM section")


# ---------------------------------------------------------------------------
# (d) GT450 <-> AT2 bidirectional GAN runs (record only)
# ---------------------------------------------------------------------------

def item_d(cohort: pd.DataFrame, set40: pd.DataFrame) -> None:
    item = "d_gan_reverse_gt450"
    locked = pd.read_csv(STUDY / "06_learned_baselines/00_contract/locked_image_evaluation_index.csv.gz",
                         dtype={"slide_id": str})
    same40 = set(zip(locked.slide_id, locked.location_index)) == set(zip(set40.slide_id, set40.location_index))
    check(item, "d_locations", "internal evaluation uses the locked 40-location index (= set40), not set20",
          same40, "Protocol rule 1 evaluates on set20; these runs are on 40 locations (4,120) and are "
          "recorded as-is, not re-aggregated")
    design_int = ("Pix2Pix: one model per direction and outer fold (10 runs); CycleGAN: one bidirectional "
                  "model per outer fold (5 runs). 200 passes, candidate every 5 passes, checkpoint chosen "
                  "by inner-validation image criterion only (Pix2Pix: paired target L1; CycleGAN: pair-blind "
                  "marginal score). Evaluation on set40 (103 slides, 4,120 locations), frozen UNI v1; "
                  "20,000 slide bootstraps.")
    design_ext = ("Fold models applied to PLISM GT450/AT2 (13 sections, 2,387 locations) without "
                  "refitting; location -> core -> section, 5 fold models averaged within section; "
                  "20,000 section bootstraps.")
    for method, root, script in (("pix2pix", PIX2PIX_BI, "src/scanner_gan/train.py, predict.py, "
                                  "evaluate_uni.py, summarize_bidirectional.py, plism_bidirectional.py"),
                                 ("cyclegan", CYCLEGAN_BI, "src/scanner_gan/train_cyclegan.py, predict.py, "
                                  "evaluate_uni.py, summarize_cyclegan.py, plism_bidirectional.py")):
        uni = pd.read_csv(root / "05_summary/uni_summary.csv")
        uni = uni[uni.analysis_role == "primary_translation"]
        for record in uni.itertuples(index=False):
            if record.metric in {"method_to_raw_distance"}:
                continue
            add(item, root / "05_summary/uni_summary.csv", script, design_int, dataset="PanNormal",
                pfm="UNI v1", direction=record.direction.replace("_to_", " -> ").upper(),
                scanner="gt450", method=method, metric=record.metric, value=record.estimate,
                ci_low=record.ci_low, ci_high=record.ci_high, n_units=int(record.n_physical_slides),
                unit="physical slide", bootstrap="20,000 slide")
        ens = pd.read_csv(root / "10_plism_external_bidirectional/02_aggregate/translation_ensemble_summary.csv")
        ens = ens[(ens.quadrant == "overall") & ens.metric.isin(
            ["raw_to_target_distance", "method_to_target_distance", "gain_to_target", "fractional_closure"])]
        for record in ens.itertuples(index=False):
            add(item, root / "10_plism_external_bidirectional/02_aggregate/translation_ensemble_summary.csv",
                script, design_ext, dataset="PLISM", pfm="UNI v1",
                direction=record.direction.replace("_to_", " -> ").upper(), scanner="gt450", method=method,
                metric=record.metric, value=record.estimate, ci_low=record.ci_low, ci_high=record.ci_high,
                n_units=int(record.sections), unit="PLISM section", bootstrap="20,000 section")
        contrast = pd.read_csv(root / "10_plism_external_bidirectional/02_aggregate/paired_direction_comparison.csv")
        record = contrast[contrast.quadrant == "overall"].iloc[0]
        add(item, root / "10_plism_external_bidirectional/02_aggregate/paired_direction_comparison.csv",
            script, design_ext, dataset="PLISM", pfm="UNI v1", direction="AT2->GT450 minus GT450->AT2",
            scanner="gt450", method=method, metric="gain_forward_minus_reverse", value=record.estimate,
            ci_low=record.ci_low, ci_high=record.ci_high, n_units=13, unit="PLISM section",
            bootstrap="20,000 section")
    summary = json.loads((PIX2PIX_BI / "05_summary/summary.json").read_text())["paired_direction_comparison"]
    add(item, PIX2PIX_BI / "05_summary/summary.json", "src/scanner_gan/summarize_bidirectional.py",
        design_int, dataset="PanNormal", pfm="UNI v1", direction="AT2->GT450 minus GT450->AT2",
        scanner="gt450", method="pix2pix", metric="gain_forward_minus_reverse", value=summary["estimate"],
        ci_low=summary["ci_low"], ci_high=summary["ci_high"], n_units=103, unit="physical slide",
        bootstrap="20,000 slide")
    review = pd.read_csv(FOLLOWUP / "existing_review/gt450_direction_contrasts.csv").set_index("method")
    add(item, FOLLOWUP / "existing_review/gt450_direction_contrasts.csv",
        "discussion_existing_review.py (.Trash/2026-09-29_paper_code_cleanup)", design_int + " Contrast "
        "re-bootstrapped by the discussion review (5,000 slide bootstraps).", dataset="PanNormal",
        pfm="UNI v1", direction="AT2->GT450 minus GT450->AT2", scanner="gt450", method="cyclegan",
        metric="gain_forward_minus_reverse", value=review.loc["cyclegan", "forward_minus_reverse_uni_gain"],
        ci_low=review.loc["cyclegan", "ci_low"], ci_high=review.loc["cyclegan", "ci_high"], n_units=103,
        unit="physical slide", bootstrap="5,000 slide")

    # Checkpoint selection per run, from the training logs.
    flags = []
    for method, root, pattern, key in (
            ("pix2pix", PIX2PIX_BI, "01_training/runs/*/fold_*/seed_*/metrics.jsonl", "validation_l1_normalized"),
            ("cyclegan", CYCLEGAN_BI, "01_training/runs/*/fold_*/seed_*/metrics.jsonl", "marginal_score")):
        training = pd.read_csv(root / "05_summary/training_summary.csv")
        for path in sorted(root.glob(pattern)):
            records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if method == "pix2pix":
                candidates = {r["completed_passes"]: r[key] for r in records if r["completed_passes"] % 5 == 0}
            else:
                candidates = {r["completed_passes"]: r["validation"][key] for r in records if "validation" in r}
            best = min(candidates, key=candidates.get)
            direction = path.parts[-4]
            fold = int(path.parts[-3].split("_")[1])
            ranked = sorted(candidates, key=candidates.get)
            detail = (f"best pass {best} (criterion {candidates[best]:.5f}); pass 200 {candidates[200]:.5f}; "
                      f"next best passes {ranked[1:4]}")
            if best <= 25:
                flags.append(f"{method} {direction} fold {fold}: {detail}")
            add(item, path, "src/scanner_gan/train.py" if method == "pix2pix" else "src/scanner_gan/train_cyclegan.py",
                design_int, dataset="PanNormal", pfm="none (image criterion)",
                direction=direction.replace("_to_", " -> ").replace("_", "<->").upper(), scanner="gt450",
                method=method, metric=f"selected_checkpoint_pass (fold {fold})", value=best, n_units=None,
                unit="inner validation fold")
        if method == "pix2pix":
            reported = training.set_index(["direction", "test_fold"]).best_pass
            flagged = int(reported.loc[("gt450_to_at2", 4)])
            check(item, "d_fold4_pass5", "Pix2Pix GT450->AT2 fold-4 checkpoint selected at pass 5 of 200",
                  flagged == 5, "training_summary.csv best_pass=5; the criterion is minimal at pass 5 "
                  "(0.10388) vs 0.11658 at pass 200, so the selection followed the rule; the fold-4 "
                  "model is effectively an early-training generator")
    check(item, "d_early_checkpoints", "runs whose image-only checkpoint was selected at pass <= 25 of 200",
          None, " | ".join(flags))


def write_summary(table: pd.DataFrame, audit: pd.DataFrame) -> None:
    """Render results/existing_evidence/summary.md from the verified tables."""

    def pick(item: str, **match) -> pd.Series:
        frame = table[table["item"] == item]
        for key, value in match.items():
            frame = frame[frame[key] == value]
        if len(frame) != 1:
            raise ValueError(f"{item} {match}: {len(frame)} rows")
        return frame.iloc[0]

    def ci(row: pd.Series, digits: int = 3) -> str:
        if pd.isna(row.ci_low):
            return f"{row.value:+.{digits}f}"
        return f"{row.value:+.{digits}f} ({row.ci_low:+.{digits}f} to {row.ci_high:+.{digits}f})"

    counts = audit.status.value_counts().to_dict()
    lines = [
        "# RV06 Existing results with verified provenance", "",
        "**Status:** complete. No analysis was re-run.",
        "**Code:** `analysis/revision/existing_evidence.py` (+ `.sbatch`). Code identity with the "
        "run-time byte code comes from `analysis/revision/restore_code_identity.py` "
        "(`../provenance_restoration/code_identity_cp39.csv`, `code_identity_cp313.csv`).",
        f"**Outputs:** `existing_evidence.csv` ({len(table)} rows: source path, producer script, design "
        "notes, metric, value, CI, n, unit, bootstrap); `verification_checks.csv` "
        f"({len(audit)} checks: {counts.get('pass', 0)} pass, {counts.get('note', 0)} notes, "
        f"{counts.get('FAIL', 0)} fail); `manifest.json` (hashes of the locked inputs and of the outputs).",
        "",
        "## What was checked", "",
        "- **Integrity.** Output hashes match the ones recorded in each run's manifest or summary: "
        "(a) 5 outputs and 103 shards; (c) the 103 `03_uni` evaluation shards, which match "
        "`input_audit.csv`.",
        "- **Design.** Checked against the locked cohort (103 slides, 37 tissue types), the locked "
        "folds (`slide_folds.csv`) and `set20`/`set40`.",
        "- **Summaries re-derived.** Each reported summary was re-derived from its frozen per-unit file, "
        "using the frozen estimator and seed. The pooled distance and probe accuracies in (a) and the "
        "internal and external summaries in (c) reproduce to within 1e-12. The summaries of (b) are "
        "re-derived under RV-P0d (ii) (`../provenance_restoration/`).",
        "- **Reported numbers.** The numbers quoted in the brief were compared with the frozen files "
        "at the brief's rounding.",
        "",
    ]
    if counts.get("FAIL", 0):
        lines += ["**Failed checks:** " + ", ".join(audit[audit.status == "FAIL"].check_id), ""]

    a = "a_symmetric_lowpass"
    lines += [
        "## (a) Symmetric low-pass filtering: UNI v1, PanNormal set20", "",
        "Source: `outputs/scanner_batch_effect_analysis_2026-09-17/12_manuscript_completion/"
        "04_frequency_mechanism/02_matched_information_removal/`. The same filter is applied to all "
        "six scanner images at a location.", "",
        "- **Design:** matches the protocol. 103 slides, set20 (2,060 locations), locked folds "
        "(all verified).",
        "- **Differences from the protocol:**",
        "  - The CIs use 20,000 slide bootstraps instead of 2,000.",
        "  - The probes are fitted on slide x scanner mean embeddings (618 rows).",
        "  - The tissue probe covers 102 slides and 36 classes. Slide 2-8_28 is excluded because its "
        "tissue type is absent from the training folds.",
        "",
        "| Condition | AT2-target distance (95% CI) | Tissue BA | Delta tissue BA vs raw (95% CI) | Scanner BA |",
        "| --- | --- | ---: | --- | ---: |",
    ]
    for condition in ("raw", "lowpass_0p25", "lowpass_0p18", "lowpass_0p125", "bandstop_high_rms"):
        distance = pick(a, method_or_condition=condition, metric="pooled_scanner_cosine_distance")
        tissue = pick(a, method_or_condition=condition, metric="tissue_type_balanced_accuracy")
        scanner = pick(a, method_or_condition=condition, metric="scanner_balanced_accuracy")
        delta = ("0" if condition == "raw" else
                 ci(pick(a, method_or_condition=condition, metric="tissue_type_balanced_accuracy_minus_raw")))
        lines.append(f"| {condition} | {distance.value:.4f} ({distance.ci_low:.4f}-{distance.ci_high:.4f}) | "
                     f"{tissue.value:.3f} | {delta} | {scanner.value:.3f} |")
    detail = audit.set_index("check_id")
    lines += [
        "",
        f"The values quoted in the brief reproduce (check `a_reported_values`: "
        f"{detail.loc['a_reported_values', 'status']}).",
        "",
        "**Provenance.** The run was jobs 22827864 (shards) and 22827866 (aggregate, finished "
        "2026-09-18 18:43 EDT) under implementation manifest v3.",
        "",
        "- **Current code has changed.** `frequency_aggregate.py` and `perturbation.py` were edited "
        "later (manifests v4/v5, 22:18-22:23 EDT). Those were aggregation-only fixes for the "
        "band-sensitivity analysis. The v3 source files are no longer on disk.",
        "- **Old byte code survives.** A cp313 cache compiled from the 2026-09-17 23:56 EDT sources, "
        "5 minutes before the v3 freeze, is still available. It differs from the current files only "
        "in `aggregate_sensitivity` and `verify_implementation_manifest`. The second change only adds "
        "one more accepted manifest status.",
        "- **Conclusion.** `aggregate_matched` and its estimators are byte-code identical to that "
        "pre-freeze cache. The cache is most likely the v3 code that ran, but that cannot be proven "
        "because the v3 hash cannot be recomputed. Re-deriving the summaries with the current "
        "functions gives exactly the frozen values.",
        "",
    ]

    b = "b_plism_gan_four_pfms"
    lines += [
        "## (b) PLISM Pix2Pix/CycleGAN in four PFMs", "",
        "Source: `outputs/plism_gan_crossencoder_2026-09-25/`.", "",
        "- **Producer:** found in `.Trash/2026-09-29_paper_code_cleanup/scripts/"
        "review_plism_gan_crossencoder{,_aggregate}.py`. It ran as array 23670688 (tasks 0-17) and job "
        "23670935. Compared with the run-time byte code, it differs only in one relocated path "
        "constant per file.",
        "- **Design:** 13 sections, 2,387 locations (the same index as the GT450 bidirectional PLISM "
        "runs), AT2 -> target.",
        "  - The 5 PanNormal fold generators are averaged within each section.",
        "  - 'all' is the mean of the 3 directions per section.",
        "  - CIs use 4,000 section bootstraps, not 2,000.",
        "  - The UNI v1 row comes from `table2_baseline_summary.csv`: same hierarchy, no CI.",
        "",
        "| PFM | Pix2Pix gain, 3 directions (95% CI) | CycleGAN gain (95% CI) |",
        "| --- | --- | --- |",
    ]
    for pfm in ("UNI v1", "UNI2-h", "Virchow2", "H-optimus-1"):
        cells = [ci(pick(b, pfm=pfm, scanner="all", method_or_condition=m, metric="target_gain"))
                 for m in ("pix2pix", "cyclegan")]
        lines.append(f"| {pfm} | {cells[0]} | {cells[1]} |")
    lines += ["", "**'All gains negative' holds for the pooled gains only.** "
              + detail.loc["b_claim_all_negative", "detail"].split("; per direction, ")[1] + ".", ""]

    c = "c_feature_reverse_40"
    lines += [
        "## (c) Feature correction, scanner -> AT2, fitted on 40 locations, evaluated on 20", "",
        "Sources: `outputs/discussion_followup_2026-09-25/feature40_reverse/` and "
        "`feature40_external_reverse/`. These are not the 20-training-location `feature_reverse/` "
        "runs; the two sets of runs differ materially ("
        + detail.loc["c_not_20_location_run", "detail"] + ").", "",
        "**Design:**",
        "- Maps are fitted on set40 embeddings of the training-fold slides and evaluated on set20 of "
        "the held-out slides. Both location sets were verified per slide.",
        "- Folds are the locked folds. The 5,000-draw bootstrap uses seed 20260925.",
        "- Training and evaluation embeddings come from different UNI extractions (see RV-P0c).",
        "- In PLISM, 2,387 locations per scanner and fold were verified.",
        "",
        "| Method | PanNormal gain, 5 directions (95% CI) | PLISM gain, 3 directions (95% CI) |",
        "| --- | --- | --- |",
    ]
    for method, label in (("featmap_ridge", "Ridge affine"), ("featmap_ols", "Affine OLS"),
                          ("procrustes", "Procrustes"), ("combat", "ComBat")):
        internal = pick(c, dataset="PanNormal", scanner="pooled (5)", method_or_condition=method)
        external = pick(c, dataset="PLISM", scanner="pooled (3)", method_or_condition=method)
        lines.append(f"| {label} | {ci(internal, 4)} | {ci(external, 4)} |")
    lines += [
        "",
        "**Producer:** `src/manuscript_completion/feature_uni_v1.py` and `feature_uni_external.py`. "
        "At run time these were `discussion_feature_compare` and `discussion_feature_external`. "
        "Compared with the run-time byte code:",
        "- Both `main` functions are byte-code identical.",
        "- All helpers are identical except `fit_combat`, whose `neuroCombat` import moved inside the "
        "function. The logic is unchanged.",
        "",
        "**Reconstructed run command:** " + detail.loc["c_run_command", "detail"], "",
        detail.loc["c_ridge_alpha", "detail"] + ".", "",
    ]

    d = "d_gan_reverse_gt450"
    lines += [
        "## (d) GT450<->AT2 GAN runs (record only)", "",
        "- **Locations:** 40 per slide (4,120). This is set40, not set20.",
        "- **Statistics:** 20,000 bootstraps.",
        "",
        "Target gain (95% CI):", "",
        "| Method | PanNormal AT2->GT450 | PanNormal GT450->AT2 | PLISM AT2->GT450 | PLISM GT450->AT2 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for method in ("pix2pix", "cyclegan"):
        cells = [ci(pick(d, dataset=ds, direction=dr, method_or_condition=method, metric="gain_to_target"))
                 for ds in ("PanNormal", "PLISM") for dr in ("AT2 -> GT450", "GT450 -> AT2")]
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    contrasts = table[(table["item"] == d) & (table.metric == "gain_forward_minus_reverse")]
    lines += [
        "",
        "Forward-minus-reverse contrasts: " + "; ".join(
            f"{r.dataset} {r.method_or_condition} {r.value:+.3f} ({r.ci_low:+.3f} to {r.ci_high:+.3f})"
            for r in contrasts.itertuples()) + ".",
        "",
        "**Early checkpoint flag.**",
        "- " + detail.loc["d_fold4_pass5", "detail"] + ".",
        "- All runs selected at pass <= 25: " + detail.loc["d_early_checkpoints", "detail"] + ".",
        "",
        "## Not verifiable here", "",
        "- The v3 source files of the frequency code; only their byte code survives.",
        "- The sbatch files `scripts/discussion_feature40.sbatch` and "
        "`scripts/discussion_feature_external40.sbatch`.",
        "- The exact `scanner_gan/predict.py` in use on 2026-09-25 before 23:43. This matters for (b): "
        "the `load_validated_generator` function is identical in the 23:43 cache and the current "
        "file, and (b) is checked empirically by regeneration under RV-P0d (ii).",
        "",
    ]
    (OUTPUT / "summary.md").write_text("\n".join(lines))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cohort, folds, set20, set40 = locked_design()
    check("all", "cohort", "cohort.csv: 103 slides, 37 tissue types; folds agree with slide_folds.csv",
          len(cohort) == 103 and cohort.tissue_type.nunique() == 37
          and bool((cohort.fold == cohort.slide_id.map(folds)).all()))
    item_a(cohort, folds, set20)
    item_b()
    item_c(cohort, folds, set20, set40)
    item_d(cohort, set40)
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT / "existing_evidence.csv", index=False)
    audit = pd.DataFrame(checks)
    audit.to_csv(OUTPUT / "verification_checks.csv", index=False)
    inputs = {
        "cohort": {"path": rel(COHORT), "sha256": sha256(COHORT)},
        "folds": {"path": rel(FOLDS), "sha256": sha256(FOLDS)},
        "set20": {"path": rel(SET20), "sha256": sha256(SET20)},
        "set40": {"path": rel(SET40), "sha256": sha256(SET40)},
    }
    summary = {"rows": int(len(table)), "checks": audit.status.value_counts().to_dict(),
               "failed_checks": audit[audit.status == "FAIL"].check_id.tolist(), "locked_inputs": inputs,
               "outputs": {name: sha256(OUTPUT / name)
                           for name in ("existing_evidence.csv", "verification_checks.csv")}}
    (OUTPUT / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_summary(table, audit)
    print(audit.to_string(index=False, max_colwidth=160))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

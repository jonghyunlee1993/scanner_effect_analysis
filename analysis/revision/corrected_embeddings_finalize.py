#!/usr/bin/env python3
"""RV03 stage 3 (CPU): verify the assembled embeddings, summarise QC, write the manifest.

Reads only results/corrected_embeddings (final files, qc/ tables, parts/ logs) and the
published table 00_manuscript/tables/table_cross_pfm_correction.tex (read-only), and
writes qc/*summary*.csv, manifest.json ("complete": true only when every check that
defines completeness passed) and summary.md.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import corrected_embeddings_common as C


TABLE = C.PROJECT / "00_manuscript/tables/table_cross_pfm_correction.tex"
LABELS = {"Raw": "raw", "Reinhard": "reinhard", "Color + frequency": "combined",
          "Pix2Pix": "pix2pix", "CycleGAN": "cyclegan", "Ridge affine": "ridge", "ComBat": "combat"}
PFM_LABELS = {"uni_v1": "UNI v1", "uni2": "UNI2-h", "virchow2": "Virchow2", "hoptimus1": "H-optimus-1"}
THRESHOLD = 0.999


def parse_table() -> pd.DataFrame:
    rows, dataset = [], None
    for line in TABLE.read_text().splitlines():
        if "textbf{PanNormal}" in line:
            dataset = "pannormal"
            continue
        if "textbf{PLISM}" in line:
            dataset = "plism"
            continue
        if dataset is None or "&" not in line or "\\\\" not in line:
            continue
        clean = re.sub(r"\\rowcolor\{[^}]*\}", "", line).split("\\\\")[0]
        cells = [cell.strip() for cell in clean.split("&")]
        if len(cells) < 10 or cells[1] not in LABELS:
            continue
        for k, pfm in enumerate(C.PFMS):
            rows.append({"dataset": dataset, "method": LABELS[cells[1]], "pfm": pfm,
                         "table_distance": float(cells[2 + 2 * k])})
    frame = pd.DataFrame(rows)
    if len(frame) != (7 + 5) * 4:
        raise ValueError(f"unexpected table parse: {len(frame)} cells")
    return frame


def pooled(qc: Path) -> pd.DataFrame:
    rows = []
    for pfm in C.PFMS:
        pan = pd.read_csv(qc / f"distances_pannormal_{pfm}.csv.gz", dtype={"slide_id": str})
        slide = pan.groupby(["method", "slide_id"]).distance.mean().groupby("method")
        for method, values in slide:
            rows.append({"dataset": "pannormal", "pfm": pfm, "method": method,
                         "convention": "slide_mean_over_locations_and_scanners",
                         "distance": float(values.mean()), "units": int(values.size)})
        plism = pd.read_csv(qc / f"distances_plism_{pfm}.csv.gz")
        # A: location -> core -> section, folds averaged, scanners equal (table4 / Table 2).
        core = plism.groupby(["method", "scanner", "fold", "section", "core"]).distance.mean()
        section = core.groupby(["method", "scanner", "fold", "section"]).mean()
        section = section.groupby(["method", "scanner", "section"]).mean()
        section = section.groupby(["method", "section"]).mean()
        for method, values in section.groupby("method"):
            rows.append({"dataset": "plism", "pfm": pfm, "method": method,
                         "convention": "location_core_section_folds_scanners",
                         "distance": float(values.mean()), "units": int(values.size)})
        # B: location -> section per scanner x fold, then section (feature cross-encoder review).
        section = plism.groupby(["method", "scanner", "fold", "section"]).distance.mean()
        section = section.groupby(["method", "section"]).mean()
        for method, values in section.groupby("method"):
            rows.append({"dataset": "plism", "pfm": pfm, "method": method,
                         "convention": "location_section_scanner_fold",
                         "distance": float(values.mean()), "units": int(values.size)})
    return pd.DataFrame(rows)


def verify_files(root: Path, smoke: bool = False) -> tuple[pd.DataFrame, bool]:
    cohort = C.load_cohort()
    set20 = C.load_set(C.SET20, 20)
    table = C.plism_locations()
    rows, ok = [], True
    pan_names, plism_names = C.pannormal_conditions(), C.plism_conditions()
    for pfm in C.PFMS:
        for slide in cohort.itertuples(index=False):
            path = root / "pannormal" / pfm / f"{slide.slide_id}.h5"
            if smoke and not path.exists():
                continue
            problem = ""
            try:
                with h5py.File(path, "r") as store:
                    features = np.asarray(store["features"])
                    locations = np.asarray(store["location_index"])
                    wanted = set20.loc[set20.slide_id.eq(slide.slide_id), "location_index"].to_numpy()
                    if features.shape != (20, len(pan_names), C.PFM_DIMS[pfm]) or features.dtype != np.float32:
                        problem = f"shape {features.shape} {features.dtype}"
                    elif C.decode(store["condition_names"][:]) != pan_names:
                        problem = "condition names"
                    elif not np.array_equal(locations, wanted):
                        problem = "location_index"
                    elif not np.isfinite(features).all():
                        problem = "non-finite"
                    elif (store.attrs["pfm"], store.attrs["slide_id"], store.attrs["tissue_type"],
                          int(store.attrs["fold"])) != (pfm, slide.slide_id, slide.tissue_type, int(slide.fold)):
                        problem = "attrs"
            except (OSError, KeyError) as error:
                problem = f"unreadable: {error}"
            ok &= problem == ""
            rows.append({"dataset": "pannormal", "pfm": pfm, "unit": slide.slide_id, "problem": problem})
        for section in C.plism_sections():
            path = root / "plism" / pfm / f"{section}.h5"
            if smoke and not path.exists():
                continue
            wanted = table[table.section.eq(section)]
            problem = ""
            try:
                with h5py.File(path, "r") as store:
                    features = np.asarray(store["features"])
                    if features.shape != (len(wanted), len(plism_names), C.PFM_DIMS[pfm]):
                        problem = f"shape {features.shape}"
                    elif C.decode(store["condition_names"][:]) != plism_names:
                        problem = "condition names"
                    elif not np.array_equal(np.asarray(store["location"]), wanted.location.to_numpy()):
                        problem = "location"
                    elif not np.array_equal(np.asarray(store["core"]), wanted.core.to_numpy()):
                        problem = "core"
                    elif C.decode(store["tissue_type"][:]) != wanted.tissue_type.tolist():
                        problem = "tissue_type"
                    elif not np.isfinite(features).all():
                        problem = "non-finite"
            except (OSError, KeyError) as error:
                problem = f"unreadable: {error}"
            ok &= problem == ""
            rows.append({"dataset": "plism", "pfm": pfm, "unit": section, "problem": problem})
    return pd.DataFrame(rows), bool(ok)


def method_of(condition: str) -> str:
    if condition.startswith("source") or condition.startswith("target"):
        return condition.split("_")[0]
    return condition.split("_to_")[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=C.OUTPUT)
    parser.add_argument("--parts-root", type=Path, default=C.PARTS)
    parser.add_argument("--smoke", action="store_true", help="partial outputs; gates reported, not enforced")
    args = parser.parse_args()
    os.chdir(C.PROJECT)
    root = args.output_root
    qc = root / "qc"
    parts = args.parts_root

    files, files_ok = verify_files(root, args.smoke)
    C.write_frame(qc / "file_checks.csv", files)

    # QC 1/2: embedding parity.
    embed = []
    for dataset in ("pannormal", "plism"):
        for pfm in C.PFMS:
            frame = pd.read_csv(qc / f"embedding_parity_{dataset}_{pfm}.csv.gz")
            frame["dataset"], frame["pfm"] = dataset, pfm
            embed.append(frame)
    embed = pd.concat(embed, ignore_index=True)
    embed["method"] = embed.condition.map(method_of)
    embed_summary = embed.groupby(["dataset", "pfm", "check", "method"]).cosine.agg(
        n="size", min="min", p01=lambda x: float(np.quantile(x, 0.01)), median="median",
        frac_ge_0999=lambda x: float((x >= THRESHOLD).mean())).reset_index()
    C.write_frame(qc / "embedding_parity_summary.csv", embed_summary)

    # Distance parity with stored per-location values.
    dist = []
    for dataset in ("pannormal", "plism"):
        for pfm in C.PFMS:
            frame = pd.read_csv(qc / f"distance_parity_{dataset}_{pfm}.csv.gz")
            frame["dataset"], frame["pfm"] = dataset, pfm
            dist.append(frame)
    dist = pd.concat(dist, ignore_index=True)
    dist["abs_diff"] = (dist.ours - dist.stored).abs()
    dist_summary = dist.groupby(["dataset", "pfm", "reference", "method"]).agg(
        n=("abs_diff", "size"), stored_missing=("stored", lambda x: int(x.isna().sum())),
        mean_ours=("ours", "mean"), mean_stored=("stored", "mean"),
        max_abs_diff=("abs_diff", "max"), p99_abs_diff=("abs_diff", lambda x: float(np.nanquantile(x, 0.99))),
        mean_abs_diff=("abs_diff", "mean")).reset_index()
    C.write_frame(qc / "distance_parity_summary.csv", dist_summary)

    # Feature-correction fitting parity.
    alpha = pd.concat([pd.read_csv(p) for p in sorted((parts / "features").glob("*/*_ridge_selection.csv"))])
    internal = pd.concat([pd.read_csv(p, dtype={"slide_id": str})
                          for p in sorted((parts / "features").glob("*/*_internal_parity.csv"))])
    external = pd.concat([pd.read_csv(p) for p in sorted((parts / "features").glob("*/*_external_parity.csv"))])
    internal["abs_diff_paper_inputs"] = (internal.distance_paper_inputs - internal.stored_distance).abs()
    internal["abs_diff_rv03"] = (internal.distance_rv03 - internal.stored_distance).abs()
    external["abs_diff"] = (external.distance_rv03 - external.stored_distance).abs()
    feature_summary = internal.groupby(["pfm", "method"]).agg(
        stored_available=("stored_distance", lambda x: int(x.notna().sum())),
        max_abs_diff_paper_inputs=("abs_diff_paper_inputs", "max"),
        max_abs_diff_rv03=("abs_diff_rv03", "max")).reset_index()
    ext_summary = external.groupby(["pfm", "method"]).agg(
        stored_available=("stored_distance", lambda x: int(x.notna().sum())),
        max_abs_diff=("abs_diff", "max")).reset_index()
    C.write_frame(qc / "feature_alpha_selection.csv", alpha)
    C.write_frame(qc / "feature_internal_parity_summary.csv", feature_summary)
    C.write_frame(qc / "feature_external_parity_summary.csv", ext_summary)

    # Generation logs.
    pan_flags = pd.concat([pd.read_csv(p, dtype={"slide_id": str}) for p in
                           sorted((parts / "pannormal/generation").glob("*_stain_flags.csv"))])
    pan_gan = pd.concat([pd.read_csv(p, dtype={"slide_id": str}) for p in
                         sorted((parts / "pannormal/generation").glob("*_gan_checks.csv"))])
    plism_flags = pd.concat([pd.read_csv(p) for p in
                             sorted((parts / "plism/generation").glob("*_stain_flags.csv"))])
    generation = {
        "pannormal_stain_images": int(len(pan_flags)),
        "pannormal_stain_fallbacks": int(pan_flags.fallback.sum()),
        "pannormal_vahadane_nonconverged": int((~pan_flags.loc[pan_flags.method.eq("vahadane"), "converged"]).sum()),
        "pannormal_gan_pairs_checked": int(len(pan_gan)),
        "pannormal_gan_source_target_equal_cache": bool(pan_gan.source_equals_cache.all()
                                                         and pan_gan.target_equals_cache.all()),
        "plism_stain_source_fits": int(plism_flags.scanner.isna().sum()) if "scanner" in plism_flags else int(len(plism_flags)),
        "plism_stain_fallback_rows": int(plism_flags.fallback.sum()),
    }
    if "scanner" in plism_flags:
        fits = plism_flags[plism_flags.scanner.isna()]
    else:
        fits = plism_flags
    generation["plism_vahadane_nonconverged"] = int((~fits.loc[fits.method.eq("vahadane"), "converged"].astype(bool)).sum())

    # QC 3: pooled target distances versus the published table.
    pooled_frame = pooled(qc)
    C.write_frame(qc / "pooled_distances.csv", pooled_frame)
    table = parse_table()
    convention = {"pannormal": "slide_mean_over_locations_and_scanners"}
    compare = []
    for item in table.itertuples(index=False):
        if item.dataset == "pannormal":
            conv = convention["pannormal"]
        else:
            conv = ("location_section_scanner_fold" if item.method in C.FEATURE_METHODS
                    else "location_core_section_folds_scanners")
        value = pooled_frame[(pooled_frame.dataset == item.dataset) & (pooled_frame.pfm == item.pfm)
                             & (pooled_frame.method == item.method) & (pooled_frame.convention == conv)]
        ours = float(value.distance.iloc[0])
        compare.append({"dataset": item.dataset, "pfm": item.pfm, "method": item.method,
                        "convention": conv, "table_distance": item.table_distance,
                        "rv03_distance": ours, "difference": ours - item.table_distance,
                        "matches_at_4_decimals": bool(round(ours, 4) == round(item.table_distance, 4)),
                        "within_0.0005": bool(abs(ours - item.table_distance) <= 0.0005)})
    compare = pd.DataFrame(compare)
    C.write_frame(qc / "table_cross_pfm_comparison.csv", compare)

    # Completion gates: every file present/valid and fitting reproduces the paper exactly.
    gates = {
        "all_files_valid": files_ok,
        "feature_alpha_matches_stored": bool(alpha.alpha_matches_stored.all()),
        "feature_fit_reproduces_paper_inputs": bool(
            feature_summary.max_abs_diff_paper_inputs.dropna().max() < 1e-6),
        "pannormal_gan_images_equal_cache_pairs": generation["pannormal_gan_source_target_equal_cache"],
    }
    complete = all(gates.values())

    pan_conditions, plism_conditions = C.pannormal_conditions(), C.plism_conditions()
    jobs = os.environ.get("JOB_RV03_JOBS", "")
    if os.environ.get("SLURM_JOB_ID"):
        jobs = f"{jobs} (finalize SLURM_JOB_ID={os.environ['SLURM_JOB_ID']})".strip()
    manifest = {
        "analysis": "RV03 corrected-image embeddings in four PFMs",
        "protocol": "analysis/revision/README.md, section RV03",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pfms": {pfm: {"label": PFM_LABELS[pfm], "dim": C.PFM_DIMS[pfm]} for pfm in C.PFMS},
        "datasets": {
            "pannormal": {
                "path_pattern": "pannormal/<pfm>/<slide_id>.h5",
                "files_per_pfm": 103, "locations_per_file": 20, "locations": 2060,
                "datasets": {"features": "float32 [20, n_conditions, d]",
                             "condition_names": "utf-8 strings", "location_index": "int64 [20]",
                             "source_index": "int64 [20]"},
                "attrs": ["pfm", "slide_id", "tissue_type", "fold"],
                "location_order": "set20.csv sorted by location_index",
                "n_conditions": len(pan_conditions), "conditions": pan_conditions,
                "fold_rule": "each held-out slide uses the parameters/models/maps of the fold in which it is held out",
            },
            "plism": {
                "path_pattern": "plism/<pfm>/<section>.h5",
                "files_per_pfm": 13, "locations": C.PLISM_LOCATIONS,
                "datasets": {"features": "float32 [n_loc, n_conditions, d]",
                             "condition_names": "utf-8 strings", "core": "int64", "location": "int64 (PLISM benchmark location id)",
                             "tissue_type": "utf-8 (PLISM tissue label)",
                             "pannormal_tissue": "utf-8 (mapped PanNormal tissue, empty if none)"},
                "attrs": ["pfm", "section", "gan_source"],
                "location_order": "benchmark selected_locations sorted by location within section",
                "n_conditions": len(plism_conditions), "conditions": plism_conditions,
                "fold_rule": "*_fold<k>: fitted on PanNormal training folds (folds != k) and applied without refitting",
            },
        },
        "conventions": {
            "raw": "source_at2 and target_* are the stored raw embeddings (PanNormal: 40-location panel subset to set20; PLISM: stored external raw)",
            "image_conditions": "L2-normalised PFM outputs of regenerated or stored corrected images",
            "feature_conditions": "ridge/combat/ols outputs are not normalised; distances in the paper use cosine",
            "distance": "cosine distance to the target_<s> embedding of the same location",
        },
        "skipped_conditions": [],
        "provenance": {
            "cohort": str(C.COHORT.relative_to(C.PROJECT)), "folds": str(C.FOLDS.relative_to(C.PROJECT)),
            "set20": str(C.SET20.relative_to(C.PROJECT)), "set40": str(C.SET40.relative_to(C.PROJECT)),
            "set20_sha256": C.sha256(C.SET20), "set40_sha256": C.sha256(C.SET40),
            "raw_images": "data/preprocessing_patch_extraction_matching_and_QC/cache/<slide>.h5 (cache_path in cohort.csv)",
            "reinhard_frequency_combined": f"{C.PARAMETERS.relative_to(C.PROJECT)} (sha256 {C.sha256(C.PARAMETERS)}); scanner_batch_extensions.reinhard_transform / frequency_transform",
            "macenko_vahadane_references": str(C.STAIN_REFERENCES.relative_to(C.PROJECT)),
            "vahadane_selection": str(C.VAHADANE_SELECTION.relative_to(C.PROJECT)),
            "macenko_vahadane_pannormal_seed": "panel_a.stable_seed(slide_id, location_index, scanner, method) as in panel_a.stain_uni_extract",
            "macenko_vahadane_plism_seed": "one source fit per location, stable_seed('table2', section, location), as in analysis/paper/table2_benchmark.py",
            "pannormal_gan_images": f"{C.GAN_PAIRS.relative_to(C.PROJECT)} (stored predictions, subset to set20)",
            "plism_gan": {"uni_v1": "stored benchmark features (06_plism_learned_external and 06_learned_baselines/*/10_plism_external_bidirectional feature_shards)",
                          "other_pfms": "regenerated from the frozen checkpoints listed in the same checkpoint_candidates.csv (scanner_gan.plism._generate_uint8, bfloat16, batch 16)"},
            "plism_renders": "08_plism_external/pix2pix_gt450_to_at2/01_rendered_inputs (AT2, GT450); 05_plism_external_correction/01_rendered_inputs (S360, S60)",
            "plism_locations": str(C.PLISM_SELECTED.relative_to(C.PROJECT)),
            "plism_tissue": str(C.PLISM_TISSUE.relative_to(C.PROJECT)),
            "stored_raw": {"uni_v1_pannormal": "manuscript_completion.feature_panel40 (Pix2Pix 04_uni raw_source/real_target; GT450 from 09_bidirectional_full_training)",
                           "uni_v1_plism": str(C.PLISM_UNI_RAW.relative_to(C.PROJECT)),
                           "other_pfms": str(C.CROSS_RAW.relative_to(C.PROJECT))},
            "encoders": {"uni_v1": "prenorm.embedding.load_uni/embed_uni via scanner_gan.uni.embed_frozen_uni(value_range='uint8'), fp32 (the path of the stored UNI v1 benchmark features)",
                         "uni2_virchow2_hoptimus1": "scripts/features/review_multiencoder_scanner.load_encoder/embed via review_feature_crossencoder_extract.images_to_features (the path of the paper's cross-PFM analyses)",
                         "weights": "timm hf-hub MahmoodLab/uni and MahmoodLab/UNI2-h (~/.cache/huggingface), Virchow2 checkpoint in PFM_alignment hf_cache, H-optimus-1 in outputs/encoder_review_2026-09-25/hf_cache"},
            "feature_fitting": "prenorm.feature_correction.fit_affine/apply_affine/fit_combat as in manuscript_completion.feature_uni_v1 (--train-panel 40) and review_feature_crossencoder_fit",
            "code": ["analysis/revision/corrected_embeddings_common.py",
                     "analysis/revision/corrected_embeddings_images.py",
                     "analysis/revision/corrected_embeddings_features.py",
                     "analysis/revision/corrected_embeddings_assemble.py",
                     "analysis/revision/corrected_embeddings_finalize.py"],
        },
        "qc": {
            "gates": gates,
            "generation": generation,
            "embedding_parity": embed_summary.to_dict(orient="records"),
            "distance_parity": dist_summary.to_dict(orient="records"),
            "feature_alpha_all_match": bool(alpha.alpha_matches_stored.all()),
            "feature_internal_parity": feature_summary.to_dict(orient="records"),
            "feature_external_parity": ext_summary.to_dict(orient="records"),
            "table_cross_pfm_comparison": compare.to_dict(orient="records"),
        },
        "deviations_proposed": [
            "PLISM Pix2Pix/CycleGAN for UNI2-h, Virchow2 and H-optimus-1: no stored per-location features existed (only distances in outputs/plism_gan_crossencoder_2026-09-25); images were regenerated from the frozen checkpoints and embedded. UNI v1 uses the stored benchmark features.",
            "PanNormal Macenko/Vahadane use the seed convention of the stored UNI embeddings (panel_a.stain_uni_extract, no fold in the seed); table2_benchmark image metrics used a seed that includes the fold, which changes Vahadane source subsampling slightly.",
            "UNI v1 PanNormal raw source/target are the stored 40-location panel (protocol rule), not the 03_uni extraction used by the paper's UNI evaluation; feature maps are therefore applied to that source, and pooled UNI v1 distances can differ from the paper in the fourth decimal.",
            "Affine OLS for UNI2-h, Virchow2 and H-optimus-1 had no paper counterpart; fitted with the same code (zero penalty).",
            "UNI v1 images are embedded with the paper's primary UNI path (prenorm.embedding.embed_uni, no clamp after resizing), which produced 03_uni, stain_v4, 04_uni and the PLISM benchmark; the review path of review_multiencoder_scanner (clamp after resizing) is used for the other PFMs only. The table's PanNormal Pix2Pix/CycleGAN UNI v1 values come from outputs/gan_encoder_review_2026-09-25 (review path), so per-location differences of up to a few 1e-3 are expected there.",
        ],
        "jobs": jobs,
        "complete": bool(complete),
    }
    C.write_json(root / "manifest.json", manifest)
    write_summary(root, manifest, embed_summary, dist_summary, feature_summary, ext_summary,
                  compare, pooled_frame, generation, gates, files)
    print(json.dumps({"complete": complete, "gates": gates}), flush=True)
    if not complete and not args.smoke:
        raise SystemExit("RV03 completeness gates failed; see manifest.json")


def md_table(frame: pd.DataFrame, floats: int = 4) -> str:
    frame = frame.copy()
    for column in frame.columns:
        if frame[column].dtype.kind == "f":
            frame[column] = frame[column].map(lambda v: "" if pd.isna(v) else f"{v:.{floats}f}")
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *body])


def write_summary(root, manifest, embed_summary, dist_summary, feature_summary, ext_summary,
                  compare, pooled_frame, generation, gates, files) -> None:
    embed_compact = embed_summary.groupby(["dataset", "pfm", "check"]).agg(
        n=("n", "sum"), min=("min", "min"), frac_ge_0999=("frac_ge_0999", "min")).reset_index()
    embed_compact = embed_compact.rename(columns={"frac_ge_0999": "worst_method_frac_ge_0.999"})
    dist_compact = dist_summary.groupby(["dataset", "pfm", "reference"]).agg(
        n=("n", "sum"), stored_missing=("stored_missing", "sum"),
        max_abs_diff=("max_abs_diff", "max"), mean_abs_diff=("mean_abs_diff", "mean")).reset_index()
    wide = compare.pivot_table(index=["dataset", "method"], columns="pfm",
                               values=["table_distance", "rv03_distance"]).reset_index()
    wide.columns = [" ".join(c).strip() if isinstance(c, tuple) else c for c in wide.columns]
    lines = [
        "# RV03 corrected-image embeddings in four PFMs",
        "",
        f"Created {manifest['created_utc']}. Protocol: `analysis/revision/README.md` (RV03). "
        f"Complete: **{manifest['complete']}**.",
        "",
        "## What ran",
        "",
        "- `corrected_embeddings_images` (GPU array): regenerated Reinhard, frequency, colour + frequency, "
        "Macenko and Vahadane images; PanNormal Pix2Pix/CycleGAN from stored predictions (set20); "
        "PLISM Pix2Pix/CycleGAN regenerated from the frozen checkpoints; embedded raw and corrected "
        "images in UNI v1, UNI2-h, Virchow2 and H-optimus-1.",
        "- `corrected_embeddings_features` (CPU array): ridge affine, ComBat and affine OLS per PFM, "
        "target scanner and fold, fitted on set40 of training-fold slides, applied to set20 of held-out "
        "slides and to PLISM.",
        "- `corrected_embeddings_assemble` (CPU array, one task per PFM): final files and QC tables.",
        "- `corrected_embeddings_finalize`: file checks, QC summaries, manifest, this summary.",
        f"- Jobs: {manifest['jobs'] or 'not recorded'}",
        "",
        "## Outputs",
        "",
        f"- `pannormal/<pfm>/<slide_id>.h5`: 103 files per PFM, features [20, {manifest['datasets']['pannormal']['n_conditions']}, d].",
        f"- `plism/<pfm>/<section>.h5`: 13 files per PFM, features [n, {manifest['datasets']['plism']['n_conditions']}, d], 2,387 locations.",
        "- `manifest.json`: condition lists, provenance, QC, deviations. `qc/`: all QC tables. "
        "`parts/`: stage-1 intermediates (fresh raw embeddings kept for audit).",
        "- Raw source/target: stored raw embeddings. Image conditions are L2-normalised; "
        "ridge/ComBat/OLS outputs are unnormalised (use cosine).",
        "",
        "## Completion gates",
        "",
        md_table(pd.DataFrame([{"gate": k, "passed": v} for k, v in gates.items()])),
        "",
        f"File problems: {int((files.problem.fillna('') != '').sum())} of {len(files)} files.",
        "",
        "## Image generation",
        "",
        md_table(pd.DataFrame([{"item": k, "value": v} for k, v in generation.items()])),
        "",
        "## QC 1/2: embedding parity with stored embeddings (cosine; threshold 0.999)",
        "",
        md_table(embed_compact, 6),
        "",
        "Per-method detail: `qc/embedding_parity_summary.csv`.",
        "",
        "## Per-location distance parity with stored producers",
        "",
        md_table(dist_compact, 6),
        "",
        "Per-method detail: `qc/distance_parity_summary.csv`.",
        "",
        "## Feature-correction fitting parity",
        "",
        "Alpha choices equal the stored choices: "
        f"{bool(pd.read_csv(root / 'qc/feature_alpha_selection.csv').alpha_matches_stored.all())}. "
        "`paper_inputs` applies the refitted maps to the inputs the paper evaluated "
        "(UNI v1: 03_uni); `rv03` uses the stored 40-location raw subset to set20.",
        "",
        md_table(feature_summary, 6),
        "",
        md_table(ext_summary, 6),
        "",
        "## QC 3: pooled target distances versus table_cross_pfm_correction.tex",
        "",
        md_table(compare[["dataset", "pfm", "method", "table_distance", "rv03_distance",
                          "difference", "matches_at_4_decimals"]], 4),
        "",
        "PanNormal: slide means over 20 locations x 5 scanners, then over 103 slides. PLISM image "
        "methods and raw: location -> core -> section, mean over folds, scanners equal; PLISM feature "
        "methods: location -> section per scanner x fold (the producers' conventions). All methods "
        "under both conventions: `qc/pooled_distances.csv`.",
        "",
        "## Proposed deviations",
        "",
        *[f"- {item}" for item in manifest["deviations_proposed"]],
        "",
    ]
    (root / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

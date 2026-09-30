#!/usr/bin/env python3
"""RV03 stage 2 (CPU): assemble the per-location embedding contract for one PFM.

Combines, for one PFM,
* raw AT2 source and real target embeddings from the stored raw extractions
  (PanNormal: 40-location panel subset to set20; PLISM: stored external raw),
* image-corrected embeddings from stage 1 (parts/{pannormal,plism}/<pfm>),
  except PLISM Pix2Pix/CycleGAN in UNI v1, which are the stored benchmark features,
* feature-corrected embeddings from stage 1 (parts/features/<pfm>),
into results/corrected_embeddings/{pannormal,plism}/<pfm>/*.h5, and writes QC tables
(embedding and distance parity with stored values, per-location target distances) to
results/corrected_embeddings/qc/.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import corrected_embeddings_common as C
from corrected_embeddings_features import load_raw40


TABLE4 = C.PROJECT / "outputs/table4_color_frequency_crossencoder_2026-09-25/shards"
GAN_REVIEW = C.PROJECT / "outputs/gan_encoder_review_2026-09-25/shards"
PLISM_GAN_CROSS = C.PROJECT / "outputs/plism_gan_crossencoder_2026-09-25/shards"
TABLE2_PLISM = {
    "s360": C.COMPLETION / "09_table2_benchmark/plism_conventional/shards",
    "s60": C.COMPLETION / "09_table2_benchmark/plism_conventional/shards",
    "gt450": C.PROJECT / "analysis/paper/results/plism_gt450_extension/plism_conventional/shards",
}
STAIN_UNI = C.PANEL_A / "01_stain_v4_mu_convergence/uni/shards"
FEATURES_ROOT = [C.PARTS / "features"]  # overridable for smoke tests


def read_part(path: Path) -> tuple[np.ndarray, dict[str, int], h5py.AttributeManager, dict]:
    with h5py.File(path, "r") as store:
        features = np.asarray(store["features"], dtype=np.float32)
        names = C.decode(store["condition_names"][:])
        extra = {key: np.asarray(store[key]) for key in store.keys()
                 if key not in {"features", "condition_names"}}
        attrs = dict(store.attrs)
    return features, {name: i for i, name in enumerate(names)}, attrs, extra


def distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 1.0 - C.cosine(a, b)


def parity_rows(unit: str, locations, condition: str, check: str, ours, stored) -> list[dict]:
    values = C.cosine(ours, stored)
    return [{"unit": unit, "location": int(loc), "condition": condition, "check": check,
             "cosine": float(v)} for loc, v in zip(locations, values)]


# ----------------------------------------------------------------------------- PanNormal

def assemble_pannormal(pfm: str, parts: Path, out: Path, qc: Path, smoke: bool = False) -> dict:
    cohort = C.load_cohort()
    set40 = C.load_set(C.SET40, 40)
    set20 = C.load_set(C.SET20, 20)
    raw40 = load_raw40(pfm, cohort, set40)
    conditions = C.pannormal_conditions()
    index = {name: i for i, name in enumerate(conditions)}
    features_by_scanner = {}
    for scanner in C.PAN_TARGETS:
        with h5py.File(FEATURES_ROOT[0] / pfm / f"{scanner}.h5", "r") as store:
            if C.decode(store["pannormal_slide_id"][:]) != cohort.slide_id.tolist():
                raise ValueError(f"{pfm}/{scanner}: feature part slide order differs")
            if tuple(C.decode(store["methods"][:])) != C.FEATURE_METHODS:
                raise ValueError("feature method order differs")
            features_by_scanner[scanner] = (np.asarray(store["pannormal_features"], dtype=np.float32),
                                            np.asarray(store["pannormal_location_index"]))
    table4 = gan = None
    if pfm != "uni_v1":
        table4 = pd.concat([pd.read_csv(p, dtype={"slide_id": str})
                            for p in sorted((TABLE4 / "pannormal" / pfm).glob("part_*.csv.gz"))])
        table4 = table4.set_index(["slide_id", "scanner", "location_index", "arm"])["distance"]
    gan = pd.concat([pd.read_csv(p, dtype={"slide_id": str})
                     for p in sorted((GAN_REVIEW / pfm).glob("fold_*.csv.gz"))])
    gan = gan.set_index(["slide_id", "scanner", "location_index"])
    embed_rows, dist_rows, final_rows = [], [], []
    present = {p.stem for p in (parts / "pannormal" / pfm).glob("*.h5")}
    if not smoke and present != set(cohort.slide_id):
        raise ValueError(f"{pfm}: PanNormal image parts incomplete ({len(present)} of 103)")
    for i, row in enumerate(cohort.itertuples(index=False)):
        slide_id = row.slide_id
        if slide_id not in present:
            continue
        rows = set20[set20.slide_id.eq(slide_id)]
        locations = rows.location_index.to_numpy(np.int64)
        forty = set40.loc[set40.slide_id.eq(slide_id), "location_index"].tolist()
        raw20 = raw40[i, [forty.index(v) for v in locations]]
        part, names, _, extra = read_part(parts / "pannormal" / pfm / f"{slide_id}.h5")
        if not np.array_equal(extra["location_index"], locations):
            raise ValueError(f"{slide_id}: image part locations differ from set20")
        final = np.full((20, len(conditions), C.PFM_DIMS[pfm]), np.nan, dtype=np.float32)
        final[:, index["source_at2"]] = raw20[:, 0]
        for s, scanner in enumerate(C.PAN_TARGETS, start=1):
            final[:, index[f"target_{scanner}"]] = raw20[:, s]
            for method in C.IMAGE_METHODS:
                final[:, index[f"{method}_to_{scanner}"]] = part[:, names[f"{method}_to_{scanner}"]]
            values, location_check = features_by_scanner[scanner]
            if not np.array_equal(location_check[i], locations):
                raise ValueError(f"{slide_id}/{scanner}: feature part locations differ")
            for m, method in enumerate(C.FEATURE_METHODS):
                final[:, index[f"{method}_to_{scanner}"]] = values[i, :, m]
        if not np.isfinite(final).all():
            raise ValueError(f"{slide_id}: non-finite final features")
        C.write_h5(out / "pannormal" / pfm / f"{slide_id}.h5", {
            "features": final, "condition_names": np.asarray(conditions),
            "location_index": locations, "source_index": rows.source_index.to_numpy(np.int64),
        }, {"pfm": pfm, "slide_id": slide_id, "tissue_type": str(row.tissue_type),
            "fold": int(row.fold), "analysis": "RV03 corrected_embeddings",
            "note": ("source_at2/target_* are stored 40-location raw embeddings subset to set20; "
                     "image conditions are L2-normalised PFM outputs; ridge/combat/ols are the raw "
                     "outputs of the feature maps (not normalised); use cosine geometry")})

        # QC 1/2: fresh extraction versus stored embeddings.
        fresh = {"source_at2": part[:, names["source_at2"]]}
        for s, scanner in enumerate(C.PAN_TARGETS, start=1):
            fresh[f"target_{scanner}"] = part[:, names[f"target_{scanner}"]]
        stored_raw = {"source_at2": raw20[:, 0]}
        stored_raw.update({f"target_{sc}": raw20[:, s] for s, sc in enumerate(C.PAN_TARGETS, start=1)})
        for name in fresh:
            embed_rows += parity_rows(slide_id, locations, name, "fresh_vs_stored_raw40",
                                      fresh[name], stored_raw[name])
        if pfm == "uni_v1":
            with h5py.File(C.UNI_SHARDS / f"{slide_id}.h5", "r") as store:
                if not np.array_equal(np.asarray(store["location_index"]), locations):
                    raise ValueError(f"{slide_id}: 03_uni locations differ from set20")
                shard = np.asarray(store["features"], dtype=np.float32)
                sname = {n: j for j, n in enumerate(C.decode(store["condition_names"][:]))}
            pairs = [("source_at2", "source")] + [(f"target_{s}", f"target:{s}") for s in C.PAN_TARGETS]
            for name, stored_name in pairs:
                embed_rows += parity_rows(slide_id, locations, name, "fresh_vs_03_uni",
                                          fresh[name], shard[:, sname[stored_name]])
                embed_rows += parity_rows(slide_id, locations, name, "stored_raw40_vs_03_uni",
                                          stored_raw[name], shard[:, sname[stored_name]])
            for scanner in C.PAN_TARGETS:
                for method in ("reinhard", "frequency", "combined"):
                    embed_rows += parity_rows(slide_id, locations, f"{method}_to_{scanner}",
                                              "fresh_vs_03_uni",
                                              part[:, names[f"{method}_to_{scanner}"]],
                                              shard[:, sname[f"{method}:{scanner}"]])
            with h5py.File(STAIN_UNI / f"{slide_id}.h5", "r") as store:
                if not np.array_equal(np.asarray(store["location_index"]), locations):
                    raise ValueError(f"{slide_id}: stain UNI locations differ from set20")
                stain = np.asarray(store["features"], dtype=np.float32)
                tname = {n: j for j, n in enumerate(C.decode(store["condition_names"][:]))}
            for scanner in C.PAN_TARGETS:
                for method in ("macenko", "vahadane"):
                    embed_rows += parity_rows(slide_id, locations, f"{method}_to_{scanner}",
                                              "fresh_vs_stain_v4_uni",
                                              part[:, names[f"{method}_to_{scanner}"]],
                                              stain[:, tname[f"{method}:{scanner}"]])
                for method in C.GAN_METHODS:
                    with h5py.File(C.pannormal_gan_uni_path(method, scanner, slide_id), "r") as store:
                        loc40 = np.asarray(store["location_index"], dtype=np.int64)
                        take = [int(np.flatnonzero(loc40 == v)[0]) for v in locations]
                        generated = np.asarray(store["generated"], dtype=np.float32)[take]
                    embed_rows += parity_rows(slide_id, locations, f"{method}_to_{scanner}",
                                              "fresh_vs_04_uni_gan", part[:, names[f"{method}_to_{scanner}"]],
                                              generated)
        # Per-location distances: fresh (same extraction as stored producers) vs stored.
        for s, scanner in enumerate(C.PAN_TARGETS, start=1):
            target_fresh = fresh[f"target_{scanner}"]
            ours = {"raw": distance(fresh["source_at2"], target_fresh)}
            for method in C.IMAGE_METHODS:
                ours[method] = distance(part[:, names[f"{method}_to_{scanner}"]], target_fresh)
            for j, location in enumerate(locations):
                key = (slide_id, scanner, int(location))
                for method in ("raw", "pix2pix", "cyclegan"):
                    column = "raw_distance" if method == "raw" else f"{method}_distance"
                    dist_rows.append({"unit": slide_id, "scanner": scanner, "location": int(location),
                                      "method": method, "fold": -1, "reference": "gan_encoder_review",
                                      "ours": float(ours[method][j]),
                                      "stored": float(gan.loc[key, column])})
                if table4 is not None:
                    for method in ("raw", "reinhard", "frequency", "combined"):
                        dist_rows.append({"unit": slide_id, "scanner": scanner,
                                          "location": int(location), "method": method, "fold": -1,
                                          "reference": "table4_crossencoder",
                                          "ours": float(ours[method][j]),
                                          "stored": float(table4.loc[key + (method,)])})
            # Final-file distances (what downstream analyses see).
            target_final = final[:, index[f"target_{scanner}"]]
            final_rows.append({"slide_id": slide_id, "tissue_type": row.tissue_type,
                               "fold": int(row.fold), "scanner": scanner, "method": "raw",
                               "distance": float(distance(final[:, index["source_at2"]], target_final).mean())})
            for method in C.METHODS:
                final_rows.append({"slide_id": slide_id, "tissue_type": row.tissue_type,
                                   "fold": int(row.fold), "scanner": scanner, "method": method,
                                   "distance": float(distance(final[:, index[f"{method}_to_{scanner}"]],
                                                              target_final).mean())})
    C.write_frame(qc / f"embedding_parity_pannormal_{pfm}.csv.gz", pd.DataFrame(embed_rows))
    C.write_frame(qc / f"distance_parity_pannormal_{pfm}.csv.gz", pd.DataFrame(dist_rows))
    C.write_frame(qc / f"distances_pannormal_{pfm}.csv.gz", pd.DataFrame(final_rows))
    return {"files": len(cohort), "shape": [20, len(conditions), C.PFM_DIMS[pfm]]}


# ----------------------------------------------------------------------------- PLISM

def stored_plism_raw(pfm: str, section: str) -> tuple[np.ndarray, np.ndarray]:
    """Stored raw [n, 4, d] in PLISM_SCANNERS order and the location order."""
    if pfm == "uni_v1":
        blocks, location = [], None
        for scanner in C.PLISM_TARGETS:
            with h5py.File(C.PLISM_UNI_RAW / scanner / f"{section}.h5", "r") as store:
                observed = np.asarray(store["location"], dtype=np.int64)
                if location is None:
                    location = observed
                    blocks.append(np.asarray(store["target_features"], dtype=np.float32))
                elif not np.array_equal(location, observed) or not np.array_equal(
                        blocks[0], np.asarray(store["target_features"], dtype=np.float32)):
                    raise ValueError(f"UNI v1 PLISM AT2 raw differs across scanner files: {section}")
                blocks.append(np.asarray(store["source_features"], dtype=np.float32))
        return np.stack(blocks, axis=1), location
    with h5py.File(C.CROSS_RAW / "external" / pfm / f"{section}.h5", "r") as store:
        if tuple(C.decode(store["scanner_names"][:])) != C.PLISM_SCANNERS:
            raise ValueError("raw external scanner order changed")
        return (np.asarray(store["features"], dtype=np.float32),
                np.asarray(store["location_index"], dtype=np.int64))


def assemble_plism(pfm: str, parts: Path, out: Path, qc: Path, smoke: bool = False) -> dict:
    table = C.plism_locations()
    conditions = C.plism_conditions()
    index = {name: i for i, name in enumerate(conditions)}
    feature_parts = {}
    for scanner in C.PLISM_TARGETS:
        with h5py.File(FEATURES_ROOT[0] / pfm / f"{scanner}.h5", "r") as store:
            if (C.decode(store["plism_section"][:]) != table.section.tolist()
                    or not np.array_equal(np.asarray(store["plism_location"]), table.location.to_numpy())):
                raise ValueError(f"{pfm}/{scanner}: PLISM feature part order differs")
            feature_parts[scanner] = np.asarray(store["plism_features"], dtype=np.float32)
    table2 = None
    if pfm == "uni_v1":
        frames = []
        for scanner, root in TABLE2_PLISM.items():
            for path in sorted(root.glob(f"{scanner.upper()}_*.csv.gz")):
                frame = pd.read_csv(path, usecols=["location", "section", "scanner", "method",
                                                   "fold", "uni_distance"])
                frames.append(frame)
        table2 = pd.concat(frames)
        table2["scanner"] = table2.scanner.str.lower()
        table2 = table2.set_index(["section", "scanner", "location", "method", "fold"])["uni_distance"]
        table4 = gan_cross = None
    else:
        table4 = pd.concat([pd.read_csv(p) for p in sorted((TABLE4 / "plism" / pfm).glob("part_*.csv.gz"))])
        table4 = table4.drop_duplicates(["section", "location_index", "scanner", "fold", "arm"])
        table4 = table4.set_index(["section", "scanner", "location_index", "arm", "fold"])["distance"]
        gan_cross = pd.concat([pd.read_csv(p) for p in sorted(PLISM_GAN_CROSS.glob("part_*.csv.gz"))])
        gan_cross = gan_cross[gan_cross.model.eq(pfm)]
        gan_cross = gan_cross.set_index(["section", "scanner", "location_index", "method", "fold"])
    embed_rows, dist_rows, final_rows = [], [], []
    gan_provenance = []
    present = {p.stem for p in (parts / "plism" / pfm).glob("*.h5")}
    if not smoke and present != set(C.plism_sections()):
        raise ValueError(f"{pfm}: PLISM image parts incomplete ({len(present)} of 13)")
    for section in C.plism_sections():
        if section not in present:
            continue
        part, names, attrs, extra = read_part(parts / "plism" / pfm / f"{section}.h5")
        keep = len(extra["location"]) if smoke else None  # smoke parts may hold a prefix
        rows = table[table.section.eq(section)].iloc[:keep]
        where = np.flatnonzero(table.section.eq(section).to_numpy())[:keep]
        location = rows.location.to_numpy(np.int64)
        if not np.array_equal(extra["location"], location) or not np.array_equal(
                extra["core"], rows.core.to_numpy(np.int64)):
            raise ValueError(f"{section}: image part location/core order differs")
        stored, stored_location = stored_plism_raw(pfm, section)
        stored, stored_location = stored[:keep], stored_location[:keep]
        if not np.array_equal(stored_location, location):
            raise ValueError(f"{pfm}/{section}: stored raw location order differs")
        n = len(location)
        final = np.full((n, len(conditions), C.PFM_DIMS[pfm]), np.nan, dtype=np.float32)
        final[:, index["source_at2"]] = stored[:, 0]
        for s, scanner in enumerate(C.PLISM_TARGETS, start=1):
            final[:, index[f"target_{scanner}"]] = stored[:, s]
            for method in C.IMAGE_METHODS:
                for fold in C.FOLD_IDS:
                    name = f"{method}_to_{scanner}_fold{fold}"
                    if pfm == "uni_v1" and method in C.GAN_METHODS:
                        candidate = C.plism_gan_candidate(method, scanner, fold)
                        path = C.plism_gan_feature_path(method, scanner, fold, section)
                        with h5py.File(path, "r") as store:
                            if str(store.attrs["checkpoint_sha256"]) != str(candidate.checkpoint_sha256):
                                raise ValueError(f"stored UNI GAN features use another checkpoint: {path}")
                            if not np.array_equal(np.asarray(store["location"])[:keep], location):
                                raise ValueError(f"stored UNI GAN location order differs: {path}")
                            generated = np.asarray(store["generated"], dtype=np.float32)[:keep]
                            stored_source = np.asarray(store["source"], dtype=np.float32)[:keep]
                            stored_target = np.asarray(store["paired_target"], dtype=np.float32)[:keep]
                        final[:, index[name]] = generated
                        embed_rows += parity_rows(section, location, name,
                                                  "regenerated_vs_stored_gan_features",
                                                  part[:, names[name]], generated)
                        if method == "pix2pix" and fold == 0:
                            embed_rows += parity_rows(section, location, "source_at2",
                                                      f"stored_gan_eval_vs_stored_raw_{scanner}",
                                                      stored_source, stored[:, 0])
                            embed_rows += parity_rows(section, location, f"target_{scanner}",
                                                      "stored_gan_eval_vs_stored_raw",
                                                      stored_target, stored[:, s])
                        gan_provenance.append({"section": section, "condition": name,
                                               "source": "stored UNI v1 benchmark features",
                                               "path": str(path)})
                    else:
                        final[:, index[name]] = part[:, names[name]]
            for m, method in enumerate(C.FEATURE_METHODS):
                for fold in C.FOLD_IDS:
                    final[:, index[f"{method}_to_{scanner}_fold{fold}"]] = feature_parts[scanner][where, fold, m]
        if not np.isfinite(final).all():
            raise ValueError(f"{section}: non-finite final features")
        C.write_h5(out / "plism" / pfm / f"{section}.h5", {
            "features": final, "condition_names": np.asarray(conditions),
            "core": rows.core.to_numpy(np.int64), "location": location,
            "tissue_type": rows.tissue_type.to_numpy(), "pannormal_tissue": rows.pannormal_tissue.to_numpy(),
        }, {"pfm": pfm, "section": section, "analysis": "RV03 corrected_embeddings",
            "gan_source": ("stored UNI v1 benchmark features" if pfm == "uni_v1"
                           else "regenerated from frozen checkpoints (no stored features)"),
            "note": ("source_at2/target_* are stored raw embeddings; image conditions are "
                     "L2-normalised PFM outputs; ridge/combat/ols are unnormalised map outputs; "
                     "location = PLISM benchmark location id; *_fold<k> = PanNormal fold-k fit")})

        # QC: fresh raw extraction versus stored raw.
        for s, scanner in enumerate(C.PLISM_SCANNERS):
            name = "source_at2" if scanner == "at2" else f"target_{scanner}"
            embed_rows += parity_rows(section, location, name, "fresh_vs_stored_raw",
                                      part[:, names[name]], stored[:, s])
        cores = rows.core.to_numpy(np.int64)
        for s, scanner in enumerate(C.PLISM_TARGETS, start=1):
            target_fresh = part[:, names[f"target_{scanner}"]]
            raw_fresh = distance(part[:, names["source_at2"]], target_fresh)
            for j, loc in enumerate(location):
                if table2 is not None:
                    stored_value = table2.get((section, scanner, int(loc), "raw", -1), np.nan)
                    dist_rows.append({"unit": section, "scanner": scanner, "location": int(loc),
                                      "method": "raw", "fold": -1, "reference": "table2_plism_uni",
                                      "ours": float(raw_fresh[j]), "stored": float(stored_value)})
                else:
                    dist_rows.append({"unit": section, "scanner": scanner, "location": int(loc),
                                      "method": "raw", "fold": -1, "reference": "table4_crossencoder",
                                      "ours": float(raw_fresh[j]),
                                      "stored": float(table4.get((section, scanner, int(loc), "raw", 0), np.nan))})
            for method in C.IMAGE_METHODS:
                for fold in C.FOLD_IDS:
                    ours = distance(part[:, names[f"{method}_to_{scanner}_fold{fold}"]], target_fresh)
                    for j, loc in enumerate(location):
                        reference, value = None, np.nan
                        if table2 is not None and method in C.CONVENTIONAL:
                            reference = "table2_plism_uni"
                            value = table2.get((section, scanner, int(loc), method, fold), np.nan)
                        elif table4 is not None and method in ("reinhard", "frequency", "combined"):
                            reference = "table4_crossencoder"
                            value = table4.get((section, scanner, int(loc), method, fold), np.nan)
                        elif gan_cross is not None and method in C.GAN_METHODS:
                            reference = "plism_gan_crossencoder"
                            key = (section, scanner, int(loc), method, fold)
                            value = gan_cross["corrected_distance"].get(key, np.nan)
                        if reference is not None:
                            dist_rows.append({"unit": section, "scanner": scanner, "location": int(loc),
                                              "method": method, "fold": fold, "reference": reference,
                                              "ours": float(ours[j]), "stored": float(value)})
            # Final-file per-location distances.
            target_final = final[:, index[f"target_{scanner}"]]
            raw_final = distance(final[:, index["source_at2"]], target_final)
            for j, loc in enumerate(location):
                final_rows.append({"section": section, "core": int(cores[j]), "location": int(loc),
                                   "scanner": scanner, "method": "raw", "fold": -1,
                                   "distance": float(raw_final[j])})
            for method in C.METHODS:
                for fold in C.FOLD_IDS:
                    d = distance(final[:, index[f"{method}_to_{scanner}_fold{fold}"]], target_final)
                    final_rows.extend({"section": section, "core": int(cores[j]), "location": int(loc),
                                       "scanner": scanner, "method": method, "fold": fold,
                                       "distance": float(d[j])} for j, loc in enumerate(location))
        print(f"{pfm} PLISM {section}: {n} locations", flush=True)
    C.write_frame(qc / f"embedding_parity_plism_{pfm}.csv.gz", pd.DataFrame(embed_rows))
    C.write_frame(qc / f"distance_parity_plism_{pfm}.csv.gz", pd.DataFrame(dist_rows))
    C.write_frame(qc / f"distances_plism_{pfm}.csv.gz", pd.DataFrame(final_rows))
    if gan_provenance:
        C.write_frame(qc / f"plism_gan_provenance_{pfm}.csv", pd.DataFrame(gan_provenance))
    return {"files": len(C.plism_sections()), "locations": int(len(table)),
            "conditions": len(conditions)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, required=True, help="0..3 = PFM")
    parser.add_argument("--parts-root", type=Path, default=C.PARTS)
    parser.add_argument("--output-root", type=Path, default=C.OUTPUT)
    parser.add_argument("--datasets", default="pannormal,plism")
    parser.add_argument("--smoke", action="store_true", help="allow partial parts (smoke test)")
    parser.add_argument("--features-root", type=Path)
    args = parser.parse_args()
    os.chdir(C.PROJECT)
    pfm = C.PFMS[args.task_index]
    FEATURES_ROOT[0] = args.features_root or args.parts_root / "features"
    qc = args.output_root / "qc"
    start = time.time()
    report = {"pfm": pfm}
    if "pannormal" in args.datasets:
        report["pannormal"] = assemble_pannormal(pfm, args.parts_root, args.output_root, qc, args.smoke)
        print(f"{pfm}: PanNormal assembled ({time.time() - start:.0f}s)", flush=True)
    if "plism" in args.datasets:
        report["plism"] = assemble_plism(pfm, args.parts_root, args.output_root, qc, args.smoke)
        print(f"{pfm}: PLISM assembled ({time.time() - start:.0f}s)", flush=True)
    report["seconds"] = round(time.time() - start, 1)
    C.write_json(qc / f"assemble_{pfm}.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

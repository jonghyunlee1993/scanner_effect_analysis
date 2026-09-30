#!/usr/bin/env python3
"""RV03 stage 1 (CPU): feature-space corrections in four PFMs, fitted per fold.

One task per PFM x PanNormal target scanner. For each outer fold k the maps are fitted
on the stored raw embeddings of the set40 locations of the training-fold slides
(folds != k), exactly as src/manuscript_completion/feature_uni_v1.py (--train-panel 40)
and scripts/features/review_feature_crossencoder_fit.py:
* ridge affine: penalty in {0.01, 0.1, 1.0} x mean Gram diagonal, chosen on the inner
  validation fold (k + 1) mod 5 with the paper's 20-location validation embeddings
  (UNI v1: 03_uni shards; other PFMs: stored 40-location raw subset to set20);
* affine OLS: the same map with zero penalty;
* ComBat: neuroCombat with the target scanner as reference batch, no covariate.
The maps are applied to the stored raw AT2 embeddings of set20 of the held-out slides
(fold k) and, for GT450/S360/S60, to every PLISM AT2 embedding (stored per fold k).

Stored raw inputs: UNI v1 = the 40-location panel (manuscript_completion.feature_panel40);
UNI2-h / Virchow2 / H-optimus-1 = outputs/feature_crossencoder_review_2026-09-25/raw.
PLISM AT2: UNI v1 = 05_plism_external_correction/04_external_uni (target_features);
others = raw/external/<pfm>/<section>.h5 (scanner 0).

Output (intermediate): parts/features/<pfm>/<scanner>.h5 plus alpha and parity tables.
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


STORED_UNI_FEATURE = C.PROJECT / "outputs/discussion_followup_2026-09-25"
STORED_CROSS = C.PROJECT / "outputs/feature_crossencoder_review_2026-09-25/results"


def load_raw40(pfm: str, cohort: pd.DataFrame, set40: pd.DataFrame) -> np.ndarray:
    """Stored raw embeddings [slide, 40, scanner, d] in set40 location order."""
    dim = C.PFM_DIMS[pfm]
    out = np.empty((len(cohort), 40, 6, dim), dtype=np.float32)
    if pfm == "uni_v1":
        from manuscript_completion.feature_panel40 import load_panel40

        panel, positions = load_panel40(cohort)
        for i, slide_id in enumerate(cohort.slide_id):
            wanted = set40.loc[set40.slide_id.eq(slide_id), "location_index"].to_numpy(np.int64)
            index = {int(v): j for j, v in enumerate(positions[i])}
            out[i] = panel[i, [index[int(v)] for v in wanted]]
        return out
    for i, row in enumerate(cohort.itertuples(index=False)):
        path = C.CROSS_RAW / "internal" / pfm / f"fold_{row.fold}" / f"{row.slide_id}.h5"
        with h5py.File(path, "r") as store:
            names = tuple(C.decode(store["scanner_names"][:]))
            if names != C.SCANNERS or str(store.attrs["slide_id"]) != row.slide_id:
                raise ValueError(f"raw internal contract changed: {path}")
            locations = np.asarray(store["location_index"], dtype=np.int64)
            features = np.asarray(store["features"], dtype=np.float32)
        wanted = set40.loc[set40.slide_id.eq(row.slide_id), "location_index"].to_numpy(np.int64)
        if sorted(locations.tolist()) != sorted(wanted.tolist()):
            raise ValueError(f"{row.slide_id}: stored 40 locations differ from set40")
        index = {int(v): j for j, v in enumerate(locations)}
        out[i] = features[[index[int(v)] for v in wanted]]
    return out


def load_plism_source(pfm: str) -> tuple[np.ndarray, pd.DataFrame]:
    table = C.plism_locations()
    blocks = []
    for section in C.plism_sections():
        wanted = table.loc[table.section.eq(section), "location"].to_numpy(np.int64)
        if pfm == "uni_v1":
            with h5py.File(C.PLISM_UNI_RAW / "gt450" / f"{section}.h5", "r") as store:
                location = np.asarray(store["location"], dtype=np.int64)
                source = np.asarray(store["target_features"], dtype=np.float32)  # AT2
        else:
            path = C.CROSS_RAW / "external" / pfm / f"{section}.h5"
            with h5py.File(path, "r") as store:
                if tuple(C.decode(store["scanner_names"][:])) != C.PLISM_SCANNERS:
                    raise ValueError(f"raw external contract changed: {path}")
                location = np.asarray(store["location_index"], dtype=np.int64)
                source = np.asarray(store["features"][:, 0, :], dtype=np.float32)
        if not np.array_equal(location, wanted):
            raise ValueError(f"{pfm}/{section}: stored PLISM location order differs")
        blocks.append(source)
    source = np.concatenate(blocks)
    if source.shape != (C.PLISM_LOCATIONS, C.PFM_DIMS[pfm]):
        raise ValueError(f"{pfm}: unexpected PLISM source shape {source.shape}")
    return source, table


def load_plism_target(pfm: str, scanner: str) -> np.ndarray:
    blocks = []
    for section in C.plism_sections():
        if pfm == "uni_v1":
            with h5py.File(C.PLISM_UNI_RAW / scanner / f"{section}.h5", "r") as store:
                blocks.append(np.asarray(store["source_features"], dtype=np.float32))
        else:
            with h5py.File(C.CROSS_RAW / "external" / pfm / f"{section}.h5", "r") as store:
                blocks.append(np.asarray(store["features"][:, C.PLISM_SCANNERS.index(scanner), :],
                                         dtype=np.float32))
    return np.concatenate(blocks)


def paper_validation_arrays(pfm: str, cohort: pd.DataFrame, raw20: np.ndarray, scanner_index: int):
    """The 20-location embeddings the paper used for the inner validation loss."""
    if pfm != "uni_v1":
        return raw20[:, :, 0], raw20[:, :, scanner_index]
    from manuscript_completion.pfm_problem import (
        DEFAULT_COHORT, DEFAULT_FOLDS, DEFAULT_INPUT_ROOT, audit_and_load_raw_embeddings,
        load_contract,
    )
    contract = load_contract(DEFAULT_COHORT, DEFAULT_FOLDS)
    if contract.slide_id.tolist() != cohort.slide_id.tolist():
        raise ValueError("paper contract order differs from the RV03 cohort order")
    embeddings, _, _ = audit_and_load_raw_embeddings(contract, DEFAULT_INPUT_ROOT)
    return embeddings[:, :, 0, :], embeddings[:, :, scanner_index, :]


def main(pfm: str, scanner: str, out_root: Path) -> None:
    from prenorm.feature_correction import apply_affine, fit_affine, fit_combat, paired_distance

    os.chdir(C.PROJECT)  # feature_panel40 and pfm_problem use repository-relative paths
    start = time.time()
    cohort = C.load_cohort()
    set40 = C.load_set(C.SET40, 40)
    set20 = C.load_set(C.SET20, 20)
    scanner_index = C.SCANNERS.index(scanner)
    raw40 = load_raw40(pfm, cohort, set40)
    positions = []
    for slide_id in cohort.slide_id:
        forty = set40.loc[set40.slide_id.eq(slide_id), "location_index"].tolist()
        twenty = set20.loc[set20.slide_id.eq(slide_id), "location_index"].tolist()
        positions.append([forty.index(v) for v in twenty])
    positions = np.asarray(positions)
    raw20 = np.stack([raw40[i, positions[i]] for i in range(len(cohort))])
    dim = raw40.shape[-1]
    if not np.isfinite(raw40).all():
        raise ValueError("non-finite stored raw embeddings")
    source40 = raw40[:, :, 0].astype(np.float64)
    target40 = raw40[:, :, scanner_index].astype(np.float64)
    apply_source = raw20[:, :, 0].astype(np.float64)
    apply_target = raw20[:, :, scanner_index].astype(np.float64)
    val_source, val_target = paper_validation_arrays(pfm, cohort, raw20, scanner_index)
    val_source = np.asarray(val_source, dtype=np.float64)
    val_target = np.asarray(val_target, dtype=np.float64)
    use_plism = scanner in C.PLISM_TARGETS
    if use_plism:
        plism_source, plism_table = load_plism_source(pfm)
        plism_source = plism_source.astype(np.float64)
        plism_target = load_plism_target(pfm, scanner).astype(np.float64)
    folds = cohort.fold.to_numpy(int)
    pan = np.full((len(cohort), 20, len(C.FEATURE_METHODS), dim), np.nan, dtype=np.float32)
    plism = (np.full((C.PLISM_LOCATIONS, 5, len(C.FEATURE_METHODS), dim), np.nan, dtype=np.float32)
             if use_plism else None)
    replica = np.full((len(cohort), 20, len(C.FEATURE_METHODS), dim), np.nan, dtype=np.float32)
    alpha_rows = []
    for fold in C.FOLD_IDS:
        train = folds != fold
        test = folds == fold
        validation = folds == (fold + 1) % 5
        inner = train & ~validation
        xin = source40[inner].reshape(-1, dim)
        yin = target40[inner].reshape(-1, dim)
        xval = val_source[validation].reshape(-1, dim)
        yval = val_target[validation].reshape(-1, dim)
        losses = [float(paired_distance(apply_affine(fit_affine(xin, yin, alpha), xval), yval).mean())
                  for alpha in C.RIDGE_ALPHAS]
        alpha = C.RIDGE_ALPHAS[int(np.argmin(losses))]
        xtrain = source40[train].reshape(-1, dim)
        ytrain = target40[train].reshape(-1, dim)
        ridge = fit_affine(xtrain, ytrain, alpha)
        ols = fit_affine(xtrain, ytrain, 0.0)
        xtest = apply_source[test].reshape(-1, dim)
        shape = (int(test.sum()), 20, dim)
        blocks = [xtest]
        replica_test = val_source[test].reshape(-1, dim)  # the paper's own evaluation input
        if pfm == "uni_v1":
            blocks.append(replica_test)
        if use_plism:
            blocks.append(plism_source)
        combat_all = fit_combat(xtrain, ytrain, np.concatenate(blocks))
        n_test = len(xtest)
        pan[test, :, 0] = apply_affine(ridge, xtest).reshape(shape)
        pan[test, :, 1] = combat_all[:n_test].reshape(shape)
        pan[test, :, 2] = apply_affine(ols, xtest).reshape(shape)
        offset = n_test
        if pfm == "uni_v1":
            replica[test, :, 0] = apply_affine(ridge, replica_test).reshape(shape)
            replica[test, :, 1] = combat_all[offset:offset + n_test].reshape(shape)
            replica[test, :, 2] = apply_affine(ols, replica_test).reshape(shape)
            offset += n_test
        else:
            replica[test] = pan[test]
        if use_plism:
            plism[:, fold, 0] = apply_affine(ridge, plism_source)
            plism[:, fold, 1] = combat_all[offset:offset + C.PLISM_LOCATIONS]
            plism[:, fold, 2] = apply_affine(ols, plism_source)
        alpha_rows.append({"pfm": pfm, "scanner": scanner, "fold": fold,
                           "chosen_relative_alpha": alpha, "inner_losses": json.dumps(losses),
                           "train_slides": int(train.sum()), "train_pairs": int(len(xtrain))})
        print(f"{pfm} {scanner} fold {fold}: alpha={alpha} ({time.time() - start:.0f}s)", flush=True)
    if not np.isfinite(pan).all() or (use_plism and not np.isfinite(plism).all()):
        raise ValueError("non-finite corrected features")

    # Parity with the stored paper results (per slide / per section-fold distances).
    alpha = pd.DataFrame(alpha_rows)
    if pfm == "uni_v1":
        stored_alpha = pd.read_csv(STORED_UNI_FEATURE / "feature40_forward/ridge_selection.csv")
        stored_alpha = stored_alpha.rename(columns={"outer_fold": "fold"})
    else:
        stored_alpha = pd.read_csv(STORED_CROSS / pfm / scanner / "ridge_selection.csv")
    stored_alpha = stored_alpha[stored_alpha.scanner.eq(scanner)].set_index("fold")
    alpha["stored_alpha"] = alpha.fold.map(stored_alpha["chosen_relative_alpha"])
    alpha["alpha_matches_stored"] = np.isclose(alpha.chosen_relative_alpha, alpha.stored_alpha)

    rows = []
    stored_slide = (pd.read_csv(STORED_UNI_FEATURE / "feature40_forward/per_slide.csv",
                                dtype={"slide_id": str}) if pfm == "uni_v1" else
                    pd.read_csv(STORED_CROSS / pfm / scanner / "per_slide.csv", dtype={"slide_id": str}))
    stored_slide = stored_slide[stored_slide.scanner.eq(scanner)]
    names = {"ridge": "featmap_ridge", "combat": "combat", "ols": "featmap_ols"}
    for m, method in enumerate(C.FEATURE_METHODS):
        ours = paired_distance(pan[:, :, m].astype(np.float64), apply_target).mean(axis=1)
        paper_input = paired_distance(replica[:, :, m].astype(np.float64), val_target).mean(axis=1)
        reference = stored_slide[stored_slide.method.eq(names[method])].set_index("slide_id")
        for i, slide_id in enumerate(cohort.slide_id):
            rows.append({"pfm": pfm, "scanner": scanner, "method": method, "slide_id": slide_id,
                         "distance_rv03": float(ours[i]),
                         "distance_paper_inputs": float(paper_input[i]),
                         "stored_distance": float(reference["corrected_distance"].get(slide_id, np.nan))})
    internal = pd.DataFrame(rows)
    external = pd.DataFrame()
    if use_plism:
        stored_ext = (pd.read_csv(STORED_UNI_FEATURE / "feature40_external_forward/per_section_fold.csv")
                      if pfm == "uni_v1" else
                      pd.read_csv(STORED_CROSS / pfm / scanner / "external_per_section_fold.csv"))
        stored_ext = stored_ext[stored_ext.scanner.eq(scanner)].set_index(["section", "fold", "method"])
        erows = []
        sections = plism_table.section.to_numpy()
        for m, method in enumerate(C.FEATURE_METHODS):
            for fold in C.FOLD_IDS:
                d = paired_distance(plism[:, fold, m].astype(np.float64), plism_target)
                for section in C.plism_sections():
                    key = (section, fold, names[method])
                    erows.append({"pfm": pfm, "scanner": scanner, "method": method, "fold": fold,
                                  "section": section,
                                  "distance_rv03": float(d[sections == section].mean()),
                                  "stored_distance": float(stored_ext["corrected_distance"].get(key, np.nan))})
        external = pd.DataFrame(erows)

    datasets = {"pannormal_features": pan, "pannormal_slide_id": cohort.slide_id.to_numpy(),
                "pannormal_location_index": np.stack([set20.loc[set20.slide_id.eq(s), "location_index"]
                                                      .to_numpy(np.int64) for s in cohort.slide_id]),
                "methods": np.asarray(C.FEATURE_METHODS)}
    if use_plism:
        datasets.update({"plism_features": plism, "plism_section": plism_table.section.to_numpy(),
                         "plism_location": plism_table.location.to_numpy(np.int64)})
    C.write_h5(out_root / "features" / pfm / f"{scanner}.h5", datasets,
               {"pfm": pfm, "scanner": scanner, "stage": "rv03_features",
                "fit": "set40 of training-fold slides; applied to set20 held-out slides and PLISM"})
    C.write_frame(out_root / "features" / pfm / f"{scanner}_ridge_selection.csv", alpha)
    C.write_frame(out_root / "features" / pfm / f"{scanner}_internal_parity.csv", internal)
    if use_plism:
        C.write_frame(out_root / "features" / pfm / f"{scanner}_external_parity.csv", external)
    summary = {"pfm": pfm, "scanner": scanner, "seconds": round(time.time() - start, 1),
               "alpha_matches_stored": bool(alpha.alpha_matches_stored.all()),
               "internal_max_abs_diff_paper_inputs": float(
                   (internal.distance_paper_inputs - internal.stored_distance).abs().max()),
               "internal_max_abs_diff_rv03": float(
                   (internal.distance_rv03 - internal.stored_distance).abs().max())}
    if use_plism:
        summary["external_max_abs_diff"] = float(
            (external.distance_rv03 - external.stored_distance).abs().max())
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, required=True, help="pfm x scanner, 0..19")
    parser.add_argument("--output-root", type=Path, default=C.PARTS)
    args = parser.parse_args()
    tasks = [(pfm, scanner) for pfm in C.PFMS for scanner in C.PAN_TARGETS]
    pfm, scanner = tasks[args.task_index]
    main(pfm, scanner, args.output_root)

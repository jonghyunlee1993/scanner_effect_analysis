#!/usr/bin/env python3
"""RV03 stage 1 (GPU): embed raw and image-corrected patches in four frozen PFMs.

Tasks 0..pan_tasks-1 take interleaved chunks of the 103 PanNormal slides (set20,
AT2 -> VERSA/AKOYA/GT450/S360/S60, parameters of the fold in which the slide is held
out). Tasks pan_tasks..pan_tasks+12 take one PLISM section each (AT2 -> GT450/S360/S60,
every PanNormal-fitted fold k = 0..4).

Images are produced exactly as in the paper's producers and are not stored:
* Reinhard / frequency / colour + frequency: scanner_batch_extensions transforms with
  02_correction/parameters.json (as in 03_uni and table4 run.py).
* Macenko / Vahadane, PanNormal: manuscript_completion.stain.normalize with the stain_v4
  references and the seed convention of panel_a.stain_uni_extract (the producer of the
  stored UNI embeddings). PLISM: one source fit per location with seed
  stable_seed("table2", section, location) and the fold reference, as in
  analysis/paper/table2_benchmark.py and table2_gt450_extension.py.
* Pix2Pix / CycleGAN, PanNormal: stored predictions subset to set20.
  PLISM: regenerated from the frozen checkpoints with scanner_gan.plism._generate_uint8
  (bfloat16, batch 16) as in scanner_gan.plism_bidirectional.evaluate_task.
Embedding: UNI2-h, Virchow2 and H-optimus-1 use
scripts.features.review_feature_crossencoder_extract.images_to_features (load_encoder/embed
from review_multiencoder_scanner), as in the paper's cross-PFM analyses. UNI v1 uses the
paper's primary path, prenorm.embedding.load_uni/embed_uni through
scanner_gan.uni.embed_frozen_uni(value_range="uint8") (fp32, bicubic to 224, no clamp after
resizing), which produced the stored UNI v1 benchmark features.

Outputs (intermediate): results/corrected_embeddings/parts/{pannormal,plism}/<pfm>/*.h5
and generation logs in parts/{pannormal,plism}/generation/.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import get_context
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import corrected_embeddings_common as C


# ----------------------------------------------------------------------------- CPU workers

_STATE: dict = {}


def _init_worker(state: dict) -> None:
    _STATE.update(state)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:  # pragma: no cover - threadpoolctl is optional
        pass


def _pan_location(args):
    """All conventional corrections of one PanNormal AT2 patch for five targets."""
    from manuscript_completion.panel_a import stable_seed
    from manuscript_completion.stain import normalize
    from scanner_batch_extensions import frequency_transform, reinhard_transform

    slide_id, location, source = args
    fitted = _STATE["parameters"]
    references = _STATE["references"]
    vahadane_kwargs = _STATE["vahadane_kwargs"]
    images, flags = {}, []
    for scanner in C.PAN_TARGETS:
        reinhard = reinhard_transform(source, fitted[scanner])
        images[("reinhard", scanner)] = reinhard
        images[("frequency", scanner)] = frequency_transform(source, fitted[scanner])
        images[("combined", scanner)] = frequency_transform(reinhard, fitted[scanner])
        for method in ("macenko", "vahadane"):
            report = normalize(
                source, references[scanner][method], method=method,
                seed=stable_seed(slide_id, int(location), scanner, method),
                vahadane_config=vahadane_kwargs if method == "vahadane" else None,
            )
            images[(method, scanner)] = report.image
            flags.append({
                "slide_id": slide_id, "location_index": int(location), "scanner": scanner,
                "method": method, "fallback": bool(report.fallback),
                "converged": bool(report.source_parameters.converged)
                if report.source_parameters is not None else False,
                "clipped_fraction": float(report.clipped_fraction),
            })
    return int(location), images, flags


def _plism_location(args):
    """All conventional corrections of one PLISM AT2 patch for 3 targets x 5 folds."""
    from manuscript_completion.panel_a import stable_seed
    from manuscript_completion.plism_external import (
        _apply_stain_from_source, _stain_source_parameters,
    )
    from scanner_batch_extensions import frequency_transform, reinhard_transform

    section, location, source = args
    folds = _STATE["parameters"]
    references = _STATE["references"]
    fitted_source = _stain_source_parameters(
        source, stable_seed("table2", section, int(location)), _STATE["vahadane_configuration"]
    )
    images, flags = {}, []
    for method in ("macenko", "vahadane"):
        value = fitted_source[method]
        flags.append({"section": section, "location": int(location), "method": method,
                      "fallback": value is None,
                      "converged": bool(value.converged) if value is not None else False})
    for scanner in C.PLISM_TARGETS:
        for fold in C.FOLD_IDS:
            parameter = folds[str(fold)][scanner]
            reinhard = reinhard_transform(source, parameter)
            images[("reinhard", scanner, fold)] = reinhard
            images[("frequency", scanner, fold)] = frequency_transform(source, parameter)
            images[("combined", scanner, fold)] = frequency_transform(reinhard, parameter)
            for method in ("macenko", "vahadane"):
                corrected, report = _apply_stain_from_source(
                    source, fitted_source[method], references[fold][scanner][method]
                )
                images[(method, scanner, fold)] = corrected
                if report["stain_fallback"]:
                    flags.append({"section": section, "location": int(location),
                                  "method": method, "scanner": scanner, "fold": fold,
                                  "fallback": True, "converged": report["stain_converged"]})
    return int(location), images, flags


def _stain_references() -> dict:
    from manuscript_completion.stain import parameters_from_json

    references = {}
    for fold in C.FOLD_IDS:
        references[fold] = {}
        for scanner in C.PAN_TARGETS:
            payload = json.loads((C.STAIN_REFERENCES / f"fold_{fold}" / f"{scanner}.json").read_text())
            references[fold][scanner] = {
                method: parameters_from_json(payload["parameters"][method])
                for method in ("macenko", "vahadane")
            }
    return references


# ----------------------------------------------------------------------------- GPU helpers

class Encoders:
    def __init__(self, names: tuple[str, ...], batch_size: int):
        from scripts.features.review_multiencoder_scanner import load_encoder

        import torch

        self.batch_size = batch_size
        self.models = {}
        for name in names:
            start = time.time()
            if name == "uni_v1":
                # The paper's UNI v1 path (src/prenorm/embedding.py): load_uni + embed_uni via
                # scanner_gan.uni.embed_frozen_uni(value_range="uint8"), fp32, bicubic 224 without
                # post-resize clamping (03_uni, stain_v4, 04_uni GAN and PLISM benchmark features).
                from prenorm.embedding import load_uni

                model, size, mean, std = load_uni(torch.device("cuda"))
                self.models[name] = (model, "uni_v1", size, mean, std)
            else:
                model, size, mean, std, variants = load_encoder(name, C.HF_CACHE)
                if len(variants) != 1:
                    raise ValueError(f"{name}: expected one embedding variant")
                self.models[name] = (model, variants[0], size, mean, std)
            print(f"loaded {name} in {time.time() - start:.0f}s", flush=True)

    def __call__(self, images: np.ndarray, chunk: int = 512) -> dict[str, np.ndarray]:
        from scripts.features.review_feature_crossencoder_extract import images_to_features

        import torch
        from scanner_gan.uni import embed_frozen_uni

        out = {}
        for name, (model, variant, size, mean, std) in self.models.items():
            if name == "uni_v1":
                parts = [embed_frozen_uni(model, images[start:start + chunk], size, mean, std,
                                          torch.device("cuda"), batch_size=self.batch_size,
                                          value_range="uint8")
                         for start in range(0, len(images), chunk)]
            else:
                parts = [images_to_features(model, variant, name, images[start:start + chunk],
                                            size, mean, std, self.batch_size)
                         for start in range(0, len(images), chunk)]
            value = np.concatenate(parts).astype(np.float32)
            if value.shape != (len(images), C.PFM_DIMS[name]) or not np.isfinite(value).all():
                raise ValueError(f"{name}: invalid embedding output {value.shape}")
            out[name] = value
        return out


# ----------------------------------------------------------------------------- PanNormal

def pannormal_task(slides: list[str], encoders_names, args, out_root: Path) -> None:
    from manuscript_completion.panel_a import load_vahadane_selection, selected_vahadane_kwargs

    cohort = C.load_cohort().set_index("slide_id")
    set20 = C.load_set(C.SET20, 20)
    pairs = pd.read_csv(C.GAN_PAIRS, dtype={"slide_id": str})
    parameters = json.loads(C.PARAMETERS.read_text())["fold"]
    references = _stain_references()
    vahadane_kwargs = selected_vahadane_kwargs(load_vahadane_selection(C.VAHADANE_SELECTION),
                                               reference=False)
    image_conditions = C.pannormal_conditions(C.IMAGE_METHODS)
    lookup = {name: index for index, name in enumerate(image_conditions)}

    # Stage A (CPU): read and correct every assigned slide before CUDA is initialised.
    prepared = []
    for slide_id in slides:
        record = cohort.loc[slide_id]
        rows = set20[set20.slide_id.eq(slide_id)].sort_values("location_index")
        if int(rows.fold.iloc[0]) != int(record.fold) or rows.tissue_type.iloc[0] != record.tissue_type:
            raise ValueError(f"{slide_id}: set20 and cohort disagree")
        locations = rows.location_index.to_numpy(np.int64)
        if args.max_locations:
            locations = locations[:args.max_locations]
        fold = int(record.fold)
        with h5py.File(record.cache_path, "r") as cache:
            names = tuple(x.decode().lower() for x in cache["scanner_names"][:])
            if names != C.SCANNERS:
                raise ValueError(f"{slide_id}: scanner order changed {names}")
            paired = np.asarray(cache["images"][locations], dtype=np.uint8)
            source_index = np.asarray(cache["source_index"][locations], dtype=np.int64)
        if not np.array_equal(source_index, rows.source_index.to_numpy(np.int64)[:len(locations)]):
            raise ValueError(f"{slide_id}: cache source_index differs from set20")
        n = len(locations)
        images = np.zeros((len(image_conditions), n, 256, 256, 3), dtype=np.uint8)
        images[lookup["source_at2"]] = paired[:, 0]
        for position, scanner in enumerate(C.PAN_TARGETS, start=1):
            images[lookup[f"target_{scanner}"]] = paired[:, position]
        state = {"parameters": parameters[str(fold)], "references": references[fold],
                 "vahadane_kwargs": vahadane_kwargs}
        start = time.time()
        with get_context("fork").Pool(args.workers, initializer=_init_worker,
                                      initargs=(state,)) as pool:
            results = pool.map(_pan_location, [(slide_id, int(loc), paired[i, 0])
                                               for i, loc in enumerate(locations)])
        flags = []
        position_of = {int(loc): i for i, loc in enumerate(locations)}
        for location, generated, flag in results:
            for (method, scanner), image in generated.items():
                images[lookup[f"{method}_to_{scanner}"], position_of[location]] = image
            flags.extend(flag)
        gan_checks = []
        for position, scanner in enumerate(C.PAN_TARGETS, start=1):
            pair = pairs[pairs.slide_id.eq(slide_id) & pairs.scanner.eq(scanner)]
            if len(pair) != 1 or int(pair.fold.iloc[0]) != fold:
                raise ValueError(f"{slide_id}/{scanner}: missing audited GAN prediction pair")
            for method in C.GAN_METHODS:
                path = Path(pair[f"{method}_path"].iloc[0])
                with h5py.File(path, "r") as store:
                    observed = np.asarray(store["metadata/location_index"], dtype=np.int64)
                    index = {int(v): i for i, v in enumerate(observed)}
                    take = np.asarray([index[int(v)] for v in locations])
                    generated = np.asarray(store["images/generated"][:], dtype=np.uint8)[take]
                    raw_source = np.asarray(store["images/raw_source"][:], dtype=np.uint8)[take]
                    real_target = np.asarray(store["images/real_target"][:], dtype=np.uint8)[take]
                source_equal = bool(np.array_equal(raw_source, paired[:, 0]))
                target_equal = bool(np.array_equal(real_target, paired[:, position]))
                if not (source_equal and target_equal):
                    raise ValueError(f"{slide_id}/{scanner}/{method}: GAN source/target != cache")
                images[lookup[f"{method}_to_{scanner}"]] = generated
                gan_checks.append({"slide_id": slide_id, "scanner": scanner, "method": method,
                                   "prediction_path": str(path), "source_equals_cache": source_equal,
                                   "target_equals_cache": target_equal})
        print(f"PanNormal {slide_id}: {n} locations, images ready in {time.time() - start:.0f}s",
              flush=True)
        prepared.append((slide_id, record, locations, source_index, images, flags, gan_checks))

    # Stage B (GPU): embed with all PFMs.
    encoders = Encoders(encoders_names, args.batch_size)
    for slide_id, record, locations, source_index, images, flags, gan_checks in prepared:
        start = time.time()
        n = len(locations)
        flat = images.reshape(-1, 256, 256, 3)
        embedded = encoders(flat)
        for name, vectors in embedded.items():
            features = vectors.reshape(len(image_conditions), n, -1).transpose(1, 0, 2)
            C.write_h5(out_root / "pannormal" / name / f"{slide_id}.h5", {
                "features": features, "condition_names": np.asarray(image_conditions),
                "location_index": locations, "source_index": source_index,
            }, {"pfm": name, "slide_id": slide_id, "tissue_type": str(record.tissue_type),
                "fold": int(record.fold), "stage": "rv03_images"})
        C.write_frame(out_root / "pannormal/generation" / f"{slide_id}_stain_flags.csv",
                      pd.DataFrame(flags))
        C.write_frame(out_root / "pannormal/generation" / f"{slide_id}_gan_checks.csv",
                      pd.DataFrame(gan_checks))
        print(f"PanNormal {slide_id}: embedded {flat.shape[0]} images x {len(embedded)} PFMs "
              f"in {time.time() - start:.0f}s", flush=True)


# ----------------------------------------------------------------------------- PLISM

def _read_render(scanner: str, section: str, renders: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    path = C.plism_render_path(scanner, section)
    observed = C.sha256(path)
    if scanner == "gt450":
        summary = json.loads(path.with_suffix(".summary.json").read_text())
        expected = summary["output_sha256"]
    else:
        match = renders[renders.scanner.str.lower().eq(scanner) & renders.stain.astype(str).eq(section)]
        if len(match) != 1 or Path(match.path.iloc[0]) != path:
            raise ValueError(f"render manifest mismatch: {scanner}/{section}")
        expected = match.sha256.iloc[0]
    if observed != expected:
        raise ValueError(f"frozen PLISM render changed: {path}")
    with h5py.File(path, "r") as store:
        images = np.asarray(store["images"], dtype=np.uint8)
        location = np.asarray(store["location"], dtype=np.int64)
        core = np.asarray(store["core"], dtype=np.int64)
    return images, location, core, {"path": str(path), "sha256": observed}


def plism_task(section: str, encoders_names, args, out_root: Path) -> None:
    import torch
    from manuscript_completion.panel_a import load_vahadane_selection
    from scanner_gan.plism import _generate_uint8
    from scanner_gan.predict import load_validated_generator
    from scanner_gan.train_cyclegan import load_validated_cyclegan_generator

    table = C.plism_locations()
    table = table[table.section.eq(section)].sort_values("location")
    renders = pd.read_csv(C.PLISM_RENDERS)
    source, location, core, provenance = _read_render("at2", section, renders)
    if not np.array_equal(location, table.location.to_numpy(np.int64)):
        raise ValueError(f"{section}: AT2 render locations differ from the locked selection")
    if not np.array_equal(core, table.core.to_numpy(np.int64)):
        raise ValueError(f"{section}: AT2 render cores differ from the locked selection")
    targets, sources = {}, {"at2": provenance}
    for scanner in C.PLISM_TARGETS:
        images, observed, _, info = _read_render(scanner, section, renders)
        if not np.array_equal(observed, location):
            raise ValueError(f"{section}/{scanner}: paired render locations differ")
        targets[scanner] = images
        sources[scanner] = info
    if args.max_locations:
        keep = slice(0, args.max_locations)
        source, location, core = source[keep], location[keep], core[keep]
        targets = {key: value[keep] for key, value in targets.items()}
    n = len(location)
    image_conditions = C.plism_conditions(C.IMAGE_METHODS)
    lookup = {name: index for index, name in enumerate(image_conditions)}
    images = np.zeros((len(image_conditions), n, 256, 256, 3), dtype=np.uint8)
    images[lookup["source_at2"]] = source
    for scanner in C.PLISM_TARGETS:
        images[lookup[f"target_{scanner}"]] = targets[scanner]

    selection = load_vahadane_selection(C.VAHADANE_SELECTION)
    state = {"parameters": json.loads(C.PARAMETERS.read_text())["fold"],
             "references": _stain_references(),
             "vahadane_configuration": selection["selected_configuration"]}
    start = time.time()
    with get_context("fork").Pool(args.workers, initializer=_init_worker, initargs=(state,)) as pool:
        results = pool.map(_plism_location, [(section, int(loc), source[i])
                                             for i, loc in enumerate(location)], chunksize=4)
    flags = []
    position_of = {int(loc): i for i, loc in enumerate(location)}
    for loc, generated, flag in results:
        for (method, scanner, fold), image in generated.items():
            images[lookup[f"{method}_to_{scanner}_fold{fold}"], position_of[loc]] = image
        flags.extend(flag)
    print(f"PLISM {section}: conventional images for {n} locations in {time.time() - start:.0f}s",
          flush=True)

    device = torch.device("cuda")
    gan_rows = []
    for method in C.GAN_METHODS:
        loader = load_validated_generator if method == "pix2pix" else load_validated_cyclegan_generator
        for scanner in C.PLISM_TARGETS:
            for fold in C.FOLD_IDS:
                candidate = C.plism_gan_candidate(method, scanner, fold)
                checkpoint = Path(str(candidate.checkpoint_path))
                digest = C.sha256(checkpoint)
                if digest != str(candidate.checkpoint_sha256):
                    raise ValueError(f"frozen {method} checkpoint changed: {checkpoint}")
                generator, _, _ = loader(checkpoint, source_scanner="at2", target_scanner=scanner,
                                         test_fold=fold, device=device)
                generated = _generate_uint8(generator, source, device, batch_size=16,
                                            amp="bfloat16")
                images[lookup[f"{method}_to_{scanner}_fold{fold}"]] = generated
                gan_rows.append({"section": section, "method": method, "scanner": scanner,
                                 "fold": fold, "checkpoint_path": str(checkpoint),
                                 "checkpoint_sha256": digest})
                del generator
                torch.cuda.empty_cache()
    print(f"PLISM {section}: GAN images regenerated", flush=True)

    encoders = Encoders(encoders_names, args.batch_size)
    start = time.time()
    embedded = encoders(images.reshape(-1, 256, 256, 3))
    for name, vectors in embedded.items():
        features = vectors.reshape(len(image_conditions), n, -1).transpose(1, 0, 2)
        C.write_h5(out_root / "plism" / name / f"{section}.h5", {
            "features": features, "condition_names": np.asarray(image_conditions),
            "location": location, "core": core,
        }, {"pfm": name, "section": section, "stage": "rv03_images",
            "renders": json.dumps(sources)})
    C.write_frame(out_root / "plism/generation" / f"{section}_stain_flags.csv", pd.DataFrame(flags))
    C.write_frame(out_root / "plism/generation" / f"{section}_gan_checkpoints.csv",
                  pd.DataFrame(gan_rows))
    print(f"PLISM {section}: embedded {images.shape[0] * n} images in {time.time() - start:.0f}s",
          flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--pan-tasks", type=int, default=6)
    parser.add_argument("--pfms", default=",".join(C.PFMS))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    parser.add_argument("--max-slides", type=int)
    parser.add_argument("--max-locations", type=int)
    parser.add_argument("--output-root", type=Path, default=C.PARTS)
    args = parser.parse_args()
    names = tuple(item for item in args.pfms.split(",") if item)
    if not set(names) <= set(C.PFMS):
        raise ValueError(names)
    os.chdir(C.PROJECT)
    if args.task_index < args.pan_tasks:
        slides = C.load_cohort().slide_id.tolist()[args.task_index::args.pan_tasks]
        if args.max_slides:
            slides = slides[:args.max_slides]
        pannormal_task(slides, names, args, args.output_root)
    else:
        sections = C.plism_sections()
        section = sections[args.task_index - args.pan_tasks]
        plism_task(section, names, args, args.output_root)
    print(json.dumps({"status": "complete", "task_index": args.task_index}), flush=True)


if __name__ == "__main__":
    main()

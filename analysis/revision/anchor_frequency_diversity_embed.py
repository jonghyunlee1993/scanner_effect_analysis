#!/usr/bin/env python3
"""RV13 stage 1 (GPU): render every image condition once per slide and embed it with the new models.

PanNormal `set20` (2,060 locations), 81 images per location:
* raw six scanners (``source_at2``, ``target_<scanner>``) from the paired patch cache;
* Reinhard, frequency and colour + frequency, AT2 -> each of the five targets, produced exactly
  as RV03 (``corrected_embeddings_images._pan_location``: ``scanner_batch_extensions``
  ``reinhard_transform`` / ``frequency_transform`` with ``02_correction/parameters.json`` of
  the fold in which the slide is held out);
* the 60 band manipulations of RV02 (``band_manipulation_set20_features.render_location``,
  imported unchanged: Reinhard colour match, low-mid / mid / high OD band, decrease and
  increase, doses 0.25 and 0.50 of the weakest band RMS); names
  ``band_<band>_<dec|inc>_d<dose>_to_<scanner>``, target direction in ``render/``.
Each image enters every model as float RGB in [0, 1] (uint8 / 255; band renders unquantized).
EXAONEPath additionally gets the released Macenko normalization of every image (CPU workers;
failures recorded per image in ``render/<slide>_macenko.csv.gz``).

Models (``anchor_frequency_diversity_models``): CONCH and SEAL-CONCH (pre-projection and
projected from one pass), SEAL-UNI2, PLIP, DINOv2 ViT-L/14, EXAONEPath as released and
without Macenko. Outputs: ``embeddings/<model>/<slide>.h5`` (``features`` [20, 81, d] unit
vectors, ``condition_names``, ``location_index``, ``source_index``).

In-task QC (``qc/``): (a) the regenerated Reinhard / frequency / colour + frequency images are
embedded with UNI v1 through the RV03 path and compared with the RV03 shards (fatal below
cosine 0.999: proves identical inputs); (b) the band renders are compared row by row with the
RV02 shards (dose, achieved dose, coefficient, target direction; fatal if not identical) and
their colour-matched base with the RV03-path Reinhard image (fatal if not identical);
(c) CONCH, SEAL-CONCH and SEAL-UNI2 at the review's three locations per slide are compared with
``outputs/encoder_review_2026-09-25/features`` (recorded; threshold 0.999 read in aggregation).
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
REVISION = Path(__file__).resolve().parent
for _path in (PROJECT, PROJECT / "src", PROJECT / "scripts", REVISION):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import corrected_embeddings_common as C  # noqa: E402
from anchor_frequency_diversity_models import DIMS, ENCODERS, NEW_MODELS, Encoder, ReleasedMacenko  # noqa: E402

OUTPUT = REVISION / "results/anchor_frequency_diversity"
REVIEW_FEATURES = PROJECT / "outputs/encoder_review_2026-09-25/features"
RV02_SHARDS = REVISION / "results/band_manipulation_set20/shards/uni_v1"
RV03_SHARDS = REVISION / "results/corrected_embeddings/pannormal/uni_v1"
TARGETS = C.PAN_TARGETS
BANDS = ("low_mid", "mid", "high")
DOSES = (0.25, 0.5)
RV03_METHODS = ("reinhard", "frequency", "combined")
UNI_PARITY_MIN = 0.999


def band_name(band: str, sign: int, dose: float, scanner: str) -> str:
    return f"band_{band}_{'inc' if sign > 0 else 'dec'}_d{dose:g}_to_{scanner}"


CONDITIONS = (["source_at2"] + [f"target_{s}" for s in TARGETS]
              + [f"{m}_to_{s}" for s in TARGETS for m in RV03_METHODS]
              + [band_name(b, sign, d, s) for s in TARGETS for d in DOSES for b in BANDS for sign in (-1, 1)])
if len(CONDITIONS) != 81 or len(set(CONDITIONS)) != 81:
    raise AssertionError("condition list")

_STATE: dict = {}


def _init_worker() -> None:
    import torch

    torch.set_num_threads(1)


def _location(job):
    """All 81 images of one location (+ Macenko versions and band metadata)."""
    from band_manipulation_set20_features import render_location
    from scanner_batch_extensions import frequency_transform, reinhard_transform

    slide_id, location, paired, fitted = job
    source = paired[0]
    images: dict[str, np.ndarray] = {"source_at2": source.astype(np.float32) / 255.0}
    uint8: dict[str, np.ndarray] = {}
    for position, scanner in enumerate(TARGETS, start=1):
        images[f"target_{scanner}"] = paired[position].astype(np.float32) / 255.0
        reinhard = reinhard_transform(source, fitted[scanner])
        uint8[f"reinhard_to_{scanner}"] = reinhard
        uint8[f"frequency_to_{scanner}"] = frequency_transform(source, fitted[scanner])
        uint8[f"combined_to_{scanner}"] = frequency_transform(reinhard, fitted[scanner])
    for name, value in uint8.items():
        images[name] = value.astype(np.float32) / 255.0
    rendered, slots = render_location((location, paired, fitted))
    rows = []
    for slot in slots:
        scanner = slot["scanner"]
        if not np.array_equal(rendered[slot["base"]], images[f"reinhard_to_{scanner}"]):
            raise ValueError(f"{slide_id}/{location}/{scanner}: RV02 colour-matched base != RV03 Reinhard")
        if not np.array_equal(rendered[slot["target"]], images[f"target_{scanner}"]):
            raise ValueError(f"{slide_id}/{location}/{scanner}: RV02 target != cache target")
        for item in slot["perturbations"]:
            name = band_name(item["band"], item["sign"], item["dose_fraction"], scanner)
            images[name] = rendered[item["index"]]
            rows.append({"slide_id": slide_id, "location_index": int(location), "scanner": scanner,
                         "condition": name, **{k: v for k, v in item.items() if k != "index"}})
    macenko, status = {}, []
    normalizer = _STATE["macenko"]
    for name in CONDITIONS:
        normalized, flag = normalizer(images[name])
        macenko[name] = normalized
        status.append({"slide_id": slide_id, "location_index": int(location), "condition": name, "status": flag})
    return int(location), images, macenko, uint8, rows, status


def uni_parity(slide_id: str, locations: np.ndarray, uint8: list[dict], uni) -> list[dict]:
    with h5py.File(RV03_SHARDS / f"{slide_id}.h5", "r") as store:
        names = C.decode(store["condition_names"][:])
        stored_locations = np.asarray(store["location_index"], dtype=np.int64)
        stored = np.asarray(store["features"], dtype=np.float32)
    keys = [f"{m}_to_{s}" for s in TARGETS for m in RV03_METHODS]
    batch = np.stack([uint8[i][key] for i in range(len(locations)) for key in keys])
    vectors = uni(batch)["uni_v1"]
    rows = []
    row_of = {int(v): i for i, v in enumerate(stored_locations)}
    for i, location in enumerate(locations):
        for j, key in enumerate(keys):
            reference = stored[row_of[int(location)], names.index(key)]
            rows.append({"slide_id": slide_id, "location_index": int(location), "condition": key,
                         "cosine": float(C.cosine(vectors[i * len(keys) + j], reference))})
    return rows


def review_parity(slide, locations: np.ndarray, features: dict[str, np.ndarray]) -> list[dict]:
    """Compare with the review's stored features at its three locations per slide."""
    arms = {"source": lambda s: "source_at2", "target": lambda s: f"target_{s}",
            "reinhard": lambda s: f"reinhard_to_{s}", "combined": lambda s: f"combined_to_{s}"}
    sources = {"conch_pre": ("conch", "conch_pre"), "conch_projected": ("conch", "conch_projected"),
               "seal_conch_pre": ("seal_conch", "seal_conch_pre"),
               "seal_conch_projected": ("seal_conch", "seal_conch_projected"),
               "seal_uni2": ("uni2_pair", "seal_uni2")}
    position = {int(v): i for i, v in enumerate(locations)}
    rows = []
    for stream, (job, variant) in sources.items():
        path = REVIEW_FEATURES / job / f"fold_{int(slide.fold)}" / f"{slide.slide_id}_{variant}.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as stored:
            for row, (location, scanner) in enumerate(zip(stored["location_index"], stored["scanner"])):
                for arm, condition in arms.items():
                    reference = np.asarray(stored[arm][row], dtype=np.float32)
                    ours = features[stream][position[int(location)], CONDITIONS.index(condition(str(scanner)))]
                    rows.append({"slide_id": str(slide.slide_id), "model": stream, "location_index": int(location),
                                 "scanner": str(scanner), "arm": arm, "cosine": float(C.cosine(ours, reference))})
    return rows


def band_parity(slide_id: str, rows: list[dict]) -> dict:
    ours = pd.DataFrame(rows)
    reference = pd.read_csv(RV02_SHARDS / f"{slide_id}.csv.gz", dtype={"slide_id": str})
    key = ["location_index", "scanner", "band", "dose_fraction", "sign"]
    merged = ours.merge(reference, on=key, how="outer", suffixes=("", "_rv02"), indicator=True)
    both = merged[merged._merge.eq("both")]
    result = {"slide_id": slide_id, "rows": len(ours), "rv02_rows": len(reference), "matched": len(both),
              "target_like_agree": bool((both.target_like == both.target_like_rv02).all())}
    for column in ("target_rms_od", "achieved_rms_od", "coefficient"):
        result[f"{column}_max_abs_diff"] = float((both[column] - both[f"{column}_rv02"]).abs().max())
    # tolerance only for the float64 CSV round trip of the RV02 shards
    result["values_identical"] = bool(result["matched"] == result["rows"] and result["target_like_agree"]
                                      and max(result[f"{c}_max_abs_diff"] for c in ("target_rms_od", "achieved_rms_od", "coefficient")) <= 1e-12)
    result["complete"] = bool(result["matched"] == result["rv02_rows"])
    result["identical"] = result["values_identical"] and result["complete"]
    return result


def process_slide(pool, slide, rows20: pd.DataFrame, fitted: dict, encoders: dict, uni, out: Path,
                  max_locations: int | None) -> None:
    slide_id = str(slide.slide_id)
    streams = list(NEW_MODELS)
    if all((out / "embeddings" / s / f"{slide_id}.h5").exists() for s in streams):
        print(f"skip {slide_id}", flush=True)
        return
    started = time.time()
    locations = rows20.location_index.to_numpy(np.int64)
    if max_locations:
        locations = locations[:max_locations]
    with h5py.File(slide.cache_path, "r") as cache:
        names = tuple(x.decode().lower() for x in cache["scanner_names"][:])
        if names != C.SCANNERS:
            raise ValueError(f"{slide_id}: scanner order changed {names}")
        paired = np.asarray(cache["images"][locations], dtype=np.uint8)
        source_index = np.asarray(cache["source_index"][locations], dtype=np.int64)
    if not np.array_equal(source_index, rows20.source_index.to_numpy(np.int64)[:len(locations)]):
        raise ValueError(f"{slide_id}: cache source_index differs from set20")
    results = pool.map(_location, [(slide_id, int(loc), paired[i], fitted) for i, loc in enumerate(locations)])
    if [item[0] for item in results] != [int(v) for v in locations]:
        raise ValueError(f"{slide_id}: worker results out of order")
    timing = {"render_s": round(time.time() - started, 1)}
    flat = [item[1][name] for item in results for name in CONDITIONS]
    flat_macenko = [item[2][name] for item in results for name in CONDITIONS]
    uint8 = [item[3] for item in results]
    band_rows = [row for item in results for row in item[4]]
    status = pd.DataFrame([row for item in results for row in item[5]])
    n = len(locations)
    features: dict[str, np.ndarray] = {}
    for name, encoder in encoders.items():
        start = time.time()
        out_streams = encoder(flat, flat_macenko if name == "exaonepath" else None)
        for stream, value in out_streams.items():
            if value.shape != (n * len(CONDITIONS), DIMS[stream]) or not np.isfinite(value).all():
                raise ValueError(f"{slide_id} {stream}: invalid embeddings {value.shape}")
            features[stream] = value.reshape(n, len(CONDITIONS), -1)
        timing[f"{name}_s"] = round(time.time() - start, 1)
    if set(features) != set(streams):
        raise ValueError(f"streams {sorted(features)}")
    del flat, flat_macenko

    start = time.time()
    uni_rows = uni_parity(slide_id, locations, uint8, uni)
    timing["uni_v1_parity_s"] = round(time.time() - start, 1)
    worst = min(row["cosine"] for row in uni_rows)
    C.write_frame(out / "qc" / f"{slide_id}_uni_v1_parity.csv", pd.DataFrame(uni_rows))
    if worst < UNI_PARITY_MIN:
        raise ValueError(f"{slide_id}: UNI v1 parity with RV03 {worst:.6f} < {UNI_PARITY_MIN}")
    band = band_parity(slide_id, band_rows)
    C.write_json(out / "qc" / f"{slide_id}_band_parity.json", band)
    if not band["values_identical"] or (not band["complete"] and not max_locations):
        raise ValueError(f"{slide_id}: band renders differ from RV02: {band}")
    C.write_frame(out / "qc" / f"{slide_id}_review_parity.csv", pd.DataFrame(review_parity(slide, locations, features)))
    C.write_frame(out / "render" / f"{slide_id}_band.csv.gz", pd.DataFrame(band_rows))
    C.write_frame(out / "render" / f"{slide_id}_macenko.csv.gz", status)
    ok = status.pivot(index="location_index", columns="condition", values="status").reindex(
        index=locations, columns=CONDITIONS).eq("ok").to_numpy()
    for stream, value in features.items():
        datasets = {"features": value.astype(np.float32), "condition_names": np.asarray(CONDITIONS),
                    "location_index": locations, "source_index": source_index}
        if stream == "exaonepath":
            datasets["macenko_ok"] = ok.astype(np.uint8)
        C.write_h5(out / "embeddings" / stream / f"{slide_id}.h5", datasets,
                   {"pfm": stream, "slide_id": slide_id, "tissue_type": str(slide.tissue_type),
                    "fold": int(slide.fold), "stage": "rv13_embed"})
    timing["total_s"] = round(time.time() - started, 1)
    print(f"saved {slide_id}: {n} locations x {len(CONDITIONS)} images; UNI parity min {worst:.6f}; "
          f"band identical {band['identical']}; macenko not ok {int((~ok).sum())}; {json.dumps(timing)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--task-count", type=int, required=True)
    parser.add_argument("--max-slides", type=int, default=0)
    parser.add_argument("--max-locations", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    args = parser.parse_args()
    import torch

    if not torch.cuda.is_available() or not 0 <= args.task_index < args.task_count:
        raise RuntimeError("valid CUDA task index required")
    os.chdir(PROJECT)
    torch.set_num_threads(1)
    cohort = C.load_cohort()
    set20 = C.load_set(C.SET20, 20)
    parameters = json.loads(C.PARAMETERS.read_text())["fold"]
    slides = cohort.iloc[args.task_index::args.task_count]
    if args.max_slides:
        slides = slides.iloc[:args.max_slides]
    start = time.time()
    _STATE["macenko"] = ReleasedMacenko()
    print(f"Macenko reference fitted in {time.time() - start:.0f}s: HERef {_STATE['macenko'].HERef.tolist()} "
          f"maxCRef {_STATE['macenko'].maxCRef.tolist()}", flush=True)
    # Fork the CPU workers before any CUDA context exists.
    pool = mp.get_context("fork").Pool(max(1, args.workers), initializer=_init_worker)
    encoders = {}
    for name in ENCODERS:
        start = time.time()
        encoders[name] = Encoder(name)
        print(f"loaded {name} in {time.time() - start:.0f}s", flush=True)
    from corrected_embeddings_images import Encoders

    uni = Encoders(("uni_v1",), batch_size=32)
    for slide in slides.itertuples(index=False):
        rows20 = set20[set20.slide_id.eq(str(slide.slide_id))].sort_values("location_index")
        if len(rows20) != 20 or int(rows20.fold.iloc[0]) != int(slide.fold):
            raise ValueError(f"{slide.slide_id}: set20 locations or fold inconsistent")
        process_slide(pool, slide, rows20, parameters[str(int(slide.fold))], encoders, uni,
                      args.output_root, args.max_locations or None)
    pool.close()
    pool.join()
    print(json.dumps({"status": "complete", "task_index": args.task_index, "slides": len(slides)}), flush=True)


if __name__ == "__main__":
    main()

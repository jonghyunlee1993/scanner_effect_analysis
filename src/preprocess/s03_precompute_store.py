"""Build crop-aligned, RGB-only version 3 slide stores.

The v3 contract separates three concepts for every scanner: ``present`` says that
real tissue is available, ``geom_ok`` says registration geometry is usable, and
``q_reg`` is a continuous appearance-robust confidence. Focus and image similarity
remain diagnostics and never select training tuples.

SLURM array mode processes the slide indexed by ``SLURM_ARRAY_TASK_ID``. Without an
array task, slides are processed locally with the configured worker count.
"""
import os
import shutil
import sys
import math
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import _common
from utils import align, grid, qc
from utils.config import config_hash, load_config, resolve_config_path
from utils.store import write_slide
from utils.wsi_io import open_wsi


def _zeros_rgb(size):
    """Return the placeholder used for an absent scanner slot."""
    return np.zeros((size, size, 3), np.uint8)


def _stage_a(cfg, handles, candidates, scanners, ref, size, margin, w0, h0, slide_id):
    """Estimate a constant or affine slide-level offset prior for each scanner."""
    settings = cfg.align
    offsets, stats, probes = align.slide_offsets(
        handles, candidates, scanners, ref, size, margin, w0, h0,
        n_probe=int(settings.n_probe),
        probe_min_ncc=float(settings.probe_min_ncc),
        min_tissue=float(cfg.qc.min_tissue_frac),
        rng=np.random.default_rng(_common.slide_seed(cfg, slide_id)),
        tissue_fn=qc.tissue_fraction,
    )
    coefficients = {}
    for scanner in scanners:
        if scanner == ref:
            continue
        coefficients[scanner], fit = align.fit_offset_field(probes[scanner])
        stat = stats[scanner]
        model = "affine" if fit["affine"] else "constant"
        print(
            f"[s03] {slide_id}/{scanner}: offset {offsets[scanner]}, "
            f"confident {stat['n_conf']}/{stat['n_probe']}, model {model}",
            flush=True,
        )
    return offsets, coefficients


def _select_tiles(cfg, tiles, slide_id):
    """Apply only the spatial budget, without quality-ranked selection."""
    budget = int(cfg.budget.tiles_per_slide)
    if len(tiles) <= budget:
        return tiles
    coords = np.asarray([[tile["x"], tile["y"]] for tile in tiles])
    rng = np.random.default_rng(_common.slide_seed(cfg, slide_id))
    picked = _common.spatial_sample(coords, budget, int(cfg.budget.grid_cells), rng)
    return [tiles[i] for i in picked]


def _spatial_bin(x, y, width, height, n_bins):
    """Return a deterministic spatial-bin key for bounded tile retention."""
    bx = min(int(x / max(width, 1) * n_bins), n_bins - 1)
    by = min(int(y / max(height, 1) * n_bins), n_bins - 1)
    return bx, by


def process_slide(cfg, slide_id, row, candidates):
    """Build one v3 HDF5 store and its sidecar parquet file."""
    scanners = list(cfg.scanners)
    ref = cfg.reference_scanner
    size = int(cfg.patch.size)
    margin = int(cfg.patch.margin)
    settings = cfg.align
    max_residual = float(getattr(settings, "max_residual_shift", 8.0))
    scanner_index = {scanner: i for i, scanner in enumerate(scanners)}
    ref_index = scanner_index[ref]
    n_scanners = len(scanners)

    intersect = bool(getattr(getattr(cfg, "grid", {}), "intersect_scanners", True))
    grid_info = grid.build_grid(
        cfg, slide_id, pd.Series(row), intersect=intersect, margin=margin
    )
    width, height = grid_info["level0_width"], grid_info["level0_height"]
    ncols = max(width // size, 1)
    handles = {scanner: open_wsi(row[f"{scanner}_path"]) for scanner in scanners}
    offsets, coefficients = _stage_a(
        cfg, handles, candidates, scanners, ref, size, margin, width, height, slide_id
    )

    # Never retain every eligible RGB tuple in memory: a large slide can have
    # tens of thousands of eligible candidates (each carries four 256x256 RGB
    # arrays).  Reservoir sampling within spatial bins preserves the intended
    # spatial budget while bounding retained patch memory to ~tiles_per_slide.
    budget = int(cfg.budget.tiles_per_slide)
    n_bins = int(cfg.budget.grid_cells)
    bin_capacity = max(1, math.ceil(budget / (n_bins * n_bins)))
    reservoirs = defaultdict(list)
    reservoir_rng = np.random.default_rng(_common.slide_seed(cfg, slide_id))
    n_eligible = 0
    n_oob = 0
    n_low_tissue = 0
    for x, y in candidates:
        if not align.in_bounds(x, y, size, margin, width, height):
            n_oob += 1
            continue
        ref_ext = align.read_ext(handles[ref], x, y, size, margin)
        ref_crop = align.centre(ref_ext, size, margin)
        tissue_density = qc.tissue_fraction(ref_crop)
        if tissue_density < float(cfg.qc.min_tissue_frac):
            n_low_tissue += 1
            continue

        offset = np.zeros((n_scanners, 2), np.int16)
        present = np.zeros(n_scanners, bool)
        geom_ok = np.zeros(n_scanners, bool)
        q_reg = np.zeros(n_scanners, np.float32)
        ncc_lp = np.zeros(n_scanners, np.float32)
        phase_response = np.zeros(n_scanners, np.float32)
        residual_shift = np.zeros(n_scanners, np.float32)
        focus = np.zeros(n_scanners, np.float32)
        pad_fraction = np.zeros(n_scanners, np.float32)
        align_peak_ncc = np.ones(n_scanners, np.float32)
        peak_boundary = np.zeros(n_scanners, bool)
        patches = {ref: ref_crop}

        present[ref_index] = not align.is_blank(ref_crop)
        geom_ok[ref_index] = present[ref_index]
        q_reg[ref_index] = 1.0
        ncc_lp[ref_index] = 1.0
        phase_response[ref_index] = 1.0
        focus[ref_index] = qc.focus_score(ref_crop)

        for scanner in scanners:
            if scanner == ref:
                continue
            j = scanner_index[scanner]
            ext = align.read_ext(handles[scanner], x, y, size, margin)
            prior = align.prior_at(x, y, offsets[scanner], coefficients[scanner])
            found = align.tile_offset(
                ext,
                ref_crop,
                margin,
                prior,
                radius=int(settings.refine_radius),
                min_ncc=float(cfg.qc.min_ncc),
                max_dist=int(settings.max_dist),
            )
            crop = align.crop_at(ext, found["dy"], found["dx"], size, margin)
            has_content, pad = align.content_ok(crop, ref_crop, max_pad=float(settings.max_pad))
            registration = qc.registration_quality(ref_crop, crop, sigma=3.0)

            patches[scanner] = crop
            offset[j] = found["dy"], found["dx"]
            present[j] = has_content
            pad_fraction[j] = pad
            align_peak_ncc[j] = found["ncc"]
            peak_boundary[j] = found["peak_boundary"]
            focus[j] = qc.focus_score(crop)
            q_reg[j] = registration["q_reg"]
            ncc_lp[j] = registration["ncc_lp"]
            phase_response[j] = registration["phase_response"]
            residual_shift[j] = registration["residual_shift"]
            geom_ok[j] = (
                has_content
                and pad <= float(settings.max_pad)
                and registration["residual_shift"] <= max_residual
                and not found["peak_boundary"]
            )

        eligible = geom_ok[ref_index] and bool(np.delete(geom_ok, ref_index).any())
        if eligible:
            n_eligible += 1
            tile = {
                "x": int(x),
                "y": int(y),
                "tissue_density": float(tissue_density),
                "offset": offset,
                "present": present,
                "geom_ok": geom_ok,
                "q_reg": q_reg,
                "ncc_lp": ncc_lp,
                "phase_response": phase_response,
                "residual_shift": residual_shift,
                "focus": focus,
                "pad_frac": pad_fraction,
                "align_peak_ncc": align_peak_ncc,
                "peak_boundary": peak_boundary,
                "patches": patches,
            }
            key = _spatial_bin(x, y, width, height, n_bins)
            priority = float(reservoir_rng.random())
            bucket = reservoirs[key]
            if len(bucket) < bin_capacity:
                bucket.append((priority, tile))
            else:
                worst = max(range(len(bucket)), key=lambda index: bucket[index][0])
                if priority < bucket[worst][0]:
                    bucket[worst] = (priority, tile)

    tiles = [tile for bucket in reservoirs.values() for _, tile in bucket]
    selected = _select_tiles(cfg, tiles, slide_id)
    records = []
    metadata = []
    for tuple_id, tile in enumerate(selected):
        tile_id = int((tile["y"] // size) * ncols + tile["x"] // size)
        record = {
            "coords": (tile["x"], tile["y"]),
            "tile_id": tile_id,
            "present": tile["present"].astype(np.uint8),
            "geom_ok": tile["geom_ok"].astype(np.uint8),
            "q_reg": tile["q_reg"],
        }
        for scanner in scanners:
            j = scanner_index[scanner]
            rgb = tile["patches"][scanner] if tile["present"][j] else _zeros_rgb(size)
            record[scanner] = {"rgb": rgb}
        records.append(record)
        metadata.append({
            "slide_id": slide_id,
            "tuple_id": tuple_id,
            "tile_id": tile_id,
            "x": tile["x"],
            "y": tile["y"],
            "tissue_density": tile["tissue_density"],
            "present": tile["present"].tolist(),
            "geom_ok": tile["geom_ok"].tolist(),
            "q_reg": tile["q_reg"].tolist(),
            "ncc_lp": tile["ncc_lp"].tolist(),
            "phase_response": tile["phase_response"].tolist(),
            "residual_shift": tile["residual_shift"].tolist(),
            "focus": tile["focus"].tolist(),
            "pad_frac": tile["pad_frac"].tolist(),
            "align_peak_ncc": tile["align_peak_ncc"].tolist(),
            "peak_boundary": tile["peak_boundary"].tolist(),
            "off_dy": tile["offset"][:, 0].tolist(),
            "off_dx": tile["offset"][:, 1].tolist(),
        })

    for handle in handles.values():
        handle.close()
    if not records:
        print(
            f"[s03] {slide_id}: no eligible tuples (oob {n_oob}, low-tissue {n_low_tissue})",
            flush=True,
        )
        return slide_id, 0, 0

    attrs = {
        "level0_width": int(width),
        "level0_height": int(height),
        "patch_size": size,
        "target_mag": int(cfg.patch.target_mag),
        "margin": margin,
        "reference_scanner": ref,
        "config_hash": config_hash(cfg),
    }
    store_dir = Path(cfg.paths.output_store)
    store_dir.mkdir(parents=True, exist_ok=True)
    write_slide(store_dir / f"{slide_id}.h5", slide_id, scanners, records, attrs)
    sidecar_dir = Path(cfg.paths.qc_dir) / "sidecar"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metadata).to_parquet(sidecar_dir / f"{slide_id}.parquet", index=False)

    present_rate = np.stack([tile["present"] for tile in selected]).mean(0)
    geom_rate = np.stack([tile["geom_ok"] for tile in selected]).mean(0)
    print(
        f"[s03] {slide_id}: candidates {len(candidates)} (oob {n_oob}, "
            f"low-tissue {n_low_tissue}) -> eligible {n_eligible} -> retained {len(tiles)} "
            f"-> kept {len(records)}\n"
        + "       present "
        + " ".join(f"{s}={present_rate[scanner_index[s]] * 100:.0f}%" for s in scanners)
        + "   geom_ok "
        + " ".join(f"{s}={geom_rate[scanner_index[s]] * 100:.0f}%" for s in scanners),
        flush=True,
    )
    return slide_id, len(records), n_eligible


def _preflight(cfg, n_slides):
    """Abort before WSI reads when the target filesystem is clearly too small."""
    estimate = n_slides * int(cfg.budget.tiles_per_slide) * len(cfg.scanners) * 180_000
    path = Path(cfg.paths.output_store)
    while not path.exists():
        path = path.parent
    free = shutil.disk_usage(path).free
    if free < estimate * 1.2:
        raise RuntimeError(
            f"disk preflight: need about {estimate * 1.2 / 1e9:.0f}GB, free {free / 1e9:.0f}GB"
        )


def _candidates(cfg, slide_id):
    """Load candidate coordinates emitted by s02."""
    path = Path(cfg.paths.qc_dir) / "candidates" / f"{slide_id}.parquet"
    return pd.read_parquet(path)[["x", "y"]].to_numpy()


def main():
    """Run v3 preprocessing for an array task or the configured slide set."""
    cfg = load_config(resolve_config_path())
    registry = _common.load_registry(cfg)
    slides = _common.slide_list(cfg, registry)
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task is not None:
        slides = [slides[int(task)]]
    _preflight(cfg, len(slides))

    if task is not None or int(cfg.runtime.num_workers) <= 1:
        for slide_id in slides:
            result = process_slide(
                cfg, slide_id, registry.loc[slide_id].to_dict(), _candidates(cfg, slide_id)
            )
            print(f"[s03] {result[0]}: kept {result[1]} / eligible {result[2]}")
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=int(cfg.runtime.num_workers)) as executor:
            futures = {
                executor.submit(
                    process_slide,
                    cfg,
                    slide_id,
                    registry.loc[slide_id].to_dict(),
                    _candidates(cfg, slide_id),
                ): slide_id
                for slide_id in slides
            }
            for future in as_completed(futures):
                result = future.result()
                print(f"[s03] {result[0]}: kept {result[1]} / eligible {result[2]}")
    print(f"[s03] done: {len(slides)} slide(s)")


if __name__ == "__main__":
    main()

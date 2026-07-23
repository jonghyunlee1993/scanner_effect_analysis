"""Canonical sampling grid. All scanners are pixel-registered into the reference
(at2) frame, so a coord means the same place in every scanner. We validate the
256-lattice and that each scanner WSI matches the reference level-0 dimensions.

The grid is the INTERSECTION of the per-scanner curated coord sets, not the
reference's alone: at2/gt450/versa share an identical coord set on every slide, but
akoya's is a strict subset (mean 86.7% of at2's, median 98.3%, min 10.5%; 11/50 slides
below 80%). Sampling from at2's set alone puts tiles where akoya was never scanned --
6.2% of the old train store, 11.5% of test. `margin` additionally drops coords whose
margin-extended read window would fall outside the level-0 frame.
"""
import h5py
import numpy as np

from utils.wsi_io import open_wsi


def coords_path(cfg, scanner, slide_id):
    """Map cfg.paths.curated_coords to a concrete path."""
    return cfg.paths.curated_coords.format(scanner=scanner, slide_id=slide_id)


def load_coords(path):
    """Read the curated 'coords' dataset (N,2 int64, level-0 top-left) and its attrs."""
    with h5py.File(path, "r") as f:
        d = f["coords"]
        coords = d[:].astype(np.int64)
        attrs = dict(d.attrs)
    return coords, attrs


def build_grid(cfg, slide_id, registry_row, intersect=True, margin=None):
    """Canonical grid + frame dims. Validates the lattice and cross-scanner level-0 dims.

    intersect: keep only coords present in EVERY scanner's curated set (drops the regions
               akoya never scanned). margin: also drop coords whose (size+2*margin)^2 read
               window leaves the frame. Reports what each filter removed."""
    ref = cfg.reference_scanner
    coords, attrs = load_coords(coords_path(cfg, ref, slide_id))
    size = cfg.patch.size
    assert (coords % size == 0).all(), f"{slide_id}: coords not on {size}px lattice"
    w0, h0 = int(attrs["level0_width"]), int(attrs["level0_height"])
    for s in cfg.scanners:
        slide = open_wsi(registry_row[f"{s}_path"])
        dims = slide.level_dimensions[0]
        slide.close()
        assert dims == (w0, h0), f"{slide_id}/{s}: level0 {dims} != ref {(w0, h0)}"

    n_ref = len(coords)
    if intersect:
        keep = set(map(tuple, coords.tolist()))
        for s in cfg.scanners:
            if s == ref:
                continue
            keep &= set(map(tuple, load_coords(coords_path(cfg, s, slide_id))[0].tolist()))
        coords = np.array(sorted(keep), np.int64).reshape(-1, 2)
    n_int = len(coords)
    if margin:
        m = int(margin)
        ok = ((coords[:, 0] - m >= 0) & (coords[:, 1] - m >= 0)
              & (coords[:, 0] + size + m <= w0) & (coords[:, 1] + size + m <= h0))
        coords = coords[ok]
    return {
        "coords": coords,
        "n_ref": n_ref, "n_intersect": n_int, "n_final": len(coords),
        "level0_width": w0,
        "level0_height": h0,
        "patch_size": int(attrs["patch_size"]),
        "target_magnification": int(attrs.get("target_magnification", cfg.patch.target_mag)),
    }

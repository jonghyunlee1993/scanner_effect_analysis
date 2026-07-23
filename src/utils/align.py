"""Crop-based cross-scanner tile alignment (margin read -> integer shift -> crop).

The crop-based path avoids wrapped tissue and keeps appearance scores separate from
the v3 geometric usability flag.

Here every scanner is read with a `margin` of slack around the canonical (x, y) window,
each scanner's true integer offset vs the reference is found inside that slack, and the
final `size x size` tile is CROPPED at the shifted position. Result: all scanners cover
the identical tissue, with zero wrapped pixels and zero invalid pixels.

Offsets are found in two stages, because a brute-force per-tile NCC peak is unreliable on
low-texture / near-blank tiles (that is what produced the bogus 33px "rail" offsets):
  A. `slide_offsets`  - one global (dy,dx) per scanner, the median of confident per-tile
                        peaks over a probe sample (akoya is ~9px slide-coherent).
  B. `tile_offset`    - per-tile refinement restricted to a small radius around the slide
                        offset; a low-confidence tile falls back to the slide offset
                        (NOT to (0,0), which is what the old cache did).

NCC is `cv2.matchTemplate(..., TM_CCOEFF_NORMED)` on low-pass grayscale. It is used
only to locate the crop; the v3 supervision confidence is computed independently
from the aligned crops.
The whole (2*margin+1)^2 score map costs one DFT correlation, not 16k crops.

NOTE: alignment is INTEGER only, by design. Sub-pixel resampling of the non-reference
scanners alone would inject a scanner-correlated interpolation blur -- a new batch effect
of exactly the kind this project exists to remove. Any sub-pixel / elastic remainder is
left for the training objective to handle.
"""
import cv2
import numpy as np

from utils.wsi_io import read_patch


# --- primitives ---------------------------------------------------------------
def gray_blur(rgb, sigma=3.0):
    """Return low-pass float32 grayscale for appearance-robust alignment."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return cv2.GaussianBlur(g, (0, 0), sigma) if sigma > 0 else g


def in_bounds(x, y, size, margin, w0, h0):
    """True when the margin-extended window lies fully inside the level-0 frame."""
    return (x - margin >= 0 and y - margin >= 0
            and x + size + margin <= w0 and y + size + margin <= h0)


def read_ext(slide, x, y, size, margin):
    """Read the margin-extended (size+2m)^2 patch whose CENTRE is the canonical window."""
    return read_patch(slide, x - margin, y - margin, size + 2 * margin)


def crop_at(ext, dy, dx, size, margin):
    """Crop the size^2 window displaced by (dy,dx) from the centre. |dy|,|dx| <= margin."""
    r, c = margin + dy, margin + dx
    return ext[r:r + size, c:c + size]


def centre(ext, size, margin):
    """The un-shifted window == exactly what the old pipeline read at (x, y)."""
    return crop_at(ext, 0, 0, size, margin)


# --- offset search ------------------------------------------------------------
def ncc_map(ext_rgb, ref_crop_rgb, sigma=3.0):
    """(2m+1, 2m+1) NCC map. Entry (r,c) scores ext[r:r+S, c:c+S] against ref_crop,
    so the peak index IS the crop position: (dy,dx) = (r-m, c-m). One DFT correlation."""
    m = cv2.matchTemplate(gray_blur(ext_rgb, sigma), gray_blur(ref_crop_rgb, sigma),
                          cv2.TM_CCOEFF_NORMED)
    return np.nan_to_num(m, nan=-1.0, posinf=-1.0, neginf=-1.0)   # blank tile -> zero variance


def peak(nmap, margin, around=None, radius=None):
    """Argmax of `nmap`, optionally restricted to `radius` around the `around`=(dy,dx)
    prior. Returns (dy, dx, ncc)."""
    if around is None or radius is None:
        r, c = np.unravel_index(int(np.argmax(nmap)), nmap.shape)
    else:
        cy, cx = margin + around[0], margin + around[1]
        y0, y1 = max(cy - radius, 0), min(cy + radius + 1, nmap.shape[0])
        x0, x1 = max(cx - radius, 0), min(cx + radius + 1, nmap.shape[1])
        sub = nmap[y0:y1, x0:x1]
        rr, cc = np.unravel_index(int(np.argmax(sub)), sub.shape)
        r, c = y0 + rr, x0 + cc
    return int(r - margin), int(c - margin), float(nmap[r, c])


def slide_offsets(handles, coords, scanners, ref, size, margin, w0, h0,
                  n_probe=80, probe_min_ncc=0.6, min_tissue=0.25, rng=None, tissue_fn=None):
    """Stage A: probe the slide and summarise each non-ref scanner's offset field.

    Probes are drawn from `coords` (already restricted to the all-scanner grid), skipping
    out-of-bounds windows and low-tissue reference patches. Returns
    (offsets: {scanner: (dy,dx)} median, stats: {scanner: {...}},
     probes: {scanner: (N,5) array of x, y, dy, dx, ncc})."""
    rng = rng or np.random.default_rng(0)
    cand = [c for c in coords if in_bounds(c[0], c[1], size, margin, w0, h0)]
    order = rng.permutation(len(cand))
    peaks = {s: [] for s in scanners if s != ref}
    used = 0
    for i in order:
        if used >= n_probe:
            break
        x, y = cand[int(i)]
        ext_ref = read_ext(handles[ref], x, y, size, margin)
        ref_crop = centre(ext_ref, size, margin)
        if tissue_fn is not None and tissue_fn(ref_crop) < min_tissue:
            continue
        used += 1
        for s in peaks:
            dy, dx, v = peak(ncc_map(read_ext(handles[s], x, y, size, margin), ref_crop),
                             margin)
            if v >= probe_min_ncc:
                peaks[s].append((x, y, dy, dx, v))
    offsets, stats, probes = {ref: (0, 0)}, {}, {}
    for s, p in peaks.items():
        probes[s] = np.array(p, float) if p else np.zeros((0, 5))
        if p:
            a = probes[s]
            offsets[s] = (int(np.median(a[:, 2])), int(np.median(a[:, 3])))
            stats[s] = {"n_conf": len(p), "n_probe": used,
                        "median_ncc": float(np.median(a[:, 4])),
                        "iqr_dy": float(np.subtract(*np.percentile(a[:, 2], [75, 25]))),
                        "iqr_dx": float(np.subtract(*np.percentile(a[:, 3], [75, 25])))}
        else:
            offsets[s] = (0, 0)
            stats[s] = {"n_conf": 0, "n_probe": used, "median_ncc": float("nan"),
                        "iqr_dy": float("nan"), "iqr_dx": float("nan")}
    return offsets, stats, probes


def fit_offset_field(probe, min_n=12, min_r2=0.5):
    """Is the scanner's offset a CONSTANT slide shift, or does it vary linearly with
    position (a residual rotation / scale in the upstream registration)?

    Least-squares fit of (dy, dx) ~ 1 + x + y over the confident probes. Returns
    (coef (3,2) | None, info). `coef` is returned only when both axes fit well enough to
    beat the constant-median prior; otherwise the caller keeps the median."""
    info = {"n": int(len(probe)), "r2_dy": float("nan"), "r2_dx": float("nan"),
            "resid_std_dy": float("nan"), "resid_std_dx": float("nan"), "affine": False}
    if len(probe) < min_n:
        return None, info
    x, y, dy, dx = probe[:, 0], probe[:, 1], probe[:, 2], probe[:, 3]
    A = np.stack([np.ones_like(x), x, y], 1)
    coef, *_ = np.linalg.lstsq(A, np.stack([dy, dx], 1), rcond=None)     # (3,2)
    pred = A @ coef
    r2, rs = [], []
    for k, t in enumerate((dy, dx)):
        ss_res = float(((t - pred[:, k]) ** 2).sum())
        ss_tot = float(((t - t.mean()) ** 2).sum())
        r2.append(1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0)
        rs.append(float(np.std(t - pred[:, k])))
    info.update(r2_dy=r2[0], r2_dx=r2[1], resid_std_dy=rs[0], resid_std_dx=rs[1])
    info["affine"] = max(r2) >= min_r2
    return (coef if info["affine"] else None), info


def prior_at(x, y, median_off, coef=None):
    """Per-tile prior: the fitted affine field when it held up, else the slide median."""
    if coef is None:
        return median_off
    p = np.array([1.0, x, y]) @ coef
    return int(round(p[0])), int(round(p[1]))


def tile_offset(ext_rgb, ref_crop_rgb, margin, slide_off, radius=24, min_ncc=0.5,
                max_dist=None):
    """Stage B: per-tile offset from the same (already computed) NCC map.

    The GLOBAL argmax is preferred: restricting the search to `radius` around the slide
    prior loses the true peak whenever the prior is loose (versa's probe IQR is ~20px, vs
    gt450's ~3px). The prior is used only to (a) reject an implausibly distant peak via
    `max_dist` and (b) supply the fallback when no peak clears `min_ncc` -- falling back to
    the slide offset, never to (0,0) as the old cache did.

    Spurious far peaks (the old "rail" junk: NCC~0.58 at 33px on a near-blank tile) are
    handled upstream by dropping blank/padded tiles, not by clamping the search.

    Returns dict(dy, dx, ncc, mode, conf, peak_boundary) with mode in {global, local}. The returned
    offset is ALWAYS an argmax -- never the bare prior, which scored 0.02-0.19 NCC on the
    tiles where the restricted argmax scored 0.44-0.50."""
    nmap = ncc_map(ext_rgb, ref_crop_rgb)
    fy, fx, fv = peak(nmap, margin)                                   # global argmax
    far = max_dist is not None and (abs(fy - slide_off[0]) + abs(fx - slide_off[1])) > max_dist
    if fv >= min_ncc and not far:
        boundary = abs(fy) >= margin or abs(fx) >= margin
        return {"dy": fy, "dx": fx, "ncc": fv, "mode": "global", "conf": True,
                "peak_boundary": boundary}
    ry, rx, rv = peak(nmap, margin, around=slide_off, radius=radius)  # near the prior
    boundary = (abs(ry) >= margin or abs(rx) >= margin
                or abs(ry - slide_off[0]) >= radius
                or abs(rx - slide_off[1]) >= radius)
    return {"dy": ry, "dx": rx, "ncc": rv, "mode": "local", "conf": rv >= min_ncc,
            "peak_boundary": boundary}


# --- blank / content checks ---------------------------------------------------
def flat_mask(rgb, win=5, thresh=1.0):
    """Pixels sitting in a PERFECTLY FLAT neighbourhood (local std < `thresh`).

    Registration padding is a constant-valued region -- measured on the akoya band in
    12.5_4/(44032,31232) it is a single RGB triple (220,228,226) with local std exactly 0.
    Note it is pale GREY, not white: `frac_white` (>=240) sees 0.000 of it, so a brightness
    test misses partially padded tiles entirely."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    m = cv2.blur(g, (win, win))
    v = cv2.blur(g * g, (win, win)) - m * m
    return np.sqrt(np.maximum(v, 0.0)) < thresh


def flat_frac(rgb, win=5, thresh=1.0):
    return float(flat_mask(rgb, win, thresh).mean())


def stained_mask(ref_rgb, min_sat=0.12, max_gray=220):
    """Reference pixels that carry actual stain -- absolute thresholds, NOT Otsu.

    Otsu (as in `qc.tissue_fraction`) always splits the histogram, so on an all-tissue or
    all-background tile its mask is meaningless; here we need a hard "the reference has
    something here" test."""
    hsv = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1].astype(np.float32) / 255.0
    gray = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2GRAY)
    return (sat >= min_sat) & (gray <= max_gray)


def pad_frac(crop, ref_crop, min_ref_px=256):
    """Padding = FLAT and BRIGHT (or black) in the scanner, where the REFERENCE is stained.
    Returned as a fraction of the reference's stained area. All three clauses are needed:

    * flat alone -- on tissue-sparse slides the registered TIFFs carry a perfectly constant
      background, so `flat_frac` is 0.19-0.32 on healthy tiles (8-12_5); and akoya, being
      over-saturated, has locally flat regions *inside dense tissue* (flat pixels there have
      gray median 98-129, i.e. dark). Both look identical to padding by flatness alone.
    * brightness alone -- real white background is bright too.
    * the reference clause -- ordinary background is background in EVERY scanner at once,
      whereas padding is flat exactly where at2 shows stained tissue.

    Measured separation on the three pilot slides: healthy tiles score 0.0000, the two
    genuinely padded tiles score 0.31 and 0.56. Padding is a single constant value (RGB
    (220,228,226) -> gray 225 in 12.5_4; gray 240 in 2-8_1) covering 32% / 65% of the tile.

    Tissue-fraction ratios cannot be used instead: gt450 is desaturated (mean HSV S 0.193 vs
    at2's 0.469) so its Otsu tissue fraction is systematically low and a ratio test rejects
    healthy gt450 tiles, while akoya scores up to 6x the reference. Flatness is a texture
    property and is indifferent to the scanner's colour transfer function."""
    ref_stained = stained_mask(ref_crop)
    n = int(ref_stained.sum())
    if n < min_ref_px:
        return 0.0
    g = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    missing = flat_mask(crop) & ((g >= 200) | (g <= 20))
    return float((missing & ref_stained).sum() / n)


def blank_stats(rgb):
    """Cheap emptiness descriptors. Registration padding is a constant-valued region;
    real background is bright but carries sensor texture."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return {"gray_std": float(g.std()),
            "frac_white": float((g >= 240).mean()),
            "frac_black": float((g <= 15).mean()),
            "flat_frac": float(flat_frac(rgb))}


def is_blank(rgb, min_gray_std=3.0, max_flat=0.98):
    """True when the patch carries essentially no content (padding or a dead region)."""
    b = blank_stats(rgb)
    return b["gray_std"] < min_gray_std or (b["frac_white"] + b["frac_black"]) > max_flat


def content_ok(crop, ref_crop, max_pad=0.01):
    """Does THIS scanner actually cover the reference's tissue in this tile?

    Per-scanner, not per-tuple: a padded scanner is marked unusable and its siblings stay.

    Rejects (a) fully blank crops -- the scanner was never scanned there -- and (b) PARTIALLY
    padded crops, where a tile straddles the edge of the scanned area, or the margin crop
    pulls padding in from just outside. Intersecting the per-scanner curated coord sets
    handles (a) at 256px granularity; nothing catches (b) except a pixel test.

    `max_pad` is deliberately strict (1% of the reference's stained area): healthy tiles
    score exactly 0.0000, so any nonzero signal is real missing data.

    Returns (ok, pad_frac)."""
    if is_blank(crop):
        return False, 1.0
    pf = pad_frac(crop, ref_crop)
    return pf <= max_pad, pf

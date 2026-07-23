"""Shared glue for the Phase-0 preprocessing entry scripts: registry access,
slide-list selection, deterministic per-slide seeding, and spatial-stratified
sampling. Heavy logic stays in utils/; this is orchestration glue only."""
import hashlib
from collections import defaultdict

import numpy as np
import pandas as pd


def load_registry(cfg):
    """Registered-pair registry indexed by slide_id; only fully paired slides."""
    df = pd.read_csv(cfg.paths.registry_csv, dtype={"slide_id": str})
    return df[df["paired_all_scanners"]].set_index("slide_id")


def slide_list(cfg, reg):
    """Slides to process. Explicit policy -> the union of split.slides (in registry);
    otherwise every paired slide, sorted for determinism."""
    if cfg.split.policy == "explicit" and getattr(cfg.split, "slides", None) is not None:
        sl = cfg.split.slides
        want = [*sl.train, *sl.val, *sl.test]
        return [s for s in want if s in reg.index]
    return sorted(reg.index)


def slide_seed(cfg, slide_id):
    """Stable per-slide seed = runtime.seed + md5(slide_id), so sampling is
    deterministic and independent of slide processing order."""
    h = int.from_bytes(hashlib.md5(slide_id.encode()).digest()[:4], "little")
    return (int(cfg.runtime.seed) + h) & 0xFFFFFFFF


def spatial_sample(coords, n, n_bins, rng):
    """Pick ~n indices from coords (M,2) spread across an n_bins x n_bins spatial
    grid via round-robin over shuffled bins. Deterministic given rng. Returns a
    sorted index array; all indices when n >= M."""
    m = len(coords)
    if n <= 0:
        return np.array([], dtype=int)
    if n >= m:
        return np.arange(m)
    xy = np.asarray(coords, np.float64)
    lo = xy.min(0)
    span = np.maximum(xy.max(0) - lo, 1.0)
    b = np.clip(np.floor((xy - lo) / span * n_bins).astype(int), 0, n_bins - 1)
    bin_id = b[:, 0] * n_bins + b[:, 1]
    groups = defaultdict(list)
    for i, bid in enumerate(bin_id):
        groups[bid].append(i)
    order = list(groups.values())
    for g in order:
        rng.shuffle(g)
    rng.shuffle(order)
    picked = []
    while len(picked) < n:
        progressed = False
        for g in order:
            if g:
                picked.append(g.pop())
                progressed = True
                if len(picked) >= n:
                    break
        if not progressed:
            break
    return np.array(sorted(picked), dtype=int)

"""Fit the fixed AT2 stain basis used by the differentiable input builder.

Only training-split AT2 pixels contribute. A seeded chunk-wise reservoir prevents
large or densely tiled slides from requiring unbounded memory while preserving a
uniform global pixel sample.
"""
import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from utils.config import config_hash, load_config
from utils.store import SCHEMA_VERSION, decode

PUBLISHED_HE = np.asarray(
    [[0.650, 0.072], [0.704, 0.990], [0.286, 0.105]], dtype=np.float64
)
PUBLISHED_HE /= np.linalg.norm(PUBLISHED_HE, axis=0, keepdims=True)


def optical_density(rgb):
    """Convert uint8 RGB pixels to optical density."""
    return -np.log((rgb.astype(np.float64) + 1.0) / 256.0)


def tissue_pixels(rgb, beta=0.15):
    """Return pixels carrying enough optical density to represent stain."""
    flat = rgb.reshape(-1, 3)
    od = optical_density(flat)
    return flat[np.linalg.norm(od, axis=1) > beta]


class PixelReservoir:
    """Uniform fixed-size reservoir with exact chunk-wise sampling."""

    def __init__(self, capacity, seed):
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.pixels = np.empty((0, 3), np.uint8)
        self.seen = 0

    def add(self, pixels):
        """Merge a chunk while preserving a uniform sample of all seen pixels."""
        chunk_size = len(pixels)
        if not chunk_size:
            return
        combined = self.seen + chunk_size
        if combined <= self.capacity:
            self.pixels = np.concatenate([self.pixels, pixels], axis=0)
            self.seen = combined
            return

        # NumPy's hypergeometric implementation rejects populations >= 1e9.
        # For the full cohort the reservoir (1e6) is tiny relative to the
        # multi-billion-pixel population, so the binomial limit is an accurate
        # sparse-sampling approximation while keeping memory fixed.
        if self.seen < 1_000_000_000:
            incoming = int(self.rng.hypergeometric(chunk_size, self.seen, self.capacity))
        else:
            incoming = int(self.rng.binomial(self.capacity, chunk_size / combined))
        old_keep = self.capacity - incoming
        if len(self.pixels) > old_keep:
            keep = self.rng.choice(len(self.pixels), old_keep, replace=False)
            retained = self.pixels[keep]
        else:
            retained = self.pixels
        take = self.rng.choice(chunk_size, incoming, replace=False)
        self.pixels = np.concatenate([retained, pixels[take]], axis=0)
        self.seen = combined


def fit_macenko(pixels):
    """Fit a two-column Macenko basis and order it as hematoxylin, eosin."""
    od = optical_density(pixels)
    covariance = np.cov(od, rowvar=False)
    _, vectors = np.linalg.eigh(covariance)
    plane = vectors[:, -2:]
    projected = od @ plane
    angles = np.arctan2(projected[:, 1], projected[:, 0])
    low, high = np.percentile(angles, [1.0, 99.0])
    candidates = np.stack(
        [plane @ np.asarray([np.cos(low), np.sin(low)]),
         plane @ np.asarray([np.cos(high), np.sin(high)])],
        axis=1,
    )
    candidates *= np.where(candidates.sum(axis=0, keepdims=True) < 0, -1.0, 1.0)
    candidates /= np.linalg.norm(candidates, axis=0, keepdims=True)

    direct = float(np.sum(candidates * PUBLISHED_HE))
    swapped = float(np.sum(candidates[:, ::-1] * PUBLISHED_HE))
    return candidates if direct >= swapped else candidates[:, ::-1]


def concentration_scale(pixels, matrix):
    """Return the positive 99th-percentile concentration of each stain."""
    concentrations = np.linalg.lstsq(matrix, optical_density(pixels).T, rcond=None)[0].T
    concentrations = np.maximum(concentrations, 0.0)
    return np.percentile(concentrations, 99.0, axis=0).astype(np.float32)


def collect_pixels(cfg, index, scanner, capacity, seed):
    """Collect a uniform reservoir from present AT2 train tuples."""
    reservoir = PixelReservoir(capacity, seed)
    rows = index[(index["split"] == "train") & index[f"present_{scanner}"]]
    for slide_id, group in rows.groupby("slide_id", sort=True):
        path = Path(cfg.paths.output_store) / f"{slide_id}.h5"
        with h5py.File(path, "r") as handle:
            if int(handle.attrs.get("schema_version", -1)) != SCHEMA_VERSION:
                raise ValueError(f"{path} is not a schema_version=3 store")
            for tuple_id in group["tuple_id"].to_numpy(np.int64):
                rgb = decode(handle[scanner]["rgb"][int(tuple_id)])
                reservoir.add(tissue_pixels(rgb))
    return reservoir.pixels


def parse_args():
    """Parse artifact fitting options while retaining PRENORM_CONFIG fallback."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--max-pixels", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    """Fit and save the immutable AT2 stain reference artifact."""
    args = parse_args()
    config_path = args.config or os.environ.get(
        "PRENORM_CONFIG", "configs/main/phase_0_preprocessing_10slide_v3.yaml"
    )
    cfg = load_config(config_path)
    scanner = cfg.reference_scanner
    seed = int(cfg.runtime.seed if args.seed is None else args.seed)
    index = pd.read_parquet(cfg.paths.output_index)
    if "schema_version" not in index or not (index["schema_version"] == SCHEMA_VERSION).all():
        raise ValueError("stain fitting requires a schema_version=3 index")
    pixels = collect_pixels(cfg, index, scanner, args.max_pixels, seed)
    if len(pixels) < 3:
        raise ValueError("not enough AT2 tissue pixels to fit a stain reference")

    matrix = fit_macenko(pixels).astype(np.float32)
    scale = concentration_scale(pixels, matrix)
    output = Path(cfg.paths.stain_reference)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        matrix=matrix,
        concentration_scale=scale,
        scanner=np.asarray(scanner),
        split=np.asarray("train"),
        sample_count=np.asarray(len(pixels), np.int64),
        seed=np.asarray(seed, np.int64),
        config_hash=np.asarray(config_hash(cfg)),
        schema_version=np.asarray(SCHEMA_VERSION, np.int64),
    )
    print(f"[stain] fitted {len(pixels):,} {scanner} pixels -> {output}")


if __name__ == "__main__":
    main()

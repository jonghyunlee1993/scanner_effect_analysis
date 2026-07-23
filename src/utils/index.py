"""Version 3 global tuple index.

The index stores independent content presence, geometric usability, and continuous
registration confidence for each scanner. Legacy ``valid`` columns are rejected
instead of being reinterpreted as geometric supervision.
"""
import numpy as np
import pandas as pd

SCHEMA_VERSION = 3
_REQUIRED = ("present", "geom_ok", "q_reg")
_DIAGNOSTIC = (
    "off_dy", "off_dx", "ncc_lp", "phase_response", "residual_shift",
    "focus", "pad_frac", "align_peak_ncc", "peak_boundary",
)
_BOOL = ("present", "geom_ok", "peak_boundary")
_INT = ("off_dy", "off_dx")


def build_index(records, scanners):
    """Flatten v3 per-tuple sidecar records into scanner-specific columns."""
    rows = []
    for record in records:
        if "valid" in record:
            raise ValueError("legacy 'valid' metadata cannot be converted to geom_ok")
        missing = [name for name in _REQUIRED if name not in record]
        if missing:
            raise ValueError(f"missing v3 metadata fields: {missing}")
        row = {
            "schema_version": SCHEMA_VERSION,
            "slide_id": record["slide_id"],
            "tuple_id": record["tuple_id"],
            "tile_id": record["tile_id"],
            "x": record["x"],
            "y": record["y"],
            "tissue_density": record["tissue_density"],
        }
        for j, scanner in enumerate(scanners):
            for name in _REQUIRED + _DIAGNOSTIC:
                if name not in record:
                    continue
                value = record[name][j]
                if name in _BOOL:
                    value = bool(value)
                elif name in _INT:
                    value = int(value)
                else:
                    value = float(value)
                row[f"{name}_{scanner}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def assign_split(df, policy, fractions=None, seed=0, explicit=None):
    """Assign complete slides to deterministic train, validation, and test splits."""
    if policy == "explicit":
        mapping = {slide: split for split, slides in explicit.items() for slide in slides}
        df["split"] = df["slide_id"].map(mapping)
    else:
        slides = sorted(df["slide_id"].unique())
        np.random.default_rng(seed).shuffle(slides)
        raw = np.asarray([fractions["train"], fractions["val"], fractions["test"]], float)
        if not np.isclose(raw.sum(), 1.0):
            raise ValueError(f"split fractions must sum to one, got {raw.sum()}")
        counts = np.floor(raw * len(slides)).astype(int)
        # Distribute the remaining slides by largest fractional remainder.  This
        # preserves the requested ratio as closely as possible (109 -> 87/11/11
        # for an 8:1:1 split), while retaining deterministic tie-breaking.
        for index in np.argsort(-(raw * len(slides) - counts), kind="stable")[:len(slides) - counts.sum()]:
            counts[index] += 1
        n_train, n_val = counts[:2]
        mapping = {
            slide: "train" if i < n_train else "val" if i < n_train + n_val else "test"
            for i, slide in enumerate(slides)
        }
        df["split"] = df["slide_id"].map(mapping)
    return df


def validate_index(df):
    """Reject legacy or incomplete indexes and return the input frame unchanged."""
    if "schema_version" not in df or not (df["schema_version"] == SCHEMA_VERSION).all():
        raise ValueError("expected a schema_version=3 index")
    if any(column.startswith("valid_") for column in df.columns):
        raise ValueError("legacy valid_* columns are not supported by the v3 index")
    scanners = [column.removeprefix("q_reg_") for column in df if column.startswith("q_reg_")]
    if not scanners:
        raise ValueError("v3 index has no q_reg_<scanner> columns")
    missing = [f"{name}_{scanner}" for scanner in scanners for name in _REQUIRED
               if f"{name}_{scanner}" not in df]
    if missing:
        raise ValueError(f"v3 index is missing required columns: {missing}")
    return df


def write_index(df, path):
    """Validate and write an index without its pandas row index."""
    validate_index(df).to_parquet(path, index=False)


def read_index(path):
    """Read and validate a version 3 index."""
    return validate_index(pd.read_parquet(path))

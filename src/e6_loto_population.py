"""Shared identity and exact-subtraction helpers for E6 LOTO transfer."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from e5_comparator_population import (
    fitted_frequency_gain,
    mean_std,
    solve_od_affine,
)


E6_LOTO_VERSION = "e6_loto_transfer_v1"


def load_tissue_annotation(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"slide_id", "tissue_type"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"geometry annotation must contain {sorted(required)}")
    pairs = {(str(row["slide_id"]), str(row["tissue_type"])) for row in rows}
    slide_to_tissues = {}
    for slide_id, tissue_type in pairs:
        slide_to_tissues.setdefault(slide_id, set()).add(tissue_type)
    if len(slide_to_tissues) != 109 or any(len(values) != 1 for values in slide_to_tissues.values()):
        raise ValueError("LOTO annotation must contain one tissue for each of 109 slides")
    mapping = {
        slide_id: next(iter(values)) for slide_id, values in slide_to_tissues.items()
    }
    tissues = sorted(set(mapping.values()))
    if len(tissues) != 37:
        raise ValueError(f"expected 37 tissue folds, got {len(tissues)}")
    return mapping, tissues


def heldout_tissue_indices(slide_ids, slide_id: str, tissue_by_slide: dict):
    normalized = [str(value) for value in slide_ids]
    if len(normalized) != 109 or normalized.count(str(slide_id)) != 1:
        raise ValueError(f"{slide_id}: absent or duplicated in sufficient statistics")
    if set(normalized) != set(tissue_by_slide):
        raise ValueError("sufficient-statistic and tissue-annotation slide sets differ")
    tissue = tissue_by_slide[str(slide_id)]
    indices = np.asarray(
        [index for index, value in enumerate(normalized) if tissue_by_slide[value] == tissue],
        dtype=int,
    )
    if len(indices) < 1 or len(indices) > 6:
        raise ValueError(f"{tissue}: invalid held-out tissue size {len(indices)}")
    return tissue, indices


def exact_tissue_image_parameters(
    statistics: dict,
    slide_id: str,
    tissue_by_slide: dict,
    gain_cap: float,
):
    tissue, indices = heldout_tissue_indices(
        statistics["slide_ids"], slide_id, tissue_by_slide
    )
    lab_sum = statistics["lab_sum"].sum(axis=0) - statistics["lab_sum"][indices].sum(axis=0)
    lab_square = statistics["lab_square_sum"].sum(axis=0) - statistics["lab_square_sum"][indices].sum(axis=0)
    lab_count = statistics["lab_count"].sum(axis=0) - statistics["lab_count"][indices].sum(axis=0)
    lab_stats = [
        mean_std(lab_sum[index], lab_square[index], int(lab_count[index]))
        for index in range(6)
    ]
    od_xtx = statistics["od_xtx"].sum(axis=0) - statistics["od_xtx"][indices].sum(axis=0)
    od_xty = statistics["od_xty"].sum(axis=0) - statistics["od_xty"][indices].sum(axis=0)
    od_coefficients = [
        solve_od_affine(od_xtx[index], od_xty[index]) for index in range(5)
    ]
    radial = statistics["radial_power"].sum(axis=0) - statistics["radial_power"][indices].sum(axis=0)
    frequency = statistics["radial_frequency"]
    frequency_gains = [
        fitted_frequency_gain(radial[index], radial[0], frequency, gain_cap)
        for index in range(1, 6)
    ]
    train_slides = 109 - len(indices)
    return {
        "lab": lab_stats,
        "od": od_coefficients,
        "frequency": frequency,
        "frequency_gain": frequency_gains,
        "heldout_tissue": tissue,
        "heldout_tissue_slides": int(len(indices)),
        "train_slides": int(train_slides),
    }

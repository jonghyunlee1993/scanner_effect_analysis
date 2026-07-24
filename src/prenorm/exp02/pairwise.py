"""Pairwise low-frequency transforms on registered scanner lattices."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from prenorm.exp02.projection import local_range_project


def apply_affine(low: torch.Tensor, affine: torch.Tensor) -> torch.Tensor:
    """Apply an RGB affine with bias to an ``[N,3,H,W]`` coarse tensor."""
    values = low.permute(0, 2, 3, 1)
    values = torch.cat([values, torch.ones_like(values[..., :1])], dim=-1)
    return (values @ affine.to(values)).permute(0, 3, 1, 2)


def solve_affine(
    source_low: torch.Tensor,
    target_low: torch.Tensor,
    ridge: float = 1e-4,
) -> torch.Tensor:
    """Fit a ``[4,3]`` RGB affine from paired coarse tensors."""
    source = source_low.permute(0, 2, 3, 1).double()
    target = target_low.permute(0, 2, 3, 1).double()
    design = torch.cat([source, torch.ones_like(source[..., :1])], dim=-1)
    x = design.reshape(-1, 4)
    y = target.reshape(-1, 3)
    regularizer = torch.diag(
        torch.tensor([ridge, ridge, ridge, 0.0], dtype=torch.float64)
    )
    return torch.linalg.solve(x.T @ x + regularizer, x.T @ y).float()


def load_lattice_batch(
    dataset,
    selection: Mapping,
    positions: Sequence[int],
) -> dict[str, torch.Tensor]:
    """Load all scanner images at the same selected lattice positions."""
    images = {}
    first_items = None
    for scanner_index, scanner in enumerate(selection["scanners"]):
        indices = selection["indices"][scanner]
        items = [dataset[indices[position]] for position in positions]
        images[scanner] = torch.stack([item["source"] for item in items])
        if scanner_index == 0:
            first_items = items
    if first_items is None:
        raise ValueError("a lattice selection must contain at least one scanner")
    images["at2"] = torch.stack([item["reference"] for item in first_items])
    return images


@torch.no_grad()
def fit_pairwise_affines(
    dataset,
    pyramid,
    selections: Mapping[str, Mapping],
    batch_size: int = 16,
    ridge: float = 1e-4,
) -> dict[str, dict[str, torch.Tensor]]:
    """Fit every directed scanner transform independently within each lattice."""
    fitted = {}
    for lattice_id, selection in selections.items():
        scanners = ("at2", *selection["scanners"])
        statistics = {
            f"{source}_to_{target}": {
                "xtx": torch.zeros(4, 4, dtype=torch.float64),
                "xty": torch.zeros(4, 3, dtype=torch.float64),
            }
            for source in scanners
            for target in scanners
            if source != target
        }
        location_count = len(selection["keys"])
        for start in range(0, location_count, batch_size):
            positions = list(range(start, min(start + batch_size, location_count)))
            images = load_lattice_batch(dataset, selection, positions)
            lows = {
                scanner: pyramid.coarsest(image).permute(0, 2, 3, 1).double()
                for scanner, image in images.items()
            }
            for source in scanners:
                design = torch.cat(
                    [lows[source], torch.ones_like(lows[source][..., :1])],
                    dim=-1,
                ).reshape(-1, 4)
                for target in scanners:
                    if source == target:
                        continue
                    response = lows[target].reshape(-1, 3)
                    stats = statistics[f"{source}_to_{target}"]
                    stats["xtx"] += design.T @ design
                    stats["xty"] += design.T @ response

        regularizer = torch.diag(
            torch.tensor([ridge, ridge, ridge, 0.0], dtype=torch.float64)
        )
        fitted[lattice_id] = {
            direction: torch.linalg.solve(
                stats["xtx"] + regularizer, stats["xty"]
            ).float()
            for direction, stats in statistics.items()
        }
    return fitted


def low_transform(
    pyramid,
    source: torch.Tensor,
    affine: torch.Tensor,
    steps: int,
) -> dict[str, torch.Tensor | list[torch.Tensor]]:
    """Transform only the low band and retain source detail coefficients."""
    low, bands = pyramid.decompose(source)
    corrected_unsafe = apply_affine(low, affine)
    projected = local_range_project(
        pyramid, source, corrected_unsafe, steps=steps
    )
    return {
        **projected,
        "source_low": low,
        "bands": bands,
    }


def clipped_low_transform(
    pyramid,
    source: torch.Tensor,
    affine: torch.Tensor,
) -> dict[str, torch.Tensor | list[torch.Tensor]]:
    """Apply the full affine low correction and make the composite operational.

    Unlike :func:`low_transform`, this policy never falls back toward the raw
    source. It reconstructs the requested affine low band with the source
    detail coefficients and explicitly clips the resulting RGB image to the
    valid input range. Clipping is part of the transform contract and can alter
    the image's re-decomposed low and detail coefficients, so callers must
    audit that distortion rather than call the output detail-exact.
    """
    low, bands = pyramid.decompose(source)
    affine_low = apply_affine(low, affine)
    preclip = pyramid.reconstruct(affine_low, bands)
    output = preclip.clamp(-1.0, 1.0)
    return {
        "output": output,
        "preclip": preclip,
        "source_low": low,
        "affine_low": affine_low,
        "bands": bands,
    }


def detail_retention_frontier(
    pyramid,
    coarse: torch.Tensor,
    bands: Sequence[torch.Tensor],
    retention: Sequence[float],
) -> dict[float, torch.Tensor]:
    """Reconstruct images while monotonically attenuating source detail."""
    return {
        float(value): pyramid.reconstruct(
            coarse, [float(value) * band for band in bands]
        )
        for value in retention
    }


def clipped_detail_retention_frontier(
    pyramid,
    coarse: torch.Tensor,
    bands: Sequence[torch.Tensor],
    retention: Sequence[float],
) -> dict[str, dict[float, torch.Tensor]]:
    """Reconstruct requested detail levels and explicitly clip every output."""
    preclip = detail_retention_frontier(
        pyramid, coarse, bands, retention
    )
    return {
        "preclip": preclip,
        "clipped": {
            value: image.clamp(-1.0, 1.0)
            for value, image in preclip.items()
        },
    }

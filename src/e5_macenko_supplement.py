"""Pure Macenko operations for the frozen E5 Supplement-only comparator."""

from __future__ import annotations

import numpy as np
import torch

from e5_comparator_population import clipping_report, od_to_rgb01, rgb01_to_od


MACENKO_VERSION = "e5_macenko_supplement_v1"
BETA = 0.15
ALPHA = 1.0
MIN_PIXELS = 50


def macenko_parameters(od_pixels: torch.Tensor):
    values = od_pixels.reshape(-1, 3).double()
    selected = values[torch.isfinite(values).all(dim=1) & (values.sum(dim=1) > BETA)]
    if len(selected) < MIN_PIXELS:
        return None
    centered = selected - selected.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / float(len(selected) - 1)
    try:
        _, eigenvectors = torch.linalg.eigh(covariance)
        plane = eigenvectors[:, -2:]
        projection = selected @ plane
        angles = torch.atan2(projection[:, 1], projection[:, 0])
        low = torch.quantile(angles, ALPHA / 100.0)
        high = torch.quantile(angles, 1.0 - ALPHA / 100.0)
        first = plane @ torch.stack([torch.cos(low), torch.sin(low)])
        second = plane @ torch.stack([torch.cos(high), torch.sin(high)])
        stains = torch.stack(
            [first, second] if first[0] >= second[0] else [second, first], dim=1
        )
        stains = stains / torch.linalg.vector_norm(stains, dim=0, keepdim=True).clamp_min(1e-8)
        # The OD directions should point into the positive octant on average.
        signs = torch.where(stains.sum(dim=0, keepdim=True) < 0, -1.0, 1.0)
        stains = stains * signs
        concentration = (selected @ torch.linalg.pinv(stains).T).T.clamp_min(0.0)
        maximum = torch.quantile(concentration, 0.99, dim=1).clamp_min(1e-4)
    except RuntimeError:
        return None
    if not torch.isfinite(stains).all() or not torch.isfinite(maximum).all():
        return None
    return stains.float(), maximum.float()


def aggregate_loso_reference(stains: np.ndarray, maximum: np.ndarray, heldout: int):
    stain_values = np.asarray(stains, dtype=np.float64)
    maximum_values = np.asarray(maximum, dtype=np.float64)
    valid = np.isfinite(stain_values).all(axis=(1, 2)) & np.isfinite(maximum_values).all(axis=1)
    valid[int(heldout)] = False
    if int(valid.sum()) < 100:
        raise ValueError("insufficient valid train-slide Macenko references")
    target_stains = stain_values[valid].mean(axis=0)
    target_stains /= np.maximum(np.linalg.norm(target_stains, axis=0, keepdims=True), 1e-8)
    target_maximum = np.exp(np.log(np.maximum(maximum_values[valid], 1e-8)).mean(axis=0))
    return target_stains.astype(np.float32), target_maximum.astype(np.float32), int(valid.sum())


def macenko_normalize_batch(
    rgb01: torch.Tensor,
    target_stains: torch.Tensor,
    target_maximum: torch.Tensor,
    stride: int = 8,
    offset: int = 4,
):
    outputs = []
    preclips = []
    fallback = []
    for image in rgb01:
        sampled = image[offset::stride, offset::stride]
        parameters = macenko_parameters(rgb01_to_od(sampled))
        if parameters is None:
            preclip = image
            fallback.append(True)
        else:
            source_stains, source_maximum = parameters
            od = rgb01_to_od(image).reshape(-1, 3).double()
            source_stains = source_stains.to(od).double()
            concentration = (od @ torch.linalg.pinv(source_stains).T).T.clamp_min(0.0)
            scaled = concentration * (
                target_maximum.to(concentration)[:, None]
                / source_maximum.to(concentration)[:, None].clamp_min(1e-4)
            )
            corrected_od = target_stains.to(concentration).double() @ scaled
            preclip = od_to_rgb01(corrected_od.T).reshape_as(image).float()
            fallback.append(False)
        preclips.append(preclip)
        outputs.append(preclip.clamp(0.0, 1.0))
    preclip = torch.stack(preclips)
    output = torch.stack(outputs)
    report = clipping_report(preclip, output)
    report["fallback"] = torch.as_tensor(fallback, dtype=torch.bool, device=rgb01.device)
    return report

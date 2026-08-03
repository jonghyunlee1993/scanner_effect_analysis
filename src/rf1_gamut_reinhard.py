"""Pure gamut-safe Reinhard operations for the RF1 image-only pilot."""

from __future__ import annotations

import torch

from e5_comparator_population import lab_to_rgb01, rgb01_to_lab


GAMUT_REINHARD_VERSION = "rf1_gamut_reinhard_source_ray_v1"


def source_ray_rgb_projection(
    source_rgb: torch.Tensor,
    proposed_rgb: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Project a proposed RGB correction along its ray from the valid source.

    A single scale is used for all three channels of a pixel. Unlike independent
    channel clipping, this preserves the direction of the proposed RGB change.
    """
    if source_rgb.shape != proposed_rgb.shape or source_rgb.shape[-1] != 3:
        raise ValueError("source and proposed RGB tensors must have equal channel-last shape")
    if not torch.isfinite(source_rgb).all() or not torch.isfinite(proposed_rgb).all():
        raise ValueError("gamut projection input contains non-finite values")
    tolerance = 2e-6
    if torch.any(source_rgb < -tolerance) or torch.any(source_rgb > 1.0 + tolerance):
        raise ValueError("source RGB must already lie in [0,1]")

    source = source_rgb.clamp(0.0, 1.0)
    delta = proposed_rgb - source
    infinity = torch.full_like(delta, torch.inf)
    positive_limit = torch.where(
        delta > 0.0,
        (1.0 - source) / delta.clamp_min(torch.finfo(delta.dtype).tiny),
        infinity,
    )
    negative_limit = torch.where(
        delta < 0.0,
        -source / delta.clamp_max(-torch.finfo(delta.dtype).tiny),
        infinity,
    )
    scale = torch.minimum(positive_limit, negative_limit).amin(dim=-1).clamp(0.0, 1.0)
    projected = source + scale[..., None] * delta
    output = projected.clamp(0.0, 1.0)

    reduce_dims = tuple(range(1, source.ndim - 1))
    rgb_reduce_dims = tuple(range(1, source.ndim))
    return {
        "output": output,
        "projected_rgb": projected,
        "scale": scale,
        "limited_pixel_fraction": (scale < 1.0 - 1e-7).float().mean(dim=reduce_dims),
        "mean_scale_loss": (1.0 - scale).mean(dim=reduce_dims),
        "projection_rgb_mae": (projected - proposed_rgb)
        .abs()
        .mean(dim=rgb_reduce_dims),
        "final_clamp_mae": (output - projected).abs().mean(dim=rgb_reduce_dims),
    }


def reinhard_lab_source_ray(
    rgb01: torch.Tensor,
    source_mean: torch.Tensor,
    source_std: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Apply Reinhard Lab moments followed by source-ray RGB gamut mapping."""
    lab = rgb01_to_lab(rgb01)
    corrected_lab = (lab - source_mean) / source_std.clamp_min(1e-3) * target_std + target_mean
    proposed_rgb = lab_to_rgb01(corrected_lab)
    report = source_ray_rgb_projection(rgb01, proposed_rgb)
    report["proposed_rgb"] = proposed_rgb
    report["corrected_lab"] = corrected_lab
    return report


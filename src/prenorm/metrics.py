"""Image metrics used by the active frequency analyses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def per_image_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window: int = 7,
) -> torch.Tensor:
    """Return the local-window RGB SSIM score for each NCHW image."""
    pad = window // 2
    mu_x = F.avg_pool2d(prediction, window, 1, pad)
    mu_y = F.avg_pool2d(target, window, 1, pad)
    var_x = F.avg_pool2d(prediction.square(), window, 1, pad) - mu_x.square()
    var_y = F.avg_pool2d(target.square(), window, 1, pad) - mu_y.square()
    covariance = (
        F.avg_pool2d(prediction * target, window, 1, pad) - mu_x * mu_y
    )
    c1, c2 = 0.02**2, 0.06**2
    score = ((2 * mu_x * mu_y + c1) * (2 * covariance + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2)
    ).clamp_min(1e-6)
    return score.flatten(1).mean(1)

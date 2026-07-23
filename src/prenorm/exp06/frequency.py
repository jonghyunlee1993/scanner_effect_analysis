"""Fixed perfect-reconstruction pyramid used by Exp-06.

The analysis low-pass is a separable binomial filter followed by decimation.
Perfect reconstruction is guaranteed algebraically: each Laplacian coefficient
is the difference between a level and the deterministic expansion of the next
coarser level.  Learned code may replace only the final coarse tensor; copied
coefficients are never recomputed inside the model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedLaplacianPyramid(nn.Module):
    """A fixed N-level pyramid with exact-by-construction synthesis."""

    def __init__(self, levels: int = 4):
        super().__init__()
        if int(levels) < 1:
            raise ValueError("levels must be positive")
        self.levels = int(levels)
        kernel_1d = torch.tensor([1, 4, 6, 4, 1], dtype=torch.float32) / 16.0
        self.register_buffer("kernel", kernel_1d[:, None] * kernel_1d[None, :])

    def reduce(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4:
            raise ValueError("pyramid input must be NCHW")
        if min(value.shape[-2:]) < 5:
            raise ValueError("pyramid level is too small for the fixed filter")
        weight = self.kernel.to(value).reshape(1, 1, 5, 5)
        weight = weight.expand(value.shape[1], 1, 5, 5)
        blurred = F.conv2d(F.pad(value, (2, 2, 2, 2), mode="reflect"), weight,
                           groups=value.shape[1])
        return blurred[:, :, ::2, ::2]

    @staticmethod
    def expand(value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(value, size=size, mode="bilinear", align_corners=False)

    def decompose(self, value: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        current = value
        bands: list[torch.Tensor] = []
        for _ in range(self.levels):
            coarse = self.reduce(current)
            bands.append(current - self.expand(coarse, current.shape[-2:]))
            current = coarse
        return current, bands

    def coarsest(self, value: torch.Tensor) -> torch.Tensor:
        for _ in range(self.levels):
            value = self.reduce(value)
        return value

    def reconstruct(self, coarse: torch.Tensor, bands: list[torch.Tensor]) -> torch.Tensor:
        if len(bands) != self.levels:
            raise ValueError(f"expected {self.levels} bands, got {len(bands)}")
        current = coarse
        for band in reversed(bands):
            current = self.expand(current, band.shape[-2:]) + band
        return current

    def copied_detail(self, value: torch.Tensor) -> torch.Tensor:
        """Return the full-resolution aggregate of every copied band."""
        coarse, bands = self.decompose(value)
        zero = torch.zeros_like(coarse)
        return self.reconstruct(zero, bands)

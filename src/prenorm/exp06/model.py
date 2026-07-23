"""Minimal context-conditioned coarse-band residual model for Exp-06."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .frequency import FixedLaplacianPyramid


class ContextSetEncoder(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, context: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if context.ndim != 5:
            raise ValueError("coarse context must be NKCHW")
        n, k = context.shape[:2]
        encoded = self.net(context.reshape(n * k, *context.shape[2:])).reshape(n, k, -1)
        weights = mask.to(encoded.dtype).unsqueeze(-1)
        return (encoded * weights).sum(1) / weights.sum(1).clamp_min(1.0)


class CoarseResidualNet(nn.Module):
    def __init__(self, width: int, residual_scale: float):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.context = ContextSetEncoder(width)
        self.input = nn.Conv2d(3 + width, width, 3, padding=1)
        self.body = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
                nn.Conv2d(width, width, 3, padding=1),
            ) for _ in range(3)
        ])
        self.output = nn.Conv2d(width, 3, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, query: torch.Tensor, context: torch.Tensor, mask: torch.Tensor):
        context_code = self.context(context, mask)
        broadcast = context_code[:, :, None, None].expand(-1, -1, *query.shape[-2:])
        value = F.silu(self.input(torch.cat([query, broadcast], dim=1)))
        for block in self.body:
            value = value + block(value)
        return self.residual_scale * torch.tanh(self.output(F.silu(value)))


class LowFrequencyHarmonizer(nn.Module):
    """Replace only the coarsest allowed component and copy every finer band."""

    def __init__(self, levels: int = 4, width: int = 32, residual_scale: float = 0.12):
        super().__init__()
        self.pyramid = FixedLaplacianPyramid(levels)
        self.correction = CoarseResidualNet(width, residual_scale)

    def context_coarse(self, context: torch.Tensor) -> torch.Tensor:
        n, k = context.shape[:2]
        low = self.pyramid.coarsest(context.reshape(n * k, *context.shape[2:]))
        return low.reshape(n, k, *low.shape[1:])

    def correct_coarse(self, coarse: torch.Tensor, context_coarse: torch.Tensor,
                       context_mask: torch.Tensor):
        residual = self.correction(coarse, context_coarse, context_mask)
        return coarse + residual, residual

    def correct_context_set(self, context_coarse: torch.Tensor,
                            context_mask: torch.Tensor):
        """Apply one pipeline pass to every member of its slide context set."""
        n, k = context_coarse.shape[:2]
        queries = context_coarse.reshape(n * k, *context_coarse.shape[2:])
        contexts = context_coarse[:, None].expand(
            n, k, k, *context_coarse.shape[2:]
        ).reshape(n * k, k, *context_coarse.shape[2:])
        masks = context_mask[:, None].expand(n, k, k).reshape(n * k, k)
        corrected, residual = self.correct_coarse(queries, contexts, masks)
        return (
            corrected.reshape(n, k, *corrected.shape[1:]),
            residual.reshape(n, k, *residual.shape[1:]),
        )

    def forward(self, source: torch.Tensor, context: torch.Tensor,
                context_mask: torch.Tensor, is_reference: torch.Tensor | None = None,
                clip: bool = True):
        if source.ndim != 4 or context.ndim != 5:
            raise ValueError("source/context must be NCHW/NKCHW")
        if context.shape[0] == 1 and source.shape[0] != 1:
            context = context.expand(source.shape[0], -1, -1, -1, -1)
            context_mask = context_mask.expand(source.shape[0], -1)
        coarse, bands = self.pyramid.decompose(source)
        context_low = self.context_coarse(context)
        corrected_low, residual = self.correct_coarse(coarse, context_low, context_mask)
        preclip = self.pyramid.reconstruct(corrected_low, bands)
        output = preclip.clamp(-1.0, 1.0) if clip else preclip
        if is_reference is not None:
            selector = is_reference.to(torch.bool).reshape(-1, 1, 1, 1)
            output = torch.where(selector, source, output)
            preclip = torch.where(selector, source, preclip)
            corrected_low = torch.where(selector, coarse, corrected_low)
            residual = torch.where(selector, torch.zeros_like(residual), residual)
        return {
            "output": output,
            "preclip": preclip,
            "source_low": coarse,
            "corrected_low": corrected_low,
            "residual": residual,
            "context_low": context_low,
            "copied_bands": bands,
        }
